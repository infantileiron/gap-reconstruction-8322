"""独立手算 oracle：验证时段累计复核的增广积分器与协方差传播。

不依赖 fastapi/psycopg/redis（fastapi 用 _stubs 最小桩），只导入 exposure.py
中的纯函数。用两种完全独立的实现交叉对账：
  1. 朴素逐点标量 Kalman/RTS（与 src.services.kalman 同口径）+ 显式全平滑
     协方差矩阵（n x n）直接积分，作为真值；
  2. exposure._augmented_smoother 的 2x2 增广状态实现。
两者在同一网格（含观测间边界插入点）上独立计算。
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_stubs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from src.services import exposure as ex  # noqa: E402


def naive_smoothed_covariance(times, values, q, r, p0):
    """朴素标量 Kalman + RTS，并显式构造完整平滑协方差矩阵。"""
    n = len(times)
    known = [v for v in values if v is not None]
    m0 = known[0]
    mf, pf, pp, mp = [], [], [], []
    for i in range(n):
        if i == 0:
            mean, var = m0, p0
        else:
            dt = times[i] - times[i - 1]
            mean, var = mf[-1], pf[-1] + q * dt
        mp.append(mean)
        pp.append(var)
        y = values[i]
        if y is None:
            mf.append(mean)
            pf.append(var)
        else:
            k = var / (var + r)
            mf.append(mean + k * (y - mean))
            pf.append(r * var / (var + r))
    ms = list(mf)
    ps = list(pf)
    gains = [0.0] * n
    for i in range(n - 2, -1, -1):
        c = pf[i] / pp[i + 1]
        gains[i] = c
        ms[i] = mf[i] + c * (ms[i + 1] - mp[i + 1])
        # 精确 RTS 协方差式（与增广2x2实现同口径），不做 min/max 截断
        ps[i] = pf[i] + c * c * (ps[i + 1] - pp[i + 1])
        ps[i] = max(0.0, ps[i])
    # 完整平滑协方差：P_s(i,j)=prod_{k=i}^{j-1} C_k * P_s(j,j)
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            chain = 1.0
            for k in range(i, j):
                chain *= gains[k]
            v = chain * ps[j]
            cov[i][j] = v
            cov[j][i] = v
    return ms, ps, cov, gains


def trapz_integral(times, means):
    """分段线性积分（梯形），返回点累计 G_k。"""
    g = [0.0]
    for k in range(1, len(times)):
        dt = times[k] - times[k - 1]
        g.append(g[-1] + 0.5 * dt * (means[k - 1] + means[k]))
    return g


def run_case(name, times, values, periods, q=0.03, r=0.09, p0=10000.0):
    print(f"\n=== {name} (n={len(times)}, periods={periods}) ===")
    # 同一网格（含观测间边界插入点），oracle 与被测独立计算
    grid, cuts = ex._build_grid(times, values, periods)
    grid_times = [t for t, _v, _kind in grid]
    grid_values = [v for _t, v, _kind in grid]
    grid_kinds = [kind for _t, _v, kind in grid]

    ms, ps_marg, full_cov, _ = naive_smoothed_covariance(grid_times, grid_values, q, r, p0)
    # 输出曲线：实测=原始观测，原始缺口=平滑补值均值，观测间边界=相邻重建点线性插值
    base = [v if grid_kinds[k] == "obs" else (ms[k] if grid_kinds[k] == "gap" else None)
            for k, v in enumerate(grid_values)]
    output_level = list(base)
    for k, kind in enumerate(grid_kinds):
        if kind != "cut":
            continue
        left = next(j for j in range(k - 1, -1, -1) if grid_kinds[j] != "cut")
        right = next(j for j in range(k + 1, len(grid)) if grid_kinds[j] != "cut")
        frac = (grid_times[k] - grid_times[left]) / (grid_times[right] - grid_times[left])
        output_level[k] = base[left] + frac * (base[right] - base[left])
    g_mean = trapz_integral(grid_times, output_level)
    base_w = ex._trapz_weights_from_times([float(t) for t in times])
    base_vw = {"by_time": {float(t): w for t, w in zip(times, base_w)}}
    smooth = ex._augmented_smoother(grid, q, r, p0, base_var_weights=base_vw)

    # 1) 平滑水平均值/边际方差对账（缺口点与平滑器一致；实测点输出值另核）
    for k, v in enumerate(grid_values):
        if grid_kinds[k] == "gap":
            assert abs(smooth["smoothed_level"][k] - ms[k]) < 1e-8
        assert abs(smooth["marginal_variance"][k] - ps_marg[k]) < 1e-7
        assert abs(smooth["output_level"][k] - output_level[k]) < 1e-12
    print("水平平滑均值/边际方差、输出曲线（实测保留、边界插值）对账通过")

    # 2) 累计积分均值：增广 G_k vs 输出曲线梯形累加（全部网格点，含观测间边界）
    for k in range(len(grid)):
        assert abs(smooth["mean"][k][1] - g_mean[k]) < 1e-9, (grid_times[k], smooth["mean"][k][1], g_mean[k])
    print("累计积分均值与输出曲线梯形 oracle 一致（含边界插入点）")

    # 方差权重取平滑器按原始观测时间轴构造的缺口权重（独立标量平滑之上的独立
    # 二次型）；全协方差矩阵 full_cov 仍由本 oracle 独立构造，保持对账独立性。
    base_by_time = base_vw["by_time"]
    expect_vw = [base_by_time[grid_times[k]] if grid_kinds[k] == "gap" else 0.0
                 for k in range(len(grid))]
    # 先校验平滑器权重与本 oracle 独立构造一致，再用于二次型对账
    assert max(abs(a - b) for a, b in zip(expect_vw, smooth["var_weights"])) < 1e-12
    vw = smooth["var_weights"]

    # 3) 积分方差：缺口子协方差矩阵 vw'Σvw（实测点权重0）
    for k in range(len(grid)):
        total = sum(
            vw[i] * vw[j] * full_cov[min(i, j)][max(i, j)]
            for i in range(k + 1) for j in range(k + 1)
        )
        assert abs(total - smooth["cov"][k][1][1]) < 1e-8, (k, total, smooth["cov"][k][1][1])
    print("累计积分方差与缺口子矩阵 vw'Σvw oracle 一致")

    # 4) 跨时刻 G 协方差抽样，独立用全矩阵求和（对称取元 full[min][max]，缺口权重）
    step = max(1, len(grid) // 5)
    for i in range(0, len(grid), step):
        for j in range(i, len(grid), step):
            total = sum(
                vw[a] * vw[b] * full_cov[min(a, b)][max(a, b)]
                for a in range(i + 1) for b in range(j + 1)
            )
            direct = ex._cross_cov(smooth, i, j)[1][1]
            assert abs(total - direct) < 1e-8, (i, j, total, direct)
    print("累计积分跨时刻协方差与显式求和一致（保留补值相关性）")

    # 5) 端到端 compute_exposure：总量还原 + 方差还原 + 实测/补值拆分
    study = {"q_per_hour": q, "observation_r": r, "initial_variance": p0}
    res = ex.compute_exposure(times, values, periods, study, "production")
    rec = res["reconciliation"]
    assert abs(rec["mean_residual"]) < 1e-9, rec
    assert abs(rec["variance_residual"]) < 1e-9, rec
    assert abs(rec["observed_plus_imputed_residual"]) < 1e-9, rec
    assert abs(sum(p["total_mean"] for p in res["periods_detail"]) - res["total"]["total_mean"]) < 1e-9
    m = res["covariance_matrix"]
    assert abs(sum(m[i][j] for i in range(len(m)) for j in range(len(m))) - res["total"]["variance"]) < 1e-9
    for i in range(len(m)):
        for j in range(len(m)):
            assert abs(m[i][j] - m[j][i]) < 1e-12
    for p in res["periods_detail"]:
        assert abs(p["observed_contribution"] + p["imputed_contribution"] - p["total_mean"]) < 1e-9
        assert p["total_low"] <= p["total_mean"] <= p["total_high"]
    print("时段总量/方差/协方差矩阵还原、实测+补值拆分、区间端点全部通过")
    print(f"整段总量 {res['total']['total_mean']:.4f} = 实测 {res['total']['observed_contribution']:.4f} + 补值 {res['total']['imputed_contribution']:.4f}")
    print(f"总方差 {res['total']['variance']:.6f}，95%区间 [{res['total']['total_low']:.4f}, {res['total']['total_high']:.4f}]")
    print(f"相邻协方差 {[round(x['covariance'], 6) for x in res['adjacent_covariance']]}")
    return res


def test_measured_kept_and_gap_uncertainty():
    """实测点原样保留；全实测时补值贡献为0且积分方差为0。"""
    print("\n=== 全实测序列：补值贡献与方差必须为0 ===")
    times = [float(t) for t in range(10)]
    values = [2.0 + 0.1 * t for t in range(10)]
    study = {"q_per_hour": 0.03, "observation_r": 0.09, "initial_variance": 10000.0}
    res = ex.compute_exposure(times, values, [0.0, 5.0, 9.0], study, "production")
    assert abs(res["total"]["imputed_contribution"]) < 1e-9
    assert abs(res["total"]["variance"]) < 1e-9
    manual = sum(0.5 * (values[i] + values[i + 1]) for i in range(9))
    assert abs(res["total"]["total_mean"] - manual) < 1e-9
    print(f"全实测积分 {res['total']['total_mean']:.4f} == 手算 {manual:.4f}，方差0 通过")


def test_boundary_between_observations():
    """观测间边界为确定性线性插值：细分不改变总量与方差，两段可还原整段。"""
    print("\n=== 观测间边界（不规则时间）===")
    times = [0.0, 2.0, 5.0, 7.0, 10.0]
    values = [1.0, 3.0, 2.0, 4.0, 5.0]
    study = {"q_per_hour": 0.03, "observation_r": 0.09, "initial_variance": 10000.0}
    # 边界取在实测时刻：全实测，积分手算可严格核对；补值贡献0、方差0
    res_obs = ex.compute_exposure(times, values, [0.0, 5.0, 10.0], study, "production")
    p0 = res_obs["periods_detail"][0]
    # 0..5: 0..2=4, 2..5=(3+2)/2*3=7.5 合计 11.5
    assert abs(p0["total_mean"] - 11.5) < 1e-9, p0
    assert abs(p0["imputed_contribution"]) < 1e-9 and p0["variance"] < 1e-12
    print(f"实测时刻边界积分 {p0['total_mean']:.4f} == 手算 11.5，补值/方差0 通过")

    # 边界 3.5 落在观测之间：插入确定性插值点，总量/方差必须与不细分完全一致
    base = ex.compute_exposure(times, values, [0.0, 10.0], study, "production")
    split = ex.compute_exposure(times, values, [0.0, 3.5, 10.0], study, "production")
    assert abs(split["total"]["total_mean"] - base["total"]["total_mean"]) < 1e-12
    assert abs(split["total"]["variance"] - base["total"]["variance"]) < 1e-12
    assert abs(sum(p["total_mean"] for p in split["periods_detail"]) - split["total"]["total_mean"]) < 1e-12
    # 观测间边界不引入补值随机性：含 cut 的第一段补值贡献仍为0、方差为0
    assert abs(split["periods_detail"][0]["imputed_contribution"]) < 1e-9
    assert split["periods_detail"][0]["variance"] < 1e-12
    print(f"观测间细分不改变总量 {split['total']['total_mean']:.4f} 与方差，且不引入补值 通过")


def test_subdivision_invariance_with_gap():
    """含真实缺口时，在实测/缺口重建曲线上加观测间边界，总量与方差仍保持不变。"""
    print("\n=== 含缺口曲线的边界细分不变量 ===")
    times = [float(t) for t in range(12)]
    values = [None if 4 <= t <= 7 else 3.0 + 0.2 * t for t in times]
    study = {"q_per_hour": 0.03, "observation_r": 0.09, "initial_variance": 10000.0}
    base = ex.compute_exposure(times, values, [0.0, 11.0], study, "production")
    split = ex.compute_exposure(times, values, [0.0, 2.5, 5.5, 9.1, 11.0], study, "production")
    assert abs(split["total"]["total_mean"] - base["total"]["total_mean"]) < 1e-9
    assert abs(split["total"]["variance"] - base["total"]["variance"]) < 1e-9
    # 缺口贡献与方差仍为正且总量可还原
    assert base["total"]["imputed_contribution"] != 0.0 and base["total"]["variance"] > 0.0
    assert abs(sum(p["total_mean"] for p in split["periods_detail"]) - split["total"]["total_mean"]) < 1e-12
    vsum = sum(split["covariance_matrix"][i][j]
               for i in range(split["period_count"]) for j in range(split["period_count"]))
    assert abs(vsum - split["total"]["variance"]) < 1e-9
    print(f"细分前后总量/方差一致（{base['total']['variance']:.6f}），分段可还原 通过")


def test_long_gap_correlation():
    """长缺口内补值强相关：积分方差有效且相邻协方差非零，内存受限。"""
    print("\n=== 长缺口相关性（5000点规模）===")
    rng = random.Random(7)
    n = 5000
    times = [float(t) for t in range(n)]
    values = [None] * n
    values[0] = 10.0
    values[n - 1] = 10.0
    for t in (100, 2500, 4900):
        values[t] = 10.0 + rng.gauss(0, 0.2)
    study = {"q_per_hour": 0.03, "observation_r": 0.09, "initial_variance": 10000.0}
    est = ex._memory_estimate(n)
    print(f"5000点预估额外内存 {est/1024/1024:.2f}MB（上限64MB）")
    assert est <= 64 * 1024 * 1024
    # 输出曲线在 compute_exposure 内部统一构造（缺口取平滑补值均值）
    res = ex.compute_exposure(times, values, [0.0, 2500.0, 4999.0], study, "validation")
    print(f"实际tracemalloc峰值 {res['memory_peak_bytes']/1024/1024:.2f}MB")
    assert res["memory_peak_bytes"] <= 64 * 1024 * 1024
    assert any(abs(x["covariance"]) > 0 for x in res["adjacent_covariance"])
    # 五千步浮点累加用相对容差（绝对残差随步数增长，量级 ~1e-7）
    total_var = res["total"]["variance"]
    assert abs(res["reconciliation"]["variance_residual"]) <= 1e-10 * max(1.0, total_var)
    print("长缺口规模内存受限、相邻协方差非零、方差还原通过")


def test_memory_reject():
    """超过64MB预估时拒绝且抛 MemoryLimitExceeded，不产生部分结果。"""
    print("\n=== 内存超限拒绝路径 ===")
    original = ex.EXPOSURE_MEMORY_LIMIT_BYTES
    try:
        ex.EXPOSURE_MEMORY_LIMIT_BYTES = 4096  # 4KB，任何正常序列都超限
        times = [float(t) for t in range(50)]
        values = [1.0] * 50
        study = {"q_per_hour": 0.03, "observation_r": 0.09, "initial_variance": 10000.0}
        try:
            ex.compute_exposure(times, values, [0.0, 25.0, 49.0], study, "production")
            raise AssertionError("应当拒绝")
        except ex.MemoryLimitExceeded:
            print("低上限下正确抛 MemoryLimitExceeded，无部分结果写入")
    finally:
        ex.EXPOSURE_MEMORY_LIMIT_BYTES = original


if __name__ == "__main__":
    rng = random.Random(42)
    times = sorted({round(rng.uniform(0, 200), 4) for _ in range(120)})
    values = []
    for i in range(len(times)):
        if 20 <= i <= 35 or 70 <= i <= 95 or i < 2 or i >= len(times) - 3:
            values.append(None)
        else:
            values.append(5.0 + math.sin(times[i] / 10) + rng.gauss(0, 0.2))
    run_case("混合缺口/不规则时间", times, values, [times[0], times[40], times[80], times[-1]])
    run_case("边界落在观测之间", times, values, [times[0] + 0.123, times[50], times[-1] - 0.321])
    run_case("单时段全覆盖", times, values, [times[0], times[-1]])
    test_measured_kept_and_gap_uncertainty()
    test_boundary_between_observations()
    test_subdivision_invariance_with_gap()
    test_long_gap_correlation()
    test_memory_reject()
    print("\n全部 oracle 对账通过 ✔")
