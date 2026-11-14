"""时段累计暴露量复核。

质控组复核累计暴露量时，逐点区间无法反映长缺口对总量的影响。本模块沿重建
曲线按实际时间作分段线性（梯形）积分，支持把观测之间的时刻作为时段边界：
实测值原样保留，只有原始缺口的补值进入不确定度传播。

数学口径
--------
把到网格点 k 的累计积分 G_k = ∫_{t_0}^{t_k} m(t) dt 增广为第二维状态
z_k=(x_k, G_k)。局部水平模型在网格上仍为线性高斯：

  x_k = x_{k-1} + η,  η~N(0, q·dt)
  G_k = G_{k-1} + dt·(x_{k-1}+x_k)/2
      = G_{k-1} + dt·x_{k-1} + (dt/2)·η

实测点用原始观测更新；缺口点不更新，只把补值状态当作潜在随机量。增广滤波/
RTS 与逐点重建使用同一 q/r/初值配置，但训练数组独立构建：遮挡复核把掩码目标
置空后重新初始化，不复用生产状态。这样区间保留了补值之间的跨时刻相关性，
而不是把方差当作独立量相加。
"""
import math
import tracemalloc

from src.constants.domain import (
    INTERVAL_Z,
    EXPOSURE_MEMORY_LIMIT_BYTES,
    EXPOSURE_MEMORY_SAFETY_BYTES,
    EXPOSURE_REVIEW_LIMIT,
    EXPOSURE_MAX_PERIODS,
)
from src.utils import repository as repo
from src.utils.identity import new_id, now


class MemoryLimitExceeded(Exception):
    """单次累计复核预计或实际额外内存超过64MB硬上限。"""


# --------------------------------------------------------------------------- #
# 平滑小工具。状态顺序 (水平x, 累计积分G)。
#
# 不直接对增广 (x,G) 协方差做 2x2 求逆：在线性局部水平模型下 G 与 x 近共线，
# 先验协方差条件数可达数百万，通用求逆会数值崩溃。改为先用与 src.services.kalman
# 同口径的稳定标量 RTS 得到水平状态的平滑边际/跨时刻协方差，再把累计积分 G 作为
# 水平平滑状态的梯形线性组合 G_k=Σ_{j≤k} w_j·x_j 解析传播，全程无矩阵求逆。
# --------------------------------------------------------------------------- #
def _scalar_smoother(grid, q, r, initial_variance, initial_level=None):
    """标量局部水平 Kalman 滤波 + RTS，返回水平平滑均值、边际方差与RTS增益。

    口径与 src.services.kalman.kalman_rts 一致（含缺失仅预测、自动初值取首个
    有值观测），但额外保留每步预测方差与增益，供跨时刻协方差重建。
    """
    n = len(grid)
    known = [item[1] for item in grid if item[1] is not None]
    if not known:
        raise ValueError("没有训练观测，不能初始化累计复核")
    level0 = known[0] if initial_level is None else float(initial_level)
    mf, pf, pp, mp = [], [], [], []
    for k, (t, y, _kind) in enumerate(grid):
        if k == 0:
            mean, var = level0, initial_variance
        else:
            dt = t - grid[k - 1][0]
            mean, var = mf[-1], pf[-1] + q * dt
        mp.append(mean)
        pp.append(var)
        if y is None:
            mf.append(mean)
            pf.append(var)
        else:
            gain = var / (var + r)
            mf.append(mean + gain * (y - mean))
            pf.append(r * var / (var + r))
    ms = list(mf)
    ps = list(pf)
    gains = [0.0] * n
    for k in range(n - 2, -1, -1):
        c = pf[k] / pp[k + 1]
        gains[k] = c
        ms[k] = mf[k] + c * (ms[k + 1] - mp[k + 1])
        # 与生产代码相同的稳定非负分解，保证与逐点重建口径一致
        ps[k] = min(pf[k], max(0.0, pf[k] * (1.0 - c) + c * c * ps[k + 1]))
    return {"mean": ms, "marginal_variance": ps, "gains": gains}


def _trapz_weights_from_times(ts):
    """给定严格递增时间轴，返回每点在累计梯形积分中的权重。"""
    n = len(ts)
    w = [0.0] * n
    for k in range(n):
        if k > 0:
            w[k] += 0.5 * (ts[k] - ts[k - 1])
        if k < n - 1:
            w[k] += 0.5 * (ts[k + 1] - ts[k])
    return w


def _trapz_weights(grid):
    """每个网格点在到该点为止的累计梯形积分中的权重（含插入边界）。"""
    return _trapz_weights_from_times([t for t, _v, _kind in grid])


def _augmented_smoother(grid, q, r, initial_variance, initial_level=None, base_var_weights=None):
    """构建 (水平x, 累计积分G) 的平滑均值与完整 2x2 协方差（含跨时刻）。

    均值口径：积分沿输出曲线 m̂ 作分段线性（梯形）积分——实测点原样保留原始观测
    （确定性常数），原始缺口取 RTS 平滑补值均值，观测间边界取相邻重建点线性插值。
    不确定度口径：只有原始缺口（kind='gap'）的补值是随机潜在量，方差二次型只对
    缺口点赋权，实测/边界权重置0，因此全实测时段积分方差严格为0，区间不确定性
    完全归因于补值并保留其跨时刻相关性。

    base_var_weights：原始观测时间轴上每个时点的梯形权重（按时间映射进网格）。
    使用原始时间轴而非含插入边界的网格，保证缺口点的方差系数与边界细分无关——
    在曲线上插入观测间边界不会改变积分总量也不会改变方差。
    水平跨时刻协方差 P_s(a,b)=L(a,b)·Ps(b,b)（a≤b，L 为增益链）。
    """
    scalar = _scalar_smoother(grid, q, r, initial_variance, initial_level)
    ms_x = scalar["mean"]
    ps_x = scalar["marginal_variance"]
    gains = scalar["gains"]
    n = len(grid)
    weights = _trapz_weights(grid)
    kinds = [kind for _t, _v, kind in grid]

    # 输出水平 m̂：实测点原样保留原始观测，原始缺口取 RTS 平滑补值均值。
    base_level = [
        v if kind == "obs" else (ms_x[k] if kind == "gap" else None)
        for k, (_, v, kind) in enumerate(grid)
    ]
    # 观测间边界（cut）取相邻重建点的线性插值，保证梯形积分对边界细分严格不变：
    # 在一条分段线性曲线上插入分点，[a,c]+[c,b] 与 [a,b] 的积分完全相等。
    output_level = list(base_level)
    for k, kind in enumerate(kinds):
        if kind != "cut":
            continue
        left = next(j for j in range(k - 1, -1, -1) if kinds[j] != "cut")
        right = next(j for j in range(k + 1, n) if kinds[j] != "cut")
        t_l, t_r = grid[left][0], grid[right][0]
        frac = (grid[k][0] - t_l) / (t_r - t_l)
        output_level[k] = base_level[left] + frac * (base_level[right] - base_level[left])

    # 方差权重：仅原始缺口（gap）携带补值不确定性。权重取原始观测时间轴上的梯形
    # 系数（按时间映射），使方差对边界细分严格不变；cut/obs 权重为0。
    if base_var_weights is not None:
        bw_by_time = base_var_weights["by_time"]
        var_weights = [
            bw_by_time[grid[k][0]] if kind == "gap" else 0.0
            for k, kind in enumerate(kinds)
        ]
    else:
        var_weights = [weights[k] if kind == "gap" else 0.0 for k, kind in enumerate(kinds)]

    # 累计积分均值沿输出曲线 m̂ 梯形累加，与实测/补值贡献拆分严格自洽
    g_mean = [0.0] * n
    for k in range(1, n):
        dt = grid[k][0] - grid[k - 1][0]
        g_mean[k] = g_mean[k - 1] + 0.5 * dt * (output_level[k - 1] + output_level[k])

    mean = [(output_level[k], g_mean[k]) for k in range(n)]

    # 对角 2x2 块，全程 O(n)。方差二次型仅用缺口点的 var_weights。
    # 维护 H_k=Σ_{a≤k} vw_a L(a,k)，则 Cov(x_k,G_k)=H_k·Ps(k,k)；
    # Var(G_k) 按新点 k 的增量递推，与缺口子矩阵上的 vw'Σvw 全求和等价。
    h_prefix = 0.0
    var_g = 0.0
    cov = [((0.0, 0.0), (0.0, 0.0))] * n
    for k in range(n):
        vw = var_weights[k]
        if k == 0:
            h_prefix = vw
            var_g = vw * vw * ps_x[0]
        else:
            sum_chain = gains[k - 1] * h_prefix      # Σ_{a<k} vw_a L(a,k)
            var_g += vw * vw * ps_x[k] + 2.0 * vw * ps_x[k] * sum_chain
            h_prefix = vw + gains[k - 1] * h_prefix
        cov_xg = h_prefix * ps_x[k]
        var_g = max(0.0, var_g)
        cov[k] = ((ps_x[k], cov_xg), (cov_xg, var_g))

    return {
        "mean": mean,
        "output_level": output_level,
        "smoothed_level": ms_x,
        "cov": cov,
        "gains": gains,
        "weights": weights,
        "var_weights": var_weights,
        "marginal_variance": ps_x,
    }


def _cross_cov(smooth, i, j):
    """平滑状态在网格 i<=j 两个时刻的完整 2x2 跨时刻协方差（独立对账参考实现）。

    仅用于 qa-scripts 的全矩阵交叉验证，不是生产热路径：生产路径用 O(n) 的
    _cumulative_gg。水平跨协方差身份 P_s(a,b)=L(a,b)·Ps(b,b)（a≤b，L 为增益
    链乘），任意 a,b 取 lo=min、hi=max，含 G 的块据此逐项求和。
    """
    if i > j:
        return _mt(_cross_cov(smooth, j, i))
    gains = smooth["gains"]
    ps_x = smooth["marginal_variance"]
    # 含 G 的不确定度只加权缺口点（var_weights），实测点为确定性常数
    vw = smooth["var_weights"]

    def chain_to(a, b):
        """L(a,b)=∏_{t=a}^{b-1} C_t，a≤b。"""
        c = 1.0
        for t in range(a, b):
            c *= gains[t]
        return c

    if i == j:
        return smooth["cov"][i]
    c_xx = chain_to(i, j) * ps_x[j]

    # Cov(x_i, G_j)=Σ_{b≤j} vw_b P_s(i,b)，统一按 lo/hi 取元
    c_xg = 0.0
    for b in range(j + 1):
        if vw[b] == 0.0:
            continue
        lo, hi = (i, b) if i <= b else (b, i)
        c_xg += vw[b] * chain_to(lo, hi) * ps_x[hi]

    # Cov(G_i, x_j)=Σ_{a≤i} vw_a P_s(a,j)，j≥i 故恒为 L(a,j)·Ps(j)
    c_gx = ps_x[j] * sum(vw[a] * chain_to(a, j) for a in range(i + 1))

    # Cov(G_i, G_j)=Σ_{a≤i,b≤j} vw_a vw_b P_s(a,b)，按 lo/hi 取右端点边际。
    # 仅缺口点权重非零；cut 点数有界（≤128），O(i·j) 可接受且与全矩阵 oracle 同构。
    c_gg = 0.0
    for a in range(i + 1):
        if vw[a] == 0.0:
            continue
        wa = vw[a]
        for b in range(j + 1):
            if vw[b] == 0.0:
                continue
            lo, hi = (a, b) if a <= b else (b, a)
            c_gg += wa * vw[b] * chain_to(lo, hi) * ps_x[hi]
    return ((c_xx, c_xg), (c_gx, max(0.0, c_gg)))


def _mt(A):
    return ((A[0][0], A[1][0]), (A[0][1], A[1][1]))



def _build_grid(times, values, boundaries):
    """把时段边界并入观测时刻网格。

    返回 (grid, cuts)，grid 每项为 (time, value, kind)：kind='obs' 为原始实测、
    'gap' 为原始缺失（补值随机量）、'cut' 为落在观测之间的边界（确定性线性插值，
    不携带补值不确定性）。边界恰好等于观测时刻时保留实测，不重复加点。
    """
    t0, t1 = times[0], times[-1]
    cuts = sorted({float(b) for b in boundaries})
    if any(not math.isfinite(b) or b < t0 or b > t1 for b in cuts):
        raise ValueError("时段边界必须为有限值且落在重建时间范围内")
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError("时段边界必须严格递增且互不重叠")
    grid = []
    ti = 0
    nb = len(cuts)
    bi = 0
    while ti < len(times) or bi < nb:
        if bi < nb and (ti >= len(times) or cuts[bi] < times[ti]):
            grid.append((cuts[bi], None, "cut"))
            bi += 1
        elif ti < len(times) and (bi >= nb or times[ti] < cuts[bi]):
            v = values[ti]
            grid.append((float(times[ti]), v, "obs" if v is not None else "gap"))
            ti += 1
        else:
            # 边界恰好等于观测时刻：保留实测，不重复加点
            v = values[ti]
            grid.append((float(times[ti]), v, "obs" if v is not None else "gap"))
            ti += 1
            bi += 1
    return grid, cuts


def _cumulative_gg(smooth, query_points):
    """构建查询点上的累计积分协方差 Cov(G_i,G_j)，全程近 O(n+p^2)。

    先用 O(n) 递推得到每个网格点的累计积分方差 VG_k 与链前缀 H_k（仅缺口点
    方差权重非零），再对每个左查询点 i 用随 b 单乘增益的 K_b 线性扫描右点，
    避免任何 O(n^2) 双层求和。返回 dict[(i,j)]（i,j 为网格下标，i<=j）。
    """
    gains = smooth["gains"]
    ps_x = smooth["marginal_variance"]
    vw = smooth["var_weights"]
    n = len(gains)
    points = sorted(set(int(k) for k in query_points))

    # O(n) 链前缀 H_b=Σ_{a≤b} vw_a L(a,b) 与累计积分方差 VG_b
    h_prefix = [0.0] * n
    vg = [0.0] * n
    for b in range(n):
        v = vw[b]
        if b == 0:
            h_prefix[b] = v
            vg[b] = v * v * ps_x[0]
        else:
            prev_h = h_prefix[b - 1]
            h_prefix[b] = v + gains[b - 1] * prev_h
            vg[b] = (
                vg[b - 1]
                + v * v * ps_x[b]
                + 2.0 * v * ps_x[b] * gains[b - 1] * prev_h
            )
        vg[b] = max(0.0, vg[b])

    result = {}
    # 对每个左点 i 扫描其后的查询点 j；K_b=Σ_{a≤i} vw_a L(a,b)，b>i 时单乘增益
    for i in points:
        result[(i, i)] = vg[i]
        later = [j for j in points if j > i]
        if not later:
            continue
        k_acc = gains[i] * h_prefix[i]      # K_{i+1}=C_i·H_i
        acc = 0.0
        b = i + 1
        for j in later:
            while b <= j:
                if b > i + 1:
                    k_acc = gains[b - 1] * k_acc
                acc += vw[b] * ps_x[b] * k_acc
                b += 1
            result[(i, j)] = max(0.0, vg[i] + acc)
    return result


def _period_stats(smooth, grid, cuts, observed_integral):
    """计算每个时段的均值、补值/实测贡献、区间与时段间完整协方差。"""
    n = len(grid)
    # 网格点在 cuts 中的下标映射，梯形权重只依赖相邻网格时间差
    cut_index = {t: k for k, (t, _v, _kind) in enumerate(grid)}
    cut_k = [cut_index[float(b)] for b in cuts]

    # 一次性构建所有 cut 点对的累计 G 协方差（O(n+p^2)，内存有界）
    gg_lookup = _cumulative_gg(smooth, cut_k)

    def gg(i, j):
        return gg_lookup[(i, j)] if i <= j else gg_lookup[(j, i)]

    # G(a,b) 的均值与方差：累计积分状态之差
    rows = []
    period_var = []
    for idx in range(len(cuts) - 1):
        a, b = cut_k[idx], cut_k[idx + 1]
        mean_ab = smooth["mean"][b][1] - smooth["mean"][a][1]
        var_gg = smooth["cov"][a][1][1] + smooth["cov"][b][1][1] - 2.0 * gg(a, b)
        var_gg = max(0.0, var_gg)
        obs_part = observed_integral[idx]
        rows.append({
            "index": idx,
            "start_hour": cuts[idx],
            "end_hour": cuts[idx + 1],
            "duration_hour": cuts[idx + 1] - cuts[idx],
            "total_mean": mean_ab,
            "observed_contribution": obs_part,
            "imputed_contribution": mean_ab - obs_part,
            "variance": var_gg,
            "sd": math.sqrt(var_gg),
            "total_low": mean_ab - INTERVAL_Z * math.sqrt(var_gg),
            "total_high": mean_ab + INTERVAL_Z * math.sqrt(var_gg),
        })
        period_var.append(var_gg)

    # 时段积分两两协方差：G(a,b) 与 G(c,d) 的线性组合，系数为 ±1 的累计G差分
    pcount = len(rows)
    cov_matrix = [[0.0] * pcount for _ in range(pcount)]
    for i in range(pcount):
        ai, bi = cut_k[i], cut_k[i + 1]
        for j in range(i, pcount):
            aj, bj = cut_k[j], cut_k[j + 1]
            cov_ij = (
                gg(bi, bj)
                - gg(ai, bj)
                - gg(bi, aj)
                + gg(ai, aj)
            )
            cov_matrix[i][j] = cov_ij
            cov_matrix[j][i] = cov_ij
        cov_matrix[i][i] = period_var[i]
    return rows, cov_matrix, cut_k


def _observed_piece_integral(grid, cuts, outputs):
    """实测贡献的确定性梯形积分：实测点用原始输出，缺口点权重记到补值侧。

    outputs[k] 为该网格点参与积分的取值（生产模式实测点=原始观测，缺口=平滑
    均值；遮挡模式目标点视为缺口）。实测点固定、缺口点为随机潜在量，因此实测
    积分只累加两端均为实测的梯形块（整块 0.5·dt·(y_a+y_b)）；任一端为缺口时，
    该块全部进入补值侧。按 cuts 边界归属到各时段，边界点不重复计入。
    """
    n = len(grid)
    # 确定性点=实测(obs)与观测间边界(cut)；仅原始缺口(gap)进入补值侧。
    # cut 是相邻重建点的线性插值，不携带随机性，其相邻梯形块仍属确定性积分。
    deterministic = [kind != "gap" for _t, _v, kind in grid]
    # 每个梯形块 [k,k+1] 的确定性积分（两端均非真实缺口才计入），归属左边界所在时段
    block_obs = [0.0] * (n - 1)
    for k in range(n - 1):
        if deterministic[k] and deterministic[k + 1]:
            dt = grid[k + 1][0] - grid[k][0]
            block_obs[k] = 0.5 * dt * (outputs[k] + outputs[k + 1])
    totals = []
    for idx in range(len(cuts) - 1):
        lo, hi = cuts[idx], cuts[idx + 1]
        totals.append(sum(
            block_obs[k] for k in range(n - 1)
            if lo <= grid[k][0] < hi
        ))
    return totals


def _memory_estimate(n):
    """计算额外内存的保守预估（字节），覆盖含插入边界的最坏情形。

    按点分配的标量数组（增益/边际方差/权重/方差权重/输出与平滑水平/累计均值/
    H、VG 前缀等）约12条，加 mean/cov 二元组内浮点，合计约每点24个浮点；另计
    grid 三元组、cuts、查表字典与时段协方差矩阵，最后乘2.2倍计 Python 列表/
    元组/字典与 tracemalloc 计量开销，确保预估不低于实际峰值从而提前拦截。
    """
    floats = 0.0
    floats += n * 24.0   # 标量数组 + mean/cov 元组内浮点
    # 时段协方差矩阵（周期数受 EXPOSURE_MAX_PERIODS 约束，最坏 min(n, 上限)^2）
    floats += float(min(n, EXPOSURE_MAX_PERIODS) ** 2)
    floats += n * 6.0    # grid 三元组、cuts、查表字典等结构开销
    return int(floats * 8.0 * 2.2)


def compute_exposure(times, values, periods, study, mode):
    """执行一次时段累计复核。

    values: 训练数组（实测值保留、原始缺失为 None；遮挡模式目标已置空）。
    输出曲线（实测=原始观测、缺口=平滑补值、观测间边界=线性插值）在内部统一构造，
    调用方无需另传输出数组。mode 为 production / validation。
    """
    if len(times) != len(values) or len(times) < 2:
        raise ValueError("累计复核至少需要2个时点")
    if not periods or len(periods) < 2:
        raise ValueError("请提供至少两个边界，形成一个或多个连续时段")
    if len(periods) - 1 > EXPOSURE_MAX_PERIODS:
        raise ValueError(f"单次累计复核最多{EXPOSURE_MAX_PERIODS}个时段")
    periods = [float(x) for x in periods]
    if any(not math.isfinite(x) for x in periods):
        raise ValueError("时段边界必须为有限值")

    estimate = _memory_estimate(len(times))
    if estimate > EXPOSURE_MEMORY_LIMIT_BYTES - EXPOSURE_MEMORY_SAFETY_BYTES:
        raise MemoryLimitExceeded(
            f"本次累计复核预计额外内存约{estimate // 1024}KB，超过64MB上限，已拒绝；已有复核结果保留"
        )

    grid, cuts = _build_grid(times, values, periods)
    # 原始观测时间轴上的梯形权重（按时间映射），保证方差对边界细分不变
    base_weights = _trapz_weights_from_times([float(t) for t in times])
    base_var_weights = {"by_time": {float(t): w for t, w in zip(times, base_weights)}}
    tracemalloc.start()
    try:
        smooth = _augmented_smoother(
            grid,
            float(study["q_per_hour"]),
            float(study["observation_r"]),
            float(study["initial_variance"]),
            base_var_weights=base_var_weights,
        )
        grid_values = [item[1] for item in grid]
        # 输出曲线由平滑器统一构造：实测点=原始观测，缺口点=平滑补值均值
        grid_outputs = smooth["output_level"]
        observed_integral = _observed_piece_integral(grid, cuts, grid_outputs)
        rows, cov_matrix, cut_k = _period_stats(smooth, grid, cuts, observed_integral)
        _, peak = tracemalloc.get_traced_memory()
        if peak > EXPOSURE_MEMORY_LIMIT_BYTES:
            raise MemoryLimitExceeded(
                f"本次累计复核实际额外内存峰值{peak // 1024}KB，超过64MB上限，已拒绝；已有复核结果保留"
            )
    finally:
        tracemalloc.stop()

    # 整段总量：分段可还原
    total_mean = sum(r["total_mean"] for r in rows)
    total_observed = sum(r["observed_contribution"] for r in rows)
    total_imputed = sum(r["imputed_contribution"] for r in rows)
    pcount = len(rows)
    total_variance = sum(cov_matrix[i][j] for i in range(pcount) for j in range(pcount))
    total_variance = max(0.0, total_variance)
    # 独立路径对账：整段方差应等于首末累计积分状态之差的方差，
    # Var(G_b-G_a)=VG_a+VG_b-2Cov(G_a,G_b)，Cov 由 O(n) 累计协方差构建器给出。
    gg_whole = _cumulative_gg(smooth, [cut_k[0], cut_k[-1]])
    direct_var = max(
        0.0,
        smooth["cov"][cut_k[0]][1][1]
        + smooth["cov"][cut_k[-1]][1][1]
        - 2.0 * gg_whole[(min(cut_k[0], cut_k[-1]), max(cut_k[0], cut_k[-1]))],
    )

    adjacent = []
    for i in range(pcount - 1):
        adjacent.append({
            "left_index": i,
            "right_index": i + 1,
            "covariance": cov_matrix[i][i + 1],
            "correlation": (
                cov_matrix[i][i + 1] / math.sqrt(rows[i]["variance"] * rows[i + 1]["variance"])
                if rows[i]["variance"] > 0 and rows[i + 1]["variance"] > 0 else None
            ),
        })

    reconciliation = {
        "sum_period_mean": total_mean,
        "direct_total_mean": smooth["mean"][cut_k[-1]][1] - smooth["mean"][cut_k[0]][1],
        "mean_residual": total_mean - (smooth["mean"][cut_k[-1]][1] - smooth["mean"][cut_k[0]][1]),
        "sum_period_variance": total_variance,
        "direct_total_variance": direct_var,
        "variance_residual": total_variance - direct_var,
        "adjacent_covariance_sum": sum(x["covariance"] for x in adjacent),
        # 仅相邻协方差无法还原总量方差（非相邻时段也相关），明确给出该口径残差
        "adjacent_only_variance_residual": (
            total_variance - (sum(r["variance"] for r in rows) + 2.0 * sum(x["covariance"] for x in adjacent))
        ),
        "observed_plus_imputed_residual": total_mean - (total_observed + total_imputed),
    }

    return {
        "mode": mode,
        "periods": cuts,
        "period_count": pcount,
        "grid_point_count": len(grid),
        "memory_peak_bytes": peak,
        "memory_limit_bytes": EXPOSURE_MEMORY_LIMIT_BYTES,
        "z": INTERVAL_Z,
        "periods_detail": rows,
        "covariance_matrix": cov_matrix,
        "adjacent_covariance": adjacent,
        "total": {
            "start_hour": cuts[0],
            "end_hour": cuts[-1],
            "duration_hour": cuts[-1] - cuts[0],
            "total_mean": total_mean,
            "observed_contribution": total_observed,
            "imputed_contribution": total_imputed,
            "variance": total_variance,
            "sd": math.sqrt(total_variance),
            "total_low": total_mean - INTERVAL_Z * math.sqrt(total_variance),
            "total_high": total_mean + INTERVAL_Z * math.sqrt(total_variance),
        },
        "reconciliation": reconciliation,
    }


def _cache_key(periods, mode):
    return mode + "|" + ",".join(f"{float(x):.10g}" for x in periods)


def review_run(run_id, data, user):
    """对外的累计复核入口：内存守卫、去重缓存、容量淘汰，失败保留已有结果。"""
    run = repo.get("runs", run_id)
    if run["status"] != "completed":
        raise ValueError("只有已完成的运行可作时段累计复核")
    result = run["result"]
    periods = data.periods
    mode = data.mode
    if mode not in ("production", "validation"):
        raise ValueError("复核模式必须为 production 或 validation")

    reviews = result.setdefault("exposure_reviews", [])
    key = _cache_key(periods, mode)
    for old in reviews:
        if old.get("cache_key") == key:
            old["last_query_at"] = now()
            old["query_count"] = old.get("query_count", 1) + 1
            run["result"] = result
            repo.put("runs", run)
            return old, True

    if mode == "production":
        prod = result["production"]["series"]
        times = [p["time_hour"] for p in prod]
        # 生产训练数组保留全部原始观测；遮挡复核在 validation 分支独立训练
        values = [p["original_value"] for p in prod]
    else:
        val = result["validation"]["series"]
        times = [p["time_hour"] for p in val]
        # 遮挡目标在验证训练中已被置空，独立重算，不读生产状态
        values = [None if p.get("is_target") else p["training_value"] for p in val]

    try:
        detail = compute_exposure(times, values, periods, result["study"], mode)
    except MemoryLimitExceeded as exc:
        # 超限拒绝：不写任何新结果，已有复核与运行结果原样保留
        from fastapi import HTTPException
        raise HTTPException(status_code=413, detail=str(exc))

    review = {
        "id": new_id("exp"),
        "cache_key": key,
        "mode": mode,
        "created_at": now(),
        "last_query_at": now(),
        "query_count": 1,
        "created_by": user["email"],
        "station": result["station"],
        **detail,
    }
    reviews.append(review)
    # 容量淘汰：超过上限按最早创建时间移除，反复查询不导致载荷无限增长
    if len(reviews) > EXPOSURE_REVIEW_LIMIT:
        reviews.sort(key=lambda x: x["created_at"])
        removed = reviews[: len(reviews) - EXPOSURE_REVIEW_LIMIT]
        del reviews[: len(removed)]
    run["result"] = result
    repo.put("runs", run)
    return review, False


def list_reviews(run_id):
    run = repo.get("runs", run_id)
    return run.get("result", {}).get("exposure_reviews", [])
