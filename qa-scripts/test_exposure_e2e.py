"""端到端服务测试：内存守卫、缓存命中、容量淘汰、超限保留已有结果。

用临时 SQLite 仓库与最小 fastapi 桩，不依赖真实 PG/Redis/FastAPI 运行时。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_stubs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "exposure_e2e.sqlite")

from src.utils import repository as repo  # noqa: E402
from src.services import exposure as ex  # noqa: E402


class Input:
    def __init__(self, periods, mode="production"):
        self.periods = periods
        self.mode = mode


def make_run():
    import math
    import random
    rng = random.Random(5)
    n = 200
    times = [float(t) for t in range(n)]
    values = [None if 30 <= t <= 60 or 120 <= t <= 140 else 8.0 + rng.gauss(0, 0.3)
              for t in range(n)]
    study = {"q_per_hour": 0.03, "observation_r": 0.09, "initial_variance": 10000.0,
             "initial_level": None}
    # 直接用 statistics 风格的最小结果结构，仅暴露 exposure 所需字段
    from src.services.reconstruction import reconstruct
    prod = reconstruct(times, values, study)
    for original, p in zip(values, prod["series"]):
        p.update(
            original_value=original,
            is_gap=original is None,
            rts_output=original if original is not None else p["rts_mean"],
            pchip_output=original if original is not None else p["pchip"],
        )
    # validation：构造一个简单遮挡结果（目标 100..103 置空重算）
    target = set(range(100, 104))
    val_train = [None if t in target else v for t, v in enumerate(values)]
    from src.services.reconstruction import evaluate_mask
    mask = {"station": "LP-T", "target_times": [float(t) for t in target]}
    validation = evaluate_mask(
        [{"station": "LP-T", "time_hour": t, "value": v} for t, v in zip(times, values)],
        mask, study,
    )
    return {
        "station": "LP-T",
        "point_count": len(times),
        "observed_count": sum(v is not None for v in values),
        "missing_count": sum(v is None for v in values),
        "pchip_unfilled_gaps": sum(p["is_gap"] and p["pchip_output"] is None for p in prod["series"]),
        "study": study,
        "production": prod,
        "validation": validation,
        "checksum": "e2e-test-checksum",
        "limitations": [],
    }


def main():
    repo.init_db()
    result = make_run()
    run = {"id": "run-e2e", "study_id": "s1", "status": "completed", "result": result, "error": ""}
    repo.put("runs", run)
    user = {"email": "analyst@lupu.test"}

    # 1) 首次计算落库
    review1, cached1 = ex.review_run("run-e2e", Input([0.0, 100.0, 199.0]), user)
    assert cached1 is False
    rec = review1["reconciliation"]
    assert abs(rec["mean_residual"]) < 1e-9
    assert abs(rec["variance_residual"]) < 1e-9
    assert abs(rec["observed_plus_imputed_residual"]) < 1e-9
    print("首次复核：总量/方差/实测+补值对账残差≈0 通过")

    # 2) 相同时段重复查询命中缓存，不新增记录，query_count 增长
    before = len(ex.list_reviews("run-e2e"))
    review1b, cached1b = ex.review_run("run-e2e", Input([0.0, 100.0, 199.0]), user)
    assert cached1b is True and review1b["id"] == review1["id"]
    assert review1b["query_count"] == 2
    assert len(ex.list_reviews("run-e2e")) == before
    print("重复查询命中缓存、计数增长、不新增 通过")

    # 3) validation 模式独立缓存键
    review_v, cached_v = ex.review_run("run-e2e", Input([0.0, 100.0, 199.0], "validation"), user)
    assert cached_v is False and review_v["mode"] == "validation"
    assert len(ex.list_reviews("run-e2e")) == before + 1
    print("遮挡复核独立训练、独立缓存键 通过")

    # 4) 内存超限拒绝（413），已有复核结果原样保留
    original_limit = ex.EXPOSURE_MEMORY_LIMIT_BYTES
    existing_ids = {r["id"] for r in ex.list_reviews("run-e2e")}
    try:
        ex.EXPOSURE_MEMORY_LIMIT_BYTES = 2048
        try:
            ex.review_run("run-e2e", Input([0.0, 50.0, 199.0]), user)
            raise AssertionError("应当拒绝超限请求")
        except Exception as http_exc:
            assert getattr(http_exc, "status_code", None) == 413, http_exc
        after_ids = {r["id"] for r in ex.list_reviews("run-e2e")}
        assert after_ids == existing_ids, "超限不得写入或删除任何已有结果"
        print("超限返回413且已有复核结果完整保留 通过")
    finally:
        ex.EXPOSURE_MEMORY_LIMIT_BYTES = original_limit

    # 5) 容量淘汰：把上限调小，新增后按最早创建淘汰
    original_cap = ex.EXPOSURE_REVIEW_LIMIT
    try:
        ex.EXPOSURE_REVIEW_LIMIT = 2
        # 当前已有2条（prod + val），再新增2条不同周期触发淘汰
        ex.review_run("run-e2e", Input([0.0, 60.0, 199.0]), user)
        ex.review_run("run-e2e", Input([0.0, 90.0, 199.0]), user)
        kept = ex.list_reviews("run-e2e")
        assert len(kept) == 2, len(kept)
        print("容量上限淘汰到2条、反复查询不无限增长 通过")
    finally:
        ex.EXPOSURE_REVIEW_LIMIT = original_cap

    print("\n端到端服务测试全部通过 ✔")


if __name__ == "__main__":
    main()
