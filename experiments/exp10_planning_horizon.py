"""实验 10：规划视界的选择——短了近视，长了被幻觉带偏

对应章节：第 5 章（复合误差）、第 8 章（MPC）、第 16 章（失败模式）

MPC 里有一个必调的超参：规划视界 H。它是"我往前想多远"。

- H 太小：看不见远处，只能局部救火（近视）
- H 太大：推演越长，模型误差越大，最优计划建立在幻觉上（远视）

本实验扫描 H ∈ {5, 10, 20, 40}，每个 H 用两种模型做规划：
    analytic  完美模型（MuJoCo 同源）
    mlp       从数据学来的模型
然后看真实环境里的任务表现。

预期能看到两条曲线：完美模型随 H 单调变好（想得远更好），
学到的模型在某个 H 之后开始变差（想得越远，幻觉越多）。

跑法：
    ../../../.venv/Scripts/python.exe exp10_planning_horizon.py
"""

import time
import numpy as np

import wm_lib as L
import wm_nn as N
from wm_cartpole import AnalyticCartPole

SUBSTEPS = 5
DT = 0.002 * SUBSTEPS
H_SIM = 200
REPLAN = 4
U_MAX = 30.0
TH0, X0 = 0.35, 0.3
HORIZONS = [5, 10, 20, 40, 80]

ANA = AnalyticCartPole(dt=DT, substeps=SUBSTEPS)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def stage_cost(S, U):
    return 4.0 * wrap(S[..., 1]) ** 2 + 0.5 * S[..., 0] ** 2 + 0.02 * S[..., 2] ** 2


def term_cost(S):
    return 20.0 * wrap(S[:, 1]) ** 2 + 2.0 * S[:, 0] ** 2


def cem(predict, s0, rng, H, n_samp=120, n_elite=12, iters=3):
    mu = np.zeros(H)
    sd = np.ones(H) * U_MAX * 0.5
    s0 = np.asarray(s0, float)[None]

    def score(U):
        s = np.repeat(s0, len(U), 0)
        c = np.zeros(len(U))
        for t in range(H):
            s = np.atleast_2d(predict(s, U[:, t:t + 1]))
            c += stage_cost(s, U[:, t:t + 1])
        return c + term_cost(s)

    for _ in range(iters):
        U = np.clip(rng.normal(mu, sd, (n_samp, H)), -U_MAX, U_MAX)
        el = U[np.argsort(score(U))[:n_elite]]
        mu, sd = el.mean(0), el.std(0) + U_MAX * 0.02
    U = np.clip(rng.normal(mu, sd, (n_samp, H)), -U_MAX, U_MAX)
    return U[int(np.argmin(score(U)))]


def run(predict, H, seed=0):
    m, d = L.load("cartpole")
    s = np.array([X0, TH0, 0.0, 0.0])
    L.set_state(m, d, s)
    rng = np.random.default_rng(seed)
    traj = [s.copy()]
    plan = np.zeros(H)
    for t in range(H_SIM):
        if t % REPLAN == 0:
            plan = cem(predict, s, rng, H)
        u = float(np.clip(plan[t % REPLAN], -U_MAX, U_MAX))
        s = L.step(m, d, np.array([u]), substeps=SUBSTEPS)
        traj.append(s.copy())
    traj = np.array(traj)
    tail = np.abs(wrap(traj[-30:, 1]))
    return {
        "cost": float(stage_cost(traj[1:], np.zeros((len(traj) - 1, 1))).sum()),
        "tail_max": float(tail.max()),
        "solved": bool(tail.max() < 0.15),
        "traj": traj,
    }


def main():
    print("[-] 训练世界模型")
    ds = L.collect("cartpole", n_episodes=80, steps=200, seed=0,
                   init_scale=0.4, action_scale=1.0, substeps=SUBSTEPS)
    tr, te = L.split(ds, 0.85)
    scale = tr["s"].std(0) + 1e-9
    mlp = N.MLPModel(n_in=tr["s"].shape[1] + 1, n_out=tr["s"].shape[1],
                     hidden=96, depth=2, lr=3e-3, seed=0)
    mlp.fit(tr["s"], tr["a"], tr["s_next"], epochs=30, batch=256, seed=0,
            verbose=False, noise=0.05)

    # 模型在各视界上的开环误差（解释后面规划结果的关键）
    print("\n[-] 模型的开环误差 vs 步数（说明误差在哪一步开始失控）")
    e = N.rollout_errors(mlp.predict, te["s"], te["a"], te["s_next"],
                         horizon=max(HORIZONS), n_start=80, ep_len=200, scale=scale)
    for H in HORIZONS:
        print(f"  H={H:3d} 步 ({H*DT:5.2f}s): 开环 nRMSE={e[H]:.4f}")

    rows = []
    print("\n[-] 规划视界扫描")
    for name, predict in [("analytic", ANA.step), ("mlp", mlp.predict)]:
        for H in HORIZONS:
            t0 = time.time()
            r = run(predict, H)
            dt_used = time.time() - t0
            rows.append({"model": name, "horizon": H, "horizon_s": H * DT,
                         "cost": r["cost"], "tail_max": r["tail_max"],
                         "solved": r["solved"], "wall_s": round(dt_used, 1),
                         "traj": r["traj"]})
            print(f"  {name:9s} H={H:3d} ({H*DT:5.2f}s)  cost={r['cost']:8.2f}  "
                  f"末段|θ|max={r['tail_max']:.3f}  "
                  f"{'立住' if r['solved'] else '倒了'}  [{dt_used:.0f}s]")

    L.plot("exp10_horizon.png", [
        ("analytic (perfect model)", np.array([r["cost"] for r in rows if r["model"] == "analytic"])),
        ("mlp (learned model)", np.array([r["cost"] for r in rows if r["model"] == "mlp"])),
    ], title="task cost vs planning horizon (x-axis = H in steps: 5/10/20/40/80)",
       xlabel="planning horizon H", ylabel="real task cost (log)", logy=True,
       x=np.array(HORIZONS))

    L.plot("exp10_model_error.png", [("open-loop nRMSE", e[1:])],
           title="the model's own error grows with the horizon it is asked to imagine",
           xlabel="horizon (dt=0.01 s)", ylabel="normalized RMSE (log)", logy=True)

    for r in rows:
        if r["horizon"] in (5, 20, 80):
            L.plot(f"exp10_traj_{r['model']}_H{r['horizon']}.png", [
                ("pole angle [rad]", wrap(r["traj"][:, 1])),
                ("cart x [m]", r["traj"][:, 0]),
            ], title=f"{r['model']} model, planning horizon {r['horizon']}",
               xlabel="step (dt=0.01 s)", ylabel="value")

    best = {}
    for name in ("analytic", "mlp"):
        sub = [r for r in rows if r["model"] == name]
        best[name] = min(sub, key=lambda r: (not r["solved"], r["cost"]))
    L.json_dump("exp10_report.json", {
        "openloop": [float(x) for x in e],
        "rows": [{k: v for k, v in r.items() if k != "traj"} for r in rows],
        "best": {k: {"horizon": v["horizon"], "cost": v["cost"],
                     "solved": v["solved"]} for k, v in best.items()},
    })
    L.banner("exp10", [
        "完美模型：视界越长越好（想得远就能提前减速、提前回正）",
        f"  最优 H = {best['analytic']['horizon']}，代价 {best['analytic']['cost']:.2f}",
        "学到的模型：视界一长就崩（在幻觉里规划）",
        f"  最优 H = {best['mlp']['horizon']}，代价 {best['mlp']['cost']:.2f}",
        "工程结论：规划视界要设成'模型误差还小'的那个长度，再加一点前瞻",
        "  经验法则：H ≈ 模型开环误差还低于 0.1 的步数",
    ])


if __name__ == "__main__":
    main()
