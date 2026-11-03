"""实验 08：世界模型的杀手级用途——在"想象"里训练策略

对应章节：第 5 章（世界模型 × 策略）、第 11 章（在想象中学习 / Dreamer）

任务：cartpole 平衡（从倾斜 0.35 rad + 小车偏 0.3 m 恢复，保持 2 s）。

三种拿到策略的办法，比较它们的**真实世界交互次数**：

  A. random-search    暴力随机试：采一堆常数增益的 PD 参数，在真环境里跑，留最好的
  B. model-free       随机试参数但迭代（进化策略 ES）：在真环境里反复评估
  C. imagination      **在模型里**优化参数：真环境只用一次做最终验收

这是世界模型之所以重要的全部理由：把试错成本从真实世界搬到模型里。
代价是模型不准时，你会在想象的幻觉里选中一个"看起来很棒的烂策略"，
所以我们同时跑两个想象策略：
    C1: 用真实物理做 rollout（完美的"世界模型"，即上界）
    C2: 用学到的 MLP 世界模型做 rollout
如果 C2 的表现明显差于 C1，差额就是"模型误差税"。

跑法：
    ../../../.venv/Scripts/python.exe exp08_imagination_policy.py
"""

from __future__ import annotations

import time
import numpy as np

import wm_lib as L
import wm_nn as N
from wm_cartpole import AnalyticCartPole

SUBSTEPS = 5
DT = 0.002 * SUBSTEPS
H_SIM = 200
U_MAX = 30.0
TH0, X0 = 0.35, 0.3

ANA = AnalyticCartPole(dt=DT, substeps=SUBSTEPS)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def pd_action(s: np.ndarray, k: np.ndarray) -> np.ndarray:
    """线性状态反馈：u = k · [1, x, θ, ẋ, θ̇]，k 是 5 维。

    只搜 5 个数字，是个"小策略族"——但足够演示核心矛盾。
    """
    feat = np.concatenate([[1.0], s[[0, 1, 2, 3]]])
    return np.clip(feat @ k, -U_MAX, U_MAX)


def task_cost(states: np.ndarray) -> float:
    """把轨迹折成一个数：末段角度 + 小车位置 + 控制平滑。"""
    tail = states[-40:]
    return float(np.mean(wrap(tail[:, 1]) ** 2) + 0.5 * np.mean(tail[:, 0] ** 2)
                 + 0.001 * np.mean(tail[:, 2] ** 2))


# ---------------------------------------------------------------------------
# 三种优化器：都在"某个动力学"里评估策略
# ---------------------------------------------------------------------------
def rollout_true(k: np.ndarray, seed: int = 0) -> np.ndarray:
    """在 MuJoCo（真实世界）里跑一条轨迹。"""
    m, d = L.load("cartpole")
    s = np.array([X0, TH0, 0.0, 0.0])
    L.set_state(m, d, s)
    traj = [s.copy()]
    for _ in range(H_SIM):
        s = L.step(m, d, np.array([pd_action(s, k)]), substeps=SUBSTEPS)
        traj.append(s.copy())
    return np.array(traj)


def rollout_ana(k: np.ndarray) -> np.ndarray:
    """在"完美但白盒"的世界模型里跑（解析动力学）。"""
    s = np.array([X0, TH0, 0.0, 0.0])
    traj = [s.copy()]
    for _ in range(H_SIM):
        s = ANA.step(s[None], np.array([[pd_action(s, k)]]))[0]
        traj.append(s.copy())
    return np.array(traj)


def rollout_model(predict, k: np.ndarray) -> np.ndarray:
    """在模型里跑。predict(s2d, a2d) -> s_next。每步单样本调用，慢一点无所谓。"""
    s = np.array([X0, TH0, 0.0, 0.0])
    traj = [s.copy()]
    for _ in range(H_SIM):
        a = np.array([[pd_action(s, k)]])
        s = np.atleast_2d(predict(s[None], a))[0]
        traj.append(s.copy())
    return np.array(traj)


def random_search(evaluate, n_try=60, seed=0):
    """纯随机采样（不迭代）：拿它当'暴力试错'的基线。"""
    rng = np.random.default_rng(seed)
    lo = np.array([-2.0, -2.0, -60.0, -4.0, -12.0])
    hi = np.array([2.0, 2.0, 60.0, 4.0, 12.0])
    best, best_c = None, np.inf
    for _ in range(n_try):
        k = rng.uniform(lo, hi)
        c = task_cost(evaluate(k))
        if c < best_c:
            best, best_c = k, c
    return best, best_c, n_try


def cem_policy(evaluate, n_iter=12, n_samp=40, n_elite=8, seed=0):
    """CEM 在参数空间上搜。返回 (最优 k, 代价, 评估次数)。"""
    rng = np.random.default_rng(seed)
    lo = np.array([-2.0, -2.0, -60.0, -4.0, -12.0])
    hi = np.array([2.0, 2.0, 60.0, 4.0, 12.0])
    mu = (lo + hi) / 2
    sd = (hi - lo) / 2
    best, best_c, n_eval = None, np.inf, 0
    for _ in range(n_iter):
        K = np.clip(rng.normal(mu, sd, (n_samp, len(mu))), lo, hi)
        cs = []
        for k in K:
            c = task_cost(evaluate(k))
            n_eval += 1
            cs.append(c)
            if c < best_c:
                best, best_c = k.copy(), c
        elite = K[np.argsort(cs)[:n_elite]]
        mu, sd = elite.mean(0), elite.std(0) + 1e-2 * (hi - lo)
    return best, best_c, n_eval


def main():
    # ---- 训练用的世界模型（粗糙即可，重点是"想象 vs 真实"的落差）
    print("[-] 采集数据并训练 MLP 世界模型")
    ds = L.collect("cartpole", n_episodes=80, steps=200, seed=0,
                   init_scale=0.4, action_scale=1.0, substeps=SUBSTEPS)
    tr, te = L.split(ds, 0.85)
    scale = tr["s"].std(0) + 1e-9
    mlp = N.MLPModel(n_in=tr["s"].shape[1] + 1, n_out=tr["s"].shape[1], hidden=96,
                     depth=2, lr=3e-3, seed=0)
    mlp.fit(tr["s"], tr["a"], tr["s_next"], epochs=30, batch=256, seed=0,
            verbose=False, noise=0.05)
    e = N.rollout_errors(mlp.predict, te["s"], te["a"], te["s_next"],
                         horizon=50, n_start=80, ep_len=200, scale=scale)
    print(f"  模型 50 步开环 nRMSE = {e[-1]:.3f}")

    # ---- 三种路线
    print("\n[-] 路线 A：真环境纯随机搜索")
    t0 = time.time()
    kA, cA_claimed, nA = random_search(rollout_true, n_try=60)
    tA = time.time() - t0

    print("[-] 路线 B：真环境 CEM（进化策略）")
    t0 = time.time()
    kB, cB_claimed, nB = cem_policy(rollout_true)
    tB = time.time() - t0

    print("[-] 路线 C1：完美模型里的 CEM（上界）")
    t0 = time.time()
    kC1, cC1_claimed, nC1 = cem_policy(rollout_ana)
    tC1 = time.time() - t0

    print("[-] 路线 C2：学到的模型里的 CEM")
    t0 = time.time()
    kC2, cC2_claimed, nC2 = cem_policy(lambda k: rollout_model(mlp.predict, k),
                                       n_iter=8, n_samp=24, n_elite=6)
    tC2 = time.time() - t0

    # ---- 最终验收：所有人都在真环境里跑一遍
    print("\n[-] 最终验收（真环境，每条策略只跑一次）")
    rows = []
    for name, k, n_imag, t_used in [
        ("A random-search", kA, nA, tA),
        ("B CEM on real", kB, nB, tB),
        ("C1 CEM perfect model", kC1, nC1, tC1),
        ("C2 CEM learned model", kC2, nC2, tC2),
    ]:
        traj = rollout_true(k)
        cost = task_cost(traj)
        tail = np.abs(wrap(traj[-40:, 1]))
        rows.append({
            "policy": name, "real_cost": cost,
            "tail_angle_max": float(tail.max()),
            "solved": bool(tail.max() < 0.15),
            "imagined_evals": int(n_imag),
            "wall_time_s": round(t_used, 1),
            "k": k.round(3).tolist(),
        })
        print(f"  {name:22s} 真实代价={cost:8.3f}  末段|θ|max={tail.max():.3f} "
              f"{'[立住]' if tail.max() < 0.15 else '[倒了]'}  "
              f"想象评估={n_imag:5d}  [{t_used:.1f}s]")

    # ---- 图：策略在真环境里的表现
    for name, k in [("A_random", kA), ("B_cem_real", kB),
                    ("C1_perfect", kC1), ("C2_learned", kC2)]:
        traj = rollout_true(k)
        L.plot(f"exp08_traj_{name}.png", [
            ("pole angle [rad]", wrap(traj[:, 1])),
            ("cart x [m]", traj[:, 0]),
        ], title=f"policy from {name}: evaluated on the real MuJoCo world",
           xlabel="step (dt=0.01 s)", ylabel="value")

    L.plot("exp08_cost_vs_evals.png", [
        ("real task cost", np.array([r["real_cost"] for r in rows])),
    ], title="real cost of each route (A / B / C1 / C2)",
       xlabel="route index", ylabel="cost (log)", logy=True, x=np.arange(len(rows)))

    L.json_dump("exp08_report.json", {
        "model_openloop_nrmse_h50": float(e[-1]),
        "routes": rows,
    })
    L.banner("exp08", [
        "核心对比：真环境评估次数。C 系列的评估几乎全在模型里完成",
        *[f"  {r['policy']:22s} 真实代价={r['real_cost']:8.3f} "
          f"{'立住' if r['solved'] else '倒了'}" for r in rows],
        "C2 与 C1 的差距 = 模型误差税：模型越准，想象越可信",
        "A 与 B 说明：真环境试错要花数量级的真实时间",
        "这就是为什么机器人公司愿意先做世界模型",
    ])


if __name__ == "__main__":
    main()
