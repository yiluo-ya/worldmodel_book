"""实验 05：世界模型存在的意义——在脑内规划（MPC）

对应章节：第 5 章（世界模型 × 规划）、第 8 章（模型预测控制）、第 16 章（应用）

任务：小车-杆从「倾斜 0.35 rad + 小车偏 0.3 m」恢复直立并保持 2 s。

规划器统一用 CEM（交叉熵法）：采样一批动作序列 → 在模型里脑内推演 →
留下代价最低的一批 → 更新采样分布 → 迭代 4 代 → 取最优序列。每 4 步重规划。

六个控制器，同一个任务、同一个时钟、同一个随机种子：

    random        随机乱推（下界）
    linear        最小二乘线性模型
    mlp           神经网络模型（输入噪声增强）
    wrong-model   同一个网络，但训练数据的 dt 差 4 倍（"时间尺度没对齐"）
    analytic      手写白盒动力学（完美模型，等于 MuJoCo 本身）
    true          MuJoCo 逐样本前向（作为 analytic 的校验，非常慢，可选）

这个表就是整本书的论点：**规划质量的实际上限 = 模型在规划视界内的预测质量**。
模型错了，MPC 不会犹豫，它会非常自信地走向灾难。

附加：--task swingup 把初始角换成 θ=π（倒立起步）。你会看到即使完美模型
配 0.3 s 视界也摆不起来——短视界规划搞不定长时程任务，这正是后来
「模型 + 价值函数」路线（Dreamer）要补的东西。

跑法：
    ../../../.venv/Scripts/python.exe exp05_mpc_planning.py
    ../../../.venv/Scripts/python.exe exp05_mpc_planning.py --task swingup --no-true
"""

from __future__ import annotations

import argparse
import time
import numpy as np

import wm_lib as L
import wm_nn as N
from wm_cartpole import AnalyticCartPole

H_SIM = 200               # 200 × 0.01 s = 2 s
SUBSTEPS = 5              # 控制周期 dt = timestep(0.002) × 5 = 0.01 s
REPLAN_EVERY = 4
H_PLAN = 30               # 规划视界 30 × 0.01 = 0.3 s
U_MAX = 30.0

W_TH, W_X, W_XD, W_U = 4.0, 0.5, 0.02, 0.001
W_TERM_TH, W_TERM_X = 20.0, 2.0


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def stage_cost(S: np.ndarray, U: np.ndarray) -> np.ndarray:
    return (W_TH * wrap(S[..., 1]) ** 2 + W_X * S[..., 0] ** 2
            + W_XD * S[..., 2] ** 2 + W_U * (U ** 2).sum(-1))


def term_cost(S: np.ndarray) -> np.ndarray:
    return W_TERM_TH * wrap(S[:, 1]) ** 2 + W_TERM_X * S[:, 0] ** 2


def cem_plan(predict_batch, s0, rng, horizon=H_PLAN, n_samp=160, n_elite=16, iters=4):
    mu = np.zeros(horizon)
    sd = np.ones(horizon) * U_MAX * 0.5
    s0 = np.asarray(s0, float)[None]

    def score(U):
        s = np.repeat(s0, len(U), axis=0)
        c = np.zeros(len(U))
        for t in range(horizon):
            s = np.atleast_2d(predict_batch(s, U[:, t:t + 1]))
            c += stage_cost(s, U[:, t:t + 1])
        return c + term_cost(s)

    for _ in range(iters):
        U = np.clip(rng.normal(mu, sd, (n_samp, horizon)), -U_MAX, U_MAX)
        elite = U[np.argsort(score(U))[:n_elite]]
        mu, sd = elite.mean(0), elite.std(0) + U_MAX * 0.02
    U = np.clip(rng.normal(mu, sd, (n_samp, horizon)), -U_MAX, U_MAX)
    return U[int(np.argmin(score(U)))]


# ---------------------------------------------------------------------------
# 训练数据与模型
# ---------------------------------------------------------------------------
def collect_data(substeps: int, n_episodes=120, steps=200, seed=0):
    """采一份随机交互数据。substeps 决定这份数据的 dt。

    任务需要覆盖这些工况，就先造出这些工况的数据：世界模型的能力边界
    由数据决定，不能凭空外推（第 15 章）。
    """
    model, data = L.load("cartpole")
    rng = np.random.default_rng(seed)
    S, A, SN = [], [], []
    for _ in range(n_episodes):
        s = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-1.5, 1.5),
                      rng.uniform(-0.5, 0.5), rng.uniform(-1, 1)])
        L.set_state(model, data, s)
        for _ in range(steps):
            u = np.array([rng.uniform(-U_MAX, U_MAX)])
            s2 = L.step(model, data, u, substeps=substeps)
            S.append(s); A.append(u); SN.append(s2)
            s = s2
    return {"s": np.array(S), "a": np.array(A), "s_next": np.array(SN)}


def train_mlp(ds, noise=0.05, epochs=40, seed=0):
    S, A, SN = ds["s"], ds["a"], ds["s_next"]
    return N.MLPModel(n_in=S.shape[1] + A.shape[1], n_out=S.shape[1], hidden=96,
                      depth=2, lr=3e-3, seed=seed).fit(
        S, A, SN, epochs=epochs, batch=256, seed=seed, verbose=False, noise=noise)


ANA = AnalyticCartPole(dt=0.002 * SUBSTEPS, substeps=SUBSTEPS)


def _true_predict(S, A):
    """MuJoCo 逐样本前向。慢，只用来验证解析模型。"""
    import mujoco
    m, d = L.load("cartpole")
    out = np.empty_like(S)
    for i in range(len(S)):
        L.set_state(m, d, S[i])
        d.ctrl[:] = A[i]
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        out[i] = L.get_state(m, d)
    return out


MODELS = {}


def run_mpc(kind: str, theta0: float, seed: int = 0, x0: float = 0.3):
    model, data = L.load("cartpole")
    s = np.array([x0, theta0, 0.0, 0.0])
    L.set_state(model, data, s)
    rng = np.random.default_rng(seed)
    kw = dict(n_samp=32, n_elite=6, iters=2) if kind == "true" else {}
    traj = [s.copy()]
    plan = np.zeros(H_PLAN)
    for t in range(H_SIM):
        if kind == "random":
            u = rng.uniform(-U_MAX, U_MAX)
        else:
            if t % REPLAN_EVERY == 0:
                plan = cem_plan(MODELS[kind], s, rng, **kw)
            u = plan[t % REPLAN_EVERY]
        s = L.step(model, data, np.array([float(np.clip(u, -U_MAX, U_MAX))]), substeps=SUBSTEPS)
        traj.append(s.copy())
    return np.array(traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["balance", "swingup"], default="balance")
    ap.add_argument("--with-true", action="store_true",
                    help="额外跑一遍逐样本 MuJoCo 前向（非常慢，用于校验 analytic）")
    args = ap.parse_args()
    theta0 = 0.35 if args.task == "balance" else np.pi

    ds = collect_data(SUBSTEPS)
    print(f"[-] 训练数据: {len(ds['s'])} 样本, dt={0.002*SUBSTEPS:.3f}s, "
          f"θ∈[{ds['s'][:,1].min():.2f},{ds['s'][:,1].max():.2f}]")

    lin = N.LinearModel().fit(ds["s"], ds["a"], ds["s_next"])
    mlp = train_mlp(ds)
    ds_wrong = collect_data(SUBSTEPS * 4, n_episodes=60)     # dt=0.04 的数据
    wrong = train_mlp(ds_wrong)

    MODELS.update({
        "linear": lin.predict,
        "mlp": mlp.predict,
        "wrong-model": wrong.predict,
        "analytic": ANA.step,
        "true": _true_predict,
    })

    order = ["random", "wrong-model", "linear", "mlp", "analytic"]
    if args.with_true:
        order.append("true")

    results, trajs = {}, {}
    for kind in order:
        t0 = time.time()
        traj = run_mpc(kind, theta0)
        dt = time.time() - t0
        trajs[kind] = traj
        tail = np.abs(wrap(traj[-30:, 1]))
        cost = float(stage_cost(traj[1:], np.zeros((len(traj) - 1, 1))).sum())
        ok = bool(tail.max() < 0.15)
        results[kind] = {"cost": cost, "tail_angle_mean": float(tail.mean()),
                         "tail_angle_max": float(tail.max()),
                         "cart_offset_end": float(abs(traj[-1, 0])),
                         "solved": ok, "wall_time_s": round(dt, 1)}
        print(f"  {kind:12s} cost={cost:8.1f}  末段|θ|={tail.mean():.3f} rad "
              f"(max {tail.max():.3f})  末位|x|={abs(traj[-1,0]):.2f} m  "
              f"{'[立住]' if ok else '[倒了]'}  [{dt:.0f}s]")

    L.plot("exp05_angles.png", [(k, wrap(trajs[k][:, 1])) for k in order],
           title=f"pole angle, task={args.task} (target: stay at 0)",
           xlabel="step (dt=0.01 s)", ylabel="θ [rad]")
    L.plot("exp05_cart.png", [(k, trajs[k][:, 0]) for k in order],
           title="cart position", xlabel="step (dt=0.01 s)", ylabel="x [m]")
    L.plot("exp05_cost.png", [("task cost", np.array([results[k]["cost"] for k in order]))],
           title="task cost by controller (lower = better)",
           xlabel="0=random 1=wrong 2=linear 3=mlp 4=analytic 5=true",
           ylabel="cost (log)", logy=True, x=np.arange(len(order)))

    L.json_dump(f"exp05_report_{args.task}.json", results)
    L.banner("exp05", [
        f"任务={args.task}；判定线：末 0.3 s 内 |θ| 始终 < 0.15 rad",
        *[f"  {k:12s} {'立住' if results[k]['solved'] else '倒了'}   "
          f"末段|θ|={results[k]['tail_angle_mean']:.3f}  cost={results[k]['cost']:.0f}"
          for k in order],
        "对比 wrong-model 与 mlp：网络一样，只是训练数据的 dt 差 4 倍",
        "对比 analytic 与 mlp：学到的模型离完美模型还差多远 = 剩余研究空间",
    ])


if __name__ == "__main__":
    main()
