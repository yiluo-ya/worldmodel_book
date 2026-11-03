"""实验 07：混沌系统——世界模型能预测多久，是有理论上限的

对应章节：第 7 章（复合误差与可预测视界）、第 13 章（物理一致性）

双摆是无控制、无阻尼、能量守恒的混沌系统（前提是给它足够能量去翻转，
小摆动只是准周期运动，并不混沌）。它给我们一个罕见的礼物：
**"应该能预测多久"这个问题的答案是算得出来的**。

两条初始状态相差 ε 的轨迹会以 e^(λt) 分离，λ 是最大 Lyapunov 指数。
分离量增长到系统尺度 L 时预测彻底失效：

    t_pred ≈ (1/λ) · ln(L/ε)

实验做三件事：

  1. 从 MuJoCo 实测 λ（两条几乎相同初始条件的轨迹，看对数分离速度）
  2. 训练 MLP 世界模型，量它的开环误差曲线，找它什么时候饱和
  3. 把 (2) 的饱和时刻与 (1) 的理论上限对齐

附加：能量守恒探针。双摆总机械能是常数，物理上自洽的模型应该让 rollout
的能量也基本守恒。这比状态 RMSE 更能暴露"模型学会了画轨迹但没学会物理"。

跑法：
    ../../../.venv/Scripts/python.exe exp07_double_pendulum_chaos.py
"""

import numpy as np

import wm_lib as L
import wm_nn as N

EP_LEN = 600
HORIZON = 250
DT = 0.002            # 一步采样 = 一个物理步（能量守恒才干净）
SUB = 1


def chaotic_state(rng) -> np.ndarray:
    """给足能量：两杆都接近水平、初速不为零。这样系统才真的混沌。"""
    return np.array([
        rng.uniform(0, 2 * np.pi),          # 第一杆角度
        rng.uniform(0, 2 * np.pi),          # 第二杆角度
        rng.uniform(-2, 2),                 # 第一杆角速度
        rng.uniform(-2, 2),                 # 第二杆角速度
    ])


def collect_chaotic(n_episodes=40, steps=EP_LEN, seed=3):
    """自己采一份"高能量"双摆数据（exp01 那份是暖机小摆动，不够混沌）。"""
    m, d = L.load("double_pendulum")
    rng = np.random.default_rng(seed)
    S, SN = [], []
    for _ in range(n_episodes):
        L.set_state(m, d, chaotic_state(rng))
        for _ in range(steps):
            s = L.get_state(m, d)
            s2 = L.step(m, d, substeps=SUB)
            S.append(s); SN.append(s2)
    return {"s": np.array(S), "s_next": np.array(SN)}


def measure_lyapunov(eps=1e-8, steps=500, seed=0):
    """实测最大 Lyapunov 指数：两条几乎相同的轨迹，看对数分离的斜率。"""
    rng = np.random.default_rng(seed)
    s0 = chaotic_state(rng)
    s1 = s0 + np.array([eps, 0, 0, 0])

    m0, d0 = L.load("double_pendulum")
    m1, d1 = L.load("double_pendulum")
    L.set_state(m0, d0, s0)
    L.set_state(m1, d1, s1)

    sep = []
    for _ in range(steps):
        for _ in range(10):        # 每 10 个物理步记录一次分离量
            L.step(m0, d0)
            L.step(m1, d1)
        sep.append(np.linalg.norm(L.get_state(m1, d1) - L.get_state(m0, d0)))
    sep = np.array(sep)

    # 只在"还没饱和"的区间拟合斜率（分离量 < 系统尺度的 30%）
    cut = max(3, int(np.argmax(sep > 0.3 * sep.max())) if (sep > 0.3 * sep.max()).any() else len(sep))
    t = np.arange(cut) * (DT * 10)     # 采样间隔 = 10 个物理步
    lam = float(np.polyfit(t, np.log(sep[:cut]), 1)[0])
    return lam, sep, eps


def main():
    # ---- 0. 采数据（高能量双摆）
    ds = collect_chaotic()
    print(f"[-] 混沌双摆数据: {len(ds['s'])} 样本，dt={DT}s")
    L.save_npz(ds, "data_double_pendulum_chaotic.npz")

    # ---- 1. 实测 Lyapunov 指数
    lam, sep, eps = measure_lyapunov()
    Lsys = float(np.linalg.norm(ds["s"].std(0)))
    t_eps = float(np.log(Lsys / eps) / lam)
    print(f"[-] 实测最大 Lyapunov 指数 lambda ≈ {lam:.2f} /s")
    print(f"    系统尺度 L={Lsys:.2f}，初始差 eps={eps:.0e}")
    print(f"    => 理论上限（完美初始条件） t ≈ {t_eps:.2f} s")
    print(f"    （对照：1/lambda = {1/lam:.2f} s，就是'误差放大 e 倍'的时间）")

    L.plot("exp07_lyapunov.png", [("||Δs|| between two twins", sep)],
           title=f"chaos: two trajectories diverge exponentially (lambda={lam:.2f} /s)",
           xlabel="time [s]", ylabel="||Δs|| (log)", logy=True)

    # ---- 2. 训练世界模型
    tr, te = L.split(ds, 0.85)
    S, SN = tr["s"], tr["s_next"]
    A = np.zeros((len(S), 1))                    # 双摆无 actuator，给个哑通道
    te_a = np.zeros((len(te["s"]), 1))
    scale = S.std(0) + 1e-9

    print("\n[-] 训练 MLP 世界模型")
    mlp = N.MLPModel(n_in=S.shape[1] + 1, n_out=S.shape[1], hidden=128, depth=2,
                     lr=3e-3, seed=0)
    mlp.fit(S, A, SN, epochs=50, batch=256, seed=0, verbose=False, noise=0.03)

    # ---- 3. 开环误差曲线与混沌地板
    e = N.rollout_errors(mlp.predict, te["s"], te_a, te["s_next"],
                         horizon=HORIZON, n_start=200, ep_len=EP_LEN, scale=scale)
    floor = 1.0        # 完全放弃预测时，误差 = 状态标准差（归一化后 = 1）
    over = np.nonzero(e[1:] >= 0.5 * floor)[0]
    hit = int(over[0] + 1) if len(over) else HORIZON
    # ---- 3b. 用误差曲线自己的指数斜率与 Lyapunov 指数对照
    # 混沌放大区（误差从 0.08 到 0.6 之间）的斜率，就是"模型误差被系统放大
    # 的速率"。它应该接近实测的 lambda —— 这是对混沌分析最强的验证。
    mask = (e[1:] > 0.08) & (e[1:] < 0.6)
    lam_eff = float("nan")
    if mask.sum() > 5:
        tt = np.arange(1, HORIZON + 1)[mask] * DT
        lam_eff = float(np.polyfit(tt, np.log(e[1:][mask]), 1)[0])
        print(f"  误差曲线的指数斜率 lambda_eff = {lam_eff:.2f} /s"
              f"（对照系统 lambda = {lam:.2f} /s）")

    # 用 log 公式算“从单步误差涨到 0.5 倍状态尺度”需要多久
    eps_model = e[1] * Lsys
    t_model = float(np.log(0.5 * Lsys / max(eps_model, 1e-12)) / lam)
    print(f"  单步 nRMSE = {e[1]:.4f}  → 等效初始误差 eps_m = {eps_model:.3f}")
    print(f"  开环 50 步 = {e[50]:.4f}   100 步 = {e[100]:.4f}   {HORIZON} 步 = {e[-1]:.4f}")
    print(f"  实测：{hit*DT:.2f} s 后误差过半；纯指数模型预测 {t_model:.2f} s")
    print("  （实测更早：误差是每步新注入的，不只是初始扰动的一次性放大）")

    L.plot("exp07_horizon.png", [
        ("MLP world model", e[1:]),
        ("chaos floor (=1, predict nothing)", np.ones(HORIZON)),
    ], title="double pendulum: open-loop error climbs and saturates at the chaos floor",
       xlabel=f"horizon (dt={DT} s)", ylabel="normalized RMSE (log)", logy=True)
    L.plot("exp07_horizon_logtime.png", [
        ("MLP world model", e[1:50]),
        ("single-step error", np.ones(49) * e[1]),
    ], title="first 1 s in detail: error grows roughly exponentially",
       xlabel=f"horizon (dt={DT} s)", ylabel="normalized RMSE (log)", logy=True)

    # ---- 4. 物理一致性探针：能量守恒
    # 用解析公式从状态直接算机械能：E = 0.5 q̇ᵀ M(q) q̇ + Σ m_i g z_i
    # （MuJoCo 的 data.energy 读数在这个模型上有系统偏差，所以自己算。）
    mE, dE = L.load("double_pendulum")
    import mujoco

    def energy(states):
        out = []
        for s in states:
            L.set_state(mE, dE, s)
            M = np.zeros((mE.nv, mE.nv))
            mujoco.mj_fullM(mE, dE, M)
            K = 0.5 * float(dE.qvel @ M @ dE.qvel)
            V = sum(mE.body_mass[i] * 9.81 * float(dE.xipos[i][2])
                    for i in range(1, mE.nbody))
            out.append(K + V)
        return np.array(out)

    n_roll = 12
    starts = np.linspace(0, len(te["s"]) - HORIZON - 2, n_roll).astype(int)
    dr_m, dr_t = [], []
    for st in starts:
        traj = mlp.rollout(te["s"][st], te_a[st:st + HORIZON])
        truth = np.concatenate([te["s"][st][None], te["s_next"][st:st + HORIZON]])
        em, et = energy(traj), energy(truth)
        dr_m.append(np.abs(em - em[0]) / max(abs(em[0]), 1e-9))
        dr_t.append(np.abs(et - et[0]) / max(abs(et[0]), 1e-9))
    dm = np.mean(dr_m, 0)
    dtru = np.mean(dr_t, 0)
    print(f"\n  能量相对漂移（{HORIZON} 步内最大）："
          f"真实={dtru.max():.2e}  模型={dm.max():.2e}")
    if dtru.max() < 1e-9:
        print("  （真实轨迹严格守恒，模型的漂移全部来自它自己的误差）")

    L.plot("exp07_energy.png", [
        ("ground truth (RK4)", dtru[1:]),
        ("MLP world model", dm[1:]),
    ], title="physics consistency probe: |ΔE|/|E| inside the model's imagination",
       xlabel=f"rollout step (dt={DT} s)", ylabel="relative energy drift (log)", logy=True)

    L.json_dump("exp07_report.json", {
        "lyapunov_per_s": lam, "system_scale": Lsys, "eps": eps,
        "t_pred_perfect_s": t_eps, "eps_model": float(e[1] * Lsys),
        "t_pred_model_s": t_model, "lambda_eff_from_error": lam_eff,
        "one_step_nrmse": float(e[1]), "h50": float(e[50]), "h100": float(e[100]),
        "h_final": float(e[-1]), "steps_to_half_sigma": hit,
        "t_to_half_sigma_s": float(hit * DT),
        "energy_drift_true_max": float(dtru.max()),
        "energy_drift_model_max": float(dm.max()),
    })
    L.banner("exp07", [
        f"混沌给预测设了硬上限：lambda={lam:.2f}/s（误差每 {1/lam:.2f} s 放大 e 倍）",
        f"完美初始条件（eps=1e-8）可预测 {t_eps:.1f} s —— 这只是理论天花板",
        f"误差曲线在混沌区的指数斜率 lambda_eff={lam_eff:.2f}/s，与系统 lambda 对照",
        f"实测 {hit*DT:.2f} s 后误差过半，{HORIZON*DT:.1f} s 时饱和到 {e[-1]:.2f}（=1 就是瞎猜）",
        f"能量探针：模型想象的轨迹能量漂移最大 {dm.max():.0%}；真实轨迹 {dtru.max():.1e}（严格守恒）",
        "看 exp07_energy.png：物理一致性不是 RMSE 能测出来的东西",
    ])


if __name__ == "__main__":
    main()
