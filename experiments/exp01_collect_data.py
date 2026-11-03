"""实验 01：把"世界"变成数据集

对应章节：第 2 章（世界模型的数学骨架）、第 15 章（数据）

这一步常被跳过，但它决定了后面一切：世界模型能学到什么，完全取决于
日志里出现过什么。这里用三个世界各采一份日志，并检查三件事：
    1. 状态维度与动作维度
    2. 状态分布范围（决定模型的外推边界）
    3. 双摆的机械能是否守恒（后面"物理探针"的基准线）

跑法：
    ../../../.venv/Scripts/python.exe exp01_collect_data.py
"""

import os
import numpy as np

import wm_lib as L


def main():
    plans = [
        # world,           episodes, steps, substeps, action_scale, seed
        ("cartpole",        200, 150, 5,  1.0, 0),
        ("double_pendulum", 120, 400, 10, 0.0, 1),
        ("rover",           150, 120, 1,  1.0, 2),
    ]

    report = {}
    for world, ne, st, sub, asc, seed in plans:
        ds = L.collect(world, n_episodes=ne, steps=st, seed=seed,
                       action_scale=asc, substeps=sub)
        path = L.save_npz(ds, f"data_{world}.npz")
        S = ds["s"]
        report[world] = {
            "file": os.path.basename(path),
            "n_samples": int(len(S)),
            "n_state": int(S.shape[1]),
            "n_action": int(ds["a"].shape[1]),
            "dt": float(L.load(world)[0].opt.timestep * sub),
            "state_min": S.min(0).round(3).tolist(),
            "state_max": S.max(0).round(3).tolist(),
            "step_delta_rms": float(np.sqrt(((ds["s_next"] - S) ** 2).mean())),
        }
        if world == "double_pendulum":
            e = ds["energy"]
            report[world]["energy_first_last"] = [float(e[0]), float(e[-1])]
            report[world]["energy_rel_drift"] = float(abs(e[-1] - e[0]) / max(abs(e[0]), 1e-9))
        print(f"[{world:16s}] samples={len(S):6d}  n_state={S.shape[1]:2d} "
              f"n_action={ds['a'].shape[1]}  -> {path}")

    print()
    print("状态一步变化量的 RMS（越小说明 dt 越细、预测越'容易'）：")
    for k, v in report.items():
        print(f"  {k:16s} {v['step_delta_rms']:.6f}")

    # 画一条 cartpole 轨迹：状态随时间的演化
    ds = L.load_npz("data_cartpole.npz")
    n = 400
    L.plot("exp01_cartpole_traj.png", [
        ("cart x [m]", ds["s"][:n, 0] * 5),
        ("pole angle [rad]", ds["s"][:n, 1]),
    ], title="cartpole: one trajectory of random control", ylabel="value (x scaled x5)")

    e = L.load_npz("data_double_pendulum.npz")["energy"]
    L.plot("exp01_energy.png", [("mechanical energy [J]", e[:3000])],
           title="double pendulum: energy stays flat (RK4, no damping)", ylabel="E [J]")

    L.json_dump("exp01_report.json", report)
    L.banner("exp01", [
        "日志已落盘到 out/data_*.npz",
        "要点：世界模型只见过数据覆盖到的状态-动作区域",
        "双摆能量漂移 = %.2e（这是实验 07 的基准线）" % report["double_pendulum"]["energy_rel_drift"],
    ])


if __name__ == "__main__":
    main()
