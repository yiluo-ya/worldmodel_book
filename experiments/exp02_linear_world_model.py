"""实验 02：最小的世界模型——线性动力学 + 复合误差 + 时间尺度

对应章节：第 2 章（数学骨架）、第 7 章（复合误差）、第 14 章（长时漂移）

世界模型的最小形态就是一句线性映射：

    s_{t+1} = s_t + W·[s_t, a_t] + b

用最小二乘就能解出来。它一定不准，但足够让我们第一次亲眼看到两件事：

  1. 单步很准，多步就崩——误差被自己的预测喂回去（compounding error）。
  2. 把 dt 调小会让单步更准，但同样物理时长内要推更多步，
     总误差未必下降。**"精度"离开时间尺度就没有意义。**

跑法：
    ../../../.venv/Scripts/python.exe exp02_linear_world_model.py
"""

import numpy as np

import wm_lib as L
import wm_nn as N

T_SEC = 3.0  # 每条 episode 的物理时长


def build(dt: float, n_ep: int = 60, seed: int = 0):
    sub = max(1, int(round(dt / 0.002)))
    return L.collect("cartpole", n_episodes=n_ep, steps=int(T_SEC / dt),
                     substeps=sub, seed=seed)


def main():
    dts = [0.002, 0.01, 0.05]
    curves, rows = {}, []

    for dt in dts:
        ds = build(dt)
        tr, te = L.split(ds, 0.8)
        m = N.LinearModel().fit(tr["s"], tr["a"], tr["s_next"])
        scale = tr["s"].std(0) + 1e-9
        ep_len = int(ds["ep_len"])

        # 单步归一化误差
        p = m.predict(te["s"], te["a"])
        one = float(np.sqrt((((p - te["s_next"]) / scale) ** 2).mean()))

        # 开环推 1 秒
        H = int(round(1.0 / dt))
        e = N.rollout_errors(m.predict, te["s"], te["a"], te["s_next"],
                             horizon=H, n_start=120, ep_len=ep_len, scale=scale)
        curves[dt] = e
        rows.append({
            "dt": dt, "steps_per_sec": 1.0 / dt,
            "one_step_nrmse": one,
            "openloop_1s_nrmse": float(e[-1]),
        })
        print(f"dt={dt:6.3f}  1-step nRMSE={one:.4f}   open-loop-1s nRMSE={e[-1]:.4f}")

    # ---- 图 1：典型复合误差曲线（dt=0.01）。h=0 误差为 0，画图时跳过。
    dt = 0.01
    e = curves[dt][1:]
    h = np.arange(1, len(e) + 1)
    sig = rows[1]["one_step_nrmse"]
    L.plot("exp02_compound_error.png", [
        ("learned linear (dt=0.01)", e),
        ("independent-noise theory  sigma*sqrt(h)", sig * np.sqrt(h)),
        ("worst case  sigma*h", sig * h),
    ], title="compounding error, open-loop rollout on cartpole",
       xlabel="horizon h (dt=0.01s, 1 s = 100 steps)", ylabel="normalized RMSE (log)", logy=True)

    # ---- 图 2：精度 vs 时间尺度（x 轴是 dt）
    dts_arr = np.array([r["dt"] for r in rows])
    L.plot("exp02_timescale.png", [
        ("1-step", np.array([r["one_step_nrmse"] for r in rows])),
        ("open-loop 1 s", np.array([r["openloop_1s_nrmse"] for r in rows])),
    ], title="accuracy vs timescale (dt = 0.002 / 0.01 / 0.05 s)",
       xlabel="dt [s]", ylabel="normalized RMSE", logy=True, x=dts_arr)

    # ---- 一条具体轨迹：模型自己跑着跑着就飘走
    ds = build(0.01, n_ep=8, seed=7)
    tr, te = L.split(ds, 0.5)
    m = N.LinearModel().fit(tr["s"], tr["a"], tr["s_next"])
    H = 200
    traj = m.rollout(te["s"][0], te["a"][:H])
    truth = np.concatenate([te["s"][0][None], te["s_next"][:H]])
    L.plot("exp02_traj_compare.png", [
        ("true pole angle", truth[:, 1]),
        ("model pole angle", traj[:, 1]),
    ], title="same actions, model vs truth (dt=0.01)", ylabel="angle [rad]")

    L.json_dump("exp02_report.json", {"rows": rows})
    L.banner("exp02", [
        "单步误差与开环误差同时看，才知道模型能不能用",
        "1-step 与 1 s 开环的数字见上",
        "图 exp02_compound_error.png：长时误差增长介于 sqrt(h) 与 h 之间",
        "图 exp02_timescale.png：dt 越细单步越准，但别指望开环误差跟着一起降",
    ])


if __name__ == "__main__":
    main()
