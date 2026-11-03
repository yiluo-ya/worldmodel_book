"""实验 03：从线性模型到神经网络世界模型，以及"分布偏移"这一关

对应章节：第 3 章（函数逼近器）、第 4 章（训练细节）、第 5 章（复合误差）

同一份代码、同一个网络结构、同一条物理世界，只换两样东西：

  ① 模型：linear / mlp-plain / mlp-robust（+输入噪声）
  ② 数据覆盖：宽（随机控制把杆甩到 ±10 rad）/ 窄（只在原点附近小幅摆动）

你会看到一件很重要的事：**"神经网络外推崩坏"不是模型的固有缺陷，
而是"数据覆盖 vs 使用分布"的差距决定的。**

    - 覆盖宽时，mlp-plain 的开环误差甚至能打赢加了噪声的版本；
    - 覆盖窄时，mlp-plain 会飞出去几十倍，而噪声增强能救回来。

所以第 3 章的反直觉结论有前提条件，本章把它显式量出来。

跑法：
    ../../../.venv/Scripts/python.exe exp03_neural_world_model.py
    ../../../.venv/Scripts/python.exe exp03_neural_world_model.py --quick   # 跳过学习曲线
"""

from __future__ import annotations

import argparse
import os
import numpy as np

import wm_lib as L
import wm_nn as N

EP_LEN = 150
HORIZON = 100
SUBSTEPS = 5
NOISE_GRID = [0.0, 0.005, 0.01, 0.02, 0.05]


# ---------------------------------------------------------------------------
def wide_data():
    """宽覆盖：随机控制在 ±30 N 之间猛推，杆会转好几圈。"""
    path = os.path.join(L.OUT, "data_cartpole.npz")
    if os.path.exists(path):
        return L.load_npz("data_cartpole.npz")
    ds = L.collect("cartpole", n_episodes=200, steps=EP_LEN, seed=0,
                   init_scale=0.1, action_scale=1.0, substeps=SUBSTEPS)
    L.save_npz(ds, "data_cartpole.npz")
    return ds


def narrow_data():
    """窄覆盖：初始状态很小 + 动作幅度只有 1/6（相当于"温柔的演示数据"）。

    这正是真实机器人数据的常见形态：人类示范总是温和的、安全的，
    于是模型从没见过"剧烈操作"下世界会怎样。
    """
    path = os.path.join(L.OUT, "data_cartpole_narrow.npz")
    if os.path.exists(path):
        return L.load_npz("data_cartpole_narrow.npz")
    ds = L.collect("cartpole", n_episodes=200, steps=EP_LEN, seed=0,
                   init_scale=0.02, action_scale=1.0 / 6, substeps=SUBSTEPS)
    L.save_npz(ds, "data_cartpole_narrow.npz")
    return ds


def make_mlp(n_in, n_out, seed=0, hidden=64):
    return N.MLPModel(n_in=n_in, n_out=n_out, hidden=hidden, depth=2, lr=3e-3, seed=seed)


def evaluate(m, te, scale, ep_len, tag):
    p = m.predict(te["s"], te["a"])
    one = float(np.sqrt((((p - te["s_next"]) / scale) ** 2).mean()))
    e = N.rollout_errors(m.predict, te["s"], te["a"], te["s_next"],
                         horizon=HORIZON, n_start=120, ep_len=ep_len, scale=scale)
    print(f"    {tag:12s} 1-step={one:.4f}   h10={e[10]:.4f}   h100={e[-1]:.4f}")
    return one, e


# ---------------------------------------------------------------------------
def study(ds, label, ep_len, train_noise):
    tr, te = L.split(ds, 0.85)
    scale = tr["s"].std(0) + 1e-9
    n_in, n_out = tr["s"].shape[1] + tr["a"].shape[1], tr["s"].shape[1]
    th = tr["s"][:, 1]
    print(f"\n[-] 数据集 {label}: {len(ds['s'])} 样本, "
          f"θ∈[{th.min():.2f},{th.max():.2f}] rad, |θ|max={np.abs(th).max():.2f}")

    lin = N.LinearModel().fit(tr["s"], tr["a"], tr["s_next"])
    one_l, e_l = evaluate(lin, te, scale, ep_len, "linear")

    plain = make_mlp(n_in, n_out).fit(tr["s"], tr["a"], tr["s_next"], epochs=30,
                                      batch=256, seed=0, verbose=False)
    one_p, e_p = evaluate(plain, te, scale, ep_len, "mlp-plain")

    robust = make_mlp(n_in, n_out).fit(tr["s"], tr["a"], tr["s_next"], epochs=30,
                                       batch=256, seed=0, verbose=False,
                                       noise=train_noise)
    one_r, e_r = evaluate(robust, te, scale, ep_len, f"mlp-robust@{train_noise}")

    return {
        "label": label, "n_samples": int(len(ds["s"])), "ep_len": int(ep_len),
        "theta_abs_max": float(np.abs(th).max()),
        "linear": {"one_step": one_l, "h10": float(e_l[10]), "h100": float(e_l[-1])},
        "mlp_plain": {"one_step": one_p, "h10": float(e_p[10]), "h100": float(e_p[-1])},
        "mlp_robust": {"one_step": one_r, "h10": float(e_r[10]), "h100": float(e_r[-1])},
        "curves": {"linear": e_l, "mlp-plain": e_p, "mlp-robust": e_r},
        "models": {"plain": plain, "robust": robust},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过学习曲线与噪声扫描（省 3 分钟）")
    ap.add_argument("--noise", type=float, default=0.005,
                    help="mlp-robust 使用的输入噪声倍率（默认 0.005，即状态标准差的 0.5%%）")
    args = ap.parse_args()

    # ---- 主实验：宽覆盖（本书 exp01 采的那份）
    wide = study(wide_data(), "wide（随机猛推，杆会翻转）", EP_LEN, args.noise)

    L.plot("exp03_openloop_wide.png",
           [(k, v[1:]) for k, v in wide["curves"].items()],
           title="wide coverage data: all three models stay bounded",
           xlabel="horizon (dt=0.01 s, 100 steps = 1 s)",
           ylabel="normalized RMSE (log)", logy=True)

    # ---- 对照实验：窄覆盖（温和的"人类示范"数据）
    narrow = study(narrow_data(), "narrow（温和小动作）", EP_LEN, args.noise)

    L.plot("exp03_openloop_narrow.png",
           [(k, v[1:]) for k, v in narrow["curves"].items()],
           title="narrow coverage data: mlp-plain explodes, input noise rescues it",
           xlabel="horizon (dt=0.01 s, 100 steps = 1 s)",
           ylabel="normalized RMSE (log)", logy=True)

    L.plot("exp03_coverage.png", [
        ("mlp-plain", np.array([wide["mlp_plain"]["h100"], narrow["mlp_plain"]["h100"]])),
        ("mlp-robust", np.array([wide["mlp_robust"]["h100"], narrow["mlp_robust"]["h100"]])),
        ("linear", np.array([wide["linear"]["h100"], narrow["linear"]["h100"]])),
    ], title="same model, same physics: coverage decides whether it explodes",
       xlabel="0 = wide coverage, 1 = narrow coverage", ylabel="h=100 error (log)",
       logy=True, x=np.arange(2))

    # ---- 一条具体轨迹：窄数据下 plain 怎么跑飞
    tr, te = L.split(narrow_data(), 0.85)
    H = HORIZON
    acts = te["a"][:H]
    truth = np.concatenate([te["s"][0][None], te["s_next"][:H]])
    L.plot("exp03_traj.png", [
        ("truth", truth[:, 1]),
        ("mlp-robust", narrow["models"]["robust"].rollout(te["s"][0], acts)[:, 1]),
        ("mlp-plain", narrow["models"]["plain"].rollout(te["s"][0], acts)[:, 1]),
    ], title="narrow-coverage data: plain network diverges, noise-trained one tracks",
       ylabel="pole angle [rad]")

    # ---- 噪声扫描：同一噪声倍率，在两种覆盖下的收益天差地别
    scan = {}
    if not args.quick:
        print("\n[-] 噪声倍率扫描（归一化 Δs 误差）")
        for name, ds in [("wide", wide_data()), ("narrow", narrow_data())]:
            tr, te = L.split(ds, 0.85)
            scale = tr["s"].std(0) + 1e-9
            n_in, n_out = tr["s"].shape[1] + 1, tr["s"].shape[1]
            rows = []
            print(f"  [{name}]  theta_std={tr['s'][:,1].std():.2f}")
            print(f"    {'noise':>6} {'1step':>8} {'h50':>8} {'h100':>8}")
            for nz in NOISE_GRID:
                m = make_mlp(n_in, n_out).fit(tr["s"], tr["a"], tr["s_next"], epochs=30,
                                              batch=256, seed=0, verbose=False, noise=nz)
                p = m.predict(te["s"], te["a"])
                one = float(np.sqrt((((p - te["s_next"]) / scale) ** 2).mean()))
                e = N.rollout_errors(m.predict, te["s"], te["a"], te["s_next"], horizon=HORIZON,
                                     n_start=120, ep_len=EP_LEN, scale=scale)
                rows.append({"noise": nz, "one_step": one,
                             "h50": float(e[50]), "h100": float(e[-1])})
                print(f"    {nz:6.3f} {one:8.4f} {e[50]:8.4f} {e[-1]:8.4f}")
            scan[name] = rows

        L.plot("exp03_noise_scan.png", [
            (f"{name} (|θ|max={np.abs(ds['s'][:,1]).max():.1f})",
             np.array([r["h100"] for r in scan[name]]))
            for name, ds in [("wide", wide_data()), ("narrow", narrow_data())]
        ], title="input-noise strength vs 1 s open-loop error (lower is better)",
           xlabel="noise multiplier (x state std)", ylabel="h=100 nRMSE (log)",
           logy=True, x=np.array(NOISE_GRID))

    # ---- 数据量 vs 精度（用宽数据，因为窄数据无法泛化到测试集）
    lc = []
    if not args.quick:
        ds = wide_data()
        tr, te = L.split(ds, 0.85)
        scale = tr["s"].std(0) + 1e-9
        n_in, n_out = tr["s"].shape[1] + 1, tr["s"].shape[1]
        n = len(tr["s"])
        print("\n[-] 数据量 vs 50 步开环误差（宽覆盖数据）")
        for f in [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
            k = max(500, int(n * f))
            S, A, SN = tr["s"][:k], tr["a"][:k], tr["s_next"][:k]
            li = N.LinearModel().fit(S, A, SN)
            mi = make_mlp(n_in, n_out).fit(S, A, SN, epochs=30, batch=256, seed=0,
                                           verbose=False, noise=0.05)
            el = N.rollout_errors(li.predict, te["s"], te["a"], te["s_next"], horizon=50,
                                  n_start=80, ep_len=EP_LEN, scale=scale)
            em = N.rollout_errors(mi.predict, te["s"], te["a"], te["s_next"], horizon=50,
                                  n_start=80, ep_len=EP_LEN, scale=scale)
            lc.append({"frac": f, "n": k, "linear_h50": float(el[-1]),
                       "mlp_h50": float(em[-1])})
            print(f"    n={k:6d}  linear h50={el[-1]:.4f}   mlp-robust h50={em[-1]:.4f}")

        L.plot("exp03_learning_curve.png", [
            ("linear", np.array([r["linear_h50"] for r in lc])),
            ("mlp-robust", np.array([r["mlp_h50"] for r in lc])),
        ], title="data efficiency: 50-step open-loop error vs training set size",
           xlabel="training samples", ylabel="normalized RMSE (log)", logy=True,
           x=np.array([r["n"] for r in lc]))

    # ---- 训练曲线（宽数据下的 robust）
    m = wide["models"]["robust"]
    if m.history["val"]:
        L.plot("exp03_train_curve.png", [
            ("train", np.array(m.history["train"])),
            ("val", np.array(m.history["val"])),
        ], title="mlp-robust training loss (normalized delta MSE)", xlabel="epoch",
           ylabel="RMSE (log)", logy=True)

    path = N.save_mlp(wide["models"]["robust"],
                      os.path.join(L.OUT, "wm_mlp_cartpole.npz"))
    print(f"\n  模型已存：{path}")

    def pack(d):
        return {k: v for k, v in d.items() if k not in ("models", "curves")}

    L.json_dump("exp03_report.json", {
        "wide": pack(wide), "narrow": pack(narrow),
        "learning_curve": lc, "noise_scan": scan, "noise_default": args.noise,
    })

    def best(rows):
        if not rows:
            return None
        return min(rows, key=lambda r: r["h100"])

    lines = [
        "同一份代码、同一个网络：唯一的变量是数据覆盖",
        f"宽覆盖（|θ|max={wide['theta_abs_max']:.1f}）:",
        f"    plain 1步={wide['mlp_plain']['one_step']:.4f}  h100={wide['mlp_plain']['h100']:.4f}",
        f"    robust 1步={wide['mlp_robust']['one_step']:.4f}  h100={wide['mlp_robust']['h100']:.4f}",
        f"窄覆盖（|θ|max={narrow['theta_abs_max']:.2f}）:",
        f"    plain 1步={narrow['mlp_plain']['one_step']:.4f}  h100={narrow['mlp_plain']['h100']:.4f}",
        f"    robust 1步={narrow['mlp_robust']['one_step']:.4f}  h100={narrow['mlp_robust']['h100']:.4f}",
    ]
    if scan:
        for name in ("wide", "narrow"):
            b = best(scan[name])
            z = [r for r in scan[name] if r["noise"] == 0.0][0]
            lines.append(f"{name} 的最优噪声倍率 = {b['noise']}（h100 {z['h100']:.3f} → {b['h100']:.3f}，"
                         f"{z['h100']/max(b['h100'],1e-9):.1f} 倍）")
    lines += [
        "结论一：'神经网络开环崩坏'取决于数据覆盖与使用分布的差距",
        "结论二：噪声强度要小（约状态标准差的 0.5%%），过大会糊掉动力学",
        "结论三：噪声的收益也随覆盖度变化 —— 覆盖窄时才是救命稻草",
    ]
    L.banner("exp03", lines)


if __name__ == "__main__":
    main()
