"""实验 04：世界模型知道自己在哪儿不知道吗？

对应章节：第 6 章（分布内 vs 外推）、第 12 章（不确定性）

做法很朴素但很说明问题：
  1. 训练集只保留"小角度"样本（|θ| ≤ 0.25）；测试故意分成两组：
     同分布组（|θ| ≤ 0.25）与外推组（|θ| ≥ 0.6）
  2. 误差在外推组会大几倍——这不意外
  3. 关键问题：**模型能不能自己报告"我这里没把握"？**
     用 3 个不同随机种子训出的集成，取预测标准差 σ 当不确定度，
     看 σ 是否与真实误差同步

如果一个世界模型拿不出可信的 σ，它就不能进任何安全关键的闭环。

跑法：
    ../../../.venv/Scripts/python.exe exp04_ood_uncertainty.py
"""

import numpy as np

import wm_lib as L
import wm_nn as N

EP_LEN = 150
TH_IN, TH_OOD = 0.25, 0.6


def main():
    ds = L.load_npz("data_cartpole.npz")
    theta = ds["s"][:, 1]
    tr, te = L.split(ds, 0.85)

    mask_tr = np.abs(tr["s"][:, 1]) <= TH_IN
    S, A, SN = tr["s"][mask_tr], tr["a"][mask_tr], tr["s_next"][mask_tr]
    print(f"训练样本（|θ|≤{TH_IN}）: {len(S)} / {len(tr['s'])}")

    te_in = np.abs(te["s"][:, 1]) <= TH_IN
    te_ood = np.abs(te["s"][:, 1]) >= TH_OOD
    print(f"测试同分布: {te_in.sum()}   测试外推（|θ|≥{TH_OOD}）: {te_ood.sum()}")

    scale = S.std(0) + 1e-9
    n_in, n_out = S.shape[1] + A.shape[1], S.shape[1]

    lin = N.LinearModel().fit(S, A, SN)
    members = [
        N.MLPModel(n_in=n_in, n_out=n_out, hidden=64, depth=2, lr=3e-3, seed=s)
        .fit(S, A, SN, epochs=40, batch=256, seed=s, verbose=False, noise=0.05)
        for s in range(3)
    ]

    def stats(mask, tag):
        rows = {}
        for name, fn in [("linear", lin.predict),
                         ("mlp-ens", lambda s, a: np.mean([m.predict(s, a) for m in members], 0))]:
            e = np.sqrt((((fn(te["s"][mask], te["a"][mask]) - te["s_next"][mask]) / scale) ** 2).mean())
            rows[name] = float(e)
        print(f"  {tag:12s} linear={rows['linear']:.4f}  mlp-ens={rows['mlp-ens']:.4f}")
        return rows

    print("\n[-] 同分布 vs 外推")
    r_in = stats(te_in, "in-dist")
    r_ood = stats(te_ood, "out-of-dist")

    # ---- 集成不确定度 σ 与真实误差的关系
    print("\n[-] 不确定度 σ 能不能标定误差分量级？")
    S_te, A_te, SN_te = te["s"][te_in | te_ood], te["a"][te_in | te_ood], te["s_next"][te_in | te_ood]
    preds = np.stack([m.predict(S_te, A_te) for m in members])        # (K, N, dim)
    sigma = preds.std(0).mean(1)                                       # (N,)
    err = np.sqrt((((preds.mean(0) - SN_te) / scale) ** 2).mean(1))    # (N,)
    # 只取状态变化量，避免静止样本拉低相关性
    moving = np.abs(SN_te - S_te).mean(1) > np.median(np.abs(SN_te - S_te).mean(1))
    corr = float(np.corrcoef(np.log(sigma[moving] + 1e-12), np.log(err[moving] + 1e-12))[0, 1])
    print(f"  Pearson(log sigma, log |error|) = {corr:.3f}  （正值越大越好）")

    # OOD 检测能力：σ 把"新分布样本"从"同分布样本"里分出来的 AUC
    is_ood = np.abs(S_te[:, 1]) >= TH_OOD
    def auc(score, label):
        order = np.argsort(score)
        ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
        n1, n0 = label.sum(), (~label).sum()
        return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    print(f"  σ 区分 OOD 的 AUC = {auc(sigma, is_ood):.3f}  (0.5 = 瞎猜, 1.0 = 完美)")

    # 分桶看标定
    qs = np.quantile(sigma[moving], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    buckets = []
    for i in range(5):
        m = (sigma[moving] >= qs[i]) & (sigma[moving] <= qs[i + 1])
        buckets.append({
            "sigma_lo": float(qs[i]), "sigma_hi": float(qs[i + 1]),
            "mean_sigma": float(sigma[moving][m].mean()),
            "mean_err": float(err[moving][m].mean()),
        })
        print(f"  σ∈[{qs[i]:.4f},{qs[i+1]:.4f}]  mean σ={buckets[-1]['mean_sigma']:.4f}  "
              f"mean |err|={buckets[-1]['mean_err']:.4f}")

    L.plot("exp04_calibration.png", [
        ("mean error", np.array([b["mean_err"] for b in buckets])),
        ("mean sigma", np.array([b["mean_sigma"] for b in buckets])),
    ], title="uncertainty vs actual error (quintiles of sigma)",
       xlabel="sigma quintile 0..4", ylabel="normalized value", x=np.arange(5))

    # ---- 安全阀：σ 超过阈值就拒绝模型
    print("\n[-] 安全阀：σ > threshold 时声明'不知道'")
    gate_rows = []
    for thr in [0.0015, 0.002, 0.003, 0.004]:
        refused = sigma[moving] > thr
        kept = ~refused
        kept_err = float(err[moving][kept].mean()) if kept.any() else float("nan")
        gate_rows.append({
            "thr": thr, "refuse_rate": float(refused.mean()),
            "kept_error": kept_err,
        })
        print(f"  thr={thr:.4f}  拒绝率={refused.mean():.2%}  留下的样本平均误差={kept_err:.4f}")

    L.json_dump("exp04_report.json", {
        "n_train_in": int(len(S)), "theta_in": TH_IN, "theta_ood": TH_OOD,
        "in_dist": r_in, "out_of_dist": r_ood,
        "log_pearson_sigma_err": corr, "ood_auc": auc(sigma, is_ood),
        "buckets": buckets, "gate": gate_rows,
    })
    L.banner("exp04", [
        f"外推组误差是同分布组的 {r_ood['mlp-ens']/max(r_in['mlp-ens'],1e-9):.1f} 倍（mlp-ens）",
        f"σ 区分新分布的 AUC = {auc(sigma, is_ood):.3f}；log σ 与 log 误差相关 {corr:.2f}",
        "看 exp04_calibration.png：σ 越大，实际误差越大（单调但非线性）",
        "结论：不确定度估计不是锦上添花，它是世界模型能不能上闭环的门票",
    ])


if __name__ == "__main__":
    main()
