"""实验 06：像素世界模型——观测从 4 维变成 2304 维

对应章节：第 9 章（从状态到像素：latent 世界模型）、第 13 章（怎么评价一个
世界模型）

前面的实验都在"真值状态"上工作，这在现实里是奢侈品：真实机器人只有相机。
本实验把观测换成 48×48 灰度图（2304 维），然后回答四个问题：

  1. 直接在像素上做线性世界模型能做到什么程度？（必须用极大的 ridge，
     因为 2304 维特征 vs 2000 个样本是欠定的）
  2. 先用 PCA 把图像压到 96 维（这就是 latent 世界模型的编码器），
     在 latent 上做 MLP 预测，长时误差会不会更好？
  3. **像素 RMSE 是坏指标**：它奖励"什么都不预测"的懒模型。改用
     任务相关指标——检索式探针把预测帧解码回杆角 θ，看角度误差。
  4. 单帧图像有歧义：杆尖位置 = 小车位置 + L·sinθ，两者在像素里部分
     混在一起；这就是第 10 章"观测不完备"的一个具体实例。

跑法：
    ../../../.venv/Scripts/python.exe exp06_pixel_world_model.py
"""

import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "wgl")

import mujoco
from PIL import Image

import wm_lib as L
import wm_nn as N

RES = 48                 # 48×48 灰度 = 2304 维观测
LATENT = 96              # PCA 降到 96 维（4% 的维度）
U_MAX = 30.0
SUBSTEPS = 5
H_ROLL = 20              # rollout 20 步 = 0.2 s
N_TRAJ = 8               # 评估用多少条 rollout 取平均


# ---------------------------------------------------------------------------
def make_renderer():
    model, data = L.load("cartpole")
    r = mujoco.Renderer(model, RES, RES)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.0, 0.0, 0.32]
    cam.distance = 0.95
    cam.azimuth = 90.0
    cam.elevation = -5.0
    return model, data, r, cam


def to_gray(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float64).mean(2) / 255.0).ravel()


def collect_pixels(n_episodes=40, steps=80, seed=0):
    """随机策略 + 渲染，得到 (img_t, a_t, img_{t+1}) 与真值状态。

    限制 |θ| < 1.2 且 |x| < 0.8：杆一旦翻过水平线，图像里就只剩一条
    对称的线，θ 与 θ+π 从单帧无法分辨。
    """
    model, data, r, cam = make_renderer()
    rng = np.random.default_rng(seed)
    I, A, IN, S, SN = [], [], [], [], []
    for _ in range(n_episodes):
        s = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4),
                      rng.uniform(-0.2, 0.2), rng.uniform(-0.3, 0.3)])
        L.set_state(model, data, s)
        for _ in range(steps):
            r.update_scene(data, cam)
            img = to_gray(r.render())
            u = np.array([rng.uniform(-U_MAX, U_MAX)])
            s2 = L.step(model, data, u, substeps=SUBSTEPS)
            if abs(s2[1]) > 1.2 or abs(s2[0]) > 0.8:
                break
            r.update_scene(data, cam)
            img2 = to_gray(r.render())
            I.append(img); A.append(u); IN.append(img2); S.append(s); SN.append(s2)
            s = s2
    return {"img": np.array(I), "a": np.array(A), "img_next": np.array(IN),
            "s": np.array(S), "s_next": np.array(SN)}


class PCAEncoder:
    """latent 世界模型里的编码器：z = E(x)，x̂ = D(z)。

    先做"背景减除"再 PCA：静态背景占了很大一部分方差，不除掉它主成分
    就有一部分花在画墙上（第 9 章"表征很容易学到无聊的东西"）。
    """

    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray) -> "PCAEncoder":
        self.bg = X.mean(0)
        R = X - self.bg
        self.mu = R.mean(0)
        _, S, Vt = np.linalg.svd(R - self.mu, full_matrices=False)
        self.D = Vt[: self.k]
        self.explained = float((S[: self.k] ** 2).sum() / (S ** 2).sum())
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X) - self.bg - self.mu) @ self.D.T

    def inv(self, Z: np.ndarray) -> np.ndarray:
        return np.atleast_2d(Z) @ self.D + self.mu + self.bg


class KNNProbe:
    """检索式探针：把一帧映射到"数据集中最像的那一帧"的真值状态。

    为什么不用线性/MLP 回归？因为单帧里 θ 与 x 部分混叠（杆尖绝对位置 =
    车位置 + L·sinθ），线性模型学不出干净的分离；最近邻反而稳。
    这个方法零训练、可解释，是评估生成式世界模型的常用套路。
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.bg = X.mean(0)
        self.Xn = X - self.bg
        self.Xn /= (np.linalg.norm(self.Xn, axis=1, keepdims=True) + 1e-9)
        self.Y = Y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(X) - self.bg
        Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        idx = (Z @ self.Xn.T).argmax(1)
        return self.Y[idx]


def sharpness(img: np.ndarray) -> float:
    """图像高频能量：预测变糊时它会掉下去。"""
    g = np.asarray(img).reshape(RES, RES)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.abs(lap).mean())


def main():
    print("[-] 采集像素数据")
    ds = collect_pixels()
    print(f"  samples={len(ds['img'])}  obs_dim={ds['img'].shape[1]}  "
          f"θ∈[{ds['s'][:,1].min():.2f},{ds['s'][:,1].max():.2f}]")
    L.save_npz(ds, "data_pixels.npz")

    k = int(len(ds["img"]) * 0.8)
    tr = {a: v[:k] for a, v in ds.items()}
    te = {a: v[k:] for a, v in ds.items()}

    enc = PCAEncoder(LATENT).fit(tr["img"])
    recon = enc.inv(enc(te["img"]))
    print(f"  PCA k={LATENT}: 保留 {enc.explained:.1%} 方差；"
          f"单帧重构 RMSE={np.sqrt(((recon - te['img']) ** 2).mean()):.5f}")

    fg = np.abs(te["img_next"] - enc.bg) > 0.06
    print(f"  前景像素占比 {fg.mean():.1%}")

    # ---- 检索式探针（视觉状态估计）
    probe = KNNProbe(tr["img"], tr["s"][:, :2])
    est = probe(te["img"])
    px_err = float(np.sqrt(((est[:, 0] - te["s"][:, 0]) ** 2).mean()))
    th_err = float(np.sqrt(((est[:, 1] - te["s"][:, 1]) ** 2).mean()))
    print(f"  探针 latent/像素→(x,θ): x {px_err:.3f} m（中位 "
          f"{np.median(np.abs(est[:,0]-te['s'][:,0])):.3f}）, "
          f"θ {th_err:.3f} rad（中位 {np.median(np.abs(est[:,1]-te['s'][:,1])):.3f}）")

    # ---- 模型 A：像素空间线性世界模型
    print("\n[-] 模型 A：像素空间线性世界模型（2304→2304, ridge=100）")
    lin_pix = N.LinearModel(ridge=100.0).fit(tr["img"], tr["a"], tr["img_next"])
    p1 = lin_pix.predict(te["img"], te["a"])
    print(f"  1 步前景像素 RMSE = {float(np.sqrt(((p1 - te['img_next'])[fg] ** 2).mean())):.5f}")

    # ---- 模型 B：latent 空间 MLP 世界模型
    print(f"[-] 模型 B：latent 空间 MLP 世界模型（{LATENT} 维 + 输入噪声）")
    mlp = N.MLPModel(n_in=LATENT + 1, n_out=LATENT, hidden=64, depth=2, lr=3e-3, seed=0)
    mlp.fit(enc(tr["img"]), tr["a"], enc(tr["img_next"]),
            epochs=40, batch=256, seed=0, verbose=False, noise=0.05)

    # ---- 开环 rollout：多条轨迹取平均
    # z 的取值范围（训练 latent 的 1%~99% 分位）：rollout 时把预测 clamp 回这个范围。
    # 这是工程上最便宜的一招：latent 一旦漂出训练分布，PCA 重建会变成
    # 一堆高频基向量的叠加（看图里 latent 那一行的噪点），clamp 能拉回来一部分。
    Ztr = enc(tr["img"])
    z_lo, z_hi = np.quantile(Ztr, 0.01, 0), np.quantile(Ztr, 0.99, 0)

    def predict_z(z, a, clamp: bool):
        z = mlp.predict(z[None], a)[0]
        return np.clip(z, z_lo, z_hi) if clamp else z

    rng = np.random.default_rng(0)
    starts = rng.choice(max(1, len(te["img"]) - H_ROLL - 1), N_TRAJ, replace=False)
    models = ["latent", "latent+clamp", "pixel", "freeze"]
    img_err = {k2: np.zeros(H_ROLL) for k2 in models}
    th_err_r = {k2: np.zeros(H_ROLL) for k2 in models}
    sharp_ratio = {k2: np.zeros(H_ROLL) for k2 in models}
    frames = None

    for si, st in enumerate(starts):
        z = enc(te["img"][st])[0].copy()
        zc = z.copy()
        ip = te["img"][st].copy()
        rec = {"true": [te["img"][st]], "latent": [enc.inv(z)[0]],
               "latent+clamp": [enc.inv(zc)[0]], "pixel": [ip.copy()],
               "freeze": [ip.copy()]}
        for h in range(H_ROLL):
            a = te["a"][st + h:st + h + 1]
            z = predict_z(z, a, clamp=False)
            zc = predict_z(zc, a, clamp=True)
            ip = lin_pix.predict(ip[None], a)[0]
            truth = te["img_next"][st + h]
            m = fg[st + h]
            imgs = {"latent": enc.inv(z)[0], "latent+clamp": enc.inv(zc)[0],
                    "pixel": ip, "freeze": te["img"][st]}
            for tag, img in imgs.items():
                img_err[tag][h] += np.sqrt(((img - truth)[m] ** 2).mean()) / N_TRAJ
                th_est = probe(img)[0, 1]
                th_err_r[tag][h] += abs(th_est - te["s_next"][st + h, 1]) / N_TRAJ
                sharp_ratio[tag][h] += sharpness(img) / max(sharpness(truth), 1e-9) / N_TRAJ
                rec[tag].append(np.clip(img, 0, 1))
            rec["true"].append(truth)
        if frames is None:
            frames = rec

    print("\n  20 步开环误差（8 条轨迹平均）：")
    print(f"  {'model':14s} {'像素RMSE':>10s} {'像素排名':>8s} {'|Δθ| rad':>10s} {'角度排名':>8s}")
    order_pix = sorted(models, key=lambda t: img_err[t][-1])
    order_th = sorted(models, key=lambda t: th_err_r[t][-1])
    for tag in order_pix:
        print(f"  {tag:14s} {img_err[tag][-1]:10.4f} {order_pix.index(tag)+1:8d} "
              f"{th_err_r[tag][-1]:10.3f} {order_th.index(tag)+1:8d}")
    print(f"  高频能量比（越低越糊）: " + "  ".join(
        f"{t}={sharp_ratio[t][-1]:.2f}" for t in models))

    # ---- 可视化：3 行 × 5 列
    cols = [0, 5, 10, 15, 20]
    tags = ["true", "latent", "latent+clamp", "pixel", "freeze"]
    sheet = np.zeros((len(tags) * (RES + 4) + 4, len(cols) * (RES + 2) + 6), dtype=np.uint8)
    for c, h in enumerate(cols):
        for ridx, tag in enumerate(tags):
            g = (np.clip(frames[tag][h], 0, 1).reshape(RES, RES) * 255).astype(np.uint8)
            y = ridx * (RES + 4)
            sheet[y:y + RES, c * (RES + 2):c * (RES + 2) + RES] = g
    Image.fromarray(sheet).save(os.path.join(L.OUT, "exp06_rollout_frames.png"))

    L.plot("exp06_error.png", [
        ("latent-MLP", img_err["latent"]),
        ("latent-MLP + clamp", img_err["latent+clamp"]),
        ("pixel-linear", img_err["pixel"]),
        ("freeze (predict nothing)", img_err["freeze"]),
    ], title="open-loop image error: the laziest model wins on pixels",
       xlabel="horizon (dt=0.01 s)", ylabel="RMSE (log)", logy=True)

    L.plot("exp06_angle.png", [
        ("latent-MLP", th_err_r["latent"]),
        ("latent-MLP + clamp", th_err_r["latent+clamp"]),
        ("pixel-linear", th_err_r["pixel"]),
        ("freeze (predict nothing)", th_err_r["freeze"]),
    ], title="task-relevant error: pole angle decoded from predicted frames",
       xlabel="horizon", ylabel="|Δθ| [rad]")

    L.json_dump("exp06_report.json", {
        "obs_dim": int(ds["img"].shape[1]), "latent": LATENT,
        "pca_explained": enc.explained,
        "probe_x_rmse": px_err, "probe_theta_rmse": th_err,
        "one_step_pixel_rmse_fg": float(np.sqrt(((p1 - te["img_next"])[fg] ** 2).mean())),
        "roll20_image_rmse": {k2: float(v[-1]) for k2, v in img_err.items()},
        "roll20_angle_err": {k2: float(v[-1]) for k2, v in th_err_r.items()},
        "roll20_sharpness_ratio": {k2: float(v[-1]) for k2, v in sharp_ratio.items()},
        "rank_by_pixels": order_pix, "rank_by_angle": order_th,
    })
    L.banner("exp06", [
        f"观测 {ds['img'].shape[1]} 维 → latent {LATENT} 维（PCA 保留 {enc.explained:.1%}）",
        f"检索式探针：x {px_err:.3f} m / θ {th_err:.3f} rad（单帧解码本身就有歧义）",
        "按像素 RMSE 排名  : " + " < ".join(order_pix),
        "按解码角度排名  : " + " < ".join(order_th),
        "排名颠倒就是本实验的全部结论：像素 RMSE 奖励'什么都不预测'",
        "看 exp06_rollout_frames.png（5 行 = 真实/latent/clamp/像素/freeze）",
        "看 exp06_error.png 与 exp06_angle.png：两条曲线的顺序是反的",
    ])


if __name__ == "__main__":
    main()
