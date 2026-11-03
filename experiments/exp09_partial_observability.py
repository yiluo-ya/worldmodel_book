"""实验 09：部分可观测——你看不见速度，还能预测吗？

对应章节：第 11 章（部分可观测、信念状态与滤波）

前面所有实验都拿"真值状态"当输入，包括速度。真实机器人没有速度传感器，
只有编码器和相机。于是 Markov 假设 P(s'|s,a) 直接破裂：
只看一帧，你无法知道"杆正往哪边倒"。

四组对照，同一份数据、同一个网络结构，只改"模型能看到什么"：

  A. full-state        观测 = [x, θ, ẋ, θ̇]                  （作弊：有速度）
  B. position-only     观测 = [x, θ]                        （只有位置）
  C. finite-diff       观测 = [x, θ, Δx/Δt, Δθ/Δt]          （显式做差当速度）
  D. history(3)        观测 = [x_t, θ_t, x_{t-1}, θ_{t-1}, x_{t-2}, θ_{t-2}]
  E. B + 噪声 + 延迟    现实加强版：位置有噪声，观测晚 2 步

评价：单步误差与 20 步开环误差（后者才是"能不能预测"的真答案）。

跑法：
    ../../../.venv/Scripts/python.exe exp09_partial_observability.py
"""

import numpy as np

import wm_lib as L
import wm_nn as N

SUBSTEPS = 5
DT = 0.002 * SUBSTEPS
H = 20
N_HIST = 3
EP_LEN = 150
OBS_NOISE = 0.01        # 位置噪声标准差（米 / 弧度）
DELAY = 2               # 观测延迟（步）


def collect(n_episodes=80, steps=EP_LEN, seed=0):
    m, d = L.load("cartpole")
    rng = np.random.default_rng(seed)
    S, A, SN = [], [], []
    for _ in range(n_episodes):
        s = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.6, 0.6),
                      rng.uniform(-0.5, 0.5), rng.uniform(-1.0, 1.0)])
        L.set_state(m, d, s)
        for _ in range(steps):
            a = np.array([rng.uniform(-30, 30)])
            s2 = L.step(m, d, a, substeps=SUBSTEPS)
            S.append(s); A.append(a); SN.append(s2)
            s = s2
    return {"s": np.array(S), "a": np.array(A), "s_next": np.array(SN)}


# ---------------------------------------------------------------------------
# 观测表示（"传感器模型"）
# ---------------------------------------------------------------------------
class Encoder:
    """把真值状态序列变成观测序列；也负责在 rollout 里推进观测。"""

    def __init__(self, kind: str, noise: float = 0.0, delay: int = 0, seed: int = 0):
        self.kind, self.noise, self.delay = kind, noise, delay
        self.rng = np.random.default_rng(seed)

    def _measure(self, s: np.ndarray) -> np.ndarray:
        pos = s[[0, 1]].copy()
        if self.noise:
            pos = pos + self.rng.normal(0, self.noise, 2)
        return pos

    def sequence(self, S: np.ndarray) -> np.ndarray:
        """整段序列 -> 观测矩阵 (N, d_obs)。delay 用"前面补"实现。"""
        S = np.asarray(S, float)
        if self.delay:
            S = np.concatenate([np.repeat(S[:1], self.delay, 0), S[:-self.delay]], 0)
        pos = np.stack([self._measure(s) for s in S])
        if self.kind == "full-state":
            return S
        if self.kind == "position-only":
            return pos
        if self.kind == "finite-diff":
            d = np.zeros_like(pos)
            d[1:] = np.diff(pos, axis=0) / DT
            d[0] = d[1]
            return np.concatenate([pos, d], 1)
        if self.kind == "history":
            cols = [pos]
            for k in range(1, N_HIST):
                prev = np.concatenate([np.repeat(pos[:1], k, 0), pos[:-k]], 0)
                cols.append(prev)
            return np.concatenate(cols, 1)
        raise ValueError(self.kind)

    def advance(self, obs: np.ndarray, s_next: np.ndarray, prev_obs=None) -> np.ndarray:
        """rollout 用：给定预测出的下一状态，算出下一观测。"""
        pos = s_next[[0, 1]]
        if self.noise:
            pos = pos + self.rng.normal(0, self.noise, 2)
        if self.kind == "full-state":
            return s_next.copy()
        if self.kind == "position-only":
            return pos
        if self.kind == "finite-diff":
            prev_pos = obs[:2]
            return np.concatenate([pos, (pos - prev_pos) / DT])
        if self.kind == "history":
            older = obs[:2 * (N_HIST - 1)]
            return np.concatenate([pos, older])
        raise ValueError(self.kind)


def fit(X, A, SN, seed=0):
    m = N.MLPMap(n_in=X.shape[1] + 1, n_out=SN.shape[1], hidden=96, depth=2,
                 lr=3e-3, seed=seed)
    m.fit(np.concatenate([X, A], 1), SN, epochs=30, seed=seed, noise=0.05)
    return m


def predict(m, x, a):
    return m(np.concatenate([np.atleast_2d(x), np.atleast_2d(a)], 1))


CASES = [
    ("full-state", dict(kind="full-state")),
    ("position-only", dict(kind="position-only")),
    ("finite-diff", dict(kind="finite-diff")),
    ("history(3)", dict(kind="history")),
    ("pos + noise + delay", dict(kind="position-only", noise=OBS_NOISE, delay=DELAY)),
]


def main():
    ds = collect()
    k = int(len(ds["s"]) * 0.8)
    tr = {a: v[:k] for a, v in ds.items()}
    te = {a: v[k:] for a, v in ds.items()}
    s_scale = tr["s"].std(0) + 1e-9
    print(f"[-] 数据 {len(ds['s'])} 样本, dt={DT}s, episode={EP_LEN} 步")
    print(f"    噪声 σ={OBS_NOISE}, 延迟={DELAY} 步\n")

    rows = []
    for label, kw in CASES:
        enc = Encoder(**kw, seed=1)
        Xtr, Xte = enc.sequence(tr["s"]), enc.sequence(te["s"])
        m = fit(Xtr, tr["a"], tr["s_next"])

        p1 = predict(m, Xte, te["a"])
        one = float(np.sqrt((((p1 - te["s_next"]) / s_scale) ** 2).mean()))

        # 开环 20 步：起点严格落在 episode 内部，避免跨段
        starts = [e * EP_LEN + 10 for e in range(len(te["s"]) // EP_LEN)
                  if e * EP_LEN + H + 12 < len(te["s"])][:60]
        errs = []
        for st in starts:
            cur = Xte[st].copy()
            err = np.zeros(H)
            for h in range(H):
                a = te["a"][st + h:st + h + 1]
                nxt = predict(m, cur[None], a)[0]
                err[h] = float(np.sqrt((((nxt - te["s_next"][st + h]) / s_scale) ** 2).mean()))
                cur = enc.advance(cur, nxt)
            errs.append(err)
        curve = np.mean(errs, 0)
        rows.append({"obs": label, "dim": int(Xtr.shape[1]), "one_step": one,
                     "h20": float(curve[-1]), "curve": curve.tolist()})
        print(f"  {label:20s} 观测维={Xtr.shape[1]:2d}  1步={one:.4f}  20步={curve[-1]:.4f}")

    L.plot("exp09_rollout.png",
           [(r["obs"], np.array(r["curve"])) for r in rows],
           title="the same physics, different sensors: open-loop error",
           xlabel="horizon (dt=0.01 s)", ylabel="normalized RMSE (log)", logy=True)

    L.json_dump("exp09_report.json", {"rows": [{k2: v for k2, v in r.items() if k2 != "curve"}
                                               for r in rows]})
    L.banner("exp09", [
        "同一条物理系统，只改变'模型能看到什么'：",
        *[f"  {r['obs']:20s} 1步={r['one_step']:.4f}  20步={r['h20']:.4f}" for r in rows],
        "看不到速度时单步还不算太差（位置变化本来就小），20 步就露馅",
        "差分成速度能救回大部分信息；历史窗口能补一点，但不如显式估计",
        "噪声 + 延迟是现实系统的常态，它是这部分误差的地板",
    ])


if __name__ == "__main__":
    main()
