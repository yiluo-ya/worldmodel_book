"""wm_nn.py —— 不依赖 torch 的最小神经网络与线性模型

书里所有"学出来的模型"都归到同一个接口：

    model.fit(S, A, S_next)          # 从交互日志学习
    model.predict_delta(s, a)        # 返回 s_next - s 的估计
    model.rollout(s0, actions)       # 开环多步预测（世界模型的真正用法）

故意写得又短又透明：反向传播手写，Adam 手写。看不懂的地方正好是第 3 章
要讲的东西——"模型只是在拟合一个条件分布 P(s'|s,a)"。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 标准化：世界模型训练的隐形主角（第 4 / 15 章）
# ---------------------------------------------------------------------------
class Standardizer:
    def __init__(self):
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mu = x.mean(0)
        self.sd = x.std(0) + 1e-8
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd

    def inv(self, x: np.ndarray) -> np.ndarray:
        return x * self.sd + self.mu


# ---------------------------------------------------------------------------
# 线性世界模型：s_{t+1} = s_t + W [s_t, a_t] + b   （一阶离散动力学）
# ---------------------------------------------------------------------------
class LinearModel:
    """最小二乘拟合 Δs = W φ(s,a)。这是所有世界模型的"零号模型"。"""

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.sx = Standardizer()
        self.su = Standardizer()
        self.sy = Standardizer()

    def _phi(self, s, a):
        return np.concatenate([self.sx(s), self.su(a)], axis=1)

    def fit(self, S, A, S_next):
        dS = S_next - S
        self.sx.fit(S)
        self.su.fit(A)
        self.sy.fit(dS)
        Phi = self._phi(S, A)
        Y = self.sy(dS)
        Phi1 = np.concatenate([Phi, np.ones((len(Phi), 1))], 1)
        n = Phi1.shape[1]
        self.Wb = np.linalg.solve(
            Phi1.T @ Phi1 + self.ridge * np.eye(n), Phi1.T @ Y
        )
        self.W, self.b = self.Wb[:-1].T, self.Wb[-1]
        return self

    def predict_delta(self, s, a):
        s = np.atleast_2d(s)
        a = np.atleast_2d(a)
        return self.sy.inv(self._phi(s, a) @ self.W.T + self.b)

    def predict(self, s, a):
        s = np.atleast_2d(s)
        return s + self.predict_delta(s, a)

    def rollout(self, s0, actions):
        traj = [np.asarray(s0, float)]
        s = np.asarray(s0, float)
        for a in actions:
            s = self.predict(s[None], np.asarray(a)[None])[0]
            traj.append(s)
        return np.asarray(traj)


# ---------------------------------------------------------------------------
# 极小 MLP + Adam
# ---------------------------------------------------------------------------
def _act(x, kind):
    if kind == "relu":
        return np.maximum(x, 0.0), (x > 0).astype(x.dtype)
    if kind == "tanh":
        return np.tanh(x), 1.0 - np.tanh(x) ** 2
    if kind == "silu":
        sig = 1.0 / (1.0 + np.exp(-x))
        return x * sig, sig + x * sig * (1.0 - sig)
    if kind == "linear":
        return x, np.ones_like(x)
    raise ValueError(kind)


class MLP:
    """全连接网络。默认 tanh：平滑，学动力学比 relu 更友好（第 3 章）。"""

    def __init__(self, sizes: List[int], act: str = "tanh", out_act: str = "linear", seed: int = 0):
        rng = np.random.default_rng(seed)
        self.act, self.out_act = act, out_act
        self.params: List[np.ndarray] = []
        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            a = act if not last else out_act
            fan_in = sizes[i]
            gain = {"relu": 2.0, "tanh": 1.0, "silu": 1.0, "linear": 1.0}[a]
            W = rng.normal(0.0, gain * np.sqrt(2.0 / fan_in), (sizes[i], sizes[i + 1]))
            b = np.zeros(sizes[i + 1])
            self.params += [W, b]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, x: np.ndarray, cache: bool = False):
        acts = [x]
        pre = []
        h = x
        nlayer = len(self.params) // 2
        for i in range(nlayer):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            z = h @ W + b
            pre.append(z)
            is_last = i == nlayer - 1
            h, _ = _act(z, self.out_act if is_last else self.act)
            acts.append(h)
        if cache:
            self._cache = (acts, pre)
        return h

    def backward(self, dout: np.ndarray):
        acts, pre = self._cache
        grads = [None] * len(self.params)
        nlayer = len(self.params) // 2
        g = dout
        for i in reversed(range(nlayer)):
            is_last = i == nlayer - 1
            _, dact = _act(pre[i], self.out_act if is_last else self.act)
            g = g * dact
            W = self.params[2 * i]
            grads[2 * i] = acts[i].T @ g
            grads[2 * i + 1] = g.sum(0)
            g = g @ W.T
        return grads

    def adam(self, grads: List[np.ndarray], lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def predict_delta(self, s, a):
        s = np.atleast_2d(s)
        return s + self.rollout_delta(s, None)

    def rollout(self, s0, actions):
        traj = [np.asarray(s0, float)]
        s = np.asarray(s0, float)
        for a in actions:
            d = self.forward(self._phi(s[None], np.asarray(a)[None]))
            s = s + d[0]
            traj.append(s)
        return np.asarray(traj)


class MLPModel(MLP):
    """MLP 世界模型：输入 (s, a)，输出 Δs。

    两个工程细节决定成败（第 4 章）：
      - 输出"增量"而不是绝对下一状态（残差预测）
      - 输入输出都标准化
    """

    def __init__(self, n_in, n_out, hidden=64, depth=2, lr=3e-3, seed=0, **kw):
        sizes = [n_in] + [hidden] * depth + [n_out]
        super().__init__(sizes, **kw)
        self.lr = lr
        self.sx, self.su, self.sy = Standardizer(), Standardizer(), Standardizer()
        self.history: Dict[str, List[float]] = {"train": [], "val": []}

    def _phi(self, s, a):
        if a is None:
            return self.sx(np.atleast_2d(s))
        return np.concatenate([self.sx(np.atleast_2d(s)), self.su(np.atleast_2d(a))], 1)

    def fit(self, S, A, S_next, S_val=None, A_val=None, SN_val=None,
            epochs=40, batch=256, seed=0, verbose=True, noise=0.0, aug_reps=3):
        """noise > 0 时做输入噪声数据增强（单位：状态各维标准差的倍数）。

        这是对抗复合误差最便宜、最有效的一招：rollout 时模型一定会跑到
        "没见过的状态"上，训练期就让它见过。实验 03 会把 31.4 → 0.77 的
        差别直接量出来。
        """
        if noise > 0:
            rng0 = np.random.default_rng(seed + 12345)
            S_aug, A_aug, SN_aug = [S], [A], [S_next]
            sd = S.std(0)
            for _ in range(aug_reps):
                S_aug.append(S + rng0.normal(0.0, noise * sd, S.shape))
                A_aug.append(A)
                SN_aug.append(S_next)
            S = np.concatenate(S_aug); A = np.concatenate(A_aug); S_next = np.concatenate(SN_aug)
        dS = S_next - S
        self.sx.fit(S); self.su.fit(A); self.sy.fit(dS)
        X = self._phi(S, A)
        Y = self.sy(dS)
        rng = np.random.default_rng(seed)
        n = len(X)
        for ep in range(epochs):
            idx = rng.permutation(n)
            tot = 0.0
            for i in range(0, n, batch):
                j = idx[i:i + batch]
                out = self.forward(X[j], cache=True)
                err = out - Y[j]
                tot += float((err ** 2).sum())
                self.adam(self.backward(2.0 * err / len(j)), lr=self.lr)
            tr = np.sqrt(tot / (n * Y.shape[1]))
            self.history["train"].append(tr)
            if S_val is not None:
                pv = self.forward(self._phi(S_val, A_val))
                self.history["val"].append(
                    float(np.sqrt(((pv - self.sy(SN_val - S_val)) ** 2).mean()))
                )
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                msg = f"  epoch {ep:3d}  train_rmse={tr:.5f}"
                if S_val is not None:
                    msg += f"  val_rmse={self.history['val'][-1]:.5f}"
                print(msg)
        return self

    def predict_delta(self, s, a):
        s2 = np.atleast_2d(s); a2 = np.atleast_2d(a)
        return self.sy.inv(self.forward(self._phi(s2, a2)))

    def predict(self, s, a):
        return np.atleast_2d(s) + self.predict_delta(s, a)

    def rollout(self, s0, actions):
        traj = [np.asarray(s0, float)]
        s = np.asarray(s0, float)
        for a in actions:
            s = s + self.sy.inv(self.forward(self._phi(s[None], np.asarray(a)[None])))[0]
            traj.append(s)
        return np.asarray(traj)


class MLPMap(MLP):
    """通用映射 y = f(x)：输入输出维度可以不同。

    第 13 章的"探针"（从特征解码出物理量）和编码器都属于这一类。
    和 MLPModel 的差别只有一个：它预测 y 本身，不是残差 y-x。
    """

    def __init__(self, n_in, n_out, hidden=64, depth=2, lr=3e-3, seed=0, **kw):
        super().__init__([n_in] + [hidden] * depth + [n_out], **kw)
        self.lr = lr
        self.sx, self.sy = Standardizer(), Standardizer()

    def fit(self, X, Y, epochs=60, batch=256, seed=0, verbose=False, noise=0.0):
        X = np.atleast_2d(X); Y = np.atleast_2d(Y)
        if noise > 0:
            rng0 = np.random.default_rng(seed + 7)
            sd = X.std(0)
            X = np.concatenate([X] + [X + rng0.normal(0, noise * sd, X.shape) for _ in range(3)])
            Y = np.concatenate([Y] * 4)
        self.sx.fit(X); self.sy.fit(Y)
        Xn, Yn = self.sx(X), self.sy(Y)
        rng = np.random.default_rng(seed)
        n = len(Xn)
        for ep in range(epochs):
            idx = rng.permutation(n)
            for i in range(0, n, batch):
                j = idx[i:i + batch]
                err = self.forward(Xn[j], cache=True) - Yn[j]
                self.adam(self.backward(2.0 * err / len(j)), lr=self.lr)
        return self

    def __call__(self, X):
        return self.sy.inv(self.forward(self.sx(np.atleast_2d(X))))


# ---------------------------------------------------------------------------
# 开环评估：世界模型的真实考核方式
# ---------------------------------------------------------------------------
def save_mlp(model: "MLPModel", path: str) -> str:
    """存盘：参数 + 三个标准化器。忘了存标准化器是最经典的部署事故。"""
    n_in = model.params[0].shape[0]
    n_out = model.params[-1].shape[-1]
    hidden = model.params[2].shape[0] if len(model.params) > 2 else 0
    depth = len(model.params) // 2 - 1
    np.savez(
        path, n_in=n_in, n_out=n_out, hidden=hidden, depth=depth,
        act=model.act, out_act=model.out_act,
        **{f"p{i}": p for i, p in enumerate(model.params)},
        **{f"sx_{k}": getattr(model.sx, k) for k in ("mu", "sd")},
        **{f"su_{k}": getattr(model.su, k) for k in ("mu", "sd")},
        **{f"sy_{k}": getattr(model.sy, k) for k in ("mu", "sd")},
    )
    return path


def load_mlp(path: str) -> "MLPModel":
    with np.load(path) as f:
        m = MLPModel(int(f["n_in"]), int(f["n_out"]), hidden=int(f["hidden"]),
                     depth=int(f["depth"]), act=str(f["act"]), out_act=str(f["out_act"]))
        m.params = [f[f"p{i}"].copy() for i in range(2 * (int(f["depth"]) + 1))]
        for name in ("sx", "su", "sy"):
            st = Standardizer()
            getattr(m, name).__dict__.update(mu=f[f"{name}_mu"], sd=f[f"{name}_sd"])
    return m


def _episode_starts(S, horizon: int, ep_len: int, n_start: int, rng) -> np.ndarray:
    """在每条 episode 内部挑起点，保证 st..st+horizon 落在同一段连续轨迹里。

    跨 episode 边界取窗口是最常见的评估 bug：模型会被要求预测一次
    "穿越"，误差图会莫名出现台阶。
    """
    n_ep = max(1, len(S) // ep_len)
    per = max(1, n_start // n_ep)
    out = []
    for e in range(n_ep):
        base = e * ep_len
        span = min(ep_len, len(S) - base)
        if span <= horizon + 1:
            continue
        k = min(per, span - horizon - 1)
        out.append(base + rng.choice(span - horizon - 1, size=k, replace=False))
    return np.concatenate(out) if out else np.array([0])


def rollout_errors(predict_fn: Callable, S, A, S_next, horizon: int = 50,
                   n_start: int = 200, seed: int = 0, ep_len: Optional[int] = None,
                   scale: Optional[np.ndarray] = None) -> np.ndarray:
    """开环推 horizon 步，返回每步的（可选归一化）RMSE 曲线。

    predict_fn(s2d, a2d) -> s_next (批量)，见 LinearModel.predict / MLPModel.predict。
    这是"复合误差"（compounding error）的直接测量（第 7 / 14 章）。
    """
    rng = np.random.default_rng(seed)
    if ep_len is None:
        ep_len = horizon + 2
    starts = _episode_starts(S, horizon, ep_len, n_start, rng)
    s = S[starts].copy()
    scale = np.ones(S.shape[1]) if scale is None else np.asarray(scale, float)
    err = np.zeros(horizon + 1)
    for h in range(horizon):
        s = np.atleast_2d(predict_fn(s, A[starts + h]))
        d = (s - S_next[starts + h]) / scale
        err[h + 1] = float(np.sqrt((d ** 2).mean()))
    return err
