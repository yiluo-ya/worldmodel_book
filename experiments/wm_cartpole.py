"""wm_cartpole.py —— 白盒（解析）推车-杆动力学，批量、可微不需

为什么需要它：教科书里"世界模型 vs 真实环境"的对照，需要一张"尽善尽美
的模型"作为上界。直接用 MuJoCo 逐样本步进太慢（规划要几万次前向），所以
这里把同一套刚体方程手写出来，参数从 MuJoCo 模型里读，误差只在积分器。

方程（θ=0 为竖直向上，杆质心距铰点 r，绕质心惯量 I）：
    (m_c + m) ẍ + m r cosθ θ̈ - m r sinθ θ̇² = F - c_x ẋ
    m r cosθ ẍ + (I + m r²) θ̈ - m g r sinθ      = -c_θ θ̇

这正是第 13 章"结构化先验"的一个例子：把已知物理写进模型，比从数据里
硬学要省几个数量级的数据。
"""

from __future__ import annotations

import numpy as np

import wm_lib as L


class AnalyticCartPole:
    def __init__(self, dt: float = 0.01, substeps: int = 5):
        model, _ = L.load("cartpole")
        self.m_c = float(model.body_mass[1])
        self.m_p = float(model.body_mass[2])
        # 杆质心相对铰点的距离（capsule 质心在 fromto 中点）
        self.r = 0.275
        self.I = float(model.body_inertia[2][1])          # 绕 y 轴
        self.c_x = float(model.dof_damping[0])
        self.c_th = float(model.dof_damping[1])
        self.g = float(-model.opt.gravity[2])
        self.dt = dt
        self.sub = substeps
        self.h = dt / substeps

    # ---- 动力学：返回 [ẋ, θ̇, ẍ, θ̈]
    def deriv(self, x, th, xd, thd, F):
        A = self.m_c + self.m_p
        B = self.m_p * self.r
        C = self.I + self.m_p * self.r ** 2
        s, c = np.sin(th), np.cos(th)
        r1 = F - self.c_x * xd + B * s * thd ** 2
        r2 = self.m_p * self.g * self.r * s - self.c_th * thd
        det = A * C - (B * c) ** 2
        xdd = (C * r1 - B * c * r2) / det
        thdd = (A * r2 - B * c * r1) / det
        return xd, thd, xdd, thdd

    def _rk4(self, st, F):
        x, th, xd, thd = st[..., 0], st[..., 1], st[..., 2], st[..., 3]
        h = self.h
        k1 = self.deriv(x, th, xd, thd, F)
        k2 = self.deriv(x + h / 2 * k1[0], th + h / 2 * k1[1], xd + h / 2 * k1[2], thd + h / 2 * k1[3], F)
        k3 = self.deriv(x + h / 2 * k2[0], th + h / 2 * k2[1], xd + h / 2 * k2[2], thd + h / 2 * k2[3], F)
        k4 = self.deriv(x + h * k3[0], th + h * k3[1], xd + h * k3[2], thd + h * k3[3], F)
        out = np.empty_like(st)
        for i in range(4):
            d = (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0
            out[..., i] = st[..., i] + h * d
        return out

    def step(self, S: np.ndarray, U: np.ndarray) -> np.ndarray:
        """S (B,4), U (B,1) -> S' (B,4)，推进一个 dt。"""
        S = np.atleast_2d(np.asarray(S, float))
        F = np.asarray(U, float).reshape(-1)
        out = S
        for _ in range(self.sub):
            out = self._rk4(out, F)
        return out

    __call__ = step
