"""wm_lib.py —— 世界模型教程的实验公共库

只依赖 numpy + mujoco + Pillow（本机已有，无需 torch / matplotlib）。

提供的三件事：
1. 三个 MuJoCo 世界（真正的"物理定律"）：cartpole / double pendulum / rover
2. 数据采集：随机策略探索，产出 (s_t, a_t, s_{t+1}) 三元组
3. 极简画图：把曲线画成 PNG（省掉 matplotlib 依赖）

术语对应书里的章节：
    world       = 真实环境（我们唯一的真值来源）
    dataset     = 交互日志
    model       = 学出来的世界模型（本文件不提供，各实验自己实现）
"""

from __future__ import annotations

import json
import os
import sys

# Windows 控制台默认 GBK，打不出箭头/符号。统一成 UTF-8，避免实验因为
# 一个 UnicodeEncodeError 白跑十分钟。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MUJOCO_GL", "wgl")  # Windows 离屏渲染后端，必须在 import mujoco 之前

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "out"))
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. 三个世界
# ----------------------------------------------------------------------------
# 推车-杆：最经典的受控低维系统。控制量是推车的水平力，目标是把杆立住。
CART_POLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.002" integrator="RK4"/>
  <visual>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <!-- 关阴影：否则地面亮度会随物体位置变化，像素世界模型会被"阴影"这一
         与状态高度相关的假特征带偏（真实机器人视觉也有这个问题）。 -->
    <quality shadowsize="0"/>
  </visual>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" rgba=".20 .21 .24 1"/>
    <body name="cart" pos="0 0 0.05">
      <joint name="slide" type="slide" axis="1 0 0" damping="0.05"/>
      <geom type="box" size="0.15 0.05 0.05" rgba=".55 .6 .7 1"
            contype="0" conaffinity="0"/>
      <body name="pole" pos="0 0 0.05">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.55" size="0.05" rgba=".95 .55 .2 1"
              density="200" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="push" joint="slide" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
"""

# 双摆：无控制、无阻尼、能量守恒，用来做"物理一致性探针"（第 7 / 13 章）。
DOUBLE_PENDULUM_XML = """
<mujoco model="double_pendulum">
  <option timestep="0.002" integrator="RK4"/>
  <worldbody>
    <body name="link1" pos="0 0 1.2">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" rgba=".2 .6 .9 1"/>
      <body name="link2" pos="0 0 -0.5">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" rgba=".9 .5 .2 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# 小坦克（差速驱动）：带摩擦和受限加速度的非线性受控世界，用于 MPC 规划（第 5 / 8 章）。
ROVER_XML = """
<mujoco model="rover">
  <option timestep="0.01" integrator="RK4"/>
  <worldbody>
    <geom name="floor" type="plane" size="6 6 .1" rgba=".5 .52 .55 1"/>
    <body name="chassis" pos="0 0 0.08">
      <freejoint name="root"/>
      <geom type="box" size="0.18 0.14 0.05" rgba=".35 .45 .6 1"/>
      <body name="wheel_l" pos="-0.1 0.16 -0.06">
        <joint name="wl" type="hinge" axis="0 1 0" armature="0.02"/>
        <geom type="cylinder" size="0.07 0.03" zaxis="0 1 0" rgba=".2 .2 .2 1"/>
      </body>
      <body name="wheel_r" pos="-0.1 -0.16 -0.06">
        <joint name="wr" type="hinge" axis="0 1 0" armature="0.02"/>
        <geom type="cylinder" size="0.07 0.03" zaxis="0 1 0" rgba=".2 .2 .2 1"/>
      </body>
      <body name="caster" pos="0.15 0 -0.06">
        <joint name="wc" type="hinge" axis="1 0 0"/>
        <geom type="sphere" size="0.045" rgba=".6 .6 .6 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="drive_l" joint="wl" ctrlrange="-1 1"/>
    <motor name="drive_r" joint="wr" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

WORLDS = {
    "cartpole": CART_POLE_XML,
    "double_pendulum": DOUBLE_PENDULUM_XML,
    "rover": ROVER_XML,
}


def load(world: str = "cartpole"):
    """返回 (model, data)。data 是可变的状态容器，和真实世界里的"当下"对应。"""
    model = mujoco.MjModel.from_xml_string(WORLDS[world])
    # 开启能量统计（双摆的"物理一致性探针"要用）
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    data = mujoco.MjData(model)
    return model, data


def get_state(model, data) -> np.ndarray:
    """本教程统一用"物理状态"作为观测：广义坐标 + 广义速度。

    注意这是一个很强的假设——真实机器人拿不到 qvel。第 10 章会讨论为什么
    真实系统只能拿到带噪、带延迟、需要估计的观测。
    """
    return np.concatenate([data.qpos.copy(), data.qvel.copy()])


def set_state(model, data, state: np.ndarray) -> None:
    nq = model.nq
    data.qpos[:] = state[:nq]
    data.qvel[:] = state[nq:]
    mujoco.mj_forward(model, data)


def step(model, data, ctrl: Optional[np.ndarray] = None, substeps: int = 1) -> np.ndarray:
    """推进 substeps 个物理步，返回新状态。

    substeps 决定了我们学世界模型时的"时间尺度"：dt = timestep * substeps。
    所有实验里，模型训练与规划必须用同一个 dt（第 5 章、实验 05）。
    """
    if ctrl is not None:
        data.ctrl[:] = ctrl
    for _ in range(substeps):
        mujoco.mj_step(model, data)
    return get_state(model, data)


def reset(model, data, rng: np.random.Generator, scale: float = 0.1) -> np.ndarray:
    """随机初始状态：给数据多样性，避免模型只见过一条轨迹。"""
    mujoco.mj_resetData(model, data)
    nq, nv = model.nq, model.nv
    data.qpos[:] = rng.normal(0.0, scale, nq)
    data.qvel[:] = rng.normal(0.0, scale, nv)
    mujoco.mj_forward(model, data)
    return get_state(model, data)


# ----------------------------------------------------------------------------
# 2. 数据采集
# ----------------------------------------------------------------------------
def collect(
    world: str = "cartpole",
    n_episodes: int = 40,
    steps: int = 100,
    seed: int = 0,
    init_scale: float = 0.1,
    action_scale: float = 1.0,
    policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    substeps: int = 1,
) -> Dict[str, np.ndarray]:
    """用随机（或给定）策略探索世界，返回交互日志。

    返回字段：
        s      (N, nq+nv)   动作前的状态
        a      (N, nu)      动作
        s_next (N, nq+nv)   一步之后的状态
        energy (N,)         系统总机械能（部分世界可用）
    """
    model, data = load(world)
    rng = np.random.default_rng(seed)
    S, A, SN, E = [], [], [], []

    for _ in range(n_episodes):
        if policy is None:
            mujoco.mj_resetData(model, data)
            nq, nv = model.nq, model.nv
            data.qpos[:] = rng.normal(0.0, init_scale, nq)
            data.qvel[:] = rng.normal(0.0, init_scale, nv)
            if world == "rover":  # 地面上的车，z 要抬高，姿态用单位四元数
                data.qpos[:3] = [rng.uniform(-1, 1), rng.uniform(-1, 1), 0.08]
                data.qpos[3:7] = [1, 0, 0, 0]
                data.qpos[7:] = 0.0
                data.qvel[:] = rng.normal(0.0, init_scale, model.nv)
            mujoco.mj_forward(model, data)
        s = get_state(model, data)
        for _ in range(steps):
            if policy is not None:
                a = policy(s)
            else:
                hi = model.actuator_ctrlrange[:, 1] * action_scale
                a = rng.uniform(-1, 1, model.nu) * hi
            a = np.clip(a, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            for _ in range(substeps):
                sn = step(model, data, a)
            S.append(s)
            A.append(a)
            SN.append(sn)
            E.append(
                float(data.energy[0])
                if model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_ENERGY
                else float("nan")
            )
            s = sn

    out = {
        "s": np.asarray(S, dtype=np.float64),
        "a": np.asarray(A, dtype=np.float64),
        "s_next": np.asarray(SN, dtype=np.float64),
        "energy": np.asarray(E, dtype=np.float64),
    }
    out["ep_len"] = np.asarray(steps)  # 每段连续轨迹的长度，评估时必须用
    return out


def split(ds: Dict[str, np.ndarray], frac: float = 0.8):
    """按时间顺序切分（不是随机切分）：世界模型的测试集必须是"未来"。

    随机切分会让相邻样本分居训练/测试，相当于泄漏了未来信息（第 13 章）。
    """
    n = len(ds["s"])
    k = int(n * frac)
    tr, te = {}, {}
    for key, v in ds.items():
        v = np.asarray(v)
        if v.ndim == 0:
            tr[key] = te[key] = v
        else:
            tr[key] = v[:k]
            te[key] = v[k:]
    return tr, te


def save_npz(ds: Dict[str, np.ndarray], name: str) -> str:
    path = os.path.join(OUT, name)
    np.savez_compressed(path, **ds)
    return path


def load_npz(name: str) -> Dict[str, np.ndarray]:
    path = name if os.path.isabs(name) else os.path.join(OUT, name)
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# ----------------------------------------------------------------------------
# 3. 极简画图（PIL）
# ----------------------------------------------------------------------------
def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.1e}"
    return f"{v:.3g}"


def plot(
    name: str,
    series: Sequence[Tuple[str, np.ndarray, Optional[str]]],
    title: str = "",
    xlabel: str = "step",
    ylabel: str = "",
    logy: bool = False,
    size: Tuple[int, int] = (760, 440),
    hlines: Sequence[Tuple[float, str]] = (),
    x: Optional[np.ndarray] = None,
) -> str:
    """画多条曲线存 PNG。

    series 每项 (label, y) 或 (label, y, color)；color 传 None 自动配色。
    """
    series = [(t[0], t[1], t[2] if len(t) > 2 else None) for t in series]
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
    W, H = size
    L, R, T, B = 78, 22, 42, 58
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    ys = [np.asarray(y, dtype=np.float64).ravel() for _, y, _ in series]
    if x is None:
        xs = [np.arange(len(y), dtype=np.float64) for y in ys]
    else:
        xs = [np.asarray(x, dtype=np.float64).ravel()] * len(ys)
    allv = np.concatenate(ys)
    if logy:
        _eps = 1e-12
        ys = [np.log10(np.maximum(y, _eps)) for y in ys]
        allv = np.log10(np.maximum(allv, _eps))
        hlines = [(float(np.log10(max(v, 1e-12))), lab) for v, lab in hlines]
    lo, hi = float(np.min(allv)), float(np.max(allv))
    if hlines:
        lo = min(lo, min(v for v, _ in hlines))
        hi = max(hi, max(v for v, _ in hlines))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    xmax = max(float(x[-1]) if len(x) else 1.0 for x in xs) or 1.0

    def px(x, y):
        return (L + (x / xmax) * (W - L - R), H - B - ((y - lo) / (hi - lo)) * (H - T - B))

    # 网格 + 轴刻度
    for i in range(5):
        yv = lo + (hi - lo) * i / 4
        y = H - B - (H - T - B) * i / 4
        d.line([(L, y), (W - R, y)], fill=(226, 226, 226))
        d.text((10, y - 5), _fmt(10 ** yv) if logy else _fmt(yv), fill=(70, 70, 70))
    for i in range(5):
        xv = xmax * i / 4
        x = L + (W - L - R) * i / 4
        d.line([(x, T), (x, H - B)], fill=(240, 240, 240))
        xlab = f"{xv:.0f}" if xmax >= 10 else f"{xv:g}"
        d.text((x - 12, H - B + 8), xlab, fill=(70, 70, 70))
    d.rectangle([L, T, W - R, H - B], outline=(150, 150, 150))

    for v, lab in hlines:
        y = H - B - ((v - lo) / (hi - lo)) * (H - T - B)
        for x in range(L, W - R, 8):
            d.line([(x, y), (x + 4, y)], fill=(160, 160, 160))
        d.text((W - R - 60, y - 12), lab, fill=(120, 120, 120))

    for i, ((label, _, color), yv, xv) in enumerate(zip(series, ys, xs)):
        c = color or palette[i % len(palette)]
        n = len(yv)
        if n < 2:
            continue
        stride = max(1, n // 2000)
        pts = [px(xv[j], yv[j]) for j in range(0, n, stride)]
        d.line(pts, fill=c, width=2)

    d.text((W // 2 - 30, 10), title, fill=(20, 20, 20))
    d.text((W // 2 - 30, H - 22), xlabel, fill=(70, 70, 70))
    d.text((12, T - 22), ylabel, fill=(70, 70, 70))
    # 图例左下（避开曲线与标题）
    ly = H - B - 16 * len(series) - 8
    for i, (label, _, color) in enumerate(series):
        c = color or palette[i % len(palette)]
        d.line([(L + 12, ly + 5), (L + 32, ly + 5)], fill=c, width=3)
        d.text((L + 38, ly - 1), label[:40], fill=(40, 40, 40))
        ly += 16

    path = os.path.join(OUT, name)
    img.save(path)
    return path


def banner(exp: str, lines: Sequence[str]) -> None:
    """实验输出的小标题，方便对照书里的"实验读数"。"""
    print("=" * 72)
    print(f"[{exp}]")
    for ln in lines:
        print("  " + ln)
    print("=" * 72)


def json_dump(name: str, obj: dict) -> str:
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path
