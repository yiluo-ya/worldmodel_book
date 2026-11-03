# 附录 A　实验手册

## A.1 环境

```bash
cd docs/world_model_book
# 依赖已在项目根目录的 .venv 里（mujoco 3.17.0 / numpy 2.4.6 / pillow 13.0.0）
export MUJOCO_GL=wgl      # Windows 离屏渲染后端，脚本里已自动设
```

所有脚本放在 `experiments/`，输出图与 JSON 读数写到 `../out/`。

## A.2 运行顺序与耗时

| 脚本 | 内容 | 耗时 | 依赖 |
|---|---|---|---|
| `exp01_collect_data.py` | 采集三个世界的日志 | 5 s | — |
| `exp02_linear_world_model.py` | 线性模型 + 复合误差 + 时间尺度 | 40 s | exp01 |
| `exp03_neural_world_model.py` | 覆盖度对照 + 噪声扫描 + 学习曲线 | 4 min | exp01 |
| `exp04_ood_uncertainty.py` | 分布外 + 集成不确定度 | 1 min | exp01 |
| `exp05_mpc_planning.py` | MPC 五控制器对比 | 3 min | — |
| `exp06_pixel_world_model.py` | 像素 / latent 世界模型 | 20 s | — |
| `exp07_double_pendulum_chaos.py` | Lyapunov + 能量探针 | 2.5 min | — |
| `exp08_imagination_policy.py` | 在想象中训练策略 | 2 min | — |
| `exp09_partial_observability.py` | 部分可观测五种眼睛 | 2.5 min | — |
| `exp10_planning_horizon.py` | 规划视界扫描 | 2 min | — |

总计约 18 分钟。`exp03/05/07/09` 较慢，可以并行跑。

## A.3 文件说明

| 文件 | 作用 |
|---|---|
| `wm_lib.py` | 三个 MuJoCo 世界、数据采集、PIL 画图、JSON 落盘 |
| `wm_nn.py` | 标准化器、线性世界模型、手写 MLP+Adam、开环评估 |
| `wm_cartpole.py` | 解析白盒 cartpole（与 MuJoCo 逐位一致） |
| `exp*.py` | 十个实验 |

## A.4 常见问题

**Q: 报 `UnicodeEncodeError`？**
`wm_lib` 已把 stdout 重配为 UTF-8。如果你自己写脚本，记得
`sys.stdout.reconfigure(encoding="utf-8")`，Windows 默认是 GBK。

**Q: 离屏渲染返回全黑？**
必须 `os.environ["MUJOCO_GL"] = "wgl"` 且**在 import mujoco 之前**。

**Q: 数字和书里不一样？**
只要依赖版本一致就可复现。差异通常来自：
- 随机种子（每个脚本都固定了）
- 缓存数据（`out/data_*.npz` 不会自动重采，删掉即可重跑）
- MuJoCo 版本（积分器行为有细微差别）

**Q: 想换成别的系统？**
改 `wm_lib.WORLDS` 加一段 MJCF，然后在实验里换 world 名。注意三件事：
状态维度（`nq+nv`）、动作维度（`nu`）、以及 dt（`timestep × substeps`）。

## A.5 每个实验的"看点"

| 实验 | 打开这张图 | 看什么 |
|---|---|---|
| exp02 | `exp02_compound_error.png` | 误差曲线高于 √h（有系统偏差），最后停在 1.0 |
| exp02 | `exp02_timescale.png` | dt 变小单步变准、开环反而变差 |
| exp03 | `exp03_openloop_narrow.png` | mlp-plain 飞到 4.58 |
| exp03 | `exp03_noise_scan.png` | U 形：最优点在 0.005 |
| exp04 | `exp04_calibration.png` | σ 越大、误差越大（单调） |
| exp05 | `exp05_angles.png` | 只有 analytic 收敛到 0 |
| exp06 | `exp06_rollout_frames.png` | 5 行：真实/latent/clamp/pixel/freeze |
| exp06 | `exp06_error.png` vs `exp06_angle.png` | 两条曲线的排名相反 |
| exp07 | `exp07_lyapunov.png` | 对数坐标下的直线 |
| exp07 | `exp07_energy.png` | 真实水平 vs 模型暴涨 |
| exp08 | `exp08_traj_C2_learned.png` | 想象出的"最优策略"实际在打转 |
| exp09 | `exp09_rollout.png` | 五条曲线的分层 |
| exp10 | `exp10_horizon.png` | 完美模型单调降、学到的模型全高 |

## A.6 一分钟速查：世界模型代码骨架

```python
# ---------- 采集 ----------
S, A, SN = [], [], []
for ep in range(N_EP):
    s = random_init()
    for t in range(STEPS):
        a = policy_or_random(s)
        sn = env.step(s, a)
        S.append(s); A.append(a); SN.append(sn)
        s = sn

# ---------- 训练 ----------
dS = SN - S
sx, su, sy = Stats(S), Stats(A), Stats(dS)          # 标准化
X, Y = concat(sx(S), su(A)), sy(dS)
augment_input_noise()                                # 关键
train_mlp(X, Y)

# ---------- 使用 ----------
def predict(s, a):
    return s + sy.inv(mlp(concat(sx(s), su(a))))

def rollout(s0, actions):
    traj, s = [s0], s0
    for a in actions:
        s = predict(s, a); traj.append(s)
    return traj

# ---------- 评价（别只看单步） ----------
err_curve = open_loop_rmse(predict, test_data, horizon=100)
plan_horizon = first_h(err_curve, 0.1)               # 你的视界预算
```

---

# 附录 B　数学补充

## B.1 复合误差的一阶分析

设单步误差 $e_t$、动力学雅可比 $F_t = \partial f/\partial s$。rollout 误差：

$$
\delta_{t+1} = F_t \delta_t + e_t
\quad\Rightarrow\quad
\delta_h = \Big(\prod_{i<h} F_i\Big)\delta_0 + \sum_{j<h}\Big(\prod_{i>j} F_i\Big)e_j
$$

三种情形：

1. $\|F\| < 1$（稳定）：误差有界，$\|\delta\| \lesssim \frac{\sigma}{1-\|F\|}$
2. $\|F\| \approx 1$（临界）：$\|\delta\| \approx \sigma\sqrt{h}$
3. $\|F\| > 1$（不稳定/混沌）：$\|\delta\| \approx \sigma e^{\lambda h}$

cartpole 的小车位置近乎随遇（$\|F\| \approx 1$），杆角是稳定的，
所以 exp02 的曲线介于 1 和 2 之间。

## B.2 归一化误差的定义

$$
\text{nRMSE} = \sqrt{\frac{1}{Nd}\sum_{i=1}^{N}\sum_{k=1}^{d}
\Big(\frac{\hat s_{i,k} - s_{i,k}}{\sigma_k}\Big)^2}
$$

其中 $\sigma_k$ 是第 $k$ 维在**训练集**上的标准差。这个量无量纲，
1.0 ≈ "完全瞎猜"（误差等于状态的典型波动）。

## B.3 集成不确定度

$$
\mu(s,a) = \frac1K\sum_{k} f_k(s,a), \quad
\sigma^2(s,a) = \frac1K\sum_k \|f_k(s,a) - \mu(s,a)\|^2
$$

$\sigma$ 是认知不确定性的近似（用"模型之间的分歧"代理"数据不足"）。
它的有效性依赖：成员之间**只在初始化/数据子集上不同**，不能是不同结构
（不同结构的分歧混入了模型偏差）。

## B.4 Lyapunov 指数量测

两条轨迹的分离 $\|\delta(t)\|$ 在半对数坐标下是直线：

$$
\ln\|\delta(t)\| = \ln\|\delta_0\| + \lambda t
$$

用最小二乘拟合斜率即得 $\lambda$。注意三点：
- 初始差要小（$10^{-8}$ 量级），否则一开始就非线性
- 只在"未饱和"区间拟合（分离量 < 系统尺度的 30%）
- 单条轨迹的估计有噪声，最好对多条初值取平均

## B.5 CEM（交叉熵法）

```
初始化 mu=0, sigma=u_max/2
重复 iters 次:
    U ~ clip(N(mu, sigma))          采样 N 条动作序列
    评估 score(U) = sum of stage costs + terminal cost
    elite = 取 score 最小的 K 条
    mu, sigma = elite.mean(), elite.std() + eps
最后再采一次，返回 score 最小的那条
```

它的优点：不用梯度（模型可以黑箱）、能处理非凸、实现 20 行。
缺点：高维动作空间需要指数级样本（这就是为什么它难以直接用于
高自由度机械臂）。

---

# 附录 C　术语表

| 术语 | 英文 | 含义 |
|---|---|---|
| 世界模型 | world model | 预测 $(s,a)\to s'$ 的函数 |
| 复合误差 | compounding error | 多步 rollout 时误差被自身输入放大 |
| 开环误差 | open-loop error | 不用真值纠偏的多步预测误差 |
| MPC | model predictive control | 用模型在脑内规划、执行第一步、重规划 |
| CEM | cross-entropy method | 采样式优化器，本书的规划核心 |
| latent | latent variable | 观测压缩后的隐向量 |
| RSSM | recurrent state space model | Dreamer 的模型结构（确定+随机两条路径） |
| 部分可观测 | partial observability | 观测不足以确定状态 |
| 信念状态 | belief state | 对状态的分布估计 $P(s_t\mid o_{0:t})$ |
| 认知不确定性 | epistemic uncertainty | "我没见过"，可由数据消除 |
| 偶然不确定性 | aleatoric uncertainty | "本来就随机"，不可消除 |
| 分布外 | out-of-distribution (OOD) | 输入落在训练分布之外 |
| 系统辨识 | system identification | 用数据拟合已知模型结构的参数 |
| 灰盒模型 | grey-box model | 物理骨架 + 学习残差 |
| 可预测视界 | prediction horizon | 误差超过阈值的步数 |
| Lyapunov 指数 | Lyapunov exponent | 混沌系统误差的指数放大率 |
| 模型误差税 | model error tax | 用不完美模型优化策略带来的性能损失 |
| Dyna 循环 | Dyna loop | 真实数据 ↔ 模型训练 ↔ 策略优化的交替 |
| 探针 | probe | 把表征解码成可解释量的辅助模型 |

---

# 附录 D　这本书的局限（诚实的部分）

1. **所有实验都在低维仿真里**。4 维状态、19 维状态的系统无法体现
   真实机器人的复杂度（接触、多体、传感器噪声、延迟）。
2. **用的是 PCA 和 4 层 MLP**，不是现代的视频 Transformer / 扩散模型。
   这是为了透明，代价是规模上的说服力不足。
3. **没有做真实的策略学习**（exp08 的策略族只有 5 个参数）。
   Dreamer 级别的实验需要 GPU 和数小时训练，超出了"能在一台笔记本上
   跑完"的目标。
4. **没有涉及多智能体、语言条件、跨任务泛化**——这些是当前研究前沿，
   本书只点到为止。

如果你把这些限制都理解了，那你也理解了这本书的定位：
**它不是前沿综述，是把"世界模型为什么会失败、怎么判断它能不能用"
这件事讲到你能自己动手的程度。**
