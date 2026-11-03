# 世界模型：从白板到能用的预测器

一本带可跑实验的世界模型教学书。

## 直接读

**打开 `out/world_model_book.html`（双击即可，单文件、完全离线）。**

带侧边栏目录、深浅主题、字号调节、KaTeX 公式、实验图内嵌、`←/→` 翻页、点图放大。

```
docs/world_model_book/
├── out/world_model_book.html      ← 成品书（约 1.2 MB，单文件离线）
├── chapters/*.md                  ← 18 章源文 + 附录
├── experiments/                   ← 10 个可跑实验
├── build_html.py                  ← 改完 md 后重新生成 HTML
├── run_all.py                     ← 一键跑全部实验 + 重新出书
├── vendor/                        ← KaTeX 0.16.9（离线内嵌用）
└── out/*.png / *.json             ← 实验图与读数
```

## 重新生成

```bash
cd docs/world_model_book
/d/01_code/02_work_code/mujoco_learn/.venv/Scripts/python.exe build_html.py   # 只重出书
/d/01_code/02_work_code/mujoco_learn/.venv/Scripts/python.exe run_all.py      # 跑全部实验 + 出书
```

## 跑实验

```bash
# 依赖已在项目根的 .venv（mujoco 3.17.0 / numpy 2.4.6 / pillow 13.0.0）
.venv/Scripts/python.exe experiments/exp01_collect_data.py    # 先采数据
.venv/Scripts/python.exe experiments/exp02_linear_world_model.py
# ... exp03 - exp10，见附录 A
```

本机实测总耗时约 18 分钟。

## 全书结构

| 部分 | 章节 | 讲什么 |
|---|---|---|
| 一 地基 | 1-3 | 世界模型是什么、数学骨架、函数逼近器 |
| 二 能预测多久 | 4-7 | 训练细节、复合误差、分布外、混沌硬上限 |
| 三 用模型做事 | 8-11 | MPC 规划、想象中训练、像素与 latent、部分可观测 |
| 四 工程与研究 | 12-18 | 评价、结构化先验、数据、范式、失败模式、应用、前沿 |
| 附录 | A-D | 实验手册、数学补充、术语表、本书局限 |

## 核心结论（一句话版）

> 世界模型不是"学一个更小的物理引擎"，而是**学一个足够准、并且知道自己
> 什么时候不准的预测器**；它的价值不体现在单步误差上，而体现在
> "在这个模型里试错比在现实里试错便宜多少"。

## 实验读数速查

| 实验 | 主题 | 关键读数 |
|---|---|---|
| exp01 | 把世界变成数据集 | 3 万 / 4.8 万 / 1.8 万样本，状态 4 / 4 / 19 维 |
| exp02 | 最小世界模型 + 复合误差 | 单步 0.027 → 100 步 0.98；dt 换不来长时精度 |
| exp03 | 覆盖度对照 + 噪声扫描 | 窄覆盖 plain 4.58，加 0.005 噪声后 0.97（4.7 倍） |
| exp04 | 分布外与不确定度 | 外推误差 36.4 倍；σ 检测 OOD 的 AUC 0.948 |
| exp05 | MPC 规划 | 完美模型立住（0.075 rad），四个学到的模型全比随机差 |
| exp06 | 像素 / latent 世界模型 | 像素 RMSE 最好的是"什么都不预测" |
| exp07 | 混沌与可预测视界 | λ=1.71/s；真实能量守恒 3e-8，模型漂移 25% |
| exp08 | 在想象里训练策略 | 真机 CEM 立住；模型里优化的策略真实代价 46.6（倒） |
| exp09 | 部分可观测 | 有速度 20 步 0.30，只有位置 0.76 |
| exp10 | 规划视界 | 完美模型 H=80 最优；学到的模型全线失败 |
