"""run_all.py —— 按依赖顺序跑完全部实验，最后重新生成 HTML 书

    ../../../.venv/Scripts/python.exe run_all.py            # 全跑（约 15 分钟）
    ../../../.venv/Scripts/python.exe run_all.py --fast     # 只跑快的（约 1 分钟）
    ../../../.venv/Scripts/python.exe run_all.py --only exp03 exp05

输出：out/ 下的图与 JSON 读数，以及 out/world_model_book.html
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, "experiments")

# (脚本, 说明, 是否快)
STEPS = [
    ("exp01_collect_data.py", "采集三个世界的交互日志", True),
    ("exp02_linear_world_model.py", "线性世界模型 + 复合误差 + 时间尺度", True),
    ("exp06_pixel_world_model.py", "像素 / latent 世界模型", True),
    ("exp03_neural_world_model.py", "线性 / MLP / 噪声鲁棒", False),
    ("exp04_ood_uncertainty.py", "分布外 + 集成不确定度", False),
    ("exp05_mpc_planning.py", "MPC 六控制器对比", False),
    ("exp09_partial_observability.py", "部分可观测五种眼睛", False),
    ("exp10_planning_horizon.py", "规划视界扫描", False),
    ("exp08_imagination_policy.py", "在想象中训练策略", False),
    ("exp07_double_pendulum_chaos.py", "Lyapunov + 能量探针", False),
]


def run(script: str, title: str) -> tuple[bool, float]:
    print(f"\n{'='*72}\n[run] {script}  —— {title}\n{'='*72}")
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(EXP_DIR, script)],
                       cwd=EXP_DIR)
    dt = time.time() - t0
    ok = p.returncode == 0
    print(f"[{'ok ' if ok else 'FAIL'}] {script}  {dt:.1f}s")
    return ok, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="只跑快实验")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定脚本（前缀匹配，如 exp03）")
    ap.add_argument("--no-build", action="store_true", help="不重新生成 HTML")
    args = ap.parse_args()

    steps = [s for s in STEPS if (not args.fast or s[2])]
    if args.only:
        steps = [s for s in STEPS if any(s[0].startswith(o) for o in args.only)]

    results = []
    for script, title, _ in steps:
        ok, dt = run(script, title)
        results.append((script, ok, dt))
        if not ok:
            print(f"[warn] {script} 失败，继续后面的实验")

    print(f"\n{'='*72}\n汇总\n{'='*72}")
    for script, ok, dt in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {script:36s} {dt:6.1f}s")
    total = sum(d for _, _, d in results)
    print(f"  合计 {total/60:.1f} 分钟")

    if not args.no_build:
        print("\n[build] 重新生成 HTML 书 …")
        subprocess.run([sys.executable, os.path.join(HERE, "build_html.py")],
                       cwd=HERE)
        print(f"  → {os.path.join(HERE, 'out', 'world_model_book.html')}")

    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
