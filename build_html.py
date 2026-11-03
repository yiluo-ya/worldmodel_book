"""build_html.py —— 把 chapters/*.md 编成一本单文件 HTML 书

设计目标
--------
1. 单文件、可离线携带：CSS 内联、实验图片全部 base64 内嵌
2. 好看：侧边栏目录、进度条、深浅主题、代码卡片、表格、公式（KaTeX）
3. 无刷新切章：所有章节放在一个 HTML 里，用 hash 路由切换

用法
----
    ../../../.venv/Scripts/python.exe build_html.py
    → 生成 out/world_model_book.html（双击即可阅读）
"""

from __future__ import annotations

import base64
import html
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CH_DIR = os.path.join(HERE, "chapters")
OUT_DIR = os.path.join(HERE, "out")
VENDOR = os.path.join(HERE, "vendor")
OUT_NAME = os.path.join(OUT_DIR, "world_model_book.html")

# 章节顺序（文件名 → 显示信息）
CHAPTERS = [
    ("ch01_what_is_world_model.md", "第 1 章　世界模型是什么", "定义、三种类型、第一个实验"),
    ("ch02_math_skeleton.md", "第 2 章　数学骨架", "MDP、确定性假设、dt 的陷阱"),
    ("ch03_function_approximators.md", "第 3 章　函数逼近器", "线性→MLP，以及一次反直觉的失败"),
    ("ch04_training_details.md", "第 4 章　训练细节", "残差、标准化、输入噪声"),
    ("ch05_compounding_error.md", "第 5 章　复合误差", "误差怎么复利、视界怎么定"),
    ("ch06_ood_uncertainty.md", "第 6 章　分布外与不确定性", "外推差 50 倍、集成 σ、安全阀"),
    ("ch07_chaos_horizon.md", "第 7 章　混沌与硬上限", "Lyapunov 指数、能量探针"),
    ("ch08_mpc_planning.md", "第 8 章　用模型做规划", "MPC + CEM，五模型对比"),
    ("ch09_learning_in_imagination.md", "第 9 章　在想象中学习", "想象 vs 真实的账本"),
    ("ch10_pixels_latent.md", "第 10 章　像素与 latent", "2304 维观测、指标打架"),
    ("ch11_partial_observability.md", "第 11 章　部分可观测", "看不见速度怎么办"),
    ("ch12_evaluation.md", "第 12 章　怎么评价", "六层指标体系与报告模板"),
    ("ch13_structured_priors.md", "第 13 章　结构化先验", "白盒、灰盒、哈密顿网络"),
    ("ch14_data.md", "第 14 章　数据", "量、覆盖、质量，与时间对齐"),
    ("ch15_training_paradigms.md", "第 15 章　训练范式", "离线/Dyna/联合训练"),
    ("ch16_failure_modes.md", "第 16 章　失败模式与配平", "视界、漏洞、自欺"),
    ("ch17_applications.md", "第 17 章　应用地图", "机器人、自驾、游戏、科学"),
    ("ch18_frontier.md", "第 18 章　前沿与路线", "争议、开放问题、12 周计划"),
    ("appendix.md", "附录　实验手册与术语表", "环境、FAQ、数学、术语"),
]

# 每章附带的实验图：(文件名, 图注)
FIGURES = {
    "ch02_math_skeleton.md": [
        ("exp02_compound_error.png", "图 2-1　复合误差：开环推演的误差随视界增长，介于 √h 与 h 之间"),
        ("exp02_timescale.png", "图 2-2　dt 越小单步越准，但同样 1 秒的开环误差几乎不变"),
    ],
    "ch03_function_approximators.md": [
        ("exp03_openloop_narrow.png", "图 3-1　窄覆盖数据：mlp-plain 直接飞出去，加噪声的版本跟得住"),
        ("exp03_noise_scan.png", "图 3-2　噪声倍率扫描：两种覆盖的最优点都在 0.005，但收益相差 4 倍"),
        ("exp03_learning_curve.png", "图 3-3　数据量 vs 精度：线性从头到尾平坦，MLP 单调改善"),
    ],
    "ch04_training_details.md": [
        ("exp03_openloop_wide.png", "图 4-1　宽覆盖数据（杆会翻转）下三个模型的开环误差"),
        ("exp03_traj.png", "图 4-2　窄覆盖数据下同一条动作序列：plain 跑飞，robust 跟住"),
    ],
    "ch05_compounding_error.md": [
        ("exp02_traj_compare.png", "图 5-1　同一条真实动作序列，模型自己的 rollout 如何飘走"),
    ],
    "ch06_ood_uncertainty.md": [
        ("exp04_calibration.png", "图 6-1　集成 σ 与实际误差：单调相关，但数值不是上界"),
    ],
    "ch07_chaos_horizon.md": [
        ("exp07_lyapunov.png", "图 7-1　两条几乎相同的轨迹，分离量在半对数坐标下是直线——这就是 λ"),
        ("exp07_horizon.png", "图 7-2　开环误差：爬升后贴在混沌地板上（1.0 = 等于瞎猜）"),
        ("exp07_energy.png", "图 7-3　能量探针：真实轨迹严格守恒，模型想象出的轨迹能量暴涨"),
    ],
    "ch08_mpc_planning.md": [
        ("exp05_angles.png", "图 8-1　五个控制器的杆角轨迹：只有 analytic（完美模型）收敛到 0 并保持"),
        ("exp05_cart.png", "图 8-2　小车位置：失败的那些在乱冲"),
    ],
    "ch09_learning_in_imagination.md": [
        ("exp08_traj_C2_learned.png", "图 9-1　在学到的模型里优化出的“最优策略”，在真实世界里其实在打转"),
        ("exp08_traj_B_cem_real.png", "图 9-2　同样优化，但在真实环境里评估——立刻能立住"),
    ],
    "ch10_pixels_latent.md": [
        ("exp06_rollout_frames.png", "图 10-1　5 行 × 5 列（0..20 步）：真实 / latent / latent+clamp / 像素线性 / freeze"),
        ("exp06_error.png", "图 10-2　像素 RMSE：排第一的是 freeze（什么都不预测）"),
        ("exp06_angle.png", "图 10-3　换成任务相关指标（解码出的杆角误差），排名翻转"),
    ],
    "ch11_partial_observability.md": [
        ("exp09_rollout.png", "图 11-1　同一物理、不同“眼睛”：五种观测表示的开环误差"),
    ],
    "ch16_failure_modes.md": [
        ("exp10_horizon.png", "图 16-1　规划视界扫描：完美模型越长越好，学到的模型全线失败"),
        ("exp10_model_error.png", "图 16-2　模型自己的误差随视界增长——这就是视界预算的依据"),
    ],
    "ch07_chaos_horizon.md#extra": [],
}


# ---------------------------------------------------------------------------
# Markdown → HTML
# ---------------------------------------------------------------------------
def esc(s: str) -> str:
    return html.escape(s, quote=False)


class MD:
    """够用的 markdown 转换器：标题/段落/列表/表格/引用/代码/公式。"""

    def __init__(self):
        self.codes: list[tuple[str, str]] = []
        self.maths: list[str] = []
        self.inlines: list[str] = []

    def convert(self, md: str) -> str:
        md = md.replace("\r\n", "\n")
        md = self._stash_code(md)
        md = self._stash_math(md)
        lines = md.split("\n")
        out: list[str] = []
        i = 0
        n = len(lines)

        def flush_para(buf):
            if buf:
                out.append("<p>" + self._inline(" ".join(x.strip() for x in buf)) + "</p>")
                buf.clear()

        para: list[str] = []
        while i < n:
            line = lines[i]
            s = line.strip()

            if s == "":
                flush_para(para)
                i += 1
                continue

            # 代码块占位
            m = re.fullmatch(r"\x00CODE(\d+)\x00", s)
            if m:
                flush_para(para)
                out.append(self._render_code(int(m.group(1))))
                i += 1
                continue

            # 独立公式占位
            m = re.fullmatch(r"\x00MATH(\d+)\x00", s)
            if m:
                flush_para(para)
                out.append('<div class="math-block">$$' + esc(self.maths[int(m.group(1))]) + '$$</div>')
                i += 1
                continue

            # 分隔线
            if re.fullmatch(r"-{3,}|\*{3,}", s):
                flush_para(para)
                out.append('<hr class="sep">')
                i += 1
                continue

            # 标题
            m = re.match(r"^(#{1,6})\s+(.*)$", s)
            if m:
                flush_para(para)
                lvl = len(m.group(1))
                txt = self._inline(m.group(2))
                anchor = "h-" + re.sub(r"[^\w\u4e00-\u9fff]+", "-", m.group(2))[:40]
                out.append(f'<h{lvl} id="{anchor}">{txt}</h{lvl}>')
                i += 1
                continue

            # 表格
            if s.startswith("|") and i + 1 < n and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
                flush_para(para)
                html_tbl, i = self._render_table(lines, i)
                out.append(html_tbl)
                continue

            # 引用块
            if s.startswith(">"):
                flush_para(para)
                buf = []
                while i < n and lines[i].strip().startswith(">"):
                    buf.append(lines[i].strip().lstrip(">").strip())
                    i += 1
                body = " ".join(x for x in buf if x)
                out.append('<blockquote>' + self._inline(body) + "</blockquote>")
                continue

            # 列表
            if re.match(r"^([-*+]|\d+\.)\s+", s):
                flush_para(para)
                html_list, i = self._render_list(lines, i)
                out.append(html_list)
                continue

            para.append(line)
            i += 1

        flush_para(para)
        body = "\n".join(out)
        body = self._unstash_math(body)
        return body

    # -- 块级渲染 --
    def _render_table(self, lines, i):
        def cells(row: str):
            row = row.strip()
            if row.startswith("|"):
                row = row[1:]
            if row.endswith("|"):
                row = row[:-1]
            return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", row)]

        header = cells(lines[i])
        i += 2
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            rows.append(cells(lines[i]))
            i += 1
        head = "".join(f"<th>{self._inline(c)}</th>" for c in header)
        body = ""
        for r in rows:
            body += "<tr>" + "".join(f"<td>{self._inline(c)}</td>" for c in r) + "</tr>"
        return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>', i

    def _render_list(self, lines, i):
        ordered = bool(re.match(r"^\d+\.\s+", lines[i].strip()))
        tag = "ol" if ordered else "ul"
        items = []
        cur = None
        n = len(lines)
        while i < n:
            s = lines[i].strip()
            if s == "":
                # 空行后若还是列表项则继续
                if i + 1 < n and re.match(r"^([-*+]|\d+\.)\s+", lines[i + 1].strip()):
                    i += 1
                    continue
                break
            m = re.match(r"^([-*+]|\d+\.)\s+(.*)$", s)
            if m:
                if cur is not None:
                    items.append(cur)
                cur = [m.group(2)]
            elif cur is not None:
                cur.append(s)          # 续行
            else:
                break
            i += 1
        if cur is not None:
            items.append(cur)
        body = "".join("<li>" + self._inline(" ".join(it)) + "</li>" for it in items)
        return f"<{tag}>{body}</{tag}>", i

    def _render_code(self, idx):
        lang, code = self.codes[idx]
        cls = "code-block" + (f" lang-{lang}" if lang else "")
        body = highlight_python(code) if lang in ("python", "py", "") else esc(code)
        return (f'<div class="{cls}"><div class="code-head">'
                f'<span class="dot"></span><span class="dot"></span><span class="dot"></span>'
                f'<span class="lang">{esc(lang or "text")}</span></div>'
                f'<pre><code>{body}</code></pre></div>')

    # -- 行内 --
    def _inline(self, s: str) -> str:
        s = esc(s)
        s = re.sub(r"`([^`]+)`", r'<code class="inline">\1</code>', s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noreferrer">\1</a>', s)

        # 公式放最后：避免 LaTeX 里的 * _ | 被 markdown 正则破坏
        def inl(m):
            return ('<span class="math-inline">$' + esc(self.inlines[int(m.group(1))]) + "$</span>")
        return re.sub(r"\x00INL(\d+)\x00", inl, s)

    # -- 占位 --
    def _stash_code(self, md):
        def f(m):
            self.codes.append((m.group(1) or "", m.group(2)))
            return f"\x00CODE{len(self.codes)-1}\x00"
        return re.sub(r"```(\w*)\n(.*?)```", f, md, flags=re.S)

    def _stash_math(self, md):
        def fd(m):
            self.maths.append(m.group(1))
            return f"\n\x00MATH{len(self.maths)-1}\x00\n"
        md = re.sub(r"\$\$(.+?)\$\$", fd, md, flags=re.S)

        def fi(m):
            self.inlines.append(m.group(1))
            return f"\x00INL{len(self.inlines)-1}\x00"
        md = re.sub(r"(?<![\\$])\$([^\$\n]+?)\$(?![0-9])", fi, md)
        return md

    def _unstash_math(self, body):
        return body


PY_KEYWORDS = {
    "def", "class", "return", "if", "elif", "else", "for", "while", "in", "not",
    "and", "or", "import", "from", "as", "with", "try", "except", "finally",
    "lambda", "None", "True", "False", "self", "pass", "break", "continue",
    "yield", "global", "raise", "assert", "is", "del", "async", "await",
}


def highlight_python(code: str) -> str:
    """极简高亮：字符串 → 注释 → 关键字 → 数字，用占位符避免嵌套。"""
    s = esc(code)
    slots: list[str] = []

    def keep(frag: str, cls: str) -> str:
        slots.append(f'<span class="{cls}">{frag}</span>')
        return f"\x01{len(slots)-1}\x01"

    # 三引号字符串、普通字符串、注释
    s = re.sub(r'""".*?"""', lambda m: keep(m.group(0), "s-str"), s, flags=re.S)
    s = re.sub(r"'''.*?'''", lambda m: keep(m.group(0), "s-str"), s, flags=re.S)
    s = re.sub(r'"[^"\n]*"', lambda m: keep(m.group(0), "s-str"), s)
    s = re.sub(r"'[^'\n]*'", lambda m: keep(m.group(0), "s-str"), s)
    s = re.sub(r"#[^\n]*", lambda m: keep(m.group(0), "s-com"), s)
    # 关键字（单词边界）
    def kw(m):
        w = m.group(0)
        return keep(w, "s-kw") if w in PY_KEYWORDS else w
    s = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", kw, s)
    # 数字要避开占位符里的索引数字，否则会把占位符拆碎
    s = re.sub(r"(?<!\x01)\b\d+\.?\d*\b(?!\x01)", lambda m: keep(m.group(0), "s-num"), s)

    def back(m):
        return slots[int(m.group(1))]
    for _ in range(3):                    # 占位符可能被再次包住，展开几轮
        s = re.sub(r"\x01(\d+)\x01", back, s)
    return s


# ---------------------------------------------------------------------------
# 图片
# ---------------------------------------------------------------------------
def img_tag(path: str, caption: str) -> str:
    full = os.path.join(OUT_DIR, path)
    if not os.path.exists(full):
        return f'<figure class="missing"><figcaption>（缺图：{esc(path)}）</figcaption></figure>'
    with open(full, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (
        '<figure class="fig">'
        f'<img src="data:image/png;base64,{b64}" alt="{esc(caption)}" loading="lazy" '
        'onclick="openLightbox(this.src)">'
        f"<figcaption>{esc(caption)}</figcaption></figure>"
    )


# ---------------------------------------------------------------------------
# KaTeX 内嵌（做成真正可离线的单文件）
# ---------------------------------------------------------------------------
def inline_katex() -> tuple[str, str, str]:
    """返回 (css, katex_js, auto_render_js)。vendor 缺失时返回空串。"""
    css_path = os.path.join(VENDOR, "katex.min.css")
    js_path = os.path.join(VENDOR, "katex.min.js")
    ar_path = os.path.join(VENDOR, "contrib", "auto-render.min.js")
    if not (os.path.exists(css_path) and os.path.exists(js_path)):
        print("[warn] vendor/katex 不存在，公式将依赖 CDN（离线打不开）")
        return "", "", ""

    with open(css_path, encoding="utf-8") as f:
        css = f.read()

    font_dir = os.path.join(VENDOR, "fonts")

    def font_url(m):
        rel = m.group(1).strip("'\"")
        name = os.path.basename(rel)
        path = os.path.join(font_dir, name)
        if not name.endswith(".woff2") or not os.path.exists(path):
            return "url(about:blank)"      # woff/ttf 一律丢掉，只留 woff2
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"url(data:font/woff2;base64,{b64}) format('woff2')"

    # 字体声明形式：src:url(fonts/X.woff2) format('woff2'),url(fonts/X.woff)...
    css = re.sub(r"url\(([^)]+?\.(?:woff2|woff|ttf))\)(?:\s*format\([^)]*\))?", font_url, css)
    css = re.sub(r",\s*url\(about:blank\)(?:\s*format\([^)]*\))?", "", css)

    with open(js_path, encoding="utf-8") as f:
        kjs = f.read()
    ar = ""
    if os.path.exists(ar_path):
        with open(ar_path, encoding="utf-8") as f:
            ar = f.read()
    return css, kjs, ar


# ---------------------------------------------------------------------------
# 模板
# ---------------------------------------------------------------------------
CSS = r"""
:root{
  --bg:#fbfaf7; --bg2:#ffffff; --ink:#1c1c1e; --ink2:#5b5b60; --line:#e6e3dc;
  --accent:#c2410c; --accent2:#0f766e; --code-bg:#1e2430; --code-ink:#dfe6f2;
  --sidebar:#f3f0ea; --shadow:0 1px 3px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.05);
}
html[data-theme="dark"]{
  --bg:#15171a; --bg2:#1c1f24; --ink:#e8e6e1; --ink2:#9aa0a6; --line:#2c3036;
  --accent:#fb923c; --accent2:#2dd4bf; --code-bg:#0f1216; --code-ink:#d6dee9;
  --sidebar:#191c20; --shadow:0 1px 3px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;background:var(--bg);color:var(--ink);
  font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Songti SC","Noto Serif CJK SC",Georgia,serif;
  font-size:17px;line-height:1.78;-webkit-font-smoothing:antialiased;
}
#progress{position:fixed;top:0;left:0;height:3px;width:0;background:linear-gradient(90deg,var(--accent),var(--accent2));z-index:60;transition:width .1s}
#sidebar{
  position:fixed;top:0;left:0;bottom:0;width:308px;background:var(--sidebar);
  border-right:1px solid var(--line);overflow-y:auto;padding:26px 18px 60px;z-index:40;
}
#sidebar h1{font-size:15.5px;margin:0 0 4px;letter-spacing:.02em}
#sidebar .sub{font-size:12px;color:var(--ink2);margin-bottom:18px;line-height:1.5}
#sidebar nav a{
  display:block;padding:7px 10px;border-radius:7px;color:var(--ink2);text-decoration:none;
  font-size:13.4px;line-height:1.45;border-left:2px solid transparent;
}
#sidebar nav a:hover{background:rgba(194,65,12,.08);color:var(--ink)}
#sidebar nav a.active{background:rgba(194,65,12,.12);color:var(--accent);border-left-color:var(--accent);font-weight:600}
#sidebar nav a .d{display:block;font-size:11.4px;color:var(--ink2);font-weight:400;margin-top:2px}
#sidebar .tools{display:flex;gap:8px;margin:14px 0 16px}
#sidebar .tools button{
  flex:1;border:1px solid var(--line);background:var(--bg2);color:var(--ink2);
  border-radius:7px;padding:6px 8px;font-size:12px;cursor:pointer;font-family:inherit;
}
#sidebar .tools button:hover{color:var(--ink);border-color:var(--accent)}
main{margin-left:308px;padding:52px 40px 140px}
main > article{max-width:840px;margin:0 auto}
body.js-ready article{display:none;scroll-margin-top:80px}
body.js-ready article.show{display:block}
.hero{
  margin:-14px 0 40px;padding:56px 40px 44px;border-radius:20px;
  background:radial-gradient(120% 140% at 0% 0%, #fde9d9 0%, #f7f2e7 45%, #eef4f1 100%);
  color:#211a14;box-shadow:var(--shadow);position:relative;overflow:hidden;
}
html[data-theme="dark"] .hero{
  background:radial-gradient(120% 140% at 0% 0%, #33221a 0%, #1c1f24 50%, #14201f 100%);
  color:#f2ece4;
}
.hero::after{
  content:"";position:absolute;right:-60px;bottom:-80px;width:280px;height:280px;border-radius:50%;
  background:radial-gradient(circle,rgba(194,65,12,.22),transparent 65%);
}
.hero-tag{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.14em;text-transform:uppercase;opacity:.6}
.hero-title{font-size:58px;line-height:1.05;margin:16px 0 10px;letter-spacing:-.02em;font-weight:700}
.hero-sub{font-size:16.5px;line-height:1.7;opacity:.85}
.hero-sub span{display:block;margin-top:8px;font-size:14px;opacity:.7;font-style:italic}
.hero-stats{display:flex;flex-wrap:wrap;gap:26px;margin:28px 0 30px}
.hero-stats div{display:flex;flex-direction:column}
.hero-stats b{font-size:26px;line-height:1;color:var(--accent)}
.hero-stats span{font-size:12px;opacity:.65;margin-top:5px}
.hero-btn{
  display:inline-block;padding:11px 22px;border-radius:999px;background:var(--accent);color:#fff;
  font-size:14.5px;font-weight:600;text-decoration:none;border:0;transition:transform .15s;
}
.hero-btn:hover{transform:translateX(3px)}
@media (max-width:860px){ .hero{padding:36px 22px 30px} .hero-title{font-size:40px} }
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
article h1{font-size:31px;line-height:1.3;margin:0 0 26px;letter-spacing:-.01em}
article h2{font-size:22.5px;margin:44px 0 14px;padding-bottom:7px;border-bottom:1px solid var(--line)}
article h3{font-size:18.5px;margin:30px 0 10px;color:var(--accent)}
article h4{font-size:16.5px;margin:22px 0 8px}
p{margin:13px 0}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid rgba(194,65,12,.3)}
a:hover{border-bottom-color:var(--accent)}
strong{font-weight:700}
code.inline{
  font-family:"JetBrains Mono",Consolas,"SF Mono",Menlo,monospace;font-size:.86em;
  background:rgba(120,120,120,.13);padding:1.5px 5px;border-radius:4px;
}
blockquote{
  margin:18px 0;padding:12px 18px;border-left:3px solid var(--accent);
  background:rgba(194,65,12,.055);border-radius:0 8px 8px 0;color:var(--ink);
}
blockquote p{margin:4px 0}
ul,ol{margin:13px 0;padding-left:26px}
li{margin:6px 0}
hr.sep{border:0;border-top:1px dashed var(--line);margin:34px 0}
.table-wrap{overflow-x:auto;margin:20px 0;border-radius:10px;border:1px solid var(--line);background:var(--bg2)}
table{border-collapse:collapse;width:100%;font-size:14.6px}
th,td{padding:9px 13px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{background:rgba(120,120,120,.07);font-weight:600;font-size:13.6px;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:rgba(120,120,120,.045)}
.figure-set{margin:30px 0 8px}
figure.fig{margin:22px 0;padding:0}
figure.fig img{
  width:100%;display:block;border-radius:10px;border:1px solid var(--line);
  background:#fff;cursor:zoom-in;box-shadow:var(--shadow);
}
figure.fig figcaption{font-size:13px;color:var(--ink2);margin-top:8px;text-align:center;line-height:1.55}
.code-block{
  margin:20px 0;border-radius:11px;overflow:hidden;background:var(--code-bg);
  box-shadow:var(--shadow);border:1px solid rgba(255,255,255,.06);
}
.code-head{display:flex;align-items:center;gap:6px;padding:9px 13px;background:rgba(255,255,255,.045)}
.code-head .dot{width:9px;height:9px;border-radius:50%;background:#4b5563;display:inline-block}
.code-head .lang{margin-left:auto;font-size:11.5px;color:#8892a4;letter-spacing:.06em;text-transform:uppercase;font-family:monospace}
.code-block pre{margin:0;padding:15px 17px;overflow-x:auto;line-height:1.62}
.code-block code{
  font-family:"JetBrains Mono",Consolas,"SF Mono",Menlo,monospace;
  font-size:13.2px;color:var(--code-ink);white-space:pre;
}
.s-kw{color:#c792ea;font-weight:600}
.s-str{color:#a5d6a7}
.s-com{color:#6b7c93;font-style:italic}
.s-num{color:#ffcb6b}
.math-block{margin:20px 0;text-align:center;overflow-x:auto;padding:6px 0}
.math-inline{padding:0 1px}
.toc-inline{display:none}
main .lede{font-size:15px;color:var(--ink2);margin:-14px 0 30px}
#lightbox{
  position:fixed;inset:0;background:rgba(8,10,14,.92);display:none;align-items:center;
  justify-content:center;z-index:80;padding:24px;cursor:zoom-out;
}
#lightbox.on{display:flex}
#lightbox img{max-width:100%;max-height:100%;border-radius:8px}
.nav-bottom{display:flex;justify-content:space-between;gap:12px;margin:52px 0 0;padding-top:22px;border-top:1px solid var(--line)}
.nav-bottom a{
  flex:1;padding:12px 16px;border:1px solid var(--line);border-radius:10px;background:var(--bg2);
  font-size:13.6px;color:var(--ink2);border-bottom:1px solid var(--line);
}
.nav-bottom a:hover{border-color:var(--accent);color:var(--accent)}
.nav-bottom .next{text-align:right}
#menu-btn{
  position:fixed;left:14px;top:12px;z-index:70;display:none;border:1px solid var(--line);
  background:var(--bg2);color:var(--ink);border-radius:8px;padding:6px 10px;cursor:pointer;
}
@media (max-width:1080px){ main{padding-left:28px;padding-right:28px} }
@media (max-width:860px){
  #sidebar{transform:translateX(-100%);transition:transform .22s;width:280px;box-shadow:var(--shadow)}
  #sidebar.open{transform:none}
  main{margin-left:0;padding:64px 20px 120px}
  #menu-btn{display:block}
  body.js-ready article{scroll-margin-top:76px}
  article h1{font-size:25px}
  body{font-size:16px}
}
@media print{
  #sidebar,#progress,#menu-btn,.nav-bottom{display:none}
  main{margin:0;padding:0;max-width:none}
  article{display:block !important;page-break-after:always}
}
"""

JS = r"""
/* app */
(function(){
if (window.__wmBook) return;
window.__wmBook = true;
document.body.classList.add('js-ready');
const chapters = Array.from(document.querySelectorAll('article'));
const links = Array.from(document.querySelectorAll('#sidebar nav a'));

function show(id, push){
  let idx = chapters.findIndex(a => a.id === id);
  if (idx < 0) idx = 0;
  chapters.forEach((a,i) => a.classList.toggle('show', i===idx));
  links.forEach((l,i) => l.classList.toggle('active', i===idx));
  if (push) history.replaceState(null,'','#'+chapters[idx].id);
  window.scrollTo({top:0, behavior:'instant'});
  document.title = chapters[idx].dataset.title + ' · 世界模型';
  try{ localStorage.setItem('wm-ch', chapters[idx].id); }catch(e){}
}
function currentIndex(){
  return chapters.findIndex(a=>a.classList.contains('show'));
}
links.forEach(l => l.addEventListener('click', e => {
  e.preventDefault();
  show(l.getAttribute('href').slice(1), true);
  if (window.innerWidth < 860) document.getElementById('sidebar').classList.remove('open');
}));

window.addEventListener('hashchange', () => show(location.hash.slice(1) || 'ch0', false));

document.getElementById('menu-btn').onclick = () =>
  document.getElementById('sidebar').classList.toggle('open');

// 主题
const root = document.documentElement;
function setTheme(t){
  root.dataset.theme = t;
  try{ localStorage.setItem('wm-theme', t); }catch(e){}
}
document.getElementById('theme-btn').onclick = () =>
  setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark');
try{
  setTheme(localStorage.getItem('wm-theme') ||
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
}catch(e){ setTheme('light'); }

// 字号
let fs = 17;
document.getElementById('font-btn').onclick = () => {
  fs = fs >= 20 ? 15 : fs + 1;
  document.body.style.fontSize = fs + 'px';
};

// 阅读进度
const bar = document.getElementById('progress');
window.addEventListener('scroll', () => {
  const h = document.documentElement.scrollHeight - window.innerHeight;
  bar.style.width = (h > 0 ? (window.scrollY / h) * 100 : 0) + '%';
}, {passive:true});

// 图片放大（挂到 window：img 的 onclick 需要全局可见）
window.openLightbox = function(src){
  const lb = document.getElementById('lightbox');
  lb.querySelector('img').src = src;
  lb.classList.add('on');
};
document.getElementById('lightbox').onclick = e =>
  e.currentTarget.classList.remove('on');
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('on');
  if (e.key === 'ArrowRight'){ const i = currentIndex(); if (i < chapters.length-1) show(chapters[i+1].id, true); }
  if (e.key === 'ArrowLeft'){ const i = currentIndex(); if (i > 0) show(chapters[i-1].id, true); }
});

// 初始化
let last = null;
try{ last = localStorage.getItem('wm-ch'); }catch(e){}
const startId = location.hash.slice(1) || last || chapters[0].id;
show(startId, false);

// 公式渲染（KaTeX 加载完成后）
function renderMath(){
  if (!window.renderMathInElement) return;
  renderMathInElement(document.body, {
    delimiters: [
      {left:'$$', right:'$$', display:true},
      {left:'$', right:'$', display:false}
    ],
    throwOnError: false,
    ignoredTags: ['script','noscript','style','textarea','pre','code']
  });
}
window.addEventListener('load', () => setTimeout(renderMath, 120));

// 暴露给调试/外部使用
window.__wmShow = show;
window.__wmRenderMath = renderMath;
})();
"""

HTML_HEAD = """<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>世界模型：从白板到能用的预测器</title>
__KATEX_CSS__
<style>__CSS__</style>
</head>
<body>
<div id="progress"></div>
<button id="menu-btn">☰ 目录</button>
<aside id="sidebar">
  <h1>世界模型</h1>
  <div class="sub">从白板到能用的预测器<br>18 章 · 10 个可跑实验</div>
  <div class="tools">
    <button id="theme-btn">深/浅色</button>
    <button id="font-btn">字号</button>
  </div>
  <nav>__NAV__</nav>
</aside>
<main>
__CONTENT__
</main>
<div id="lightbox"><img alt=""></div>
__KATEX_JS__
<script>
__JS__
</script>
</body>
</html>
"""


def build_cover() -> str:
    hero = """
<div class="hero">
  <div class="hero-tag">interactive book · 18 chapters · 10 runnable experiments</div>
  <div class="hero-title">世界模型</div>
  <div class="hero-sub">从白板到能用的预测器<br><span>“如果在模型里试错，比在现实里试错便宜，那它就值得做。”</span></div>
  <div class="hero-stats">
    <div><b>18</b><span>章</span></div>
    <div><b>10</b><span>个实验</span></div>
    <div><b>60+</b><span>条实测读数</span></div>
    <div><b>15</b><span>分钟跑完全部代码</span></div>
  </div>
  <a class="hero-btn" href="#ch1">开始阅读 →</a>
</div>
"""
    readme = os.path.join(HERE, "README.md")
    body = ""
    if os.path.exists(readme):
        with open(readme, encoding="utf-8") as f:
            md = f.read()
        md = re.sub(r"^#\s+.*?\n", "", md, count=1)   # 大标题已在 hero 里
        body = MD().convert(md)
    return hero + body


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    arts = []
    nav = []

    # 封面
    arts.append(
        '<article id="ch0" class="show" data-title="封面">'
        + build_cover() + "</article>"
    )
    nav.append('<a href="#ch0"><strong>封面与导读</strong><span class="d">一句话总结全书</span></a>')

    for idx, (fname, title, desc) in enumerate(CHAPTERS, start=1):
        path = os.path.join(CH_DIR, fname)
        if not os.path.exists(path):
            print(f"[warn] 缺章节: {fname}")
            continue
        with open(path, encoding="utf-8") as f:
            md = f.read()

        # 去掉 md 的第一个一级标题（HTML 里用章节标题）
        md = re.sub(r"^#\s+.*?\n", "", md, count=1)

        body = MD().convert(md)

        figs = FIGURES.get(fname, [])
        if figs:
            body += '<div class="figure-set"><h2>本章实验输出</h2>'
            body += "".join(img_tag(p, c) for p, c in figs)
            body += "</div>"

        prev_link = f'<a class="prev" href="#ch{idx-1}">← 上一章</a>' if idx > 1 else '<a class="prev" href="#ch0">← 封面</a>'
        nxt = idx + 1 if idx < len(CHAPTERS) else None
        next_link = (f'<a class="next" href="#ch{nxt}">下一章 →</a>' if nxt
                     else '<a class="next" href="#ch0">回到封面 →</a>')
        body += f'<div class="nav-bottom">{prev_link}{next_link}</div>'

        arts.append(f'<article id="ch{idx}" data-title="{esc(title)}">'
                    f'<h1>{esc(title)}</h1><p class="lede">{esc(desc)}</p>{body}</article>')
        nav.append(f'<a href="#ch{idx}">{esc(title)}<span class="d">{esc(desc)}</span></a>')

    kcss, kjs, kar = inline_katex()
    if kcss:
        katex_css = "<style>\n" + kcss + "\n</style>"
        katex_js = ("<script>\n" + kjs + "\n</script>\n"
                    + "<script>\n" + kar + "\n</script>")
    else:
        katex_css = ('<link rel="stylesheet" '
                     'href="https://registry.npmmirror.com/katex/0.16.9/files/dist/katex.min.css">')
        katex_js = ('<script src="https://registry.npmmirror.com/katex/0.16.9/files/dist/katex.min.js"></script>\n'
                    '<script src="https://registry.npmmirror.com/katex/0.16.9/files/dist/contrib/auto-render.min.js"></script>')

    page = (HTML_HEAD.replace("__KATEX_CSS__", katex_css)
                   .replace("__KATEX_JS__", katex_js)
                   .replace("__CSS__", CSS)
                   .replace("__NAV__", "\n".join(nav))
                   .replace("__CONTENT__", "\n".join(arts))
                   .replace("__JS__", JS))
    with open(OUT_NAME, "w", encoding="utf-8") as f:
        f.write(page)

    size = os.path.getsize(OUT_NAME) / 1024
    print(f"[ok] {OUT_NAME}")
    print(f"     章节 {len(arts)-1} + 封面，共 {size:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
