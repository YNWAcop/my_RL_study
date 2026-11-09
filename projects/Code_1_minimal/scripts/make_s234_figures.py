"""生成 S2–S4 汇报用的静态图（PNG），供 Markdown 直接引用。

用法：D:/envs/conda-envs/paper/python.exe scripts/make_s234_figures.py
输出：docs/img/*.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#D5D8DC"
plt.rcParams["axes.labelcolor"] = "#2C3E50"
plt.rcParams["xtick.color"] = "#5D6D7E"
plt.rcParams["ytick.color"] = "#5D6D7E"
plt.rcParams["text.color"] = "#1B2631"

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "img")
os.makedirs(OUT, exist_ok=True)

L25 = [i * 25 for i in range(20)]

S2 = [71.0, 242.9, 241.0, 267.9, 401.2, 357.8, 429.6, 382.2, 396.7, 475.3,
      473.8, 465.9, 477.8, 483.3, 481.0, 494.4, 499.5, 495.7, 494.0, 496.1]

S3 = {
    "基线": ([71.0, 242.9, 241.0, 267.9, 401.2, 357.8, 429.6, 382.2, 396.7, 475.3,
            473.8, 465.9, 477.8, 483.3, 481.0, 494.4, 499.5, 495.7, 494.0, 496.1],
            "#1B2631", 2.8, "-"),
    "clip = 0（冻结）": ([22.5, 23.3, 22.6, 22.8, 23.3, 23.3, 23.5, 23.7, 23.3, 23.7,
                     24.4, 25.0, 25.8, 24.7, 25.5, 25.8, 24.7, 26.4, 25.5, 25.0],
                     "#C0392B", 2.8, "-"),
    "只更新 1 轮": ([33.2, 57.9, 113.5, 183.9, 230.4, 252.7, 230.6, 223.3, 286.4, 295.3,
                 270.6, 282.7, 234.4, 230.3, 280.1, 239.9, 283.6, 278.8, 306.4, 319.7],
                 "#D35400", 2.2, "-"),
    "clip = 5": ([62.0, 233.6, 346.4, 408.1, 472.5, 462.4, 463.1, 465.8, 500.0, 496.2,
                 487.1, 500.0, 497.3, 500.0, 500.0, 500.0, 500.0, 500.0, 495.7, 500.0],
                 "#2471A3", 1.4, "--"),
    "熵 = 0": ([66.7, 257.9, 251.4, 350.4, 460.0, 463.3, 483.2, 487.8, 359.7, 412.5,
               467.1, 453.9, 450.1, 374.1, 301.0, 310.2, 484.2, 497.0, 500.0, 500.0],
               "#B7950B", 1.4, "--"),
    "采样 1/8": ([104.1, 211.4, 500.0, 500.0, 500.0, 483.6, 485.0, 441.6, 399.6, 457.0,
                 472.5, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
                 "#6C3483", 1.4, "-."),
    "关优势归一化": ([67.1, 316.9, 304.6, 419.8, 341.4, 459.6, 482.7, 446.8, 486.4, 498.9,
                 444.8, 472.6, 491.6, 487.9, 500.0, 493.8, 472.6, 491.9, 496.9, 486.8],
                 "#C2185B", 1.4, "-."),
    "关梯度裁剪": ([90.8, 311.9, 253.7, 447.6, 461.4, 487.3, 375.3, 272.0, 204.7, 497.8,
                498.4, 497.3, 499.4, 500.0, 500.0, 500.0, 500.0, 500.0, 485.3, 500.0],
                "#148F77", 1.6, (0, (5, 1))),
}


def style(ax, xlabel, ylabel, ymax=520):
    ax.set_xlim(0, 486)
    ax.set_ylim(0, ymax)
    ax.grid(True, color="#EAECEE", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)


# ---------- 图 1：S2 学习曲线 ----------
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(L25, S2, color="#148F77", linewidth=3, solid_capstyle="round", zorder=3)
ax.fill_between(L25, S2, color="#148F77", alpha=0.12, zorder=2)
ax.scatter(L25, S2, s=34, color="white", edgecolor="#148F77", linewidth=1.8, zorder=4)
ax.axhline(400, color="#A93226", linestyle=(0, (5, 4)), linewidth=1.4, alpha=0.75, zorder=1)
ax.annotate("106,540 步首次连续 10 局 ≥ 400", xy=(106.5, 400), xytext=(150, 300),
            fontsize=10.5, color="#A93226",
            arrowprops=dict(arrowstyle="->", color="#A93226", lw=1.2,
                            connectionstyle="arc3,rad=-0.2"))
ax.annotate("满分 500", xy=(470, 500), xytext=(400, 505), fontsize=10, color="#148F77")
style(ax, "训练步数（千步）", "episode 回报（每 2.5 万步均值）")
ax.set_title("S2 · PPO 在 CartPole 上的学习曲线（500k 步，seed=1）",
             fontsize=13.5, pad=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "s2_curve.png"), dpi=160)
plt.close(fig)

# ---------- 图 2：S3 八条曲线 ----------
fig, ax = plt.subplots(figsize=(11, 6.0))
for name, (data, color, lw, ls) in S3.items():
    ax.plot(L25, data, label=name, color=color, linewidth=lw, linestyle=ls,
            solid_capstyle="round", zorder=3 if lw > 2 else 2)
style(ax, "训练步数（千步）", "平均回报（每 2.5 万步分桶）")
ax.set_title("S3 · 七次破坏后的回报曲线对比", fontsize=13.5, pad=14, fontweight="bold")
fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=10.5,
           bbox_to_anchor=(0.5, -0.02), handlelength=2.6)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(OUT, "s3_curves.png"), dpi=160)
plt.close(fig)

# ---------- 图 3：达标步数条形 ----------
labels = ["关梯度裁剪", "关优势归一化", "采样 1/8", "clip 放宽到 5",
          "熵系数归零", "基线", "只更新 1 轮", "clip 归零"]
vals = [31, 39, 46, 63, 90, 107, 226, 500]
colors = ["#148F77"] * 5 + ["#7F8C8D", "#D35400", "#C0392B"]
texts = ["31k", "39k", "46k", "63k", "90k", "107k", "226k", "从未达标"]

fig, ax = plt.subplots(figsize=(10.5, 5.4))
y = np.arange(len(labels))[::-1]
ax.barh(y, vals, height=0.58, color=colors, zorder=3)
for yi, v, t in zip(y, vals, texts):
    ax.text(v + 8, yi, t, va="center", fontsize=10.5,
            color="#C0392B" if v >= 500 else "#2C3E50",
            fontweight="bold" if v >= 500 else "normal")
ax.axvline(107, color="#7F8C8D", linestyle=":", linewidth=1.2, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlim(0, 560)
ax.grid(True, axis="x", color="#EAECEE", linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("达到 400 分所需训练步数（千步）", fontsize=11)
ax.set_title("S3 · 各配置的收敛速度（越短越快）", fontsize=13.5, pad=14, fontweight="bold")
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "s3_bars.png"), dpi=160)
plt.close(fig)

# ---------- 图 4：PPO 数据流 ----------
fig, ax = plt.subplots(figsize=(10.5, 7.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

boxes = [
    (7.5, "① 采样 rollout", "4 个环境 × 128 步 = 512 条转移"),
    (5.5, "② 倒序算优势", "GAE 用未来回报修正，γ=0.99 λ=0.95"),
    (3.5, "③ 归一化并摊平", "(adv − mean)/std，打乱成 16 个 minibatch"),
    (1.5, "④ 裁剪目标更新", "ratio 限在 [0.8, 1.2]，4 轮 × 4 批梯度"),
]
for y, title, sub in boxes:
    box = FancyBboxPatch((0.6, y - 0.62), 6.4, 1.24,
                         boxstyle="round,pad=0.02,rounding_size=0.12",
                         linewidth=1.2, edgecolor="#148F77", facecolor="#E8F6F1", zorder=2)
    ax.add_patch(box)
    ax.text(3.8, y + 0.24, title, ha="center", va="center",
            fontsize=14, fontweight="bold", color="#0E6655", zorder=3)
    ax.text(3.8, y - 0.26, sub, ha="center", va="center",
            fontsize=10.5, color="#1C7C63", zorder=3)

for y0, y1 in [(7.5, 5.5), (5.5, 3.5), (3.5, 1.5)]:
    ax.annotate("", xy=(3.8, y1 + 0.66), xytext=(3.8, y0 - 0.66),
                arrowprops=dict(arrowstyle="-|>", color="#148F77", lw=1.8,
                                shrinkA=0, shrinkB=0), zorder=1)

# 回流线
ax.plot([7.0, 8.6, 8.6, 7.0], [1.5, 1.5, 7.5, 7.5], color="#7F8C8D",
        linewidth=1.5, linestyle=(0, (5, 4)), zorder=1)
ax.annotate("", xy=(7.0, 7.5), xytext=(7.35, 7.5),
            arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=1.5), zorder=1)
ax.text(8.85, 4.5, "参数更新后\n回到 ①\n\n共 976 轮", fontsize=10.5,
        color="#5D6D7E", va="center", ha="left", linespacing=1.7)

ax.set_title("PPO 训练主循环：一次 iteration 里发生什么",
             fontsize=14, pad=18, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ppo_flow.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

print("已生成：")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"  {f}  {os.path.getsize(p)/1024:.0f} KB")
