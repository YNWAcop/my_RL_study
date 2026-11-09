"""生成 KNN 章节的图，供 S234_汇报.md 引用。

用法：D:/envs/conda-envs/paper/python.exe scripts/make_knn_figures.py
输出：docs/img/knn_boundary.png、docs/img/knn_eval.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#D5D8DC"
plt.rcParams["text.color"] = "#1B2631"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img")
os.makedirs(OUT, exist_ok=True)

CSV = os.path.join(ROOT, "data", "iris.csv")


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = np.asarray(X), np.asarray(y)

    def predict(self, X):
        X = np.asarray(X)
        out = []
        for x in X:
            dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            idx = dis.argsort()[: self.k]
            out.append(np.bincount(self.y[idx]).argmax())
        return np.asarray(out)


data = pd.read_csv(CSV, header=None,
                   names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
data["species"] = data["species"].map({"Iris-virginica": 0, "Iris-setosa": 1, "Iris-versicolor": 2})
data = data.drop_duplicates()

parts = {c: data[data["species"] == c].sample(len(data[data["species"] == c]), random_state=0)
         for c in (0, 1, 2)}
train = pd.concat([parts[c].iloc[:40] for c in (0, 1, 2)], axis=0)
test = pd.concat([parts[c].iloc[40:] for c in (0, 1, 2)], axis=0)

train_X, train_y = train.iloc[:, :-1], train.iloc[:, -1]
test_X, test_y = test.iloc[:, :-1], test.iloc[:, -1]

CN = {0: "virginica", 1: "setosa", 2: "versicolor"}
COLOR = {0: "#8E44AD", 1: "#2E86C1", 2: "#E67E22"}
MARKER = {0: "s", 1: "o", 2: "^"}

# ---------------- 图 1：二维决策边界 ----------------
FX, FY = "petal_length", "petal_width"
Xtr = train_X[[FX, FY]].to_numpy()
ytr = train_y.to_numpy()
Xte = test_X[[FX, FY]].to_numpy()

k_best = 7
m = KNN(k_best)
m.fit(Xtr, ytr)
acc2d = (m.predict(Xte) == test_y.to_numpy()).mean()

h = 0.02
x_min, x_max = Xtr[:, 0].min() - 0.4, Xtr[:, 0].max() + 0.4
y_min, y_max = Xtr[:, 1].min() - 0.3, Xtr[:, 1].max() + 0.3
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(9.6, 6.4))
light = ListedColormap(["#E8DAEF", "#D6EAF8", "#FDEBD0"])
ax.pcolormesh(xx, yy, Z, cmap=light, shading="auto", alpha=0.85, zorder=1)

for c in (0, 1, 2):
    sel = ytr == c
    ax.scatter(Xtr[sel, 0], Xtr[sel, 1], s=52, c=COLOR[c], marker=MARKER[c],
               edgecolor="white", linewidth=0.8, label=f"训练 · {CN[c]}", zorder=3)

wrong = 0
for i in range(len(Xte)):
    pred = m.predict(Xte[i:i+1])[0]
    ok = pred == test_y.to_numpy()[i]
    wrong += (0 if ok else 1)
    ax.scatter(Xte[i, 0], Xte[i, 1], s=170, facecolor="none",
               edgecolor="#C0392B" if not ok else "#2C3E50",
               linewidth=2.0 if not ok else 1.2,
               marker="o", zorder=4)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("花瓣长度 petal_length (cm)", fontsize=11)
ax.set_ylabel("花瓣宽度 petal_width (cm)", fontsize=11)
ax.tick_params(labelsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=10.5, markerscale=1.1)
ax.set_title(f"手写 KNN 的决策边界（k={k_best}，仅用两个花瓣特征）",
             fontsize=13.5, pad=14, fontweight="bold")
ax.text(0.985, 0.03, f"空心圈 = 27 个测试样本，红圈 = 判错（{wrong} 个）",
        transform=ax.transAxes, ha="right", fontsize=10, color="#5D6D7E")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "knn_boundary.png"), dpi=160)
plt.close(fig)

# ---------------- 图 2：k 值扫描 + 混淆矩阵 ----------------
ks = [1, 3, 5, 7, 9, 11, 15, 21, 31]
accs = []
for k in ks:
    m = KNN(k)
    m.fit(train_X, train_y)
    accs.append((m.predict(test_X) == test_y.to_numpy()).mean())

m3 = KNN(3)
m3.fit(train_X, train_y)
pred3 = m3.predict(test_X)
cm = np.zeros((3, 3), int)
for t, p in zip(test_y, pred3):
    cm[int(t), int(p)] += 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8),
                               gridspec_kw={"width_ratios": [1.35, 1]})

ax1.plot(ks, [a * 100 for a in accs], color="#148F77", linewidth=2.4,
         marker="o", markersize=7, markerfacecolor="white",
         markeredgecolor="#148F77", markeredgewidth=1.8, zorder=3)
ax1.axhline(100, color="#7F8C8D", linestyle=":", linewidth=1.1, zorder=1)
best = max(accs)
for k, a in zip(ks, accs):
    if a == best:
        ax1.scatter([k], [a * 100], s=180, facecolor="#148F77", edgecolor="white",
                    linewidth=1.6, zorder=4)
ax1.set_xlabel("邻居数 k", fontsize=11)
ax1.set_ylabel("测试集准确率 (%)", fontsize=11)
ax1.set_ylim(93, 101.5)
ax1.set_xticks(ks)
ax1.grid(True, color="#EAECEE", linewidth=0.8)
ax1.set_axisbelow(True)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
ax1.tick_params(labelsize=10)
ax1.set_title("k 值扫描（绿点 = 最优 k = 7~11）", fontsize=12, pad=12, fontweight="bold")

im = ax2.imshow(cm, cmap="YlGnBu", alpha=0.9)
ax2.set_xticks(range(3))
ax2.set_yticks(range(3))
ax2.set_xticklabels([CN[i] for i in range(3)], fontsize=10, rotation=18)
ax2.set_yticklabels([CN[i] for i in range(3)], fontsize=10)
ax2.set_xlabel("预测类别", fontsize=11)
ax2.set_ylabel("真实类别", fontsize=11)
for i in range(3):
    for j in range(3):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                 color="#C0392B" if i != j and cm[i, j] > 0 else "#1B2631",
                 fontweight="bold" if i != j and cm[i, j] > 0 else "normal")
ax2.set_title("k = 3 的混淆矩阵", fontsize=12, pad=12, fontweight="bold")
for s in ax2.spines.values():
    s.set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "knn_eval.png"), dpi=160)
plt.close(fig)

print("已生成：")
for f in ("knn_boundary.png", "knn_eval.png"):
    p = os.path.join(OUT, f)
    print(f"  {f}  {os.path.getsize(p)/1024:.0f} KB")
print(f"\n二维（花瓣两特征）k=7 准确率: {acc2d:.4f}")
