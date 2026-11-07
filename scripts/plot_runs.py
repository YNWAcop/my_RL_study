"""绘制训练 CSV，保留原始曲线及滑动均值。"""
import argparse
import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pattern", required=True)
    cli.add_argument("--metric", default="return")
    cli.add_argument("--window", type=int, default=20)
    cli.add_argument("--output", type=Path, default=Path("runs/comparison.png"))
    args = cli.parse_args()
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        cli.error("没有匹配的 CSV 文件")
    if args.window < 1:
        cli.error("window 必须大于零")
    figure, axis = plt.subplots(figsize=(9, 5))
    for path in paths:
        with open(path, encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows or args.metric not in rows[0]:
            cli.error(f"{path} 无记录或缺少指标 {args.metric}")
        groups = sorted(set(row.get("algo", "") for row in rows))
        for group in groups:
            selected = [row for row in rows if row.get("algo", "") == group]
            x = np.array([float(row["step"]) for row in selected])
            y = np.array([float(row[args.metric]) for row in selected])
            window = min(args.window, len(y))
            label = Path(path).parent.name + (f"/{group}" if group else "")
            line, = axis.plot(x[window - 1:], np.convolve(y, np.ones(window) / window, "valid"), label=label)
            axis.plot(x, y, color=line.get_color(), alpha=0.15, linewidth=0.6)
    axis.set(xlabel="Environment steps", ylabel=args.metric)
    axis.legend(fontsize=7)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
