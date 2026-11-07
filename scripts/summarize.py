"""聚合独立训练种子的最终评估，不把评估回合当作训练种子。"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pattern", required=True)
    args = cli.parse_args()
    groups = defaultdict(dict)
    for path in sorted(glob.glob(args.pattern)):
        directory = Path(path).parent
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        key = (directory.parent.name, config.get("algo", directory.parent.name),
               config.get("steps"), config.get("episodes"))
        seed = config["seed"]
        if seed in groups[key]:
            cli.error(f"同一组出现重复训练 seed={seed}，请收窄 pattern")
        groups[key][seed] = result["mean"]
    if not groups:
        cli.error("没有匹配的 evaluation.json")
    for key, values in sorted(groups.items()):
        means = list(values.values())
        std = float(np.std(means, ddof=1)) if len(means) > 1 else None
        print(key, "训练种子", sorted(values), "均值", float(np.mean(means)), "种子间样本标准差", std)


if __name__ == "__main__":
    main()
