"""只共享机械性操作，算法更新保留在章节中。"""
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--output", type=Path, default=Path("runs"))
    return result


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class Run:
    def __init__(self, chapter, args, columns):
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        label = getattr(args, "algo", chapter)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.path = args.output / chapter / f"{label}-seed{args.seed}-{stamp}"
        self.path.mkdir(parents=True)
        (self.path / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        self.file = (self.path / "train.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        self.writer.writeheader()
        print(f"日志目录 {self.path}")

    def log(self, **values):
        self.writer.writerow(values)
        self.file.flush()

    def close(self):
        self.file.close()
