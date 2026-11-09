"""按学习目录显示、运行和检查每一步。默认显示步骤清单。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def load_steps():
    return json.loads((ROOT / "lessons/study_plan.json").read_text(encoding="utf-8"))["steps"]


def get_step(steps, identifier, cli):
    if not identifier.strip():
        cli.error("步骤编号不能为空")
    identifier = identifier.split(maxsplit=1)[0].zfill(2)
    for step in steps:
        if step["id"] == identifier:
            return step
    cli.error("步骤编号不存在，可先运行 list 查看")


def command(step, quick=False, seed=None, extra=()):
    arguments = step["quick_args"] if quick else step["args"]
    result = [sys.executable, "-X", "utf8", *step["entry"], *arguments]
    if seed is not None:
        result.extend(["--seed", str(seed)])
    return [*result, *extra]


def show_step(step):
    print(step["id"], step["title"])
    print("阅读文件", ", ".join(step["read"]))
    print("阅读顺序", step["focus"])
    print("快速练习", subprocess.list2cmdline(command(step, quick=True)))
    print("正式实验", subprocess.list2cmdline(command(step)))
    print("应当观察", step["expected"])
    print("动手修改", step["exercise"])
    print("验收问题", step["question"])
    for reference in step["references"]:
        print("源码对照", reference)


def check_steps(steps, timeout):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = ROOT / "runs/study_checks" / stamp
    output.mkdir(parents=True)
    results = []
    for step in steps:
        log_path = output / (step["id"] + ".log")
        started = time.monotonic()
        print("正在检查", step["id"], step["title"], flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(command(step, quick=True), cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=timeout)
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
                log.write("\n超过检查时间上限\n")
        results.append({"id": step["id"], "title": step["title"], "exit_code": exit_code,
                        "seconds": round(time.monotonic() - started, 2), "log": str(log_path.relative_to(ROOT))})
        # 每完成一步就保存，便于中断后查看已经完成的检查。
        partial = output / "summary.json.tmp"
        partial.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        partial.replace(output / "summary.json")
        print("通过" if exit_code == 0 else "失败", "日志", log_path, flush=True)
    print("结果目录", output)
    return int(any(result["exit_code"] != 0 for result in results))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    subcommands = cli.add_subparsers(dest="action")
    subcommands.add_parser("list", help="显示全部学习步骤")
    show = subcommands.add_parser("show", help="查看一步的阅读、实验与验收要求")
    show.add_argument("step")
    run = subcommands.add_parser("run", help="运行一步，默认使用正式实验预算")
    run.add_argument("step")
    run.add_argument("--quick", action="store_true")
    run.add_argument("--seed", type=int)
    run.add_argument("--args", nargs=argparse.REMAINDER, default=[], help="向章节传递额外参数")
    check = subcommands.add_parser("check", help="用快速预算检查运行入口")
    check.add_argument("step", nargs="?")
    check.add_argument("--all", action="store_true")
    check.add_argument("--timeout", type=int, default=180)
    args = cli.parse_args()
    steps = load_steps()
    if args.action in (None, "list"):
        for step in steps:
            print(step["id"], step["title"], "|", " ".join(step["entry"]))
        return 0
    if args.action == "check":
        if args.all and args.step or args.timeout < 1:
            cli.error("选择一个步骤或 --all，并提供正数 timeout")
        selected = steps if args.all else [get_step(steps, args.step or "00", cli)]
        return check_steps(selected, args.timeout)
    step = get_step(steps, args.step, cli)
    if args.action == "show":
        show_step(step)
        return 0
    if args.seed is not None and not step["seed"]:
        cli.error("该入口使用固定模型或自检数据；请在对应章节内查看种子设置")
    print("执行", subprocess.list2cmdline(command(step, args.quick, args.seed, args.args)), flush=True)
    return subprocess.call(command(step, args.quick, args.seed, args.args), cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
