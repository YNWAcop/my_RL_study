# 作业车间排程

每个作业由一串按顺序执行的工序组成，每道工序指定机器和加工时长。每次选择一个未完成作业，把它的下一道工序追加到对应机器的队列。环境在前置工序与机器均可用时开始加工，目标是缩短全部作业完成时间（makespan）。

本项目从本仓库的六工序练习扩展到不同排程实例。派工基线包括最短加工时间优先（SPT）和最长加工时间优先（LPT）。MaskablePPO 用固定维度的 MLP 和动作掩码学习派工，CP-SAT 按工序优先约束和机器互斥求解同一实例。

在仓库根目录运行。Windows 上可将 `python` 换成 `.\.venv\Scripts\python.exe`。

```powershell
python -m pip install -r projects/job_shop/requirements.txt
python -m projects.job_shop.run evaluate --example --output runs/job_shop/example
python -m projects.job_shop.run train --jobs 3 --machines 2 --steps 50000 --seed 0 --output runs/job_shop/seed0
python -m projects.job_shop.run evaluate --model runs/job_shop/seed0/model.zip --jobs 3 --machines 2 --instances 30 --seed 10000 --output runs/job_shop/seed0_eval
```

需要加入求解器对照时，安装 OR-Tools，然后给评估命令加上 `--cp-sat --time-limit 2`。依赖版本对应本仓库的 Python 3.9 环境。

```powershell
python -m pip install -r projects/job_shop/requirements-cp-sat.txt
python -m projects.job_shop.run evaluate --example --cp-sat --output runs/job_shop/example_cp_sat
```

也可用 `--instance-file path/to/instance.json` 评估自己的实例。格式为 `[[[0,3],[1,3]],[[0,2],[1,5]],[[1,4],[0,2]]]`，每个作业内按执行顺序列出 `[机器编号, 加工时长]`。机器编号从 0 开始，每个作业访问每台机器一次。PPO 模型需与测试的作业数、机器数相同。

评估输出包括实例 JSON、逐实例的 `results.csv`、首个实例的排程 JSON 与甘特图。所有产出的排程都经过时长、工序顺序与机器冲突校验。CP-SAT 返回 `FEASIBLE` 时表示已找到可行解，`OPTIMAL` 表示已证明最优。每次使用新的输出目录。

参考 [L2D 论文实现](https://github.com/zcaicaros/L2D)、[MaskablePPO](https://sb3-contrib.readthedocs.io/en/v2.4.0/modules/ppo_mask.html) 和 [OR-Tools 排程示例](https://developers.google.com/optimization/scheduling/job_shop)。当前训练基线是 MLP-PPO，后续可以在相同实例上加入 GIN-PPO 做对照。
