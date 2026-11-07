# VS Code 点击运行

## 打开项目

在 VS Code 中选择“文件 → 打开文件夹”，打开整个 `RL` 文件夹。当前机器路径为 `E:\UserData\Documents\ChatGPT\RL`。安装 Microsoft 的 Python 扩展及 Python Debugger 扩展，工作区会推荐这两个扩展。

选择右下角的 Python 解释器，指定项目 `.venv` 中的解释器。当前机器为 `.venv\Scripts\python.exe`。公共模块已安装在此环境，源文件修改后直接生效。换用其他环境时需要完成项目环境安装。

## 点击右上角运行

打开 `lessons/01_bandit.py`，点击编辑器右上角三角形旁的下拉菜单，选择“在终端中运行 Python 文件”。程序使用该文件中的默认参数。下方终端显示结果，用户无需输入命令。

需要修改默认步数时，在源代码中调整 `default=2000`，保存后重新点击运行。`03_td_control.py` 等多算法章节的默认算法由 `--algo` 对应的 `default` 决定，也可以通过下面的算法菜单选择。

## 从算法菜单运行

点击左侧“运行和调试”图标，或按 Ctrl+Shift+D。顶部下拉菜单包含老虎机、价值迭代、Q-learning、SARSA、DQN、Double DQN、REINFORCE、Actor-Critic、PPO、DDPG、TD3 和 SAC。选择一个配置，再点击绿色三角形。

F5 会启动调试，Ctrl+F5 运行时跳过断点暂停。点击代码行号左侧添加断点后，可以查看变量、单步执行或继续运行。配置来自 [launch.json](../.vscode/launch.json)，`module` 指定章节，`args` 保存该配置使用的训练参数，`cwd` 固定为项目根目录。参见 [VS Code Python 调试说明](https://code.visualstudio.com/docs/python/debugging)。

调整菜单中的训练步数或算法时，修改相应配置的 `args` 并保存。例如把 `"--steps", "60000"` 改为 `"--steps", "5000"`。这些参数会覆盖源代码中的默认值。深度算法的完整预算需要等待训练完成，红色停止按钮可以中断运行。

## 查看结果

运行日志会打印本次结果目录。在左侧展开 `runs`，查看该目录的 `config.json` 和 `train.csv`。模型训练结束后还会生成权重与评估文件。不同运行使用独立目录。

如果仍提示找不到 `rlstudy`，先确认当前解释器确实是已安装本项目的 `.venv`。editable 安装将模块指向当前工作区中的源文件，移动仓库或重新创建虚拟环境后需重新安装项目。安装机制见 [pip editable 安装说明](https://pip.pypa.io/en/stable/topics/local-project-installs/)。
