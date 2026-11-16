# GitHub 同步

本仓库同步到 [YNWAcop/my_RL_study](https://github.com/YNWAcop/my_RL_study) 的默认分支 `codex/rl-learning-path`。仓库启用提交后钩子时，在此分支每成功提交一次，就自动向 GitHub 推送一次。

在 VS Code 的「源代码管理」中，暂存本轮改动、输入提交说明并点击「提交」即可。也可以在仓库根目录运行以下命令。

```powershell
git status --short
git add -- projects/Code_2/lesson3/dqn/agent.py
git diff --cached
git commit -m "Study DQN action selection"
```

把示例文件路径换成本轮要同步的文件。保存文件后，改动会一直留在工作区，直到你提交。模型权重、运行产物、虚拟环境和原始 ZIP 已通过 `.gitignore` 排除。

如果 GitHub 连接或验证登录失败，本地提交仍然有效。恢复网络或登录后重试。

```powershell
git push origin codex/rl-learning-path
```

在新克隆的仓库中，可以在仓库根目录启用相同的提交后同步。

```powershell
git config --local core.hooksPath scripts/git-hooks
```

如果本机已运行 GitHub 代理，Git 可以为此仓库的 `origin` 单独配置代理。将地址替换为本机实际端口。

```powershell
git config --local remote.origin.proxy http://127.0.0.1:7897
```

直连可用时，可以用 `git config --local --unset remote.origin.proxy` 清除该设置。
