# GitHub 同步

用户要求每轮代码修改完成后同步到 GitHub。完成本轮验证后，检查改动范围，只暂存本轮相关文件，在 `codex/rl-learning-path` 提交。仓库启用的提交后钩子会推送到 `origin`。若钩子未启用或自动推送失败，使用 `git push origin codex/rl-learning-path` 重试，并核对远端分支提交。

项目的远端为 `https://github.com/YNWAcop/my_RL_study.git`。保留用户的无关改动，不暂存训练权重、运行产物、本地环境、凭据或原始压缩包。
