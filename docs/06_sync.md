# GitHub 同步

目标仓库为 [YNWAcop/my_RL_study](https://github.com/YNWAcop/my_RL_study)。工作区最初为空 Git 仓库，没有提交和 origin。2026-11-07 初次连接失败，后定位到 GitHub DNS 指向回环地址，显式使用系统现有的 `http://127.0.0.1:7897` 代理后，`ls-remote` 与 `fetch` 成功，确认当时远端没有分支或提交，并配置了 origin。最终推送状态见 [验证记录](05_validation.md)。

以后提交自己的学习笔记与代码修改，可从课程根目录执行下列命令。

```powershell
git status
git add notes lessons docs
git commit -m "Record reinforcement learning study progress"
git push
```

提交前检查暂存区，确保只包含本次学习内容。`runs/`、`.venv/` 和临时 wheel 下载目录已被忽略。个人实验结论写入 `notes/`，必要时挑选小型结果表提交，模型权重与大体积训练输出保留在本地。

如果当前网络需要已配置的系统代理，可以只对本次 Git 命令传入代理选项，例如 `git -c http.proxy=http://127.0.0.1:7897 push`。端口来自本次机器的现有配置，在其他电脑上需使用其实际设置。课程不修改系统代理配置。
