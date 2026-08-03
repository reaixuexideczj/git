# Git 基础学习笔记

## 一、Git 是什么

Git 是一种分布式版本控制工具，主要用于记录文件的修改历史。

使用 Git 可以：

- 查看文件发生过哪些变化
- 保存不同阶段的版本
- 在出现错误时恢复旧版本
- 与其他开发者协作
- 将项目同步到 GitHub 或 Gitee

Git 不仅可以管理代码，也可以管理 Markdown 笔记和其他文本文件。

## 二、Git、GitHub 和 Gitee 的区别

- **Git**：安装在本地计算机上的版本控制工具。
- **GitHub**：用于在线托管 Git 仓库的平台。
- **Gitee**：与 GitHub 功能相似的国内代码托管平台。

Git 和 GitHub 不是同一个东西。没有网络时也可以使用 Git，但只有推送到 GitHub 后，远程仓库才会更新。

## 三、Git 的几个重要区域

1. **工作区**：实际编辑文件的文件夹。
2. **暂存区**：保存准备提交的修改。
3. **本地仓库**：保存已经 Commit 的版本。
4. **远程仓库**：GitHub 或 Gitee 上的仓库。

基本流程：

```text
修改文件 → Stage → Commit → Push
工作区      暂存区    本地仓库    远程仓库
```

## 四、初始化仓库

将当前文件夹初始化为 Git 仓库：

```bash
git init
```

初始化后，文件夹中会出现隐藏的 `.git` 目录。它用于保存仓库配置和版本历史，不应该随意删除。

查看仓库状态：

```bash
git status
```

## 五、连接 GitHub 仓库

添加远程仓库：

```bash
git remote add origin https://github.com/用户名/仓库名.git
```

查看远程仓库地址：

```bash
git remote -v
```

其中，`origin` 是远程仓库常用的默认名称。

## 六、Git 的日常操作

### 1. 暂存修改

暂存单个文件：

```bash
git add 文件名
```

暂存全部修改：

```bash
git add .
```

VS Code 中的 **Stage Changes** 相当于执行 `git add`。

### 2. 提交修改

```bash
git commit -m "提交说明"
```

例如：

```bash
git commit -m "添加 Git 基础学习笔记"
```

Commit 只会把版本保存到本地仓库，不会自动上传到 GitHub。

### 3. 推送到 GitHub

第一次推送：

```bash
git push -u origin main
```

以后可以直接执行：

```bash
git push
```

只有 Push 成功后，GitHub 网页才会显示最新提交。

### 4. 获取远程更新

```bash
git pull
```

该命令用于获取远程仓库中的最新内容，并合并到当前本地分支。

## 七、查看提交历史

查看详细记录：

```bash
git log
```

查看简洁记录：

```bash
git log --oneline
```

每次提交都有唯一的 Commit ID，例如：

```text
fd64470 添加 Git 学习笔记
```

## 八、分支基础

查看当前分支：

```bash
git branch
```

创建并切换到新分支：

```bash
git switch -c 分支名
```

切换分支：

```bash
git switch 分支名
```

`main` 通常是仓库的主分支。使用其他分支，可以在不影响主分支的情况下进行修改。

## 九、VS Code 中的 Git 状态

- **Changes**：文件已经修改，但尚未暂存。
- **Staged Changes**：文件已经暂存，可以进行 Commit。
- **Commit**：修改已经保存到本地 Git 历史。
- **Sync Changes / Push**：将本地提交上传到 GitHub。
- **Ahead 1**：本地比远程多一个提交，需要 Push。
- 文件显示为橙色：文件相对于上一次提交发生了修改。

在 Git Graph 中看到提交，不一定代表 GitHub 已经更新，因为 Graph 也会显示本地提交。

## 十、常见问题

### GitHub 没有显示最新内容

可能只完成了 Commit，还没有 Push：

```bash
git status
git push
```

### 无法连接 github.com:443

这通常是网络、代理或防火墙问题。本地提交不会丢失，网络恢复后重新执行：

```bash
git push
```

### 文件显示只读

从 Git Graph 或 Staged Changes 打开的可能是历史版本或差异视图。应该从 VS Code 左侧的资源管理器打开实际文件。

## 十一、常用命令总结

| 命令 | 作用 |
| --- | --- |
| `git init` | 初始化 Git 仓库 |
| `git status` | 查看仓库状态 |
| `git add .` | 暂存全部修改 |
| `git commit -m "说明"` | 提交到本地仓库 |
| `git push` | 推送到远程仓库 |
| `git pull` | 获取远程更新 |
| `git log --oneline` | 查看提交历史 |
| `git branch` | 查看分支 |
| `git remote -v` | 查看远程地址 |

## 十二、学习体会

通过实际操作，我理解了 Git 并不是简单的文件上传工具，而是用来记录文件版本变化的工具。

Stage 用于选择准备提交的修改，Commit 用于生成本地版本记录，Push 才会将提交同步到 GitHub。

虽然 VS Code 可以通过图形界面完成 Git 操作，但了解相应的 Git 命令，可以帮助我判断操作进行到了哪一步，也能在出现问题时更快地找到原因。
  