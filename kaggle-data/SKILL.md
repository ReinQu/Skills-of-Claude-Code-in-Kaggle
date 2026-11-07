---
name: kaggle-data
description: "Download, verify and lay out Kaggle competition data with the Kaggle CLI. Use when starting a competition, fetching a dataset, checking available files, or inspecting submission limits."
---

# Kaggle 数据获取与目录规范

用官方 `kaggle` CLI 拉数据。不需要 Rube / Composio，也不需要额外的 MCP。

## 环境

- 解释器：`D:/Users/24659/envs/ai/python.exe`（conda 环境 `ai`，已装 `kaggle 2.2.5`）
- CLI：`D:/Users/24659/envs/ai/Scripts/kaggle.exe`
- 凭据：`C:/Users/24659/.kaggle/access_token`（已存在，不要读取、打印或提交到仓库）

Git Bash 里 conda 没有加入 PATH，要么直接调用绝对路径，要么先激活：

```bash
source /d/Users/24659/anaconda3/etc/profile.d/conda.sh && conda activate ai
```

## 目录约定

```
D:/Kaggle/Project/
├── data/<slug>/          # 原始数据，只读
├── notebooks/            # 探索与实验
├── src/                  # 可复用脚本
├── submissions/          # 每次提交新文件，不覆盖
└── reports/              # EDA、CV、实验记录
```

## 命令

```bash
K="D:/Users/24659/envs/ai/Scripts/kaggle.exe"

# 1. 认证冒烟测试（失败先修 token）
"$K" competitions list --search "<关键词>"

# 2. 看比赛提供了哪些文件、多大
"$K" competitions files -c <slug>

# 3. 下载并解压到约定目录
"$K" competitions download -c <slug> -p "D:/Kaggle/Project/data/<slug>" --unzip

# 4. 剩余提交次数 + 历史成绩
"$K" competitions submissions -c <slug>

# 5. 榜单规则（public/private 比例）
"$K" competitions leaderboard -c <slug> --show
```

## 下载后必须核对

1. 本地文件清单与 `competitions files` 输出一致（文件名、数量、大小）。
2. `train.csv` / `test.csv` / `sample_submission.csv` 齐全；没有 `sample_submission` 时以官方 Data 页说明为准。
3. 把行数、列数、文件大小写进 `reports/data_manifest.md`，后续对不上时以它为准。
4. 原始文件不改；清洗结果写到 `data/<slug>/processed/`。

## 常见失败对照

| 现象 | 原因 |
|---|---|
| 401 / 403 | token 无效或过期 |
| 403 且 token 正常 | 比赛 Rules 还没在网页上点 Accept |
| 下载中断 / 文件偏小 | 网络问题，删掉重下，不要接着用残缺文件 |

下载前先 `files` 看一眼总大小，避免把几十 GB 拉到 C 盘或临时目录。
