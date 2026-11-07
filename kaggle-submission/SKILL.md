---
name: kaggle-submission
description: "Create and validate a Kaggle submission.csv: align sample_submission columns and row order, map test ids, assert shape/NaN/dtype, and manage submission limits. Use whenever generating or submitting a competition result."
---

# 提交文件生成与校验

Kaggle 新手翻车最多的地方不是模型，而是提交文件格式。**生成后必须跑完下面的断言，再提交。**

## 1. 先读 sample_submission

```python
import pandas as pd

sample = pd.read_csv("data/<slug>/sample_submission.csv")
test = pd.read_csv("data/<slug>/test.csv")
print(sample.columns.tolist(), sample.shape)
print(sample.head())
```

规则：

- 列名和列顺序以 `sample_submission` 为准，不是以你的预测 DataFrame 为准。
- 行数必须等于 `sample_submission` 的行数。
- id 集合必须与 `sample_submission` 完全一致，但**行顺序可以不同**；重排到 sample 的顺序最稳。
- 目标列名通常就是 sample 里除 id 之外的列。

## 2. 构建（按 sample 顺序对齐）

```python
import numpy as np
import pandas as pd

def build_submission(sample_path, id_col, test_ids, preds, target_col=None):
    sample = pd.read_csv(sample_path)
    target_col = target_col or [c for c in sample.columns if c != id_col][0]

    pred = pd.Series(np.asarray(preds), index=pd.Index(test_ids, name=id_col))
    assert pred.index.is_unique, "test id 有重复"

    sub = sample.copy()
    sub[target_col] = pred.reindex(sample[id_col]).to_numpy()
    return sub, target_col
```

## 3. 提交前断言

```python
def validate_submission(sub, sample, id_col, target_col):
    assert list(sub.columns) == list(sample.columns), "列名或列顺序不一致"
    assert len(sub) == len(sample), f"行数不一致: {len(sub)} vs {len(sample)}"
    assert sub[id_col].is_unique, "id 有重复"
    assert set(sub[id_col]) == set(sample[id_col]), "id 集合与 sample 不一致"
    assert sub[target_col].notna().all(), "预测值含 NaN"
    assert pd.api.types.is_numeric_dtype(sub[target_col]), "预测列不是数值"
    print(sub.head())
    print("rows:", len(sub), "range:", sub[target_col].min(), sub[target_col].max())
    return True
```

按任务补充：

| 任务 | 额外检查 |
|---|---|
| 概率输出 | 值域在 `[0, 1]`，必要时 `clip` |
| 多分类 | 标签属于训练集出现的类别集合 |
| 回归 | 是否需要对 `log1p` 目标做 `expm1` 还原 |
| 整数目标 | dtype 是否为 int，不要输出 `1.0` |
| 多列输出 | 每一列都做同样的检查 |

## 4. 保存与提交

文件命名带日期、模型和 CV 分数，**不覆盖历史文件**：

```bash
submissions/2026-11-06_lgbm_cv0.9123.csv
```

```bash
K="D:/Users/24659/envs/ai/Scripts/kaggle.exe"

# 提交前先看还剩几次
"$K" competitions submissions -c <slug>

"$K" competitions submit -c <slug> -f "submissions/<file>.csv" -m "lgbm cv0.9123"
```

## 5. 选择最终提交

- 以 **CV 与 LB 一致** 的模型为主，不追单次 public LB 峰值。
- 保留 2 个相关性低的方案（不同模型 / 不同特征），防止 private LB 翻转。
- 在 `reports/submissions.md` 记录每次提交的时间、文件、CV、LB、结论。
- 比赛结束前确认选择了正确的 submission（部分比赛需要手动 Select）。

## 常见错误

- 多了一列 `index`（`to_csv(index=False)`）。
- 列名大小写或空格与 sample 不一致。
- 把概率当类别提交，或反之。
- 用 OOF 顺序而不是 sample 的 id 顺序直接赋值。
- 忘记还原 log 变换。
