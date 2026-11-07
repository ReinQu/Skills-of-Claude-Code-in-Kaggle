---
name: kaggle-eda
description: "Perform exploratory data analysis for a Kaggle competition: profile train/test, find missing values, target imbalance, distribution shift, leakage and duplicate rows. Use when starting a competition or profiling a new dataset."
---

# Kaggle EDA

目标不是"画很多图"，而是回答三个问题：**数据长什么样、哪里有坑、下一步该做什么特征**。

## 步骤

1. **读数据清单**：`data/<slug>/` 下有哪些文件，各自行数、列数、大小；先读 `reports/data_manifest.md`（若存在）。
2. **看表结构**：`dtypes`、唯一值数、常量列、ID 列候选。
3. **缺失值**：每列缺失率，train/test 分开看。缺失率 > 50% 的列单独标记。
4. **目标分布**：分类看类别比例（是否不平衡），回归看偏度、长尾、是否需要 log 变换。注意比赛指标是否需要变换目标。
5. **数值特征**：describe、偏度、异常值（用 IQR 或分位数，不要直接删）。
6. **类别特征**：基数（高基数类别考虑 target/frequency encoding）、稀有类别、拼写不一致。
7. **train/test 对比**：逐特征比较分布；差异大的特征可能是分布漂移，也可能是泄漏信号。
8. **重复与泄漏**：完全重复行、train/test 重叠行、ID 是否与目标相关、时间戳是否含未来信息。
9. **关系**：与目标的相关性 / 分组统计；数值特征之间的共线性。
10. **结论**：写 3–5 条可执行的下一步（要造什么特征、用什么验证、排除哪些列）。

## 产物规范

- 代码：`src/eda.py` 或 `notebooks/01_eda.ipynb`
- 图：`reports/figures/eda/*.png`
- 结论：`reports/eda_report.md`，包含数据规模、关键发现、特征工程建议、风险
- 画图前读 `dataviz` skill，统一配色和版式；不要每张图换一种风格

```python
import pandas as pd

train = pd.read_csv("D:/Kaggle/Project/data/<slug>/train.csv")
test = pd.read_csv("D:/Kaggle/Project/data/<slug>/test.csv")

cols = [c for c in train.columns if c in test.columns]
profile = pd.DataFrame({
    "dtype": train.dtypes.astype(str),
    "nunique": train.nunique(),
    "null_rate": train[cols].isna().mean(),
    "test_null_rate": test[cols].isna().mean(),
})
```

## 不要做

- 不修改 `data/` 下的原始文件。
- 不在 EDA 阶段就对全量数据做 target encoding / 标准化（泄漏，见 `kaggle-validation`）。
- 不根据 public LB 反推 EDA 结论。
