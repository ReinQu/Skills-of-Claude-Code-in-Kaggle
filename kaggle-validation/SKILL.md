---
name: kaggle-validation
description: "Design leaderboard-aligned validation for Kaggle: KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, OOF predictions, leakage checks, and diagnosing CV-vs-LB gaps. Use before training any model."
---

# 验证策略：让本地 CV 可信

核心原则：**本地验证必须模拟 test 的划分方式，并用比赛官方指标计算。** 否则调参调的是幻觉。

## 先确定三件事

1. 官方指标（RMSE / RMSLE / LogLoss / F1 / MAP@k / QWK …），本地用同一个公式复现。
2. test 与 train 的关系：随机同分布？按组？按时间？
3. `sample_submission` 的结构：行是否必须与 test 一一对应，id 列叫什么。

## 划分方式选择

| 数据形态 | 用什么 | 说明 |
|---|---|---|
| 行之间独立同分布 | `StratifiedKFold`（分类）/ `KFold`（回归），5 折 | 分类按标签分层 |
| 同一实体多行（用户 / 商品 / 病人） | `GroupKFold` / `StratifiedGroupKFold` | 组不能跨折，否则 CV 虚高 |
| 时间序列 | `TimeSeriesSplit` 或按时间切分 | 只用过去预测未来 |
| 类别极不平衡 | `StratifiedKFold` + PR-AUC / F1 | 不要只看 accuracy |
| 数据量小 | 重复 K 折（多个 seed）取均值 ± 方差 | 单次 CV 波动大 |

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
for tr, va in skf.split(X, y):
    model.fit(X.iloc[tr], y.iloc[tr])
    oof[va] = model.predict_proba(X.iloc[va])[:, 1]

# 官方指标在完整的 oof 上算一次，而不是把各折分数简单平均
```

## 泄漏检查清单

- 目标编码 / 均值编码：只能在每折训练集上 `fit`，再 `transform` 验证集。
- 标准化、缺失填充、特征选择：同样只能 fit 训练折。
- 时间特征：不能出现未来信息（例如用全量统计量构造的特征）。
- 重复行：train/test 之间若有完全相同的行，CV 要按重复组划分。
- ID / 行号 / 文件名：确认它们不是标签的代理。
- 公开榜 probing：不要用 public LB 反复试错来选模型。

## CV 与 LB 差距怎么读

| 现象 | 可能原因 | 处理 |
|---|---|---|
| CV 高、LB 低很多 | CV 泄漏 / 分布漂移 / 过拟合 CV | 按清单排查，做 adversarial validation |
| CV 低、LB 高 | CV 过于悲观（如时间切分过严） | 检查划分是否与 test 一致 |
| CV 稳定但 LB 抖动 | public LB 样本小 | 看多次提交的方差，别追单次峰值 |
| 折间方差很大 | 数据少或划分不稳 | 增加折数、重复 CV、检查异常值 |

## 产物

每次验证保存：折划分索引、OOF 预测、每折指标、总体指标，写入 `reports/cv_<model>.json`。后续融合、stacking 和提交选择都以 OOF 为准。
