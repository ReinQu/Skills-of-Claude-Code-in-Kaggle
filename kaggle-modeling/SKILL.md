---
name: kaggle-modeling
description: "Build and improve Kaggle models: baselines, feature engineering, HistGradientBoosting/LightGBM/XGBoost/CatBoost, hyperparameter tuning, blending and stacking. Use after EDA and after a validation scheme is defined."
---

# Kaggle 建模

顺序固定：**baseline → 单模型 → 特征工程 → 调参 → 集成**。每一步都用 `kaggle-validation` 定义的 CV 比较，不看 public LB 调参。

## 0. 依赖检查

```bash
D:/Users/24659/envs/ai/python.exe -c "import pandas, numpy, sklearn; print('ok')"
```

`ai` 环境已装 `scikit-learn` / `seaborn` / `lightgbm` / `xgboost` / `imbalanced-learn` / `optuna`。需要新包时先告诉用户、由用户决定是否安装，不要擅自 `pip install`。

## 1. Baseline（必须先有）

```python
from sklearn.dummy import DummyClassifier, DummyRegressor

# 分类：预测多数类；回归：预测均值
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
```

先记录 baseline 的 CV 分数。任何模型如果打不过 baseline，说明特征或数据有问题，先回头查，不要继续调参。

## 2. 模型阶梯

| 场景 | 先用 | 升级到 |
|---|---|---|
| 需要可解释 | Logistic / Linear Regression | 树模型 |
| 中小规模表格 | `HistGradientBoosting`（sklearn 内置） | LightGBM / XGBoost / CatBoost |
| 大规模表格 | LightGBM / CatBoost | 神经网络 |
| 图像 / 文本 | 预训练骨干（timm / transformers）微调 | 集成 |
| 无监督分组 | K-Means / DBSCAN | 用 silhouette 选 k |

树模型是表格比赛的主力。`HistGradientBoosting` 不依赖第三方库，适合作为第一个非 baseline 模型。

## 3. 特征工程

每次只改一类特征，用同一套 CV 对比，避免"改了一堆不知道哪个有用"。

- **数值**：`log1p`、平方根、分箱、缺失指示列、比值 / 差值。
- **类别**：低基数 one-hot；高基数用 frequency encoding，或**折内 fit** 的 target encoding。
- **时间**：hour / dayofweek / month / is_weekend，周期特征用 sin/cos。
- **聚合**：按实体分组的 count / mean / std / max-min（注意时间泄漏）。
- **文本**：TF-IDF + 线性模型或 SVD 降维后喂树模型。
- **图像**：预训练模型提 embedding，再训练轻量分类头。

## 4. 调参

- 用 CV 目标函数，不用 test。
- 先粗后细：learning rate、树数量、深度 / 叶子数、正则化。
- 小数据优先 `GridSearchCV` / `RandomizedSearchCV`；搜索空间大时用 Optuna（需另装）。
- 早停只能用验证折，不能用 public LB。

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

search = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [3, 5, 7], "max_iter": [200, 500]},
    n_iter=20, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1,
)
search.fit(X_train, y_train)
```

## 5. 集成

优先级从低到高：

1. **seed averaging**：同模型多个随机种子取平均。
2. **fold averaging**：每折模型对 test 预测后取平均（几乎零成本，先做这个）。
3. **weighted blend**：在 OOF 上搜索权重，目标是对齐官方指标。

```python
import numpy as np
from scipy.optimize import minimize

def blend_weights(oofs, y, metric):
    n = len(oofs)
    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return metric(y, sum(wi * o for wi, o in zip(w, oofs)))
    return minimize(loss, np.ones(n) / n, method="Nelder-Mead").x
```

4. **stacking**：用 OOF 预测作为元特征，再训练二层模型；必须用嵌套 CV，否则会泄漏。

## 6. 实验记录

每次实验往 `reports/experiments.jsonl` 追加一行：模型名、关键参数、CV 分数、OOF 文件、备注。这是后续选模型和复现的依据，不要只记在脑子里。

## 禁止

- 用 `test.csv` 调参或早停。
- 在切分前对全量数据 fit 编码器 / 标准化器。
- 只报一个总分，不报折间方差。
- 不做 baseline 直接上复杂模型。
