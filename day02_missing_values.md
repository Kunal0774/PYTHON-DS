# Day 02 — Handling Missing Values

**Topic:** Preprocessing — Mean/Median/Mode Imputation vs KNN Imputer
**Dataset:** Loan Default dataset (reused from my Loan Default Risk Analyzer project)

## Goal
Compare simple imputation methods against KNN Imputer on a numeric column with missing values.

## Code
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# sample data (swap with your real loan dataset)
df = pd.DataFrame({
    "annual_income": [25000, np.nan, 41000, 39000, np.nan, 45000, 38000, 30000],
    "credit_score":  [700, 650, np.nan, 680, 710, np.nan, 690, 660]
})

# Mean imputation
mean_imputer = SimpleImputer(strategy="mean")
mean_result = pd.DataFrame(mean_imputer.fit_transform(df), columns=df.columns)

# Median imputation
median_imputer = SimpleImputer(strategy="median")
median_result = pd.DataFrame(median_imputer.fit_transform(df), columns=df.columns)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=3)
knn_result = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

print("Mean:\n", mean_result)
print("Median:\n", median_result)
print("KNN:\n", knn_result)
```

## Output (observation)
- **Mean imputation**: fast, but distorts distribution if data is skewed or has outliers.
- **Median imputation**: more robust to outliers/skew than mean — safer default for income-type columns.
- **KNN Imputer**: fills missing values based on similar rows (nearest neighbors across other columns), so it captures relationships between features instead of using one flat number — more accurate but slower on large datasets.

## Key takeaway
For small/simple datasets, median imputation is a fast and safe default. For datasets where features are correlated (like credit_score and annual_income likely are), KNN Imputer gives more realistic fill values because it uses that relationship instead of ignoring it.

## Next
- Apply KNN Imputer on the actual loan dataset and compare model ROC-AUC vs median imputation to see if it matters in practice.
