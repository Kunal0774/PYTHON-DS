# Day 01 — Feature Scaling Methods

**Topic:** Preprocessing — StandardScaler vs MinMaxScaler vs RobustScaler
**Dataset:** Loan Default dataset (reused from my Loan Default Risk Analyzer project)

## Goal
Compare how three scaling methods affect a numeric feature (`annual_income`) before feeding it into a model, and note when to use which.

## Code
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# sample data (swap with your real loan dataset)
df = pd.DataFrame({
    "annual_income": [25000, 32000, 41000, 39000, 1200000, 45000, 38000, 30000]
})

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

results = df.copy()
for name, scaler in scalers.items():
    results[name] = scaler.fit_transform(df[["annual_income"]])

print(results)
```

## Output (observation)
- `annual_income` has an outlier (1,200,000) compared to the rest (~25k–45k).
- **StandardScaler**: heavily skewed by the outlier — most values get compressed near a similar negative range.
- **MinMaxScaler**: even worse with this outlier — all normal values get squeezed near 0, outlier sits at 1.
- **RobustScaler**: uses median/IQR instead of mean/std, so it's far less affected by the outlier — normal values stay spread out and usable.

## Key takeaway
When a feature has outliers (common in income/loan data), **RobustScaler** is usually the safer default over StandardScaler/MinMaxScaler. I'll apply this to `annual_income` and similar skewed columns in my Loan Default Risk Analyzer next.

## Next
- Try this on the actual loan dataset column, not dummy data.
- Compare model performance (ROC-AUC) with each scaler to confirm it matters in practice, not just theoretically.
