# Outlier Detection — IQR Method vs Z-Score

**Topic:** EDA / Data Cleaning
**Dataset:** Loan Default dataset (reused from my Loan Default Risk Analyzer project)

## Goal
Compare two standard outlier detection methods on `annual_income` and see which flags data points more reliably when the distribution is skewed.

## Code
```python
import pandas as pd
import numpy as np
from scipy import stats

df = pd.DataFrame({
    "annual_income": [25000, 32000, 41000, 39000, 45000, 38000, 30000, 1200000, 28000, 500000]
})

# IQR method
Q1 = df["annual_income"].quantile(0.25)
Q3 = df["annual_income"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
iqr_outliers = df[(df["annual_income"] < lower_bound) | (df["annual_income"] > upper_bound)]

# Z-score method
df["z_score"] = np.abs(stats.zscore(df["annual_income"]))
z_outliers = df[df["z_score"] > 3]

print("IQR outliers:\n", iqr_outliers)
print("\nZ-score outliers:\n", z_outliers)
```

## Output (observation)
- **IQR method**: flagged both 1,200,000 and 500,000 as outliers — based purely on quartile spread, unaffected by the extreme value itself.
- **Z-score method**: with such an extreme outlier (1,200,000) in a small dataset, the mean and standard deviation themselves get pulled/inflated by that outlier, which can make the z-score method under-flag other genuine outliers (like 500,000) — a known weakness of z-score on small or heavily skewed samples.

## Key takeaway
IQR is generally safer for skewed, real-world financial data (income, loan amounts) because it's based on percentiles, not mean/std, so it isn't distorted by the very outliers it's trying to detect. Z-score works better on roughly normal distributions with fewer extreme values.

## Next
- Apply the IQR method on the actual Loan Default Risk Analyzer dataset's `annual_income` and `loan_amount` columns, and decide whether to cap (winsorize) or remove flagged rows before retraining the model.
