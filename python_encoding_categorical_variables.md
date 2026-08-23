# Encoding Categorical Variables

**Topic:** Preprocessing — One-Hot vs Label vs Target Encoding
**Dataset:** Loan Default dataset (reused from my Loan Default Risk Analyzer project)

## Goal
Compare three ways of converting a categorical column (`employment_type`) into numeric form for a model, and when each is appropriate.

## Code
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame({
    "employment_type": ["Salaried", "Self-Employed", "Salaried", "Unemployed", "Self-Employed"],
    "defaulted": [0, 1, 0, 1, 0]
})

# Label Encoding
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["employment_type"])

# One-Hot Encoding
one_hot = pd.get_dummies(df["employment_type"], prefix="emp")

# Target Encoding (mean of target per category)
target_means = df.groupby("employment_type")["defaulted"].mean()
df["target_encoded"] = df["employment_type"].map(target_means)

print(df)
print(one_hot)
```

## Output (observation)
- **Label Encoding**: assigns arbitrary integers (0, 1, 2...) — risky for tree-unaware models since it implies a false order (e.g., "Unemployed" = 2 isn't "greater than" "Salaried" = 0). Fine for tree-based models like Random Forest/XGBoost which don't assume ordinality.
- **One-Hot Encoding**: creates a separate binary column per category — safe for any model, but increases dimensionality if the column has many unique categories.
- **Target Encoding**: replaces each category with the mean target value for that category — compact and often more predictive, but risks data leakage/overfitting if not done with proper cross-validation folds (must fit encoding on train only, never on full data).

## Key takeaway
For `employment_type` with only 3 categories, One-Hot is the safest default for linear models. For tree-based models (which I use in Loan Default Risk Analyzer — Random Forest, XGBoost), Label Encoding is usually fine and simpler. Target Encoding is worth trying only with careful cross-validation to avoid leakage.

## Next
- Apply target encoding with proper K-Fold cross-validation on the real loan dataset and compare ROC-AUC against one-hot encoding.
