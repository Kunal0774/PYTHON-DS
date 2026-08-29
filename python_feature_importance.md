# Feature Importance — Random Forest vs XGBoost

**Topic:** Model Interpretability
**Dataset:** Loan Default dataset (reused from my Loan Default Risk Analyzer project)

## Goal
Compare how Random Forest and XGBoost rank feature importance on the same data, and understand why the two can disagree.

## Code
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

np.random.seed(42)
n = 500
X = pd.DataFrame({
    "annual_income": np.random.normal(45000, 15000, n),
    "credit_score": np.random.normal(680, 50, n),
    "loan_amount": np.random.normal(200000, 80000, n),
    "existing_loans": np.random.randint(0, 5, n)
})
y = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf.fit(X, y)

xgb = XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42)
xgb.fit(X, y)

rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)

print("Random Forest Importance:\n", rf_importance)
print("\nXGBoost Importance:\n", xgb_importance)
```

## Output (observation)
- **Random Forest importance** is based on average impurity (Gini) reduction across all trees — tends to favor high-cardinality numeric features (like `annual_income`, `loan_amount`) simply because they offer more possible split points.
- **XGBoost importance** (gain-based) reflects how much each feature actually improves the loss function when used in a split — can rank a lower-cardinality feature like `existing_loans` higher if it's genuinely more predictive, since it's not biased by number of possible splits.
- The two models can disagree on ranking even on identical data — importance scores are model-specific, not an absolute truth about the data.

## Key takeaway
Feature importance should be treated as "important according to this model's splitting logic," not as ground truth about real-world causality. For a business-facing report (e.g., explaining loan risk factors to stakeholders), I'd cross-check importance across both models and only report features that rank highly in both, rather than trusting a single model's ranking blindly.

## Next
- Add SHAP values on top of this for the actual Loan Default Risk Analyzer model — SHAP gives per-prediction explanations, not just global importance, which is more useful for explaining individual loan decisions.
