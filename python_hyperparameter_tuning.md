# Hyperparameter Tuning — GridSearchCV vs RandomizedSearchCV

**Topic:** Model Optimization
**Dataset:** Loan Default dataset (reused from my Loan Default Risk Analyzer project)

## Goal
Compare exhaustive vs randomized hyperparameter search for tuning a Random Forest classifier, and note the tradeoff between search quality and compute time.

## Code
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n = 500
X = pd.DataFrame({
    "annual_income": np.random.normal(45000, 15000, n),
    "credit_score": np.random.normal(680, 50, n),
    "loan_amount": np.random.normal(200000, 80000, n)
})
y = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10]
}

# GridSearchCV — tries every combination
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid, cv=3, scoring="roc_auc", n_jobs=-1
)
grid_search.fit(X_train, y_train)

# RandomizedSearchCV — samples a fixed number of combinations
random_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid, n_iter=10, cv=3, scoring="roc_auc", n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)

print("Grid Best Params:", grid_search.best_params_, "Score:", grid_search.best_score_)
print("Random Best Params:", random_search.best_params_, "Score:", random_search.best_score_)
```

## Output (observation)
- **GridSearchCV**: tests all 3×4×3 = 36 combinations exhaustively — guaranteed to find the best combination within the grid, but compute cost grows fast as parameters/values increase.
- **RandomizedSearchCV**: samples only `n_iter` random combinations (10 here) — much faster, and in practice often finds a near-optimal result without testing every combination, especially when only a few parameters actually matter.

## Key takeaway
For a small grid like this, GridSearchCV is fine. For larger search spaces (more parameters or wider ranges), RandomizedSearchCV is the practical choice — it trades a small amount of optimality for a large reduction in compute time, which matters when iterating quickly on a laptop.

## Next
- Run this on the actual Loan Default Risk Analyzer's XGBoost model instead of Random Forest, since that's the model I reported 0.85 ROC-AUC with.
