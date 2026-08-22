# Day 04 — Confusion Matrix & Classification Metrics

**Topic:** Model Evaluation — Precision, Recall, F1, ROC-AUC
**Dataset:** Loan Default dataset (reused from my Loan Default Risk Analyzer project)

## Goal
Go beyond accuracy and understand what each classification metric actually tells you, using a trained model's predictions on loan default data.

## Code
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# sample data (swap with real loan dataset)
np.random.seed(42)
n = 500
X = pd.DataFrame({
    "annual_income": np.random.normal(45000, 15000, n),
    "credit_score": np.random.normal(680, 50, n)
})
y = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
```

## Output (observation)
- **Confusion matrix** breaks down TP/FP/TN/FN — for loan default, a False Negative (predicted "won't default" but actually did) is the costly error, not a False Positive.
- **Precision**: of all applicants flagged as "will default," how many actually did. High precision = fewer false alarms, but may miss real defaulters.
- **Recall**: of all actual defaulters, how many the model caught. This is the metric that matters most for lending risk — missing a defaulter costs more than an unnecessary manual review.
- **F1**: balances precision and recall — useful as a single number when both matter somewhat equally.
- **ROC-AUC**: measures ranking quality across all thresholds, not just one cutoff — good for comparing models overall, less useful for picking an operating threshold.

## Key takeaway
Accuracy alone is misleading on imbalanced data (a model that predicts "no default" every time can still look ~85% accurate). For loan risk, I should optimize threshold/model choice around **recall on the default class**, and use ROC-AUC to compare models before deciding on a decision threshold.

## Next
- Apply this evaluation on my actual Loan Default Risk Analyzer model output and pick a threshold (not just default 0.5) based on the precision-recall tradeoff.
