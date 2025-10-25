"""
analyze_guards.py

Evaluates multiple guard functions against human and LLM safety labels.
Run this after activating your virtual environment:
    source .venv/bin/activate
    python analyze_guards.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from scipy.stats import fisher_exact

# ----------------------------
# 1. Load your dataset
# ----------------------------
# Expected CSV columns:
#   guard_1, guard_2, ..., guard_m, llm_flag, human_label
df = pd.read_csv("guards_data.csv")

# Identify which columns are guards automatically
guard_cols = [c for c in df.columns if c.startswith("guard_")]
y_human = df["human_label"]
y_llm = df["llm_flag"]

print(f"\nLoaded {len(df)} samples with {len(guard_cols)} guards.\n")

# ----------------------------
# 2. Evaluate each guard individually
# ----------------------------
results = []

for g in guard_cols:
    y_pred = df[g]
    tn, fp, fn, tp = confusion_matrix(y_human, y_pred).ravel()
    phi = matthews_corrcoef(y_human, y_pred)
    odds, pval = fisher_exact([[tp, fp], [fn, tn]])
    acc = accuracy_score(y_human, y_pred)
    prec = precision_score(y_human, y_pred, zero_division=0)
    rec = recall_score(y_human, y_pred, zero_division=0)
    f1 = f1_score(y_human, y_pred, zero_division=0)

    results.append({
        "guard": g,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "phi_corr": phi,
        "fisher_p": pval
    })

guard_df = pd.DataFrame(results).sort_values("f1", ascending=False)
print("=== Per-Guard Performance vs Human Labels ===")
print(guard_df.to_string(index=False, float_format="%.3f"))
print()

# ----------------------------
# 3. Compare each guard to the LLM’s behavior
# ----------------------------
phi_llm = {
    g: matthews_corrcoef(df[g], y_llm)
    for g in guard_cols
}
print("=== Correlation (phi) of Guards vs LLM Flag ===")
for g, v in sorted(phi_llm.items(), key=lambda x: -abs(x[1])):
    print(f"{g:15s} phi={v: .3f}")
print()

# ----------------------------
# 4. Agreement between LLM and Humans
# ----------------------------
kappa = cohen_kappa_score(y_human, y_llm)
phi_hl = matthews_corrcoef(y_human, y_llm)
print(f"=== LLM ↔ Human Agreement ===")
print(f"Cohen’s kappa: {kappa:.3f}")
print(f"Phi correlation: {phi_hl:.3f}\n")

# ----------------------------
# 5. Logistic regression ensemble of guards
# ----------------------------
X = df[guard_cols].values
model = LogisticRegression(penalty="l2", solver="liblinear")
model.fit(X, y_human)
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

print("=== Combined Guard Model (Logistic Regression) ===")
print(f"Accuracy: {accuracy_score(y_human, pred):.3f}")
print(f"F1 Score: {f1_score(y_human, pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_human, prob):.3f}\n")

# Show top contributing guards
coefs = pd.Series(model.coef_[0], index=guard_cols).sort_values(ascending=False)
print("Top Guard Coefficients (importance):")
print(coefs.head(10).to_string(float_format="%.3f"))
print()

# ----------------------------
# 6. Optional: Human-readable rules from a decision tree
# ----------------------------
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y_human)
print("=== Simple Rule-Based Tree (depth 3) ===")
print(export_text(tree, feature_names=guard_cols))
print()

print("Analysis complete ✅")
