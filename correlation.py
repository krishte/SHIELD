"""
analyze_guards.py

Evaluates continuous guard scores in [-1, 1] against human and LLM safety labels.

Expected CSV columns:
    prompt_id, prompt_text (optional),
    guard_1, guard_2, ..., guard_m  (floats in [-1, 1]),
    llm_flag (TRUE/FALSE),
    human_label (0/1)

Output:
    - Per-guard performance vs human labels
    - Per-guard alignment with LLM
    - Human ↔ LLM agreement (kappa)
    - Ensemble logistic regression using all guards
    - Interpretable rules from a shallow decision tree
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from scipy.stats import pearsonr


# ----------------------------
# 1. Load and preprocess data
# ----------------------------
df = pd.read_csv("guards_data.csv")

# strip weird whitespace in column headers just in case
df.columns = df.columns.str.strip()

# guard columns are every column that starts with "guard_"
guard_cols = [c for c in df.columns if c.startswith("guard_")]

# human_label should be 0/1 already, but we coerce just in case
y_human = (
    df["human_label"]
    .astype(str)
    .str.strip()
    .astype(int)
)

# llm_flag may be TRUE/FALSE or 1/0 as strings; normalize to int 0/1
y_llm = (
    df["llm_flag"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
    .astype(int)
)

print("Sanity check:")
print("  y_human value counts:\n", pd.Series(y_human).value_counts(dropna=False))
print("  y_llm   value counts:\n", pd.Series(y_llm).value_counts(dropna=False))
print(f"\nLoaded {len(df)} samples with {len(guard_cols)} guards.\n")


# ----------------------------
# Helper: binarize a guard by threshold for intuition
# score > 0 => unsafe (1), else safe (0)
# ----------------------------
def binarize_guard(series, thresh=0.0):
    return (series > thresh).astype(int)


# ----------------------------
# 2. Per-guard analysis vs HUMAN LABEL
# ----------------------------
per_guard_rows = []

for g in guard_cols:
    scores = df[g].astype(float)

    # do we have both classes 0 and 1 in human_label?
    unique_classes = np.unique(y_human)
    has_both_classes = (len(unique_classes) > 1)

    # 2.1 AUROC vs human_label
    if has_both_classes and scores.nunique() > 1:
        try:
            auroc_h = roc_auc_score(y_human, scores)
        except ValueError:
            auroc_h = np.nan
    else:
        auroc_h = np.nan  # not defined if only one class

    # 2.2 Pearson correlation vs human_label (point-biserial)
    # Needs variation in BOTH arrays
    if has_both_classes and scores.nunique() > 1:
        try:
            r_h, p_h = pearsonr(scores, y_human)
        except Exception:
            r_h, p_h = (np.nan, np.nan)
    else:
        r_h, p_h = (np.nan, np.nan)

    # 2.3 Snapshot classification using a fixed threshold at 0
    y_pred_bin = binarize_guard(scores, thresh=0.0)

    # Force confusion_matrix to be 2x2 so .ravel() always works
    cm = confusion_matrix(y_human, y_pred_bin, labels=[0, 1])
    # cm =
    # [[tn, fp],
    #  [fn, tp]]
    tn, fp, fn, tp = cm.ravel()

    acc  = accuracy_score(y_human, y_pred_bin)
    prec = precision_score(y_human, y_pred_bin, zero_division=0)
    rec  = recall_score(y_human, y_pred_bin,   zero_division=0)
    f1   = f1_score(y_human, y_pred_bin,       zero_division=0)

    per_guard_rows.append({
        "guard": g,
        "auroc_vs_human": auroc_h,
        "pearson_r_vs_human": r_h,
        "p_value_vs_human": p_h,
        "acc@thr0": acc,
        "prec@thr0": prec,
        "recall@thr0": rec,
        "f1@thr0": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    })

per_guard_df = pd.DataFrame(per_guard_rows)
# sort guards by F1 so we can say "these are our best guards"
per_guard_df = per_guard_df.sort_values("f1@thr0", ascending=False)

print("=== Per-Guard Performance vs HUMAN_LABEL ===")
print(per_guard_df.to_string(index=False, float_format="%.3f"))
print()


# ----------------------------
# 3. Per-guard analysis vs LLM_FLAG (agreement with model policy)
# ----------------------------
per_guard_rows_llm = []

for g in guard_cols:
    scores = df[g].astype(float)

    # check LLM labels diversity
    unique_llm = np.unique(y_llm)
    llm_has_both = (len(unique_llm) > 1)

    # AUROC vs llm_flag
    if llm_has_both and scores.nunique() > 1:
        try:
            auroc_llm = roc_auc_score(y_llm, scores)
        except ValueError:
            auroc_llm = np.nan
    else:
        auroc_llm = np.nan

    # Pearson r vs llm_flag
    if llm_has_both and scores.nunique() > 1:
        try:
            r_llm, p_llm = pearsonr(scores, y_llm)
        except Exception:
            r_llm, p_llm = (np.nan, np.nan)
    else:
        r_llm, p_llm = (np.nan, np.nan)

    per_guard_rows_llm.append({
        "guard": g,
        "auroc_vs_llm": auroc_llm,
        "pearson_r_vs_llm": r_llm,
        "p_value_vs_llm": p_llm,
    })

per_guard_llm_df = pd.DataFrame(per_guard_rows_llm)
per_guard_llm_df = per_guard_llm_df.sort_values("pearson_r_vs_llm", ascending=False)

print("=== Per-Guard Alignment with LLM_FLAG ===")
print(per_guard_llm_df.to_string(index=False, float_format="%.3f"))
print()


# ----------------------------
# 4. Human ↔ LLM agreement overall
# ----------------------------
kappa = cohen_kappa_score(y_human, y_llm)
print("=== LLM ↔ Human Agreement ===")
print(f"Cohen's kappa: {kappa:.3f}")
print()


# ----------------------------
# 5. Logistic regression using ALL guards → HUMAN_LABEL
# ----------------------------
X = df[guard_cols].astype(float).values

log_reg = LogisticRegression(
    penalty="l2",
    solver="liblinear",
)
log_reg.fit(X, y_human)

# model predictions
y_pred_model_bin  = log_reg.predict(X)
y_pred_model_prob = log_reg.predict_proba(X)[:, 1]

ensemble_acc = accuracy_score(y_human, y_pred_model_bin)
ensemble_f1  = f1_score(y_human, y_pred_model_bin, zero_division=0)

# ROC-AUC for the model only makes sense if y_human has both classes
if len(np.unique(y_human)) > 1:
    ensemble_auc = roc_auc_score(y_human, y_pred_model_prob)
else:
    ensemble_auc = np.nan

print("=== Ensemble Logistic Regression (All Guards -> HUMAN_LABEL) ===")
print(f"Accuracy: {ensemble_acc:.3f}")
print(f"F1 Score: {ensemble_f1:.3f}")
print(f"ROC-AUC: {ensemble_auc:.3f}")
print()

coef_series = pd.Series(log_reg.coef_[0], index=guard_cols).sort_values(ascending=False)
print("Guard coefficients (higher => pushes more toward 'unsafe'):")
print(coef_series.to_string(float_format='%.3f'))
print()


# ----------------------------
# 6. Simple rule extractor (decision tree)
# ----------------------------
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y_human)

rules_text = export_text(tree, feature_names=guard_cols)

print("=== Interpretable Rules (Decision Tree depth=3) ===")
print(rules_text)
print()

print("Analysis complete ✅")
