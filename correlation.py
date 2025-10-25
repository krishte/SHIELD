"""
analyze_guards.py

Pipeline:
1. Per-guard metrics (each guard alone vs human_label)
2. Equal-weight guardrail ensemble (all guards averaged, no learning)
3. Agreement comparisons:
   - LLM vs Human
   - Guardrails vs Human
   - LLM vs Guardrails
4. Learned weighting via logistic regression (feature importance)
5. Simple interpretable decision tree

Data assumptions:
- guard_* columns: floats in [-1, 1]  (more positive = more unsafe)
- human_label: 0 or 1 (our ground truth of unsafe)
- llm_flag: TRUE / FALSE (LLM said unsafe or not)
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

# normalize headers
df.columns = df.columns.str.strip()

# guard columns
guard_cols = [c for c in df.columns if c.startswith("guard_")]

# human_label: 0/1 ground truth
y_human = (
    df["human_label"]
    .astype(str)
    .str.strip()
    .astype(int)
)

# llm_flag: TRUE/FALSE → 1/0
y_llm = (
    df["llm_flag"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
    .astype(int)
)

print("Sanity check label balance:")
print("  Human labels:\n", pd.Series(y_human).value_counts(dropna=False))
print("  LLM flags:\n", pd.Series(y_llm).value_counts(dropna=False))
print(f"\nLoaded {len(df)} samples with {len(guard_cols)} guards.\n")

# matrix of guard scores
X_guards = df[guard_cols].astype(float).values


# ----------------------------
# Helper functions
# ----------------------------
def binarize_guard(series, thresh=0.0):
    """score > thresh => unsafe (1), else safe (0)"""
    return (series > thresh).astype(int)


def safe_roc_auc(y_true, scores):
    """AUROC if both classes exist and scores vary, else NaN"""
    if len(np.unique(y_true)) < 2:
        return np.nan
    if pd.Series(scores).nunique() < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, scores)
    except ValueError:
        return np.nan


def safe_pearson(x, y):
    """Pearson r if both vary, else (NaN, NaN)"""
    if pd.Series(x).nunique() < 2:
        return (np.nan, np.nan)
    if pd.Series(y).nunique() < 2:
        return (np.nan, np.nan)
    try:
        return pearsonr(x, y)
    except Exception:
        return (np.nan, np.nan)


def evaluate_binary_predictions(y_true, y_pred_bin, label_for=""):
    """Return dict of accuracy/precision/recall/F1 + confmat for reporting."""
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc  = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true, y_pred_bin,   zero_division=0)
    f1   = f1_score(y_true, y_pred_bin,       zero_division=0)

    return {
        "target": label_for,
        "acc": acc,
        "prec": prec,
        "recall": rec,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ----------------------------
# 2. Per-guard analysis vs HUMAN LABEL
# ----------------------------
per_guard_rows = []

for g in guard_cols:
    scores = df[g].astype(float)

    auroc_h = safe_roc_auc(y_human, scores)
    r_h, p_h = safe_pearson(scores, y_human)

    y_pred_bin = binarize_guard(scores, thresh=0.0)
    metrics_vs_human = evaluate_binary_predictions(y_human, y_pred_bin, label_for=g)

    per_guard_rows.append({
        "guard": g,
        "auroc_vs_human": auroc_h,
        "pearson_r_vs_human": r_h,
        "p_value_vs_human": p_h,
        "acc@thr0": metrics_vs_human["acc"],
        "prec@thr0": metrics_vs_human["prec"],
        "recall@thr0": metrics_vs_human["recall"],
        "f1@thr0": metrics_vs_human["f1"],
        "tn": metrics_vs_human["tn"],
        "fp": metrics_vs_human["fp"],
        "fn": metrics_vs_human["fn"],
        "tp": metrics_vs_human["tp"],
    })

per_guard_df = pd.DataFrame(per_guard_rows).sort_values("f1@thr0", ascending=False)

print("=== Per-Guard Performance vs HUMAN_LABEL ===")
print(per_guard_df.to_string(index=False, float_format="%.3f"))
print()


# ----------------------------
# 3. Equal-weight guardrail ensemble (no learning)
# ----------------------------
# Step 1: average all guards per row  (this is "the guardrails score")
# If avg_score > 0 => unsafe, else safe
avg_guard_score = df[guard_cols].astype(float).mean(axis=1)
guardrails_pred_bin = binarize_guard(avg_guard_score, thresh=0.0)

# We'll also keep the raw avg score for AUROC-style eval
guardrails_auc_vs_human = safe_roc_auc(y_human, avg_guard_score)
guardrails_auc_vs_llm   = safe_roc_auc(y_llm,   avg_guard_score)

# 3a. Guardrails vs Human (your second requested comparison)
guard_vs_human_metrics = evaluate_binary_predictions(
    y_true=y_human,
    y_pred_bin=guardrails_pred_bin,
    label_for="guardrails_vs_human"
)

# 3b. Guardrails vs LLM (your third requested comparison)
guard_vs_llm_metrics = evaluate_binary_predictions(
    y_true=y_llm,
    y_pred_bin=guardrails_pred_bin,
    label_for="guardrails_vs_llm"
)

print("=== Equal-weight Guardrails Ensemble (all guards averaged) ===")
print(f"AUROC guardrails vs HUMAN: {guardrails_auc_vs_human:.3f}")
print(f"AUROC guardrails vs LLM  : {guardrails_auc_vs_llm:.3f}")
print("\nGuardrails vs HUMAN_LABEL:")
print(guard_vs_human_metrics)
print("\nGuardrails vs LLM_FLAG:")
print(guard_vs_llm_metrics)
print()


# ----------------------------
# 4. Human ↔ LLM agreement (your first requested comparison)
# ----------------------------
# LLM vs Human
llm_vs_human_metrics = evaluate_binary_predictions(
    y_true=y_human,
    y_pred_bin=y_llm,    # treat LLM flag as its prediction
    label_for="llm_vs_human"
)

kappa_human_llm = cohen_kappa_score(y_human, y_llm)

print("=== LLM ↔ Human comparison ===")
print("LLM vs HUMAN metrics:")
print(llm_vs_human_metrics)
print(f"Cohen's kappa (LLM vs Human): {kappa_human_llm:.3f}")
print()


# ----------------------------
# 5. Learned weighting: Logistic regression (which guards matter most?)
# ----------------------------
log_reg = LogisticRegression(
    penalty="l2",
    solver="liblinear",
)
log_reg.fit(X_guards, y_human)

# Predictions from learned model
y_pred_lr_bin  = log_reg.predict(X_guards)
y_pred_lr_prob = log_reg.predict_proba(X_guards)[:, 1]

lr_auc = safe_roc_auc(y_human, y_pred_lr_prob)
lr_metrics_vs_human = evaluate_binary_predictions(
    y_true=y_human,
    y_pred_bin=y_pred_lr_bin,
    label_for="log_reg_vs_human"
)

print("=== Learned Ensemble (Logistic Regression, guards -> HUMAN_LABEL) ===")
print(f"AUROC vs HUMAN: {lr_auc:.3f}")
print("Metrics vs HUMAN_LABEL:")
print(lr_metrics_vs_human)
print()

# Feature importances = learned weights
coef_series = pd.Series(log_reg.coef_[0], index=guard_cols).sort_values(ascending=False)
print("Guard coefficients (higher => pushes more toward 'unsafe'):")
print(coef_series.to_string(float_format='%.3f'))
print()


# ----------------------------
# 6. Interpretable rules (Decision Tree on guards -> HUMAN_LABEL)
# ----------------------------
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_guards, y_human)

rules_text = export_text(tree, feature_names=guard_cols)

print("=== Interpretable Rules (Decision Tree depth=3) ===")
print(rules_text)
print()

print("Analysis complete ✅")
