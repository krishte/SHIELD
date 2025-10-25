"""
analyze_guards.py

Pipeline:
1. Per-guard metrics (each guard alone vs human_label) on TEST
2. Equal-weight guardrail ensemble (no learning) on TEST
3. Agreement comparisons on TEST:
   - LLM vs Human
   - Guardrails vs Human
   - LLM vs Guardrails
4. Learned weighting (logistic regression) trained on TRAIN, evaluated on TEST
5. Simple interpretable decision tree trained on TRAIN, evaluated on TEST

Data assumptions:
- guard_* columns: floats in [-1, 1]  (more positive = more unsafe)
- human_label: 0 or 1 (ground truth unsafe)
- llm_flag: TRUE / FALSE (LLM says unsafe or not)
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
from sklearn.model_selection import train_test_split
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
y_all_human = (
    df["human_label"]
    .astype(str)
    .str.strip()
    .astype(int)
)

# llm_flag: TRUE/FALSE → 1/0
y_all_llm = (
    df["llm_flag"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
    .astype(int)
)

X_all = df[guard_cols].astype(float).values

print("Sanity check label balance (full dataset):")
print("  Human labels:\n", pd.Series(y_all_human).value_counts(dropna=False))
print("  LLM flags:\n", pd.Series(y_all_llm).value_counts(dropna=False))
print(f"\nLoaded {len(df)} samples with {len(guard_cols)} guards.\n")


# ----------------------------
# 2. Train / test split (80/20)
# ----------------------------
# We split once and then use:
#   - X_train / y_train_human to train learned models
#   - X_test  / y_test_*     to report all metrics
X_train, X_test, y_train_human, y_test_human, y_train_llm, y_test_llm, df_train, df_test = train_test_split(
    X_all,
    y_all_human,
    y_all_llm,
    df,  # keep rows so we can still do per-guard analysis with original columns
    test_size=0.2,
    random_state=42,
    stratify=y_all_human if len(np.unique(y_all_human)) > 1 else None,
)

# grab guard columns for train/test DataFrames
df_train_guards = df_train[guard_cols].astype(float)
df_test_guards  = df_test[guard_cols].astype(float)

print(f"Train size: {len(df_train)} rows")
print(f"Test size : {len(df_test)} rows\n")


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


def evaluate_binary_predictions(y_true, y_pred_bin):
    """Return dict of accuracy/precision/recall/F1 + confmat for reporting."""
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc  = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true, y_pred_bin,   zero_division=0)
    f1   = f1_score(y_true, y_pred_bin,       zero_division=0)

    return {
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
# 3. Per-guard analysis vs HUMAN LABEL (TEST SET ONLY)
# ----------------------------
per_guard_rows = []

for g in guard_cols:
    scores_test = df_test[g].astype(float)

    auroc_h = safe_roc_auc(y_test_human, scores_test)
    r_h, p_h = safe_pearson(scores_test, y_test_human)

    y_pred_bin = binarize_guard(scores_test, thresh=0.0)
    metrics_vs_human = evaluate_binary_predictions(y_test_human, y_pred_bin)

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

print("=== Per-Guard Performance vs HUMAN_LABEL (TEST SET) ===")
print(per_guard_df.to_string(index=False, float_format="%.3f"))
print()


# ----------------------------
# 4. Equal-weight guardrail ensemble (TEST SET)
# ----------------------------
# average all guard scores across columns for each row
avg_guard_score_test = df_test_guards.mean(axis=1)

# turn that into a binary unsafe/safe prediction
guardrails_pred_bin_test = binarize_guard(avg_guard_score_test, thresh=0.0)

# AUROC vs human / vs llm on test
guardrails_auc_vs_human = safe_roc_auc(y_test_human, avg_guard_score_test)
guardrails_auc_vs_llm   = safe_roc_auc(y_test_llm,   avg_guard_score_test)

# Guardrails vs Human (equal-weight ensemble vs human label)
guard_vs_human_metrics = evaluate_binary_predictions(
    y_true=y_test_human,
    y_pred_bin=guardrails_pred_bin_test,
)

# Guardrails vs LLM
guard_vs_llm_metrics = evaluate_binary_predictions(
    y_true=y_test_llm,
    y_pred_bin=guardrails_pred_bin_test,
)

print("=== Equal-weight Guardrails Ensemble (TEST SET) ===")
print(f"AUROC guardrails vs HUMAN: {guardrails_auc_vs_human:.3f}")
print(f"AUROC guardrails vs LLM  : {guardrails_auc_vs_llm:.3f}")
print("\nGuardrails vs HUMAN_LABEL:")
print(guard_vs_human_metrics)
print("\nGuardrails vs LLM_FLAG:")
print(guard_vs_llm_metrics)
print()


# ----------------------------
# 5. Human ↔ LLM agreement (TEST SET)
# ----------------------------
# Treat y_test_llm as the LLM's predicted label and compare to human truth
llm_vs_human_metrics = evaluate_binary_predictions(
    y_true=y_test_human,
    y_pred_bin=y_test_llm,
)

kappa_human_llm = cohen_kappa_score(y_test_human, y_test_llm)

print("=== LLM ↔ Human comparison (TEST SET) ===")
print("LLM vs HUMAN metrics:")
print(llm_vs_human_metrics)
print(f"Cohen's kappa (LLM vs Human): {kappa_human_llm:.3f}")
print()


# ----------------------------
# 6. Learned weighting: Logistic regression
# Train on TRAIN, evaluate on TEST
# ----------------------------
log_reg = LogisticRegression(
    penalty="l2",
    solver="liblinear",
)
log_reg.fit(X_train, y_train_human)

# Predict on TEST
y_pred_lr_bin_test  = log_reg.predict(X_test)
y_pred_lr_prob_test = log_reg.predict_proba(X_test)[:, 1]

lr_auc_test = safe_roc_auc(y_test_human, y_pred_lr_prob_test)
lr_metrics_vs_human = evaluate_binary_predictions(
    y_true=y_test_human,
    y_pred_bin=y_pred_lr_bin_test,
)

print("=== Learned Ensemble (LogReg trained on TRAIN, evaluated on TEST) ===")
print(f"AUROC vs HUMAN (test): {lr_auc_test:.3f}")
print("Metrics vs HUMAN_LABEL (test):")
print(lr_metrics_vs_human)
print()

# Feature importances on the trained model
coef_series = pd.Series(log_reg.coef_[0], index=guard_cols).sort_values(ascending=False)
print("Guard coefficients (higher => pushes more toward 'unsafe'):")
print(coef_series.to_string(float_format='%.3f'))
print()


# ----------------------------
# 7. Interpretable tree: train on TRAIN, evaluate on TEST
# ----------------------------
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train_human)

rules_text = export_text(tree, feature_names=guard_cols)
print("=== Interpretable Rules (Decision Tree depth=3, trained on TRAIN) ===")
print(rules_text)
print()

# Evaluate tree binary predictions on TEST
y_pred_tree_bin_test = tree.predict(X_test)
tree_metrics_vs_human = evaluate_binary_predictions(
    y_true=y_test_human,
    y_pred_bin=y_pred_tree_bin_test,
)
tree_auc_test = safe_roc_auc(y_test_human, tree.predict_proba(X_test)[:,1] if hasattr(tree, "predict_proba") else y_pred_tree_bin_test)

print("Tree metrics vs HUMAN_LABEL (test):")
print(tree_metrics_vs_human)
print(f"Tree AUROC vs HUMAN (test): {tree_auc_test:.3f}")
print()

print("Analysis complete ✅")
