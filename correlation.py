"""
analyze_guards.py

Pipeline:
1. Per-guard metrics (each guard alone vs human_label) on TEST
2. Equal-weight guardrail ensemble (all guards averaged, no learning) on TEST
3. Agreement comparisons (TEST):
   - LLM vs Human
   - Guardrails vs Human
   - LLM vs Guardrails
4. Learned weighting (logistic regression) trained on TRAIN, evaluated on TEST
5. Interpretable rules (decision tree) trained on TRAIN, evaluated on TEST
6. Train/test sizes + sanity checks
7. Feature importance (which guards matter)
8. 10-fold cross validation (logistic regression robustness across all data)

Data assumptions:
- guard_* columns: floats in [-1, 1]  (more positive = more unsafe)
- human_label: 0 or 1 (ground truth unsafe)
- llm_flag: TRUE / FALSE (LLM judged unsafe or not)
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
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import pearsonr


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
# 1. Load and preprocess data
# ----------------------------
df = pd.read_csv("guards_data.csv")

# normalize headers
df.columns = df.columns.str.strip()

# guard columns
guard_cols = [c for c in df.columns if c.startswith("guard_")]

# human_label: 0/1
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

# guards matrix
X_all = df[guard_cols].astype(float).values

print("Sanity check label balance (full dataset):")
print("  Human labels:\n", pd.Series(y_all_human).value_counts(dropna=False))
print("  LLM flags:\n", pd.Series(y_all_llm).value_counts(dropna=False))
print(f"\nLoaded {len(df)} samples with {len(guard_cols)} guards.\n")


# ----------------------------
# 2. Train / test split (80/20)
# ----------------------------
# Stratify by y_all_human if we actually have both classes; otherwise leave unstratified
stratifier = y_all_human if len(np.unique(y_all_human)) > 1 else None

X_train, X_test, y_train_human, y_test_human, y_train_llm, y_test_llm, df_train, df_test = train_test_split(
    X_all,
    y_all_human,
    y_all_llm,
    df,
    test_size=0.2,
    random_state=42,
    stratify=stratifier,
)

df_train_guards = df_train[guard_cols].astype(float)
df_test_guards  = df_test[guard_cols].astype(float)

print(f"Train size: {len(df_train)} rows")
print(f"Test size : {len(df_test)} rows\n")


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
# average all guard scores equally
avg_guard_score_test = df_test_guards.mean(axis=1)

# treat >0 as unsafe
guardrails_pred_bin_test = binarize_guard(avg_guard_score_test, thresh=0.0)

# AUROC vs human / vs llm on test
guardrails_auc_vs_human = safe_roc_auc(y_test_human, avg_guard_score_test)
guardrails_auc_vs_llm   = safe_roc_auc(y_test_llm,   avg_guard_score_test)

# Guardrails vs Human
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
# train on TRAIN, evaluate on TEST
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

print("=== Learned Ensemble (LogReg TRAIN->TEST) ===")
print(f"AUROC vs HUMAN (test): {lr_auc_test:.3f}")
print("Metrics vs HUMAN_LABEL (test):")
print(lr_metrics_vs_human)
print()

coef_series = pd.Series(log_reg.coef_[0], index=guard_cols).sort_values(ascending=False)
print("Guard coefficients (higher => pushes more toward 'unsafe'):")
print(coef_series.to_string(float_format='%.3f'))
print()


# ----------------------------
# 7. Decision tree (TRAIN -> TEST)
# ----------------------------
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train_human)

rules_text = export_text(tree, feature_names=guard_cols)
print("=== Interpretable Rules (Decision Tree depth=3, TRAIN->TEST) ===")
print(rules_text)
print()

# Evaluate tree on TEST
y_pred_tree_bin_test = tree.predict(X_test)

tree_metrics_vs_human = evaluate_binary_predictions(
    y_true=y_test_human,
    y_pred_bin=y_pred_tree_bin_test,
)

if hasattr(tree, "predict_proba"):
    tree_auc_test = safe_roc_auc(y_test_human, tree.predict_proba(X_test)[:, 1])
else:
    tree_auc_test = safe_roc_auc(y_test_human, y_pred_tree_bin_test)

print("Tree metrics vs HUMAN_LABEL (test):")
print(tree_metrics_vs_human)
print(f"Tree AUROC vs HUMAN (test): {tree_auc_test:.3f}")
print()


# ----------------------------
# 8. 10-fold cross validation on the WHOLE DATA
# (logistic regression robustness)
# ----------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_acc    = []
fold_f1     = []
fold_auc    = []
fold_kappa  = []  # agreement between model and human in that fold

for train_idx, test_idx in kf.split(X_all):
    X_tr, X_te = X_all[train_idx], X_all[test_idx]
    y_tr, y_te = y_all_human.iloc[train_idx], y_all_human.iloc[test_idx]

    # train logreg on fold's train
    lr_cv = LogisticRegression(penalty="l2", solver="liblinear")
    lr_cv.fit(X_tr, y_tr)

    # predict on fold's held-out test
    y_pred_bin_cv  = lr_cv.predict(X_te)
    y_pred_prob_cv = lr_cv.predict_proba(X_te)[:, 1]

    # accuracy / f1
    fold_acc.append(accuracy_score(y_te, y_pred_bin_cv))
    fold_f1.append(f1_score(y_te, y_pred_bin_cv, zero_division=0))

    # AUROC (safe)
    fold_auc.append(safe_roc_auc(y_te, y_pred_prob_cv))

    # Cohen's kappa between model prediction and human in this fold
    fold_kappa.append(cohen_kappa_score(y_te, y_pred_bin_cv))

print("=== 10-Fold Cross Validation (LogReg vs HUMAN_LABEL) ===")
print(f"Accuracy: mean={np.nanmean(fold_acc):.3f}  std={np.nanstd(fold_acc):.3f}")
print(f"F1 Score: mean={np.nanmean(fold_f1):.3f}   std={np.nanstd(fold_f1):.3f}")
print(f"AUROC   : mean={np.nanmean(fold_auc):.3f}  std={np.nanstd(fold_auc):.3f}")
print(f"Kappa   : mean={np.nanmean(fold_kappa):.3f} std={np.nanstd(fold_kappa):.3f}")
print()

print("Analysis complete ✅")
