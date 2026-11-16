#!/usr/bin/env python3
"""EDA for kanvas-gpu-failure feature matrix. All numbers computed from the CSV."""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(f"{BASE}/data/kanvas_feature_matrix.csv")
FEATS = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num", "avg_cpu_usr",
         "avg_gpu_util", "avg_cpu_kernel", "avg_load_1", "cap_gpu", "avg_net_receive"]
R = {}

# --- schema / integrity ---
R["shape"] = list(df.shape)
R["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
R["missing_per_col"] = {c: int(df[c].isna().sum()) for c in df.columns}
R["duplicated_rows"] = int(df.duplicated().sum())
R["task_id_unique"] = int(df["task_id"].nunique())
R["label_counts"] = {int(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}
R["failure_rate"] = float(df["label"].mean())

# --- univariate by label ---
uni = {}
for f in FEATS:
    a = df.loc[df.label == 1, f].dropna()
    b = df.loc[df.label == 0, f].dropna()
    # Cohen's d (pooled sd)
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = roc_auc_score(df["label"], df[f])  # univariate AUC (direction as-is)
    uni[f] = {
        "mean_failed": float(a.mean()), "mean_term": float(b.mean()),
        "median_failed": float(a.median()), "median_term": float(b.median()),
        "sd_failed": float(a.std()), "sd_term": float(b.std()),
        "min": float(df[f].min()), "max": float(df[f].max()),
        "cohens_d": float(d), "mw_p": float(u.pvalue),
        "univariate_auc": float(auc),
    }
R["univariate"] = uni

# --- correlations with label (point-biserial = Pearson) and among top features ---
R["corr_with_label"] = {f: float(df[f].corr(df["label"])) for f in FEATS}
R["corr_matrix_top"] = df[["plan_gpu", "plan_mem", "plan_cpu", "inst_num", "avg_gpu_util", "avg_load_1"]].corr().round(3).to_dict()

# --- discrete / grouped failure rates ---
def rate_by(col, bins=None, labels=None):
    if bins is not None:
        g = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    else:
        g = df[col]
    t = df.groupby(g, observed=True)["label"].agg(["mean", "count"])
    return {str(k): {"rate": float(v["mean"]), "n": int(v["count"])} for k, v in t.iterrows()}

R["rate_by_plan_gpu"] = rate_by("plan_gpu")
R["rate_by_cap_gpu"] = rate_by("cap_gpu")
R["rate_by_inst_num_group"] = rate_by("inst_num", bins=[0, 1, 2, 4, 8, 16, 1e9],
                                      labels=["1", "2", "3-4", "5-8", "9-16", ">16"])
R["rate_by_plan_gpu_decile"] = rate_by("plan_gpu", bins=np.unique(np.quantile(df["plan_gpu"], np.linspace(0, 1, 11))))
for f in ["avg_gpu_util", "avg_load_1", "plan_mem", "avg_cpu_usr", "avg_net_receive"]:
    R[f"rate_by_{f}_quartile"] = rate_by(f, bins=np.unique(np.quantile(df[f], [0, .25, .5, .75, 1.0])))

# --- derived: requested GPUs as fraction of machine capacity ---
df["gpu_frac"] = (df["plan_gpu"] / 100.0) / df["cap_gpu"]
R["gpu_frac_stats"] = {
    "mean_failed": float(df.loc[df.label == 1, "gpu_frac"].mean()),
    "mean_term": float(df.loc[df.label == 0, "gpu_frac"].mean()),
}
R["rate_by_gpu_frac"] = rate_by("gpu_frac", bins=[0, 0.125, 0.25, 0.5, 1.0, 1e9],
                                labels=["<=1/8", "1/8-1/4", "1/4-1/2", "1/2-1", ">1"])
# plan_gpu > machine capacity cases (request exceeds one machine's GPUs)
R["plan_gpu_exceeds_cap_share"] = float((df["plan_gpu"] / 100.0 > df["cap_gpu"]).mean())

# --- multivariate logistic regression (standardized), random split like the card ---
X = df[FEATS].values
y = df["label"].values
mu, sd = X.mean(0), X.std(0)
Xs = (X - mu) / sd
Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
lr = LogisticRegression(max_iter=1000, C=1e6).fit(Xtr, ytr)
R["logreg_random_split_auc"] = float(roc_auc_score(yte, lr.predict_proba(Xte)[:, 1]))
lr_full = LogisticRegression(max_iter=1000, C=1e6).fit(Xs, y)
R["logreg_coef_per_sd"] = {f: float(c) for f, c in zip(FEATS, lr_full.coef_[0])}
R["logreg_or_per_sd"] = {f: float(np.exp(c)) for f, c in zip(FEATS, lr_full.coef_[0])}

# --- task_id sanity check (NOT temporal: card says ids are anonymous row numbers) ---
R["corr_taskid_label"] = float(df["task_id"].corr(df["label"]))
half = df["task_id"].median()
R["failure_rate_first_half_ids"] = float(df.loc[df.task_id <= half, "label"].mean())
R["failure_rate_second_half_ids"] = float(df.loc[df.task_id > half, "label"].mean())

# --- value concentration checks ---
R["plan_gpu_value_counts_top"] = {str(k): int(v) for k, v in df["plan_gpu"].value_counts().head(8).items()}
R["inst_num_value_counts_top"] = {str(k): int(v) for k, v in df["inst_num"].value_counts().head(8).items()}
R["cap_gpu_value_counts"] = {str(k): int(v) for k, v in df["cap_gpu"].value_counts().sort_index().items()}
R["plan_mem_unique_top"] = {str(k): int(v) for k, v in df["plan_mem"].value_counts().head(6).items()}

with open(f"{BASE}/scripts/eda_results.json", "w") as fh:
    json.dump(R, fh, indent=1)
print(json.dumps({k: R[k] for k in ["shape", "label_counts", "failure_rate",
                                    "logreg_random_split_auc", "corr_taskid_label"]}, indent=1))
print("full results -> scripts/eda_results.json")
