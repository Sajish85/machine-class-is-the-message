#!/usr/bin/env python3
"""Stratified follow-up: is the telemetry signal just machine-class confounding?"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(f"{BASE}/data/kanvas_feature_matrix.csv")
FEATS = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num", "avg_cpu_usr",
         "avg_gpu_util", "avg_cpu_kernel", "avg_load_1", "cap_gpu", "avg_net_receive"]
CTX = ["avg_cpu_usr", "avg_gpu_util", "avg_cpu_kernel", "avg_load_1", "avg_net_receive"]
S = {}

# context feature levels by machine class
S["context_means_by_cap"] = {
    str(cap): {f: float(g[f].mean()) for f in CTX}
    for cap, g in df.groupby("cap_gpu")
}
S["plan_means_by_cap"] = {
    str(cap): {f: float(g[f].mean()) for f in ["plan_cpu", "plan_mem", "plan_gpu", "inst_num"]}
    for cap, g in df.groupby("cap_gpu")
}

def quartile_rates(data, col):
    qs = np.unique(np.quantile(data[col], [0, .25, .5, .75, 1.0]))
    g = pd.cut(data[col], bins=qs, include_lowest=True)
    t = data.groupby(g, observed=True)["label"].agg(["mean", "count"])
    return {str(k): {"rate": float(v["mean"]), "n": int(v["count"])} for k, v in t.iterrows()}

for cap, g in df.groupby("cap_gpu"):
    S[f"gpu_util_quartile_rates_cap{int(cap)}"] = quartile_rates(g, "avg_gpu_util")
    S[f"load_quartile_rates_cap{int(cap)}"] = quartile_rates(g, "avg_load_1")
    # failure rate by plan_gpu value within class
    t = g.groupby("plan_gpu")["label"].agg(["mean", "count"])
    S[f"rate_by_plan_gpu_cap{int(cap)}"] = {
        str(k): {"rate": float(v["mean"]), "n": int(v["count"])}
        for k, v in t.iterrows() if v["count"] >= 100}

# within-class logistic regression (context features only, then all-but-cap)
for cap, g in df.groupby("cap_gpu"):
    feats = [f for f in FEATS if f != "cap_gpu"]
    X = g[feats].values; y = g["label"].values
    Xs = (X - X.mean(0)) / X.std(0)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
    m = LogisticRegression(max_iter=1000, C=1e6).fit(Xtr, ytr)
    S[f"logreg_auc_within_cap{int(cap)}"] = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    # context-only model within class
    Xc = g[CTX].values
    Xcs = (Xc - Xc.mean(0)) / Xc.std(0)
    Xtr, Xte, ytr, yte = train_test_split(Xcs, y, test_size=0.25, random_state=42, stratify=y)
    mc = LogisticRegression(max_iter=1000, C=1e6).fit(Xtr, ytr)
    S[f"context_only_auc_within_cap{int(cap)}"] = float(roc_auc_score(yte, mc.predict_proba(Xte)[:, 1]))

# context-only model overall (no cap_gpu) vs with cap_gpu
for name, feats in [("context_only_overall", CTX), ("context_plus_cap", CTX + ["cap_gpu"])]:
    X = df[feats].values; y = df["label"].values
    Xs = (X - X.mean(0)) / X.std(0)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
    m = LogisticRegression(max_iter=1000, C=1e6).fit(Xtr, ytr)
    S[f"logreg_auc_{name}"] = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))

# KAAM phi-curve ranges (which features the published model weights most)
phi = pd.read_csv(f"{BASE}/data/kanvas_phi_curves.csv")
S["phi_ranges"] = {
    f: {"range": float(g["phi_value"].max() - g["phi_value"].min()),
        "min": float(g["phi_value"].min()), "max": float(g["phi_value"].max())}
    for f, g in phi.groupby("feature_name")
}

with open(f"{BASE}/scripts/stratified_results.json", "w") as fh:
    json.dump(S, fh, indent=1)
print(json.dumps(S, indent=1))
