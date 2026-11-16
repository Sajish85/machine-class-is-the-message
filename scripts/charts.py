#!/usr/bin/env python3
"""Publication-quality charts for the kanvas-gpu-failure study."""
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{BASE}/charts"
df = pd.read_csv(f"{BASE}/data/kanvas_feature_matrix.csv")
FEATS = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num", "avg_cpu_usr",
         "avg_gpu_util", "avg_cpu_kernel", "avg_load_1", "cap_gpu", "avg_net_receive"]
CTX = ["avg_cpu_usr", "avg_gpu_util", "avg_cpu_kernel", "avg_load_1", "avg_net_receive"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200})
C_ALL, C2, C8 = "#444444", "#2c7fb8", "#d95f02"

def quartile_rates(data, col):
    qs = np.unique(np.quantile(data[col], [0, .25, .5, .75, 1.0]))
    g = pd.cut(data[col], bins=qs, include_lowest=True)
    return data.groupby(g, observed=True)["label"].mean().values * 100

# ---- Chart 1: Simpson's paradox — GPU-util quartile dose-response, overall vs within class
fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.arange(4)
w = 0.27
ax.bar(x - w, quartile_rates(df, "avg_gpu_util"), w, color=C_ALL, label="All tasks (n=18,890)")
ax.bar(x, quartile_rates(df[df.cap_gpu == 2], "avg_gpu_util"), w, color=C2, label="2-GPU machines (n=5,793)")
ax.bar(x + w, quartile_rates(df[df.cap_gpu == 8], "avg_gpu_util"), w, color=C8, label="8-GPU machines (n=13,097)")
ax.set_xticks(x, ["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"])
ax.set_xlabel("Mean GPU utilization of assigned machine, prior hour (quartiles within each group)")
ax.set_ylabel("Task failure rate (%)")
ax.set_title("Machine busyness predicts failure overall —\nbut not within either machine class")
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/chart1_simpsons_paradox.png"); plt.close(fig)

# ---- Chart 2: failure rate by requested GPU size, by machine class
vals = [10, 25, 50, 100]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for cap, color, lab in [(2, C2, "2-GPU machines"), (8, C8, "8-GPU machines")]:
    g = df[df.cap_gpu == cap]
    rates, ns = [], []
    for v in vals:
        s = g[g.plan_gpu == v]
        rates.append(s["label"].mean() * 100); ns.append(len(s))
    bars = ax.bar([i + (-0.19 if cap == 2 else 0.19) for i in range(len(vals))],
                  rates, 0.38, color=color, label=lab)
    for b, n in zip(bars, ns):
        ax.annotate(f"n={n:,}", (b.get_x() + b.get_width() / 2, b.get_height() + 1.2),
                    ha="center", fontsize=7.5, color="#555555")
ax.set_xticks(range(len(vals)), ["0.10 GPU", "0.25 GPU", "0.50 GPU", "1.00 GPU"])
ax.set_xlabel("GPU requested per instance (plan_gpu; 100 = 1 GPU)")
ax.set_ylabel("Task failure rate (%)")
ax.set_title("Quarter-GPU requests on 8-GPU machines fail at 54%\n— 9x the rate of the same request on 2-GPU machines")
ax.set_ylim(0, 62)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/chart2_plan_gpu_by_class.png"); plt.close(fig)

# ---- Chart 3: univariate effect sizes (Cohen's d)
import json as _json
uni = _json.load(open(f"{BASE}/scripts/eda_results.json"))["univariate"]
fam = {"plan_cpu": "Request", "plan_mem": "Request", "plan_gpu": "Request", "inst_num": "Request",
       "cap_gpu": "Machine class", "avg_cpu_usr": "Machine context (prior hour)",
       "avg_gpu_util": "Machine context (prior hour)", "avg_cpu_kernel": "Machine context (prior hour)",
       "avg_load_1": "Machine context (prior hour)", "avg_net_receive": "Machine context (prior hour)"}
names = {"plan_cpu": "CPU requested", "plan_mem": "Memory requested", "plan_gpu": "GPU requested",
         "inst_num": "Instances requested", "cap_gpu": "GPUs on machine (class)",
         "avg_cpu_usr": "Machine CPU (user)", "avg_gpu_util": "Machine GPU utilization",
         "avg_cpu_kernel": "Machine CPU (kernel)", "avg_load_1": "Machine load (1 min)",
         "avg_net_receive": "Machine network receive"}
order = sorted(uni, key=lambda f: uni[f]["cohens_d"])
colors = {"Request": "#2c7fb8", "Machine class": "#d95f02", "Machine context (prior hour)": "#999999"}
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for i, f in enumerate(order):
    ax.barh(i, uni[f]["cohens_d"], 0.62, color=colors[fam[f]])
ax.axvline(0, color="#333333", lw=0.9)
ax.set_yticks(range(len(order)), [names[f] for f in order])
ax.set_xlabel("Cohen's d (Failed minus Terminated) — univariate")
ax.set_title("Univariate effect sizes: machine class and request\nsize dominate; machine context is weaker")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
ax.legend(handles, list(colors.keys()), frameon=False, fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/chart3_effect_sizes.png"); plt.close(fig)

# ---- Chart 4: ROC — full model vs context-only vs machine class alone
y = df["label"].values
def roc_for(feats):
    X = df[feats].values
    Xs = (X - X.mean(0)) / X.std(0)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
    m = LogisticRegression(max_iter=1000, C=1e6).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p)
    return fpr, tpr, roc_auc_score(yte, p)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
for feats, lab, col in [(FEATS, "All 10 features", "#2c7fb8"),
                        (CTX + ["cap_gpu"], "Machine context + class", "#d95f02"),
                        (CTX, "Machine context only", "#999999")]:
    fpr, tpr, a = roc_for(feats)
    ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{lab} (AUC = {a:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="Chance (AUC = 0.500)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Discrimination on a random split (test n=4,723)\nContext telemetry alone is weak; request features carry the signal")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/chart4_roc.png"); plt.close(fig)
print("charts written to", OUT)
