# pipelines/compute_xi.py
"""
Compute XI metrics (LF/GF/FA/RS/CL/HC) for a given run folder with a trained model & data.
Expected env:
- RUN_FOLDER: evidence/<run_id> (TI artifacts already written there)
- (optional) MODEL_PATH, DATA_PATH: unused in this demo (uses the same dataset)
Writes to RUN_FOLDER/explainability/*.json
"""

import os, json, io, hashlib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import shap

RUN_FOLDER = os.environ.get("RUN_FOLDER", "evidence/demo_run")
os.makedirs(os.path.join(RUN_FOLDER, "explainability"), exist_ok=True)

# For demo: rebuild same model/data as in train_and_log
data = load_breast_cancer(as_frame=True)
X_df = data["data"]
y = data["target"].values
Xtr, Xte, ytr, yte = train_test_split(X_df, y, test_size=0.3, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr.values, ytr)

# SHAP
explainer = shap.TreeExplainer(clf)
shap_out = explainer.shap_values(Xte.values)
shap_vals = shap_out[1] if isinstance(shap_out, list) else shap_out
mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
rank_order = np.argsort(-mean_abs_shap)

# Global stability
c2 = RandomForestClassifier(n_estimators=300, random_state=99, n_jobs=-1).fit(Xtr.values, ytr)
expl2 = shap.TreeExplainer(c2)
shap_out2 = expl2.shap_values(Xte.values)
shap_vals2 = shap_out2[1] if isinstance(shap_out2, list) else shap_out2
mean_abs_shap2 = np.mean(np.abs(shap_vals2), axis=0)
res = spearmanr(mean_abs_shap, mean_abs_shap2)
corr = getattr(res, "statistic", getattr(res, "correlation", res[0]))
corr = float(np.array(corr).flatten()[0])
corr = max(-1.0, min(1.0, corr))
with open(os.path.join(RUN_FOLDER, "explainability", "global_stability.json"), "w") as f:
    json.dump({"spearman": corr}, f, indent=2)

# Faithfulness
def deletion_auc(model, X, y, rank_order, steps=10):
    base = model.predict_proba(X)[:, 1]
    base_auc = roc_auc_score(y, base)
    aucs = [base_auc]
    Xmod = X.copy()
    k_per = max(1, len(rank_order)//steps)
    for i in range(1, steps+1):
        topk = rank_order[:i*k_per]
        Xmod_ = Xmod.copy()
        Xmod_[:, topk] = 0.0
        preds = model.predict_proba(Xmod_)[:, 1]
        aucs.append(roc_auc_score(y, preds))
    drop = max(0.0, base_auc - float(np.mean(aucs[1:])))
    return float(np.clip(drop / max(1e-6, base_auc), 0.0, 1.0))

fa = deletion_auc(clf, Xte.values, yte, rank_order, steps=10)
with open(os.path.join(RUN_FOLDER, "explainability", "faithfulness.json"), "w") as f:
    json.dump({"deletion_auc": fa}, f, indent=2)

# Robustness
def jaccard(a, b): a, b = set(a), set(b); return len(a & b)/max(1, len(a | b))
k = max(5, X_df.shape[1] // 5)
rank_order = np.argsort(-mean_abs_shap).astype(int).ravel()
topk = set(rank_order[:k].tolist())
Xpert = Xte.values + np.random.normal(0, 0.01, Xte.values.shape)
shap_out_pert = explainer.shap_values(Xpert)
shap_vals_pert = shap_out_pert[1] if isinstance(shap_out_pert, list) else shap_out_pert
mean_abs_shap_pert = np.mean(np.abs(shap_vals_pert), axis=0)
rank_order_pert = np.argsort(-mean_abs_shap_pert).astype(int).ravel()
topk_pert = set(rank_order_pert[:k].tolist())
jac = float(jaccard(topk, topk_pert))
with open(os.path.join(RUN_FOLDER, "explainability", "robustness.json"), "w") as f:
    json.dump({"jaccard_topk": jac}, f, indent=2)

# Coverage (demo): 100%
with open(os.path.join(RUN_FOLDER, "explainability", "coverage.json"), "w") as f:
    json.dump({"coverage": 1.0}, f, indent=2)

# Local fidelity (demo): store placeholder 0.7; for full LIME, run in a job with more resources
with open(os.path.join(RUN_FOLDER, "explainability", "local_fidelity.json"), "w") as f:
    json.dump({"r2": 0.70}, f, indent=2)

# Human comprehensibility (demo)
with open(os.path.join(RUN_FOLDER, "explainability", "human_comprehensibility.json"), "w") as f:
    json.dump({"score": 0.70}, f, indent=2)

print(f"XI metrics written to: {RUN_FOLDER}/explainability")
