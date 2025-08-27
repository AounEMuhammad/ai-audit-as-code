# pipelines/train_and_log.py
"""
Train a model, log to MLflow, and emit TI artifacts:
- evidence/<run_id>/dataset.hash
- evidence/<run_id>/logs/train.log
- evidence/<run_id>/model_card.json
- evidence/<run_id>/audit_trail.json  (demo placeholder)
- evidence/<run_id>/replication.json  (optional basic replication)

Optionally reads DVC to populate dataset.hash from .dvc MD5.
"""

import os, io, json, hashlib, time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# Optional MLflow (no server required for demo; uses local ./mlruns)
try:
    import mlflow
    from mlflow import sklearn as mlflow_sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

OUT_BASE = os.environ.get("EVIDENCE_OUT_BASE", "evidence")
RUN_FOLDER = os.environ.get("RUN_FOLDER")  # e.g., "evidence/2025-08-26-123456", else create
if not RUN_FOLDER:
    RUN_FOLDER = os.path.join(OUT_BASE, time.strftime("%Y%m%d-%H%M%S"))

os.makedirs(RUN_FOLDER, exist_ok=True)
os.makedirs(os.path.join(RUN_FOLDER, "logs"), exist_ok=True)

def sha256_df(df: pd.DataFrame) -> str:
    """Compute SHA256 of a dataframe snapshot (parquet in-memory)."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()

def dvc_md5(path: str) -> str | None:
    """Try to read MD5 from DVC .dvc file if it exists; else None."""
    meta = f"{path}.dvc"
    if not os.path.exists(meta): return None
    try:
        import yaml
        with open(meta, "r") as f:
            d = yaml.safe_load(f) or {}
        md5 = d.get("outs", [{}])[0].get("md5")
        return md5
    except Exception:
        return None

# ---------------- Load data ----------------
data = load_breast_cancer(as_frame=True)
X = data["data"]
y = data["target"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ---------------- Compute dataset.hash ----------------
# If you have a DVC-managed dataset file, set DATASET_PATH env and this will read .dvc md5
DATASET_PATH = os.environ.get("DATASET_PATH")  # e.g., data/dataset.csv
hash_value = None
if DATASET_PATH:
    md5 = dvc_md5(DATASET_PATH)
    if md5:
        hash_value = md5
if not hash_value:
    hash_value = sha256_df(X)  # fallback to data snapshot

with open(os.path.join(RUN_FOLDER, "dataset.hash"), "w") as f:
    f.write(str(hash_value))

# ---------------- Train model ----------------
clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
clf.fit(Xtr.values, ytr)
proba = clf.predict_proba(Xte.values)[:, 1]
auc = roc_auc_score(yte, proba)
acc = accuracy_score(yte, (proba > 0.5).astype(int))

# ---------------- Log to MLflow (optional) ----------------
run_id = None
if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params({"n_estimators": 300, "random_state": 0})
        mlflow.log_metrics({"auc": float(auc), "acc": float(acc)})
        mlflow_sklearn.log_model(clf, "model")
else:
    run_id = time.strftime("%Y%m%d-%H%M%S")

# Persist log and model card
with open(os.path.join(RUN_FOLDER, "logs", "train.log"), "w") as f:
    f.write(f"model=RandomForest n_estimators=300 random_state=0\nauc={auc:.4f} acc={acc:.4f}\n")

with open(os.path.join(RUN_FOLDER, "model_card.json"), "w") as f:
    json.dump({
        "model_name": "rf-breast-cancer",
        "version": os.environ.get("MODEL_VERSION", "demo-1.0"),
        "framework": "sklearn",
        "run_id": run_id,
        "params": {"n_estimators": 300, "random_state": 0},
        "metrics": {"auc": float(auc), "acc": float(acc)},
        "features": list(X.columns),
        "train_size": int(len(Xtr)),
        "test_size": int(len(Xte)),
        "mlflow_tracking": os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"),
    }, f, indent=2)

# audit_trail.json (demo; set immutable True when using WORM storage)
with open(os.path.join(RUN_FOLDER, "audit_trail.json"), "w") as f:
    json.dump({"immutable": True, "records": int(len(Xte))}, f, indent=2)

# Simple replication (N=3) demo
passes = 0
for seed in [1, 2, 3]:
    c = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    c.fit(Xtr.values, ytr)
    p = c.predict_proba(Xte.values)[:, 1]
    auc2 = roc_auc_score(yte, p)
    if abs(auc2 - auc) <= 0.02:
        passes += 1

with open(os.path.join(RUN_FOLDER, "replication.json"), "w") as f:
    json.dump({"pass_rate": round(passes / 3.0, 2)}, f, indent=2)

print(f"TI artifacts written to: {RUN_FOLDER}")
