# pipelines/validate_evidence.py
import os, sys, requests, json

EVIDENCE_BASE = os.environ.get(
    "EVIDENCE_BASE",
    "https://raw.githubusercontent.com/AounEMuhammad/ai-audit-as-code/main/evidence/demo_run"
)

def must_have_json(relpath, keys=None):
    url = f"{EVIDENCE_BASE}/{relpath}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    try:
        data = r.json()
    except Exception as e:
        raise ValueError(f"{relpath}: invalid JSON ({e})")
    if keys:
        for k in keys:
            if k not in data:
                raise ValueError(f"{relpath}: missing key '{k}'")
    return True

def must_exist(relpath):
    url = f"{EVIDENCE_BASE}/{relpath}"
    r = requests.get(url, timeout=15)
    if not r.ok:
        raise ValueError(f"{relpath}: missing or not accessible")
    return True

# ---- Traceability ----
must_have_json("audit_trail.json", keys=["immutable", "records"])
must_have_json("replication.json", keys=["pass_rate"])
must_exist("dataset.hash")
must_exist("logs/train.log")
must_have_json("model_card.json")  # you can tighten schema later

# ---- Explainability ----
must_have_json("explainability/local_fidelity.json", keys=["r2"])
must_have_json("explainability/global_stability.json", keys=["spearman"])
must_have_json("explainability/faithfulness.json", keys=["deletion_auc"])
must_have_json("explainability/robustness.json", keys=["jaccard_topk"])
must_have_json("explainability/coverage.json", keys=["coverage"])
must_have_json("explainability/human_comprehensibility.json", keys=["score"])

print("Evidence schema OK.")
