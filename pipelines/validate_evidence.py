# pipelines/validate_evidence.py
import os, requests

EVIDENCE_BASE = os.environ.get(
    "EVIDENCE_BASE",
    "https://raw.githubusercontent.com/AounEMuhammad/ai-audit-as-code/main/evidence/demo_run"
)

def must_json(relpath, keys=None):
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
must_json("audit_trail.json", keys=["immutable", "records"])
must_json("replication.json", keys=["pass_rate"])
must_exist("dataset.hash")
must_exist("logs/train.log")
must_json("model_card.json")  # tighten later if you want

# ---- Explainability ----
must_json("explainability/local_fidelity.json", keys=["r2"])
must_json("explainability/global_stability.json", keys=["spearman"])
must_json("explainability/faithfulness.json", keys=["deletion_auc"])
must_json("explainability/robustness.json", keys=["jaccard_topk"])
must_json("explainability/coverage.json", keys=["coverage"])
must_json("explainability/human_comprehensibility.json", keys=["score"])

print("Evidence schema OK.")
