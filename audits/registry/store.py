# audits/registry/store.py
import json, os, time
from typing import Dict, Any

def save_run(run: Dict[str, Any], outdir: str = ".audit_runs") -> str:
    os.makedirs(outdir, exist_ok=True)
    ts = int(time.time())
    path = os.path.join(outdir, f"{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    return path
