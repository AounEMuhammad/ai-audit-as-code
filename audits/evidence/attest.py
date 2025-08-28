# audits/evidence/attest.py
import subprocess, tempfile, os

def sign_evidence(predicate_json: str, key_ref: str) -> str:
    """Return attestation output or 'UNSIGNED' if cosign unavailable."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            f.write(predicate_json.encode("utf-8"))
            path = f.name
        out = subprocess.check_output([
            "cosign","attest","--predicate", path, "--key", key_ref, "--type","evidence"
        ], stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="ignore")
    except Exception:
        return "UNSIGNED"
    finally:
        try: os.unlink(path)
        except Exception: pass
