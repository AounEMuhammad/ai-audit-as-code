# audits/lineage/emit.py
import platform, subprocess

def snapshot(container_hint: str = "docker://python:3.11-slim") -> dict:
    git = ""
    try:
        git = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
    except Exception:
        pass
    return {
        "container": container_hint,
        "python": platform.python_version(),
        "git": git,
        "seed": 1337
    }
