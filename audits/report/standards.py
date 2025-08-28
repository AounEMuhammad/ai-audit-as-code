# audits/report/standards.py
def conformity_matrix(blockers: list[dict]) -> dict:
    MAP = {
        "fairness_minority_gap": ["EUAA:Annex IV-2(b)", "NIST:MAP-3.6", "ISO42001:8.3"],
        "pii_leakage_scan":     ["EUAA:Art.10",        "NIST:MAP-3.3", "ISO42001:8.4"],
    }
    rows = []
    for b in blockers:
        name = b.get("name")
        rows.append({"control": name, "standards": MAP.get(name, [])})
    return {"matrix": rows}
