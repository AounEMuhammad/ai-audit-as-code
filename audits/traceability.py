import os, json
def score_AT_obj(obj):
    if not isinstance(obj, dict): return 0.60
    return 0.90 if obj.get("immutable") else 0.70
def score_RR_obj(obj):
    try:
        pr=float(obj.get("pass_rate",0.0))
    except: pr=0.0
    return 0.95 if pr>=1.0 else 0.80 if pr>=0.8 else 0.60 if pr>=0.5 else 0.40 if pr>=0.3 else 0.20
def compute_TI_components_from_uploads(dataset_hash, model_card, train_log, audit_trail, replication):
    return dict(
        DV=0.75 if dataset_hash else 0.30,
        MV=0.75 if model_card else 0.35,
        PL=0.75 if train_log else 0.30,
        AT=score_AT_obj(audit_trail) if audit_trail is not None else 0.30,
        RR=score_RR_obj(replication) if replication is not None else 0.20,
    )
