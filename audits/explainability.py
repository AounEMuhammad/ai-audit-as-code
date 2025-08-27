def compute_XI_components_from_uploads(lf,gf,fa,rs,cl,hc):
    def get(d,k,default): 
        try: return float(d.get(k,default)) if d is not None else float(default)
        except: return float(default)
    return dict(
        LF = get(lf,"r2",0.70),
        GF = get(gf,"spearman",0.70),
        FA = get(fa,"deletion_auc",0.72),
        RS = get(rs,"jaccard_topk",0.68),
        CL = get(cl,"coverage",0.80),
        HC = get(hc,"score",0.70),
    )
