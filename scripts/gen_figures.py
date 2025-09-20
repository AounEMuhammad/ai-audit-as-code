
# scripts/gen_figures.py
# Generate quadrant, ARS histogram, blockers bar, and per-scenario radar PNGs from a summary CSV.
# Usage:
#   python scripts/gen_figures.py --csv reports/audit_results/summary_strict.csv --out docs/figures

import os, argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure required columns exist; fill safe defaults
    defaults_float = {"TI":0.0, "XI":0.0, "ARS":0.0, "R":0.0}
    defaults_str   = {"decision":"", "recommendation":"", "policy_blockers":"", "validator_blockers":"", "scenario":""}
    for k,v in defaults_float.items():
        if k not in df.columns: df[k] = v
        df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0.0)
    for k,v in defaults_str.items():
        if k not in df.columns: df[k] = v
        df[k] = df[k].fillna("")
    # Derived: t_bottleneck = min(TI, XI)
    df["t_bottleneck"] = np.minimum(df["TI"], df["XI"])
    # Risk may be absent; if so derive a proxy from ARS and t_bottleneck (not ideal, but keeps figs working)
    if "risk" not in df.columns:
        # ARS ≈ sqrt(risk * min(TI,XI))  → risk ≈ (ARS^2)/min(TI,XI)  (clip to [0,1], guard div by zero)
        tb = df["t_bottleneck"].replace(0, np.nan)
        risk_est = (df["ARS"]**2) / tb
        risk_est = risk_est.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0,1)
        df["risk"] = risk_est
    else:
        df["risk"] = pd.to_numeric(df["risk"], errors="coerce").fillna(0.0).clip(0,1)
    # Blockers combined for convenience
    if "policy_blockers" in df.columns and "validator_blockers" in df.columns:
        df["all_blockers"] = df["policy_blockers"].fillna("").astype(str) + ";" + df["validator_blockers"].fillna("").astype(str)
    else:
        df["all_blockers"] = df.get("policy_blockers","").astype(str)
    # Scenario label fallback
    if not df["scenario"].any():
        df["scenario"] = [f"scenario_{i+1}" for i in range(len(df))]
    return df

def fig_quadrant(df: pd.DataFrame, outdir: str):
    ensure_dir(outdir)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(df["t_bottleneck"], df["risk"], c=df["ARS"], alpha=0.85)
    cbar = plt.colorbar(sc); cbar.set_label("ARS", rotation=270, labelpad=12)
    plt.xlabel("min(TI, XI)")
    plt.ylabel("Risk (CORTEX)")
    # Vertical lines for TI/XI thresholds lattice (informative grid)
    for x in [0.50, 0.60, 0.70, 0.80]: plt.axvline(x, ls="--", lw=0.8, color="gray")
    # Horizontal lines for CORTEX tiers
    for y in [0.30, 0.50, 0.70, 0.85]: plt.axhline(y, ls="--", lw=0.8, color="gray")
    plt.title("Quadrant: Risk vs min(TI, XI) (colored by ARS)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "quadrant.png"), dpi=200)
    plt.close()

def fig_hist_ars(df: pd.DataFrame, outdir: str):
    ensure_dir(outdir)
    plt.figure(figsize=(8,4))
    plt.hist(df["ARS"], bins=20, alpha=0.9)
    plt.xlabel("ARS")
    plt.ylabel("Count")
    plt.title("ARS Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ars_hist.png"), dpi=200)
    plt.close()

def fig_blockers(df: pd.DataFrame, outdir: str, topn: int = 15):
    ensure_dir(outdir)
    counts = {}
    for s in df["all_blockers"].fillna("").tolist():
        items = [x for x in s.split(";") if x]
        for it in items:
            counts[it] = counts.get(it, 0) + 1
    if not counts:
        return
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:topn]
    labels = [k for k,_ in items]; vals = [v for _,v in items]
    plt.figure(figsize=(10,4))
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Top Blockers")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "blockers.png"), dpi=200)
    plt.close()

def fig_radar(df: pd.DataFrame, outdir: str, topn: int = 6):
    ensure_dir(outdir)
    # pick top-N by risk as illustrative
    sub = df.sort_values("risk", ascending=False).head(topn)
    for _, row in sub.iterrows():
        vals = [row["TI"], row["XI"], row["ARS"]]
        labels = ["TI","XI","ARS"]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        vals += vals[:1]; angles += angles[:1]
        plt.figure(figsize=(5,5))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, vals, linewidth=2)
        ax.fill(angles, vals, alpha=0.25)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        ttl = str(row.get("scenario","radar")).replace("/","_")[:40]
        ax.set_title(f"Radar: {ttl}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"radar_{ttl}.png"), dpi=200)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to summary CSV (batch_eval output)")
    ap.add_argument("--out", required=True, help="Output folder for PNGs")
    args = ap.parse_args()
    ensure_dir(args.out)
    df = load_csv(args.csv)
    fig_quadrant(df, args.out)
    fig_hist_ars(df, args.out)
    fig_blockers(df, args.out)
    fig_radar(df, args.out, topn=6)
    print(f"Wrote figures to {args.out}")

if __name__ == "__main__":
    main()
