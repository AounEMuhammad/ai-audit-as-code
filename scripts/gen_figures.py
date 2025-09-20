
import os, argparse, pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def load_csv(p):
    df = pd.read_csv(p)
    for c in ['TI','XI','ARS','R']: 
        if c not in df: df[c]=0.0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    for c in ['decision','recommendation','policy_blockers','validator_blockers','scenario','domain']:
        if c not in df: df[c]=''
        df[c]=df[c].fillna('')
    df['t_bottleneck'] = np.minimum(df['TI'], df['XI'])
    if 'risk' not in df:
        tb = df['t_bottleneck'].replace(0,np.nan)
        rk = (df['ARS']**2)/tb
        df['risk'] = rk.replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(0,1)
    else:
        df['risk'] = pd.to_numeric(df['risk'], errors='coerce').fillna(0.0).clip(0,1)
    if 'policy_blockers' in df and 'validator_blockers' in df:
        df['all_blockers'] = df['policy_blockers'].astype(str).fillna('') + ';' + df['validator_blockers'].astype(str).fillna('')
    else:
        df['all_blockers'] = df.get('policy_blockers','').astype(str)
    if not df['scenario'].any(): df['scenario']=[f'scenario_{i+1}' for i in range(len(df))]
    return df

def quadrant(df,out):
    plt.figure(figsize=(8,6))
    sc=plt.scatter(df['t_bottleneck'], df['risk'], c=df['ARS'], alpha=0.85)
    cb=plt.colorbar(sc); cb.set_label('ARS', rotation=270, labelpad=12)
    plt.xlabel('min(TI, XI)'); plt.ylabel('Risk (CORTEX)')
    for x in [0.50,0.60,0.70,0.80]: plt.axvline(x, ls='--', lw=0.8, color='gray')
    for y in [0.30,0.50,0.70,0.85]: plt.axhline(y, ls='--', lw=0.8, color='gray')
    plt.title('Quadrant: Risk vs min(TI,XI) (colored by ARS)')
    plt.tight_layout(); plt.savefig(os.path.join(out,'quadrant.png'), dpi=200); plt.close()

def hist_ars(df,out):
    plt.figure(figsize=(8,4))
    plt.hist(df['ARS'], bins=20, alpha=0.9)
    plt.xlabel('ARS'); plt.ylabel('Count'); plt.title('ARS Distribution')
    plt.tight_layout(); plt.savefig(os.path.join(out,'ars_hist.png'), dpi=200); plt.close()

def blockers(df,out):
    counts={}
    for s in df['all_blockers'].fillna('').tolist():
        for it in [x for x in s.split(';') if x]: counts[it]=counts.get(it,0)+1
    if not counts: return
    items=sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
    labels=[k for k,_ in items]; vals=[v for _,v in items]
    plt.figure(figsize=(10,4))
    plt.bar(labels, vals); plt.xticks(rotation=45, ha='right'); plt.ylabel('Count'); plt.title('Top Blockers')
    plt.tight_layout(); plt.savefig(os.path.join(out,'blockers.png'), dpi=200); plt.close()

def radar(df,out,topn=6):
    sub=df.sort_values('risk', ascending=False).head(topn)
    for _,row in sub.iterrows():
        vals=[row['TI'],row['XI'],row['ARS']]; labels=['TI','XI','ARS']
        ang=np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()
        vals+=vals[:1]; ang+=ang[:1]
        plt.figure(figsize=(5,5)); ax=plt.subplot(111, polar=True)
        ax.plot(ang, vals, linewidth=2); ax.fill(ang, vals, alpha=0.25)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels); ax.set_yticklabels([])
        ttl=str(row.get('scenario','radar')).replace('/','_')[:40]; ax.set_title(f'Radar: {ttl}')
        plt.tight_layout(); plt.savefig(os.path.join(out, f'radar_{ttl}.png'), dpi=200); plt.close()

def main():
    import argparse, os
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv', required=True); ap.add_argument('--out', required=True)
    a=ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df=load_csv(a.csv)
    quadrant(df,a.out); hist_ars(df,a.out); blockers(df,a.out); radar(df,a.out)
    print('Wrote figures to', a.out)
if __name__=='__main__': main()
