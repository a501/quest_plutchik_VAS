#!/usr/bin/env python3
"""
2_Rest_quest_half_lmm.py
========================
ANALYSIS TYPE: SLIDING WINDOW (win1s) -- 2-min halves of the 5-min rest run.
Every output tagged  SLIDING_win1s_halves.
Only half == 'first' / 'second' used here; 'full' is for step 3.

INPUT
    analysis/rest_half_dynamics_win1s.csv      (from 1_Rest_quest_half_dynamics.py)
    <QUEST_SCORES>

ACTUAL COLUMN NAMES (from SS_rest_dynamics.py output):
    ROI column  : unit_name
    measure col : measure   (values: LZC, ACW0, ACW50, SpectralEntropy, MedianFreq)
    value col   : value
    sliding time: win_sec

OUTPUT
    analysis/SLIDING_win1s_halves_ttest.csv
    analysis/SLIDING_win1s_halves_lmm.csv
    analysis/figures/SLIDING_win1s_halves_raincloud_p<NN>.png  (300 dpi, 2x3 per page)

    export PATH="/home/ZhangJianfeng_04/.conda/envs/MEG_env/bin:$PATH"
    python3 2_Rest_quest_half_lmm.py
"""

import os
import os.path as op
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE     = '/home/ZhangJianfeng_04/emo_study/Rest_task/SourceSpace'
ANA_DIR  = op.join(BASE, 'analysis')
FIG_DIR  = op.join(ANA_DIR, 'figures')

HALF_CSV     = op.join(ANA_DIR, 'rest_half_dynamics_win1s.csv')
QUEST_SCORES = '/home/ZhangJianfeng_04/emo_study/Behav/quest/analysis/questionnaire_scores_wide.csv'

# Column names as written by 2_score_questionnaires.py -> readable names used
# in every output here. Item positions verified against the published keys:
#   PERS NegativeDuration  = items 4,10,16,22,28   NegativeIntensity = 6,12,18,24,30
#   DERS Impulse           = items 3,14,19,24R,27,32
#   EAQ BA (Bodily Awareness) = items 14-17, all reverse-scored in the EAQ-30
#   TSQ item01 = 'emotions feel like strangers', item13 = 'fast and speedy',
#   item15 = 'very irregular ... cannot foresee'
SCALE_MAP = {
    'PERS_NegativeDuration':  'PERS_Negative_Duration',
    'DERS_Impulse':           'DERS_Impulse',
    'PERS_NegativeIntensity': 'PERS_Negative_Intensity',
    'EAQ_BA':                 'EAQ_BodilyAwareness',
    'TSQ_item13':             'TSQ_fast_speedy',
    'TSQ_item15':             'TSQ_irregular',
    'TSQ_item01':             'TSQ_stranger',
}

TAG      = 'SLIDING_win1s_halves'
MEASURES = ['ACW0', 'ACW50', 'LZC', 'SpecEn', 'MedianFreq']

SCALES = [
    'PERS_Negative_Duration',    # primary
    'DERS_Impulse',              # secondary
    'PERS_Negative_Intensity',   # secondary
    'EAQ_BodilyAwareness',       # exploratory -- inverted below
    'TSQ_fast_speedy',
    'TSQ_irregular',
    'TSQ_stranger',
]
INVERT_SCALES = ['EAQ_BodilyAwareness']

# Random-effects structures tried in order; first that converges is used and
# recorded in 're_structure'. Only the first is properly calibrated.
RE_FORMULAS = ['~half_c + net_c', '~net_c', '~1']

# DMN ROIs -- names exactly as they appear in unit_name column
DMN_ROIS = [
    'EMO_L_dmPFC',            'EMO_R_dmPFC',
    'EMO_L_Precuneus',        'EMO_R_Precuneus',
    'EMO_L_Angular',          'EMO_R_Angular',
    'Left_PCC_TB',            'Left_mPFC_TB',       'Right_mPFC_TB',
    'Self_L_PCC_KK',          'Self_L_ACC_KK',
    'Self_L_SFG_anterior_KK', 'Self_L_SFG_dorsal_KK',
    'Self_L_MTG_KK',          'Self_R_MTG_KK',
    'Self_L_Cingulate_mid_KK',
    'Extero_L_mPFC_KK',
    # tier-3 additions (present once SS scripts are updated)
    'EMO_L_vmPFC',            'EMO_R_vmPFC',
    'EMO_L_PCC',              'EMO_R_PCC',
]
CTRL_ROIS = [
    'Primary_visual',
    'Primary_auditive',
    'Primary_motor',
    'Primary_sensorymotor',
]

ALPHA   = 0.05
FIGSIZE = (13.33, 7.5)
DPI     = 300
N_ROWS, N_COLS = 2, 3
PER_PAGE = N_ROWS * N_COLS

os.makedirs(FIG_DIR, exist_ok=True)


def apply_inversions(df):
    names = []
    for s in SCALES:
        if s in INVERT_SCALES and s in df.columns:
            lo, hi = df[s].min(), df[s].max()
            df[s + '_inv'] = (lo + hi) - df[s]
            names.append(s + '_inv')
        elif s in df.columns:
            names.append(s)
    return df, names


# ============================================================
# RAINCLOUD
# ============================================================
def raincloud_panel(ax, v1, v2, title, ylabel):
    colors = ['#4C72B0', '#DD8452']
    rng = np.random.default_rng(0)
    for i, (d, c) in enumerate(zip([np.asarray(v1, float),
                                     np.asarray(v2, float)], colors)):
        x = i + 1
        dd = d[np.isfinite(d)]
        if dd.size < 3:
            continue
        try:
            kde = stats.gaussian_kde(dd)
            ys  = np.linspace(dd.min(), dd.max(), 200)
            dn  = kde(ys); dn = dn / dn.max() * 0.32
            ax.fill_betweenx(ys, x+0.08, x+0.08+dn, color=c, alpha=0.45, lw=0)
        except Exception:
            pass
        bp = ax.boxplot(dd, positions=[x-0.02], widths=0.10,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='k', lw=1.2))
        bp['boxes'][0].set_facecolor(c); bp['boxes'][0].set_alpha(0.75)
        ax.scatter(rng.normal(x-0.22, 0.035, size=dd.size),
                   dd, s=11, color=c, alpha=0.65, lw=0, zorder=3)

    v1a, v2a = np.asarray(v1, float), np.asarray(v2, float)
    ok = np.isfinite(v1a) & np.isfinite(v2a)
    rng2 = np.random.default_rng(0)
    j1 = rng2.normal(1-0.22, 0.035, size=v1a.size)
    j2 = rng2.normal(2-0.22, 0.035, size=v2a.size)
    for a, b, xa, xb in zip(v1a[ok], v2a[ok], j1[ok], j2[ok]):
        ax.plot([xa, xb], [a, b], color='grey', alpha=0.25, lw=0.6, zorder=2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['1st half', '2nd half'], fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# ============================================================
# MAIN
# ============================================================
def main():
    if not op.isfile(HALF_CSV):
        sys.exit(f'Missing {HALF_CSV} -- run 1_Rest_quest_half_dynamics.py first.')

    df = pd.read_csv(HALF_CSV)
    df['subject'] = df['subject'].astype(str).str.zfill(3)

    # column might be 'ROI' (written by step 1) or 'unit_name' if loaded raw
    if 'ROI' not in df.columns and 'unit_name' in df.columns:
        df = df.rename(columns={'unit_name': 'ROI'})

    df = df[df['half'].isin(['first', 'second'])].copy()

    net = {r: 'DMN' for r in DMN_ROIS}
    net.update({r: 'Control' for r in CTRL_ROIS})
    df = df[df['ROI'].isin(net)].copy()
    df['network'] = df['ROI'].map(net)

    if df.empty:
        present = pd.read_csv(HALF_CSV)['ROI'].unique() if 'ROI' in pd.read_csv(HALF_CSV).columns \
                  else pd.read_csv(HALF_CSV)['unit_name'].unique()
        print('No DMN/control ROIs matched. ROIs in file:')
        for r in sorted(present): print(f'  {r}')
        sys.exit()

    print(f'{TAG}: {df.subject.nunique()} subjects, '
          f'{df.ROI.nunique()} ROIs, {df.measure.nunique()} measures')

    # ---- Part A: paired t-test ----
    wide = df.pivot_table(index=['subject','ROI','measure','network'],
                          columns='half', values='value').reset_index()
    wide = wide.rename(columns={'first':'h1','second':'h2'})

    rows = []
    for (roi, meas), g in wide.groupby(['ROI','measure']):
        g = g.dropna(subset=['h1','h2'])
        n = len(g)
        if n < 4: continue
        d   = g['h2'].values - g['h1'].values
        t,p = stats.ttest_rel(g['h2'], g['h1'])
        try:   w,pw = stats.wilcoxon(g['h2'], g['h1'])
        except Exception: w,pw = np.nan, np.nan
        dz  = d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else np.nan
        rows.append(dict(analysis=TAG, ROI=roi, measure=meas,
                         network=g['network'].iloc[0], n=n,
                         mean_h1=g['h1'].mean(), mean_h2=g['h2'].mean(),
                         mean_diff=d.mean(), t=t, p_uncorrected=p,
                         wilcoxon_W=w, wilcoxon_p=pw, cohens_dz=dz))

    tt = pd.DataFrame(rows).sort_values('p_uncorrected').reset_index(drop=True)
    tt.insert(1, 'rank', np.arange(1, len(tt)+1))
    f_tt = op.join(ANA_DIR, f'{TAG}_ttest.csv')
    tt.to_csv(f_tt, index=False)
    print(f'  t-tests -> {f_tt}  '
          f'({(tt.p_uncorrected < ALPHA).sum()}/{len(tt)} at p<{ALPHA}, uncorrected)')

    # ---- rainclouds: all measures pooled, 6 per page ----
    sig = tt[tt.p_uncorrected < ALPHA].sort_values('p_uncorrected').reset_index(drop=True)
    n_sig = len(sig)
    if n_sig == 0:
        print('  no cells at p<.05 -- no raincloud pages written')
    else:
        n_pages = int(np.ceil(n_sig / PER_PAGE))
        for page in range(n_pages):
            block = sig.iloc[page*PER_PAGE:(page+1)*PER_PAGE]
            fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=FIGSIZE, squeeze=False)
            flat = axes.ravel()
            for ax, (_, r) in zip(flat, block.iterrows()):
                g = wide[(wide.ROI==r.ROI)&(wide.measure==r.measure)].dropna(
                    subset=['h1','h2'])
                raincloud_panel(ax, g['h1'].values, g['h2'].values,
                    f"#{int(r['rank'])}  {r.measure} | {r.ROI} [{r.network}]\n"
                    f"dz={r.cohens_dz:.2f}  p={r.p_uncorrected:.4f}  n={int(r.n)}",
                    ylabel=r.measure)
            for ax in flat[len(block):]: ax.axis('off')
            r0 = page*PER_PAGE+1; r1 = page*PER_PAGE+len(block)
            fig.suptitle(
                f'1st vs 2nd half  |  SLIDING WINDOW win1s, 2-min halves  |  '
                f'all measures pooled, ranked by p  |  rank {r0}-{r1} of {n_sig}',
                fontsize=11)
            fig.tight_layout(rect=[0,0,1,0.93])
            f_png = op.join(FIG_DIR, f'{TAG}_raincloud_p{page+1:02d}.png')
            fig.savefig(f_png, dpi=DPI); plt.close(fig)
            print(f'  raincloud page {page+1}/{n_pages} -> {f_png}')

    # ---- Part B: LMM ----
    if not op.isfile(QUEST_SCORES):
        print(f'\n[skip LMM] {QUEST_SCORES} not found'); return

    q = pd.read_csv(QUEST_SCORES)
    qsub = next(c for c in q.columns if c.lower() in ('subject','subj','sub','id'))
    q = q.rename(columns={qsub:'subject'})
    q['subject'] = q['subject'].astype(str).str.zfill(3)
    missing = [c for c in SCALE_MAP if c not in q.columns]
    if missing:
        print(f'  WARNING: not in scores file: {missing}')
    q = q.rename(columns=SCALE_MAP)
    q, scale_names = apply_inversions(q)

    mdf = df.merge(q, on='subject', how='inner')
    mdf['half_c'] = (mdf['half']=='second').astype(float)
    mdf['net_c']  = (mdf['network']=='DMN').astype(float)

    res_rows = []
    for meas in MEASURES:
        sub = mdf[mdf.measure==meas].copy()
        if sub.empty: continue
        for scale in scale_names:
            d = sub.dropna(subset=['value', scale]).copy()
            if d.subject.nunique() < 8: continue
            d['q_z'] = (d[scale]-d[scale].mean())/d[scale].std(ddof=1)
            d['y_z'] = (d['value']-d['value'].mean())/d['value'].std(ddof=1)
            # Random slopes for half and network per subject: both are
            # within-subject factors crossed with a between-subject trait, so
            # without them the moderation terms are tested against ROI-level
            # noise (pseudo-replication, anti-conservative p-values).
            m, re_used = None, None
            for re_f in RE_FORMULAS:
                try:
                    m = smf.mixedlm('y_z ~ half_c * q_z * net_c', data=d,
                                     groups=d['subject'], re_formula=re_f,
                                     vc_formula={'ROI': '0+C(ROI)'}
                                     ).fit(reml=True, method='lbfgs')
                    re_used = re_f
                    break
                except Exception as e:
                    last_err = e
            if m is None:
                print(f'  [LMM fail] {meas} x {scale}: {last_err}'); continue
            if re_used != RE_FORMULAS[0]:
                print(f'  [fallback] {meas} x {scale}: random effects {re_used!r} '
                      f'-- moderation p-values likely ANTI-CONSERVATIVE')
            for term in m.params.index:
                if term.startswith('Group') or term.endswith('Var'): continue
                res_rows.append(dict(analysis=TAG, measure=meas, scale=scale,
                    term=term, inverted=scale.endswith('_inv'),
                    beta=m.params[term], se=m.bse[term],
                    z=m.tvalues[term], p=m.pvalues[term],
                    re_structure=re_used,
                    n_obs=int(m.nobs), n_subjects=int(d.subject.nunique())))
            key = 'half_c:q_z:net_c'
            if key in m.pvalues.index:
                print(f'  {meas:<10s} x {scale:<26s} '
                      f'half:q:net  b={m.params[key]: .3f}  p={m.pvalues[key]:.3f}')

    if res_rows:
        lmm = pd.DataFrame(res_rows)
        f_lmm = op.join(ANA_DIR, f'{TAG}_lmm.csv')
        lmm.to_csv(f_lmm, index=False)
        print(f'\n  LMM -> {f_lmm}')


if __name__ == '__main__':
    main()
