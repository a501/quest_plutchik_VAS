#!/usr/bin/env python3
"""
3_Rest_quest_full_lmm.py
========================
FULL 5-MIN REST, RUN TWICE WITH TWO ESTIMATORS:

    STATIC_full5min          measures from <NNN>_ecg_static.csv
                             (computed on the continuous 300-s series)
    SLIDING_win1s_full5min   mean across ALL 1-s windows of the recording
                             (half == 'full' in rest_half_dynamics_win1s.csv)

ACTUAL COLUMN NAMES (from SS_rest_dynamics.py output):
    static CSV : unit_name, measure, value
    sliding CSV: unit_name, win_sec, measure, stat, value
    measure values: LZC, ACW0, ACW50, SpectralEntropy (-> SpecEn), MedianFreq

Static CSV path: dynamics/<NNN>_ecg_static.csv   (no _rest_ tag)
Sliding summary: analysis/rest_half_dynamics_win1s.csv  half=='full'

Model, per estimator x measure x scale:
    value ~ questionnaire * network + (1|subject) + ROI variance component
    term of interest: q_z:net_c

INPUT
    dynamics/<NNN>_ecg_static.csv
    analysis/rest_half_dynamics_win1s.csv
    <QUEST_SCORES>

OUTPUT
    analysis/STATIC_full5min_quest_lmm.csv
    analysis/SLIDING_win1s_full5min_quest_lmm.csv
    analysis/figures/STATIC_full5min_scatter_<MEASURE>.png      (300 dpi)
    analysis/figures/SLIDING_win1s_full5min_scatter_<MEASURE>.png

    export PATH="/home/ZhangJianfeng_04/.conda/envs/MEG_env/bin:$PATH"
    python3 3_Rest_quest_full_lmm.py
"""

import os
import os.path as op
import glob
import re
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
DYN_DIR  = op.join(BASE, 'dynamics')
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

VERSION  = 'ecg'

# measure names as they appear in the static/sliding CSV 'measure' column
# SpectralEntropy is renamed to SpecEn internally
MEAS_RENAME = {'SpectralEntropy': 'SpecEn'}
MEASURES_PRIMARY = ['SpecEn', 'LZC', 'ACW0']
MEASURES_ALL     = ['ACW0', 'ACW50', 'LZC', 'SpecEn', 'MedianFreq']

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

FIGSIZE = (13.33, 7.5)
DPI     = 300
ALPHA   = 0.05
N_ROWS, N_COLS = 2, 3            # 6 panels per page, as in step 2
PER_PAGE = N_ROWS * N_COLS
# Which LMM terms qualify a cell for a scatter panel. The interaction is the
# hypothesis test (is the trait relationship specific to DMN?). Add 'q_z' to
# also plot significant overall trait effects.
PLOT_TERMS = ['q_z:net_c']

# Random-effects structures tried in order; the first that converges is used
# and recorded in the 're_structure' column. Only the first is calibrated.
RE_FORMULAS = ['~net_c', '~1']

os.makedirs(FIG_DIR, exist_ok=True)


def apply_inversions(df):
    names = []
    for s in SCALES:
        if s in INVERT_SCALES and s in df.columns:
            lo, hi = df[s].min(), df[s].max()
            df[s+'_inv'] = (lo+hi) - df[s]
            names.append(s+'_inv')
        elif s in df.columns:
            names.append(s)
    return df, names


# ============================================================
# LOADERS
# ============================================================
def load_static():
    """
    Load <NNN>_ecg_static.csv files.
    Relevant columns: unit_name, measure, value
    One row per subject x ROI x measure (+ method/level/flag cols we ignore).
    We keep only the 'mean_flip' method rows (the primary output) to avoid
    duplicates from multiple methods if present.
    """
    files = sorted(glob.glob(op.join(DYN_DIR, f'*_{VERSION}_static.csv')))
    if not files:
        return None

    out = []
    for f in files:
        m = re.match(r'(\d{3})_', op.basename(f))
        if not m: continue
        sub = m.group(1)
        d = pd.read_csv(f)

        # keep mean_flip only (primary orientation method)
        if 'method' in d.columns:
            d = d[d['method'] == 'mean_flip']

        d = d.rename(columns={'unit_name': 'ROI'})
        d['measure'] = d['measure'].replace(MEAS_RENAME)
        d = d[d['measure'].isin(MEASURES_ALL)]

        out.append(pd.DataFrame({
            'subject': sub,
            'ROI':     d['ROI'].astype(str),
            'measure': d['measure'].astype(str),
            'value':   pd.to_numeric(d['value'], errors='coerce'),
        }))

    return pd.concat(out, ignore_index=True) if out else None


def load_sliding_full():
    """
    Load half=='full' rows from the assembled half table.
    Column 'ROI' is already named correctly by step 1.
    """
    if not op.isfile(HALF_CSV):
        return None
    d = pd.read_csv(HALF_CSV)
    if 'ROI' not in d.columns and 'unit_name' in d.columns:
        d = d.rename(columns={'unit_name': 'ROI'})
    d = d[d['half'] == 'full'].copy()
    return d[['subject', 'ROI', 'measure', 'value']] if not d.empty else None


# ============================================================
# ONE ESTIMATOR
# ============================================================
def run_estimator(df, q, scale_names, tag):
    df = df.copy()
    df['subject'] = df['subject'].astype(str).str.zfill(3)

    net = {r: 'DMN' for r in DMN_ROIS}
    net.update({r: 'Control' for r in CTRL_ROIS})
    df = df[df['ROI'].isin(net)].copy()
    df['network'] = df['ROI'].map(net)

    if df.empty:
        print(f'[{tag}] no DMN/control ROIs matched')
        present = df['ROI'].unique() if 'ROI' in df.columns else []
        for r in sorted(present): print(f'  {r}')
        return

    mdf = df.merge(q, on='subject', how='inner')
    mdf['net_c'] = (mdf['network'] == 'DMN').astype(float)

    print(f'\n=== {tag} ===')
    print(f'  {mdf.subject.nunique()} subjects, {mdf.ROI.nunique()} ROIs')

    rows, fits, slope_rows = [], {}, []
    for meas in MEASURES_ALL:
        sub = mdf[mdf.measure == meas]
        if sub.empty: continue
        for scale in scale_names:
            d = sub.dropna(subset=['value', scale]).copy()
            if d.subject.nunique() < 8: continue
            d['q_z'] = (d[scale]-d[scale].mean())/d[scale].std(ddof=1)
            d['y_z'] = (d['value']-d['value'].mean())/d['value'].std(ddof=1)
            # Random slope for network per subject is REQUIRED: the trait is
            # constant within subject but repeated across every ROI, so a
            # random-intercept-only model treats ROIs as independent
            # replicates of the q x network interaction. In null simulations
            # (35 subjects, 10 ROIs) that inflated the false-positive rate to
            # ~13% at alpha=.05; with '~net_c' it is ~5%.
            m, re_used = None, None
            for re_f in RE_FORMULAS:
                try:
                    m = smf.mixedlm('y_z ~ q_z * net_c', data=d,
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
                      f'-- interaction p is likely ANTI-CONSERVATIVE')
            for term in m.params.index:
                if term.startswith('Group') or term.endswith('Var'): continue
                rows.append(dict(analysis=tag, measure=meas, scale=scale,
                    term=term, inverted=scale.endswith('_inv'),
                    primary=meas in MEASURES_PRIMARY,
                    beta=m.params[term], se=m.bse[term],
                    z=m.tvalues[term], p=m.pvalues[term],
                    re_structure=re_used,
                    n_obs=int(m.nobs), n_subjects=int(d.subject.nunique())))
            # ---- simple slopes from the fixed effects ----
            # Control slope = b_q ; DMN slope = b_q + b_qxnet
            fe = ['Intercept', 'q_z', 'net_c', 'q_z:net_c']
            if all(t in m.params.index for t in fe):
                b = m.params[fe].values.astype(float)
                V = m.cov_params().loc[fe, fe].values.astype(float)
                ctrl_b, ctrl_se = b[1], np.sqrt(V[1, 1])
                dmn_b = b[1] + b[3]
                dmn_se = np.sqrt(V[1, 1] + V[3, 3] + 2 * V[1, 3])
                for netname, sb, sse in (('DMN', dmn_b, dmn_se),
                                         ('Control', ctrl_b, ctrl_se)):
                    zz = sb / sse if sse > 0 else np.nan
                    slope_rows.append(dict(
                        analysis=tag, measure=meas, scale=scale,
                        network=netname, slope=sb, se=sse,
                        ci_lo=sb - 1.96 * sse, ci_hi=sb + 1.96 * sse,
                        z=zz, p=2 * stats.norm.sf(abs(zz)),
                        interaction_b=b[3], interaction_p=m.pvalues['q_z:net_c'],
                        n_subjects=int(d.subject.nunique())))
                fits[(meas, scale)] = dict(
                    b=b, V=V,
                    q_mean=float(d[scale].mean()), q_sd=float(d[scale].std(ddof=1)),
                    data=d[['subject', 'network', scale, 'q_z', 'y_z']].copy())
            if 'q_z:net_c' in m.pvalues.index:
                print(f'  {meas:<12s} x {scale:<26s} '
                      f'q:net  b={m.params["q_z:net_c"]: .3f}  '
                      f'p={m.pvalues["q_z:net_c"]:.3f}')

    if not rows:
        print(f'  [{tag}] no models fitted'); return

    res = pd.DataFrame(rows)
    f_out = op.join(ANA_DIR, f'{tag}_quest_lmm.csv')
    res.to_csv(f_out, index=False)
    print(f'  LMM -> {f_out}')

    slopes = pd.DataFrame(slope_rows)
    if not slopes.empty:
        f_sl = op.join(ANA_DIR, f'{tag}_simple_slopes.csv')
        slopes.to_csv(f_sl, index=False)
        print(f'  simple slopes -> {f_sl}')

    # ---- interaction plots: SIGNIFICANT q x network cells only ----
    # Lines and 95% bands are MODEL predictions from the LMM fixed effects
    # (not OLS through subject means), so the picture and the p-value come
    # from the same model. x = questionnaire in its own units (back-transformed
    # from q_z); y = measure z-scored, as modelled. Points = subject means per
    # network, shown for transparency only.
    sig = res[res['term'].isin(PLOT_TERMS) & (res['p'] < ALPHA)] \
        .sort_values('p').reset_index(drop=True)
    sig = sig[[(r.measure, r.scale) in fits for r in sig.itertuples()]]
    if sig.empty:
        print(f'  [{tag}] no {PLOT_TERMS} effects at p<{ALPHA} -- no plots written')
        return
    sig = sig.reset_index(drop=True)
    sig.insert(0, 'rank', np.arange(1, len(sig) + 1))
    sig.to_csv(op.join(ANA_DIR, f'{tag}_significant.csv'), index=False)

    colors = {'DMN': '#4C72B0', 'Control': '#C0504D'}
    netcode = {'DMN': 1.0, 'Control': 0.0}
    n_sig = len(sig)
    n_pages = int(np.ceil(n_sig / PER_PAGE))

    for page in range(n_pages):
        block = sig.iloc[page * PER_PAGE:(page + 1) * PER_PAGE]
        fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=FIGSIZE, squeeze=False)
        flat = axes.ravel()

        for ax, (_, r) in zip(flat, block.iterrows()):
            f = fits[(r.measure, r.scale)]
            b, V, dd = f['b'], f['V'], f['data']
            qz = np.linspace(dd['q_z'].min(), dd['q_z'].max(), 100)
            qx = f['q_mean'] + f['q_sd'] * qz            # raw questionnaire units
            sl = slopes[(slopes.measure == r.measure) & (slopes.scale == r.scale)]
            lab = {}
            for netname, nc in netcode.items():
                X = np.column_stack([np.ones_like(qz), qz,
                                     np.full_like(qz, nc), qz * nc])
                yhat = X @ b
                se = np.sqrt(np.einsum('ij,jk,ik->i', X, V, X))
                c = colors[netname]
                ax.fill_between(qx, yhat - 1.96 * se, yhat + 1.96 * se,
                                color=c, alpha=0.18, lw=0)
                ax.plot(qx, yhat, color=c, lw=2)
                pts = (dd[dd.network == netname].groupby('subject')
                       .agg(q=(r.scale, 'first'), y=('y_z', 'mean')))
                ax.scatter(pts['q'], pts['y'], s=12, color=c, alpha=0.45,
                           lw=0, zorder=1)
                row = sl[sl.network == netname]
                if len(row):
                    row = row.iloc[0]
                    lab[netname] = (f"{netname}: {row.slope:+.2f} "
                                    f"[{row.ci_lo:+.2f}, {row.ci_hi:+.2f}]")
                    ax.plot([], [], color=c, lw=2, label=lab[netname])
            ax.axhline(0, color='grey', lw=0.5, ls=':')
            ax.set_title(f"#{int(r['rank'])}  {r.measure} ~ {r.scale}\n"
                         f"interaction b={r.beta:+.2f}  p={r.p:.4f}  "
                         f"n={int(r.n_subjects)}", fontsize=8)
            ax.set_xlabel(r.scale, fontsize=7)
            ax.set_ylabel(f'{r.measure} (z)', fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5, frameon=False, loc='best',
                      title='simple slope [95% CI]', title_fontsize=6.5)
            for s_ in ('top', 'right'):
                ax.spines[s_].set_visible(False)

        for ax in flat[len(block):]:
            ax.axis('off')

        r0 = page * PER_PAGE + 1
        r1 = page * PER_PAGE + len(block)
        fig.suptitle(f'Questionnaire x network interaction (LMM predictions, 95% CI)  |  '
                     f'{tag}  |  full 5-min rest  |  rank {r0}-{r1} of {n_sig} '
                     f'(uncorrected p<{ALPHA})', fontsize=10.5)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        f_png = op.join(FIG_DIR, f'{tag}_interaction_p{page + 1:02d}.png')
        fig.savefig(f_png, dpi=DPI)
        plt.close(fig)
        print(f'  interaction page {page + 1}/{n_pages} -> {f_png}')

    # ---- forest summary: both simple slopes for every significant cell ----
    fs = slopes.merge(sig[['rank', 'measure', 'scale']], on=['measure', 'scale'])
    fs = fs.sort_values(['rank', 'network'])
    labels = [f"#{int(k)}  {m_} ~ {s_}" for k, m_, s_ in
              sig[['rank', 'measure', 'scale']].itertuples(index=False)]
    h = max(3.0, 0.42 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(13.33, min(h, 7.5)))
    for netname, off in (('DMN', 0.15), ('Control', -0.15)):
        g = fs[fs.network == netname].set_index('rank').reindex(sig['rank'])
        ypos = np.arange(len(sig))[::-1] + off
        ax.errorbar(g['slope'], ypos,
                    xerr=[g['slope'] - g['ci_lo'], g['ci_hi'] - g['slope']],
                    fmt='o', color=colors[netname], ms=5, capsize=2,
                    lw=1.2, label=netname)
    ax.axvline(0, color='grey', lw=0.8, ls=':')
    ax.set_yticks(np.arange(len(sig))[::-1])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('simple slope: change in measure (SD) per 1 SD of questionnaire, 95% CI',
                  fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc='best')
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.set_title(f'Simple slopes by network for significant q x network cells  |  '
                 f'{tag}  |  full 5-min rest', fontsize=10.5)
    fig.tight_layout()
    f_png = op.join(FIG_DIR, f'{tag}_simple_slopes_forest.png')
    fig.savefig(f_png, dpi=DPI)
    plt.close(fig)
    print(f'  forest -> {f_png}')


# ============================================================
# MAIN
# ============================================================
def main():
    if not op.isfile(QUEST_SCORES):
        sys.exit(f'Questionnaire scores not found: {QUEST_SCORES}')

    q = pd.read_csv(QUEST_SCORES)
    qsub = next(c for c in q.columns if c.lower() in ('subject','subj','sub','id'))
    q = q.rename(columns={qsub:'subject'})
    q['subject'] = q['subject'].astype(str).str.zfill(3)
    missing = [c for c in SCALE_MAP if c not in q.columns]
    if missing:
        print(f'  WARNING: not in scores file: {missing}')
    q = q.rename(columns=SCALE_MAP)
    q, scale_names = apply_inversions(q)
    print(f'Scales: {scale_names}')

    static = load_static()
    if static is None:
        print('[STATIC] no static CSVs found under', DYN_DIR)
    else:
        run_estimator(static, q, scale_names, 'STATIC_full5min')

    sliding = load_sliding_full()
    if sliding is None:
        print('[SLIDING] no full-recording rows -- run 1_Rest_quest_half_dynamics.py first')
    else:
        run_estimator(sliding, q, scale_names, 'SLIDING_win1s_full5min')


if __name__ == '__main__':
    main()
