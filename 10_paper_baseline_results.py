# -*- coding: utf-8 -*-
"""
10_paper_baseline_results.py
============================
Supplementary Tables and figures for the manuscript on intrinsic emotions:
do Sadness and Fear ratings vary during rest without any external stimulus?

The main text uses the baseline rest block (no preceding video); the video
blocks are the positive control (Extended Data). Nothing is re-estimated
here except the cross-block consistency (Supplementary Table 8B): every
other number is read from the outputs of the analysis scripts, so each
reported value can be traced to one table and one pipeline file.

Inputs (ANALYSIS_DIR / ...)
    vas_fluctuation_proof/     5_vas_fluctuation_proof.py (all probes)
        00_series_audit.csv, 01_series_descriptives.csv,
        02_nonnull_per_series.csv, 03_fluctuation_per_series.csv,
        04_dissociation_per_subject_block.csv, 05_group_tests.csv,
        06_prevalence.csv, 08_mixture_model_selection.csv,
        09_mixture_assignments.csv
    vas_noise_statespace/      6_vas_noise_statespace.py
        sigma_estimates.csv, sigma_profile.csv
    vas_first_probe/           7_vas_first_probe_test.py
        first_probe_tests.csv, first_probe_per_series.csv
    vas_distribution_check/    9_vas_distribution_check.py --simulate
        dist_fit.csv, dist_tests.csv, dist_sim.csv

Outputs (OUT_DIR)
    tables/ST01_sample.csv ... ST12_test_calibration.csv
    figures/Fig1 ... Fig5, ExtData_Fig1 ... ExtData_Fig5  (.png 300 dpi + .pdf)

Settings of the main text: rating noise SIGMA_MAIN = 3 (estimated from the
data, slide/Results 2), sensitivity SIGMA_SENS = 5 and 10; lapse rate 10%.

Usage
    python3 10_paper_baseline_results.py
    python3 10_paper_baseline_results.py --analysis-dir path/to/analysis
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ===================================================================
# CONFIG
# ===================================================================

ANALYSIS_DIR = Path("/home/ZhangJianfeng_04/emo_study/Behav/analysis")
OUT_NAME = "paper_baseline"
SIGMA_MAIN = 3.0
SIGMA_SENS = (5.0, 10.0)
SIGMA_GRID = (2.0, 3.0, 5.0, 7.5, 10.0, 12.5)
LAPSE_MAIN = 0.10
ALPHA = 0.05

BLOCKS = ["base", "sad", "fear"]
EMOS = ["Sadness", "Fear"]
CELLS = [(b, e) for b in BLOCKS for e in EMOS]
BLOCK_NAME = {"base": "Baseline", "sad": "After sad video", "fear": "After fear video"}
VARIANT_NAME = {"all": "all probes", "drop_first": "first probe removed",
                "detrended": "linear trend removed"}


# ===================================================================
# HELPERS
# ===================================================================

def load(d, name):
    return pd.read_csv(d / name)


def cell_rows(df, b, e=None, variant="all"):
    q = df[(df.block_role == b)]
    if e is not None:
        q = q[q.emotion == e]
    if "variant" in q.columns and variant is not None:
        q = q[q.variant == variant]
    return q


def binom_greater(k, n, p=ALPHA):
    """Exact one-sided binomial p: more significant series than alpha predicts."""
    return float(stats.binomtest(int(k), int(n), p, alternative="greater").pvalue) if n else np.nan


def clopper(k, n, conf=0.95):
    if n == 0:
        return np.nan, np.nan
    ci = stats.binomtest(int(k), int(n)).proportion_ci(conf, method="exact")
    return ci.low, ci.high


def prev_row(P, analysis, b, e, variant, par):
    q = P[(P.analysis == analysis) & (P.block_role == b) & (P.variant == variant)]
    if e is not None:
        q = q[q.emotion == e]
    if par is not None:
        q = q[np.isclose(q.parameter_value, par)]
    return q.iloc[0] if len(q) else None


def group_row(G, analysis, test, b, e, variant, par=None):
    q = G[(G.analysis == analysis) & (G.test == test) & (G.block_role == b) & (G.variant == variant)]
    if e is not None:
        q = q[q.emotion == e]
    if par is not None:
        q = q[np.isclose(q.parameter_value, par)]
    return q.iloc[0] if len(q) else None


def spearman_ci(x, y):
    """Spearman rho, p and Fisher-z 95% CI (Bonett-Wright SE)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = m.sum()
    r, p = stats.spearmanr(x, y)
    se = np.sqrt((1 + r ** 2 / 2) / (n - 3))
    z = np.arctanh(r)
    return n, r, p, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


# ===================================================================
# SUPPLEMENTARY TABLES
# ===================================================================

def st01_sample(audit, desc):
    rows = []
    d = desc[desc.variant == "all"]
    for b in BLOCKS:
        a = audit[audit.block_role == b]
        g = d[d.block_role == b]
        rows.append(dict(
            block=BLOCK_NAME[b],
            participants_with_block=a.subject.nunique(),
            series_included_sadness=int((g.emotion == "Sadness").sum()),
            series_included_fear=int((g.emotion == "Fear").sum()),
            series_excluded=int((~a.included).sum()),
            exclusion_reasons="; ".join(sorted(set(a.loc[~a.included, "reason"].astype(str)))),
            probes_per_series_median=float(g.n.median()),
            probes_per_series_min=int(g.n.min()), probes_per_series_max=int(g.n.max()),
            ratings_total=int(g.n.sum()),
            ratings_at_cursor_start_pct=100 * g.n_at_cursor_start.sum() / g.n.sum()))
    return pd.DataFrame(rows)


def st02_descriptives(desc):
    rows = []
    d = desc[desc.variant == "all"]
    for b, e in CELLS:
        g = d[(d.block_role == b) & (d.emotion == e)]
        q = lambda s, p: float(np.percentile(s, p))
        rows.append(dict(
            block=BLOCK_NAME[b], scale=e, n_series=len(g),
            mean_M=g["mean"].mean(), mean_SD=g["mean"].std(), mean_Mdn=g["mean"].median(),
            mean_Q1=q(g["mean"], 25), mean_Q3=q(g["mean"], 75),
            sd_M=g.sd.mean(), sd_SD=g.sd.std(), sd_Mdn=g.sd.median(),
            sd_Q1=q(g.sd, 25), sd_Q3=q(g.sd, 75),
            range_Mdn=g["range"].median(), distinct_values_Mdn=g.n_distinct.median(),
            probes_at_0_pct=100 * g.frac_zero.mean(),
            all_zero_series=int((g.sd == 0).sum()),
            all_zero_pct=100 * (g.sd == 0).mean(),
            rvi_M=g.rvi_sd.mean(), rvi_SD=g.rvi_sd.std(),
            at_cursor_start_pct=100 * g.n_at_cursor_start.sum() / g.n.sum()))
    return pd.DataFrame(rows)


def st03_presence(nonnull, G, P):
    rows = []
    for b, e in CELLS:
        for v in ("all", "drop_first"):
            anyr = group_row(G, "nonnull", "share_of_people_with_any_nonzero_rating", b, e, v)
            for lapse in (0.05, 0.10, 0.20):
                s = nonnull[(nonnull.block_role == b) & (nonnull.emotion == e) &
                            (nonnull.variant == v) & np.isclose(nonnull.lapse_rate, lapse)]
                if not len(s):
                    continue
                k, n = int(s.significant.sum()), len(s)
                w = group_row(G, "nonnull", "wilcoxon_frac_nonzero_above_lapse", b, e, v, lapse)
                pr = prev_row(P, "nonnull", b, e, v, lapse)
                rows.append(dict(
                    block=BLOCK_NAME[b], scale=e, variant=VARIANT_NAME[v], lapse_rate=lapse,
                    n_series=n, k_significant=k, pct_significant=100 * k / n,
                    p_binomial_vs_alpha=binom_greater(k, n),
                    people_any_nonzero=int(anyr.k) if anyr is not None else np.nan,
                    people_any_nonzero_pct=100 * anyr.estimate if anyr is not None else np.nan,
                    people_any_nonzero_ci_low=100 * anyr.ci_low if anyr is not None else np.nan,
                    people_any_nonzero_ci_high=100 * anyr.ci_high if anyr is not None else np.nan,
                    wilcoxon_median_share_minus_lapse=w.estimate if w is not None else np.nan,
                    wilcoxon_r=w.effect_size if w is not None else np.nan,
                    wilcoxon_p_fdr=w.p_fdr if w is not None else np.nan,
                    prevalence_map_pct=100 * pr.prev_map if pr is not None else np.nan,
                    prevalence_hpdi_low_pct=100 * pr.prev_hpdi_low if pr is not None else np.nan,
                    prevalence_hpdi_high_pct=100 * pr.prev_hpdi_high if pr is not None else np.nan,
                    prevalence_p_gt_half=pr.p_prev_gt_half if pr is not None else np.nan))
    return pd.DataFrame(rows)


def st04_noise(sig, dissoc):
    a = sig.rename(columns={"fit": "fit", "n_series": "n_series", "sigma_hat": "sigma",
                            "sigma_ci_low": "sigma_ci_low", "sigma_ci_high": "sigma_ci_high",
                            "sigma_noise_only": "sigma_without_rounding", "tau_hat_s": "tau_s"})
    a = a[["fit", "n_series", "sigma", "sigma_ci_low", "sigma_ci_high",
           "sigma_without_rounding", "tau_s", "sigma_at_grid_edge", "tau_at_grid_edge"]]
    rows = []
    x = dissoc[(dissoc.variant == "all") & dissoc.eligible]
    x = x.drop_duplicates(["subject", "block_role"])
    for b in BLOCKS + ["all"]:
        g = x if b == "all" else x[x.block_role == b]
        bound = (g.sd_diff / np.sqrt(2)).dropna()
        rows.append(dict(block="All blocks" if b == "all" else BLOCK_NAME[b], n_blocks=len(bound),
                         bound_Mdn=bound.median(), bound_Q1=bound.quantile(.25),
                         bound_Q3=bound.quantile(.75),
                         pct_below_3=100 * (bound < 3).mean(), pct_below_5=100 * (bound < 5).mean(),
                         pct_below_7_5=100 * (bound < 7.5).mean(),
                         pct_below_10=100 * (bound < 10).mean()))
    return a, pd.DataFrame(rows)


def st05_fluctuation(fl, G, P):
    rows = []
    for b, e in CELLS:
        for v in ("all", "drop_first", "detrended"):
            for s in SIGMA_GRID:
                g = fl[(fl.block_role == b) & (fl.emotion == e) & (fl.variant == v) &
                       np.isclose(fl.noise_sd, s)]
                if not len(g):
                    continue
                k, n = int(g.significant.sum()), len(g)
                w = group_row(G, "fluctuation", "wilcoxon_sd_above_noise_null", b, e, v, s)
                pr = prev_row(P, "fluctuation", b, e, v, s)
                rows.append(dict(
                    block=BLOCK_NAME[b], scale=e, variant=VARIANT_NAME[v], sigma=s, n_series=n,
                    k_significant=k, pct_significant=100 * k / n,
                    p_binomial_vs_alpha=binom_greater(k, n),
                    n_fluctuating=int((g.verdict == "fluctuating").sum()),
                    n_flat=int((g.verdict == "flat").sum()),
                    n_inconclusive=int((g.verdict == "inconclusive").sum()),
                    sd_excess_M=g.sd_excess.mean(), sd_excess_SD=g.sd_excess.std(),
                    wilcoxon_median_excess=w.estimate if w is not None else np.nan,
                    wilcoxon_r=w.effect_size if w is not None else np.nan,
                    wilcoxon_p_fdr=w.p_fdr if w is not None else np.nan,
                    prevalence_map_pct=100 * pr.prev_map if pr is not None else np.nan,
                    prevalence_hpdi_low_pct=100 * pr.prev_hpdi_low if pr is not None else np.nan,
                    prevalence_hpdi_high_pct=100 * pr.prev_hpdi_high if pr is not None else np.nan,
                    prevalence_p_gt_half=pr.p_prev_gt_half if pr is not None else np.nan))
    return pd.DataFrame(rows)


def st05bc_people_and_presence(fl, nonnull):
    """B: per participant, baseline: at least one scale varying beyond error /
    fluctuating, both scales flat. C: presence (Test 1, lapse 10%) x
    variation beyond error (Test 2), per cell and sigma."""
    b_rows, c_rows = [], []
    for v in ("all", "drop_first", "detrended"):
        for s in SIGMA_GRID:
            g = fl[(fl.block_role == "base") & (fl.variant == v) & np.isclose(fl.noise_sd, s)]
            p = g.groupby("subject").agg(
                any_sig=("significant", "any"),
                any_fluct=("verdict", lambda x: (x == "fluctuating").any()),
                both_flat=("verdict", lambda x: (x == "flat").all()))
            b_rows.append(dict(block="Baseline", variant=VARIANT_NAME[v], sigma=s,
                               n_participants=len(p),
                               k_any_scale_significant=int(p.any_sig.sum()),
                               k_any_scale_fluctuating=int(p.any_fluct.sum()),
                               k_both_scales_flat=int(p.both_flat.sum())))
    n = nonnull[(nonnull.variant == "all") & np.isclose(nonnull.lapse_rate, LAPSE_MAIN)]
    n = n[["subject", "block_role", "emotion", "significant"]].rename(
        columns={"significant": "present"})
    for b, e in CELLS:
        for s in SIGMA_GRID:
            g = fl[(fl.block_role == b) & (fl.emotion == e) & (fl.variant == "all") &
                   np.isclose(fl.noise_sd, s)].merge(n, on=["subject", "block_role", "emotion"])
            for pres in (True, False):
                h = g[g.present == pres]
                c_rows.append(dict(block=BLOCK_NAME[b], scale=e, sigma=s,
                                   present_test1=pres, n_series=len(h),
                                   k_vary_beyond_error=int(h.significant.sum()),
                                   pct=100 * h.significant.mean() if len(h) else np.nan))
    return pd.DataFrame(b_rows), pd.DataFrame(c_rows)


def st06_within_block(ft):
    t = ft.copy()
    t.insert(0, "block", t.block_role.map(BLOCK_NAME))
    return t.drop(columns=["block_role"])


def st07_specificity(dissoc, G, P):
    rows = []
    for b in BLOCKS:
        for v in ("all", "drop_first", "detrended"):
            x = dissoc[(dissoc.block_role == b) & (dissoc.variant == v)]
            first = x.drop_duplicates("subject")
            el = first[first.eligible]
            asym = el.asym_significant.astype(bool)
            pa = prev_row(P, "asymmetry", b, None, v, None)
            base = dict(block=BLOCK_NAME[b], variant=VARIANT_NAME[v], n_blocks=len(first),
                        n_eligible=len(el), r_M=el.corr_sad_fear.mean(),
                        r_SD=el.corr_sad_fear.std(), r_Mdn=el.corr_sad_fear.median(),
                        asym_k=int(asym.sum()),
                        asym_sadness_more=int((asym & (el.more_variable_scale == "Sadness")).sum()),
                        asym_fear_more=int((asym & (el.more_variable_scale == "Fear")).sum()),
                        asym_p_binomial_vs_alpha=binom_greater(asym.sum(), len(el)),
                        asym_prev_map_pct=100 * pa.prev_map if pa is not None else np.nan,
                        asym_prev_hpdi_low_pct=100 * pa.prev_hpdi_low if pa is not None else np.nan,
                        asym_prev_hpdi_high_pct=100 * pa.prev_hpdi_high if pa is not None else np.nan)
            for s in SIGMA_GRID:
                g = x[np.isclose(x.noise_sd, s) & x.eligible]
                w = group_row(G, "dissociation", "wilcoxon_sd_difference_above_noise_null",
                              b, None, v, s)
                pr = prev_row(P, "dissociation", b, None, v, s)
                rows.append({**base, "sigma": s, "diff_k_significant": int(g.significant.sum()),
                             "diff_pct_significant": 100 * g.significant.mean() if len(g) else np.nan,
                             "diff_wilcoxon_median_excess": w.estimate if w is not None else np.nan,
                             "diff_wilcoxon_r": w.effect_size if w is not None else np.nan,
                             "diff_wilcoxon_p_fdr": w.p_fdr if w is not None else np.nan,
                             "diff_prev_map_pct": 100 * pr.prev_map if pr is not None else np.nan,
                             "diff_prev_hpdi_low_pct": 100 * pr.prev_hpdi_low if pr is not None else np.nan,
                             "diff_prev_hpdi_high_pct": 100 * pr.prev_hpdi_high if pr is not None else np.nan})
    return pd.DataFrame(rows)


def st08_individual(desc, G):
    a = []
    for b, e in CELLS:
        for v in ("all", "drop_first", "detrended"):
            r = group_row(G, "fluctuation", "brown_forsythe_people_differ_in_variability", b, e, v)
            if r is not None:
                a.append(dict(block=BLOCK_NAME[b], scale=e, variant=VARIANT_NAME[v],
                              n_participants=int(r.n), F=r.statistic, p=r.p_value, p_fdr=r.p_fdr))
    d = desc[desc.variant == "all"]
    w = d.pivot_table(index="subject", columns=["block_role", "emotion"], values=["sd", "mean"])
    c = []
    for meas in ("sd", "mean"):
        for e in EMOS:
            for other in ("sad", "fear"):
                n, r, p, lo, hi = spearman_ci(w[(meas, "base", e)].to_numpy(),
                                              w[(meas, other, e)].to_numpy())
                c.append(dict(measure=f"series {meas.upper() if meas == 'sd' else meas}",
                              scale=e, baseline_vs=BLOCK_NAME[other], n_participants=n,
                              spearman_rho=r, ci_low=lo, ci_high=hi, p=p))
    c = pd.DataFrame(c)
    c["p_holm"] = holm(c.p)
    return pd.DataFrame(a), c


def holm(p):
    p = np.asarray(p, float)
    out = np.empty_like(p)
    run = 0.0
    for rank, k in enumerate(np.argsort(p)):
        run = max(run, (len(p) - rank) * p[k])
        out[k] = min(1.0, run)
    return out


def st09_typology(sel, asg):
    s = sel.copy()
    s.insert(0, "block", s.block_role.map(BLOCK_NAME))
    s = s.drop(columns=["block_role"])
    a = asg[asg.solution == "selected"]
    rows = []
    for (b, e), g in a.groupby(["block_role", "emotion"], sort=False):
        order = g.groupby("class_label").frac_nonzero.median().sort_values().index
        for i, cl in enumerate(order, 1):
            h = g[g.class_label == cl]
            rows.append(dict(block=BLOCK_NAME[b], scale=e, k_selected=int(h.k.iloc[0]),
                             class_by_reporting=i, n=len(h), mean_Mdn=h["mean"].median(),
                             sd_Mdn=h.sd.median(), share_nonzero_Mdn=h.frac_nonzero.median(),
                             posterior_Mdn=h.posterior_max.median()))
    return s, pd.DataFrame(rows)


def st10_conditions(dt):
    t = dt.copy()
    nice = {}
    for c in t.comparison.unique():
        a, b = c.split(" vs ")
        (ab, ae), (bb, be) = a.split(), b.split()
        nice[c] = (f"{BLOCK_NAME[ab]}: {ae} vs {be}" if ab == bb
                   else f"{ae}: {BLOCK_NAME[ab]} vs {BLOCK_NAME[bb]}")
    t.insert(2, "comparison_label", t.comparison.map(nice))
    return t


# ===================================================================
# MAIN
# ===================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analysis-dir", default=str(ANALYSIS_DIR))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    root = Path(a.analysis_dir)
    out = Path(a.out_dir) if a.out_dir else root / OUT_NAME
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fp, ns = root / "vas_fluctuation_proof", root / "vas_noise_statespace"
    fpr, dc = root / "vas_first_probe", root / "vas_distribution_check"

    audit, desc = load(fp, "00_series_audit.csv"), load(fp, "01_series_descriptives.csv")
    nonnull, fl = load(fp, "02_nonnull_per_series.csv"), load(fp, "03_fluctuation_per_series.csv")
    dissoc, G, P = (load(fp, "04_dissociation_per_subject_block.csv"),
                    load(fp, "05_group_tests.csv"), load(fp, "06_prevalence.csv"))
    sel, asg = load(fp, "08_mixture_model_selection.csv"), load(fp, "09_mixture_assignments.csv")
    sig, prof = load(ns, "sigma_estimates.csv"), load(ns, "sigma_profile.csv")
    ft, fps = load(fpr, "first_probe_tests.csv"), load(fpr, "first_probe_per_series.csv")
    dfit, dt, dsim = load(dc, "dist_fit.csv"), load(dc, "dist_tests.csv"), load(dc, "dist_sim.csv")

    T = {}
    T["ST01_sample"] = st01_sample(audit, desc)
    T["ST02_descriptives"] = st02_descriptives(desc)
    T["ST03_presence"] = st03_presence(nonnull, G, P)
    T["ST04A_noise_statespace"], T["ST04B_noise_bound_sadness_minus_fear"] = st04_noise(sig, dissoc)
    T["ST05_fluctuation"] = st05_fluctuation(fl, G, P)
    T["ST05B_fluctuation_per_participant"], T["ST05C_presence_x_fluctuation"] = \
        st05bc_people_and_presence(fl, nonnull)
    T["ST06_within_block_change"] = st06_within_block(ft)
    T["ST07_specificity"] = st07_specificity(dissoc, G, P)
    T["ST08A_people_differ"], T["ST08B_cross_block_consistency"] = st08_individual(desc, G)
    T["ST09A_typology_selection"], T["ST09B_typology_classes"] = st09_typology(sel, asg)
    T["ST10_condition_comparisons"] = st10_conditions(dt)
    T["ST11_distribution_fit"] = dfit
    T["ST12_test_calibration"] = dsim
    for name, t in T.items():
        t.to_csv(out / "tables" / f"{name}.csv", index=False, float_format="%.6g")
    print(f"-> {out / 'tables'}: {len(T)} tables")

    if not a.no_figures:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "paper_figures", Path(__file__).with_name("10b_paper_figures.py"))
        figs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(figs)
        figs.make_all(dict(desc=desc, nonnull=nonnull, fl=fl, dissoc=dissoc, G=G, P=P,
                           asg=asg, sig=sig, prof=prof, ft=ft, fps=fps, dt=dt, dsim=dsim,
                           dfit=dfit, T=T), out / "figures")
        print(f"-> {out / 'figures'}")


if __name__ == "__main__":
    main()
