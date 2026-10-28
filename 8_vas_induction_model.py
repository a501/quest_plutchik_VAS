# -*- coding: utf-8 -*-
"""
8_vas_induction_model.py
========================
Time course of the induced emotion within each rest block, and the switch
from one induced emotion to the other across the session.

MODEL (per series = participant x block x emotion, within-rest probes)
    rating(t) = L + A * exp(-(t - T_START) / tau)
    t    minutes since the start of the rest (elapsed_s / 60)
    tau  decay time constant in minutes, shared by all series of one
         block x emotion cell (people differ in L and A, not in tau)
For a given tau the model is linear in L and A, so each series is fitted by
least squares; tau is chosen per cell by minimising the pooled squared
error over a log grid (TAU_GRID). 95% CI of tau: bootstrap over participants.
L and A are extrapolations (to t = infinity) and unstable when tau is very
short or very long, so the analyses use fitted values INSIDE the block:
    start    fitted rating at T_START (0.5 min, about the first probe)
    end      fitted rating at T_END (20 min, end of the block)
    decline  start - end (> 0: the rating falls across the block)

TESTS (across participants; Holm within each family)
    A  per cell: decline > 0?     one-sample t-test, Wilcoxon
    B  induced vs other scale in the same video block, for decline, start
       and end (Sadness - Fear after sad, Fear - Sadness after fear); paired
    C  video block vs baseline, same scale, for decline, start and end;
       paired (order-confounded: baseline is always the first block)
    D  switch / carry-over: the scale induced by the FIRST video, measured
       in the SECOND video block, split by order (block_num 3 vs 2):
       e.g. Sadness start and end level in the fear block when the fear
       block came after the sad block (3rd) vs before it (2nd). Welch t,
       Mann-Whitney. Higher when it follows the other video = the first
       emotion has not returned to its level.

VERSIONS
    The first rating of the baseline block is raised for both scales alike
    (start of the session). Every analysis is run four times, with the first
    0, 1, 2 or 3 probes of the BASELINE blocks dropped (video blocks are
    never changed; t is not re-zeroed). Each version writes to its own
    folder; versions_summary.csv puts the key numbers side by side.

LIMITS
    Least squares on bounded 0-100 ratings: series at the floor give A, L
    near 0 and pull means down; they are kept so that every participant
    counts. The decay is assumed exponential; tau at the upper grid edge
    means a (near-)linear drift, at the lower edge only the first probe
    differs from the rest.

Outputs (OUT_DIR/basedrop{k}/): induction_per_series.csv, induction_tau.csv,
induction_tests.csv; OUT_DIR/versions_summary.csv

Usage
    python3 8_vas_induction_model.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ===================================================================
# CONFIG
# ===================================================================

SUBJ_MAX = 200
BASE = Path("/home/ZhangJianfeng_04/emo_study/Behav")
VAS_PERBLOCK = BASE / "analysis/behav_per_block"
OUT_DIR = BASE / "analysis/vas_induction_model"
MIN_PROBES = 15                       # after dropping, as in script 5
BASE_DROP_VERSIONS = (0, 1, 2, 3)     # first N baseline probes dropped
TAU_GRID = np.exp(np.linspace(np.log(0.5), np.log(60.0), 50))  # minutes
T_START, T_END = 0.5, 20.0            # minutes: where start / end are read
N_BOOT = 1000
SEED = 20261028

COND_ORDER = ["base", "sad", "fear"]
RATING_ORDER = ["Sadness", "Fear"]
INDUCED = {"sad": ("Sadness", "Fear"), "fear": ("Fear", "Sadness")}


# ===================================================================
# DATA
# ===================================================================

def load_probes(perblock_dir, subj_max):
    """One row per within-rest probe: subject, block_role, block_num,
    emotion, t (minutes), rating, probe_idx (0 = first probe)."""
    frames = []
    for f in sorted(Path(perblock_dir).glob("*.csv")):
        m = re.match(r"dyn(\d+)_", f.name)
        if not m or int(m.group(1)) >= subj_max:
            continue
        d = pd.read_csv(f)
        d.columns = d.columns.str.strip().str.lower()
        if "context" in d.columns:
            d = d[d["context"].astype(str).str.strip() == "within_rest"]
        d = pd.DataFrame({
            "subject": int(m.group(1)),
            "block_role": d["block_role"].astype(str).str.strip().str.lower(),
            "block_num": (pd.to_numeric(d["block_num"], errors="coerce")
                          if "block_num" in d.columns else np.nan),
            "emotion": d["emotion"].astype(str).str.strip().str.capitalize(),
            "t": pd.to_numeric(d["elapsed_s"], errors="coerce") / 60.0,
            "rating": pd.to_numeric(d["rating"], errors="coerce")})
        frames.append(d)
    p = pd.concat(frames, ignore_index=True)
    p = p[p.block_role.isin(COND_ORDER) & p.emotion.isin(RATING_ORDER)]
    p = p.dropna(subset=["t", "rating"]).sort_values(
        ["subject", "block_role", "emotion", "t"])
    p["probe_idx"] = p.groupby(["subject", "block_role", "emotion"]).cumcount()
    return p.reset_index(drop=True)


def series_dict(p, drop_base):
    """dict (subject, block, emotion) -> (t, y, block_num), baseline blocks
    without their first drop_base probes."""
    p = p[~((p.block_role == "base") & (p.probe_idx < drop_base))]
    out = {}
    for (s, b, e), g in p.groupby(["subject", "block_role", "emotion"]):
        if len(g) >= MIN_PROBES:
            num = g["block_num"].dropna()
            out[(s, b, e)] = (g["t"].to_numpy(), g["rating"].to_numpy(),
                              int(num.iloc[0]) if num.size else np.nan)
    return out


# ===================================================================
# MODEL
# ===================================================================

def fit_series(t, y, tau):
    """Least-squares L, A for a fixed tau; returns L, A, SSE."""
    X = np.column_stack([np.ones_like(t), np.exp(-(t - T_START) / tau)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef[0], coef[1], float(resid @ resid)


def fit_cells(sd, rng):
    taus, rows = [], []
    for b in COND_ORDER:
        for e in RATING_ORDER:
            keys = [k for k in sd if k[1:] == (b, e)]
            if len(keys) < 5:
                continue
            # SSE of every series at every tau, computed once; a bootstrap
            # sample of participants is then a sum over resampled rows
            sse = np.array([[fit_series(sd[k][0], sd[k][1], tau)[2] for tau in TAU_GRID]
                            for k in keys])
            j = int(np.argmin(sse.sum(axis=0)))
            tau = float(TAU_GRID[j])
            idx = rng.integers(0, len(keys), (N_BOOT, len(keys)))
            boot = TAU_GRID[np.argmin(sse[idx].sum(axis=1), axis=1)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            taus.append(dict(block_role=b, emotion=e, n=len(keys), tau_min=tau,
                             tau_ci_low=lo, tau_ci_high=hi,
                             tau_at_grid_edge=j in (0, TAU_GRID.size - 1)))
            for k in keys:
                t, y, num = sd[k]
                L, A, s = fit_series(t, y, tau)
                start = L + A
                end = L + A * np.exp(-(T_END - T_START) / tau)
                ss_tot = float(((y - y.mean()) ** 2).sum())
                rows.append(dict(subject=k[0], block_role=b, emotion=e, block_num=num,
                                 n=y.size, tau_min=tau, L=L, A=A, start=start, end=end,
                                 decline=start - end,
                                 r2=(1 - s / ss_tot) if ss_tot > 0 else np.nan))
    return pd.DataFrame(taus), pd.DataFrame(rows)


# ===================================================================
# TESTS
# ===================================================================

def holm(p):
    p = np.asarray(p, float)
    out = np.full(p.size, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    run = 0.0
    for rank, k in enumerate(order):
        run = max(run, (order.size - rank) * p[k])
        out[k] = min(1.0, run)
    return out


def one_sample(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    t = stats.ttest_1samp(x, 0.0)
    w = stats.wilcoxon(x) if np.any(x != 0) else None
    sd = x.std(ddof=1)
    return dict(n=int(x.size), mean=x.mean(), sd=sd, t=t.statistic, df=x.size - 1,
                p_t=t.pvalue, p_wilcoxon=w.pvalue if w is not None else 1.0,
                d=x.mean() / sd if sd > 0 else np.nan)


def run_tests(fits):
    w = fits.pivot_table(index="subject", columns=["block_role", "emotion"],
                         values=["decline", "start", "end"])
    rows = []
    for b in COND_ORDER:                                   # A: decline > 0
        for e in RATING_ORDER:
            g = fits[(fits.block_role == b) & (fits.emotion == e)]
            if len(g) >= 3:
                rows.append(dict(family="A decline > 0", contrast=f"{b} {e}",
                                 measure="decline", **one_sample(g["decline"])))
    for b, (ind, oth) in INDUCED.items():
        for meas in ("decline", "start", "end"):
            pairs = [("B induced vs other scale", f"{b}: {ind} - {oth}",
                      (meas, b, ind), (meas, b, oth)),
                     ("C video vs baseline (order-confounded)", f"{ind}: {b} - base",
                      (meas, b, ind), (meas, "base", ind))]
            for fam, lab, ka, kb in pairs:
                if ka in w and kb in w:
                    diff = (w[ka] - w[kb]).dropna()
                    rows.append(dict(family=fam, contrast=lab, measure=meas,
                                     **one_sample(diff)))
    for b, (ind, oth) in INDUCED.items():                  # D: carry-over
        # scale 'oth' was induced by the other video; does it stay up in
        # this block when the other video came first (block 3 vs 2)?
        for meas in ("start", "end"):
            g = fits[(fits.block_role == b) & (fits.emotion == oth)]
            x3 = g.loc[g.block_num == 3, meas].to_numpy()
            x2 = g.loc[g.block_num == 2, meas].to_numpy()
            if x3.size >= 3 and x2.size >= 3:
                t = stats.ttest_ind(x3, x2, equal_var=False)
                u = stats.mannwhitneyu(x3, x2)
                rows.append(dict(family="D carry-over (after other video vs before)",
                                 contrast=f"{oth} in {b} block: 3rd - 2nd", measure=meas,
                                 n=int(x3.size + x2.size), mean=x3.mean() - x2.mean(),
                                 sd=np.nan, t=t.statistic, df=t.df, p_t=t.pvalue,
                                 p_wilcoxon=u.pvalue, d=np.nan))
    r = pd.DataFrame(rows)
    for fam, idx in r.groupby("family").groups.items():
        r.loc[idx, "p_t_holm"] = holm(r.loc[idx, "p_t"])
    return r


# ===================================================================
# MAIN
# ===================================================================

def main():
    p = load_probes(VAS_PERBLOCK, SUBJ_MAX)
    print(f"probes loaded: {len(p)} ({p.subject.nunique()} participants)")
    summary = []
    for k in BASE_DROP_VERSIONS:
        out = OUT_DIR / f"basedrop{k}"
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(SEED)
        sd = series_dict(p, k)
        taus, fits = fit_cells(sd, rng)
        tests = run_tests(fits)
        fits.to_csv(out / "induction_per_series.csv", index=False)
        taus.to_csv(out / "induction_tau.csv", index=False)
        tests.to_csv(out / "induction_tests.csv", index=False)
        print(f"\n=== baseline: first {k} probe(s) dropped  ({len(sd)} series) ===")
        with pd.option_context("display.width", 200, "display.max_columns", 20,
                               "display.float_format", "{:.3f}".format):
            print(taus.to_string(index=False))
            print(tests[["family", "contrast", "measure", "n", "mean", "t", "df",
                         "p_t", "p_t_holm", "d"]].to_string(index=False))
        for _, r in taus.iterrows():
            summary.append(dict(base_drop=k, row=f"tau {r.block_role} {r.emotion}",
                                value=r.tau_min, ci_low=r.tau_ci_low,
                                ci_high=r.tau_ci_high, p_holm=np.nan))
        for _, r in tests.iterrows():
            summary.append(dict(base_drop=k, row=f"{r.family} | {r.contrast} | {r.measure}",
                                value=r["mean"], ci_low=np.nan, ci_high=np.nan,
                                p_holm=r.p_t_holm))
    s = pd.DataFrame(summary)
    s.to_csv(OUT_DIR / "versions_summary.csv", index=False)
    wide = s.pivot_table(index="row", columns="base_drop", values=["value", "p_holm"],
                         sort=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200,
                           "display.float_format", "{:.3f}".format):
        print("\n=== all versions side by side (value = tau in min or mean "
              "difference; p_holm per family) ===")
        print(wide.to_string())
    print(f"\n-> {OUT_DIR}/basedrop0..3/, versions_summary.csv")


if __name__ == "__main__":
    main()
