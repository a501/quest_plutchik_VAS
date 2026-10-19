# -*- coding: utf-8 -*-
"""
5_vas_fluctuation_proof.py
==========================
Do emotions exist and fluctuate at rest, without external stimulus?

Proof analyses on the WITHIN-REST VAS ratings (Sadness, Fear) of the MEG
emotion study, in the base / sad / fear rest blocks. Every test is first
run on each single rating series (one subject x block x emotion, ~25
probes) and then generalised to the population.

  PART A  NON-NULL     People report a non-zero emotion more often than an
                       occasional lapse would produce.
  PART B  FLUCTUATION  Ratings change over the block by more than rating
                       noise would produce. Level does not matter here:
                       a series moving around 50 fluctuates, a series
                       stuck at 30 does not.

PART B makes NO assumption about the time course (no autocorrelation, no
dependence on the inter-probe interval). Every statistic is invariant to
the order of the probes, except the two robustness variants that are
explicitly about order (drop_first, detrended).

STEPS
  0  load + audit       behav_per_block/*.csv, within_rest rows only,
                        subjects with ID < SUBJ_MAX
  1  descriptives       per series: level, SD, flatness, relative
                        variability index (Mestdagh et al., 2018)
  2  non-null test      exact binomial: share of ratings > 0 against a
                        lapse rate (LAPSE_GRID)
  3  fluctuation test   Monte Carlo null "constant level + rating noise"
                        (noise SD from NOISE_SD_GRID, rounded to the
                        slider step and clipped to 0-100 like real
                        ratings) + noise-corrected latent SD with
                        bootstrap CI + verdict fluctuating / flat /
                        inconclusive
  4  dissociation       two complementary tests on Sadness and Fear given at
                        the same probes:
                        a) difference test: Sadness minus Fear removes
                           everything that moves both scales together
                           (attention, response style, general negative
                           affect); what is left is tested against noise
                        b) asymmetry test: do the two scales vary by
                           different amounts? Uses the person's own other
                           scale as noise reference - no external noise
                           estimate needed (catches "Sadness moves, Fear
                           stays at 30")
  5  population level   Bayesian prevalence (Ince et al., 2021, eLife),
                        group Wilcoxon tests, Brown-Forsythe test of
                        between-person differences in variability,
                        BH-FDR across the block x emotion cells
  6  typology           Gaussian mixture with 1..MIX_K_MAX classes on
                        fluctuation amplitude and share of non-zero
                        ratings: BIC, bootstrap LRT, entropy, minimum
                        class size, bootstrap stability

All outputs are CSV files in OUT_DIR; 10_glossary.csv explains every term
in plain language.

Usage
    python3 5_vas_fluctuation_proof.py              # full run
    python3 5_vas_fluctuation_proof.py --quick      # fewer simulations (testing only)
    python3 5_vas_fluctuation_proof.py --selftest   # verify the statistics on synthetic data

Run directly on fox-c18 (or submit to SLURM); the full run takes minutes.
"""

import argparse
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp, xlogy

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===================================================================
# CONFIG
# ===================================================================

SUBJ_MAX = 200   # analyse subjects with ID < SUBJ_MAX  (e.g. 1000 = everyone)

BASE = Path("/home/ZhangJianfeng_04/emo_study/Behav")
VAS_PERBLOCK = BASE / "analysis/behav_per_block"   # within_rest probes (input)
VAS_ANALYSIS = BASE / "analysis"                   # dyn{NNN}_vas_analysis.csv (cross-check)
OUT_DIR = BASE / "analysis/vas_fluctuation_proof"

COND_ORDER = ["base", "sad", "fear"]
RATING_ORDER = ["Sadness", "Fear"]

# ---- rating scale ---------------------------------------------------
SCALE_MIN, SCALE_MAX = 0.0, 100.0
SLIDER_STEP = 5.0          # cursor moves in steps of 5. Verified on the
                           # data in step 0; if ratings are not on this
                           # grid, simulated ratings are not rounded.
CURSOR_START = 50.0        # value of an unmoved cursor (reported only)

# ---- inclusion --------------------------------------------------------
MIN_PROBES = 18            # same floor as the RT pipeline; applied to the
                           # full series so every variant uses the same
                           # subjects

# ---- PART A: non-null -------------------------------------------------
# Lapse rate = share of non-zero ratings that a person with no emotion
# could still produce by accident (mis-press, unmoved cursor, etc.).
LAPSE_GRID = (0.05, 0.10, 0.20)
LAPSE_PRIMARY = 0.10

# ---- PART B: fluctuation ---------------------------------------------
# Rating-noise SD on the 0-100 scale. ~10 = "almost a tenth of the scale"
# (Dejonckheere et al., 2022, Psychological Assessment, within-survey
# retest). The grid is a sensitivity analysis: conclusions should hold
# across it, not only at one value.
NOISE_SD_GRID = (5.0, 7.5, 10.0, 12.5)
NOISE_SD_PRIMARY = 10.0

# Smallest resolvable change = one slider step. Used for the verdict:
#   fluctuating   : lower CI of the noise-corrected latent SD > ROPE_SD
#   flat          : upper CI of the OBSERVED SD < ROPE_SD (no noise
#                   subtracted, so 'flat' is never claimed by removing
#                   variation as noise)
#   inconclusive  : neither
ROPE_SD = SLIDER_STEP

# Dissociation test: a scale with more than this share of ratings at 0 or
# 100 hides shared movement (the floor cannot go lower), so the pair is
# marked untestable instead of tested.
MAX_BOUND_FRAC = 0.20
STABLE_SD = SLIDER_STEP / 2     # "stable reference": SD at most half a step
OFFFLOOR_MARGIN = 10.0          # ...and mean between 10 and 90

# ---- variants -----------------------------------------------------------
VARIANTS_LEVEL = ("all", "drop_first")
VARIANTS_FLUCT = ("all", "drop_first", "detrended")
VARIANT_CODE = {"all": 0, "drop_first": 1, "detrended": 2}
VARIANT_PRIMARY = "all"
BLOCK_PRIMARY = "base"

# ---- inference ------------------------------------------------------------
ALPHA = 0.05
PREV_BETA = 1.0            # sensitivity of the first-level test assumed by
                           # Bayesian prevalence (Ince et al. default)
N_SIM = 10000              # Monte Carlo null draws per test
N_BOOT = 5000              # bootstrap draws per series
SEED = 20261017

# ---- PART 6: mixture typology --------------------------------------------
MIX_VARIANT = "all"
MIX_K_MAX = 4
MIX_COV_TYPE = "diag"      # diagonal covariance: fewer parameters at n~40
MIX_REG_COVAR = 1e-3       # keeps point masses (flat series) fittable
MIX_N_INIT = 20
MIX_N_INIT_BOOT = 5
MIX_N_BLRT = 199           # bootstrap likelihood-ratio test draws
MIX_N_STAB = 200           # bootstrap resamples for class stability
MIX_MIN_SUBJECTS = 12
MIN_CLASS_N = 5
MIN_ENTROPY = 0.80
MIN_STABILITY_ARI = 0.70


# ===================================================================
# SMALL UTILITIES
# ===================================================================

def _find_col(df, candidates):
    """Case-insensitive column lookup: exact match first, then substring."""
    low = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for cand in candidates:
        for lc, orig in low.items():
            if cand in lc:
                return orig
    return None


def subject_from_filename(name):
    """dyn011_block2_sad.csv -> 11. The filename is authoritative for IDs."""
    m = re.search(r"dyn(\d+)", name) or re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def make_rng(*keys):
    """Independent, reproducible random stream per test. Results for one
    subject do not change when other subjects are added or removed."""
    return np.random.default_rng(
        np.random.SeedSequence([SEED] + [int(k) for k in keys]))


def sigma_code(sigma):
    return int(round(float(sigma) * 100))


def sample_sd(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if x.size > 1 else np.nan


def step(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


# ===================================================================
# STATISTICAL BUILDING BLOCKS
# ===================================================================

def max_sd_given_mean(mean, n, lo=SCALE_MIN, hi=SCALE_MAX):
    """Largest possible sample SD (ddof=1) of n values in [lo, hi] with
    this mean (Mestdagh et al., 2018). Variance is convex, so its maximum
    over {x in [lo,hi]^n, sum x = n*mean} sits at a vertex: k values at
    hi, one value in between, the rest at lo."""
    if n < 2 or not np.isfinite(mean):
        return np.nan
    span = hi - lo
    total = (mean - lo) * n
    k = int(np.floor(total / span + 1e-9))
    k = min(max(k, 0), n)
    vals = np.full(n, lo, float)
    vals[:k] = hi
    if k < n:
        vals[k] = lo + min(max(total - k * span, 0.0), span)
    return float(np.std(vals, ddof=1))


def quantize_clip(x, step_size, lo=SCALE_MIN, hi=SCALE_MAX):
    """Turn latent values into ratings: round to the slider grid, clip."""
    if step_size:
        x = np.round(x / step_size) * step_size
    return np.clip(x, lo, hi)


def identity_rows(x, t):
    return np.atleast_2d(np.asarray(x, float))


def detrend_rows(x, t):
    """Remove each row's least-squares linear trend over time t, keeping
    the row mean. Vectorised over rows (simulations)."""
    x = np.atleast_2d(np.asarray(x, float))
    tc = np.asarray(t, float) - np.mean(t)
    denom = float(tc @ tc)
    if denom <= 0:
        return x.copy()
    xm = x.mean(axis=1, keepdims=True)
    slope = ((x - xm) @ tc) / denom
    return x - slope[:, None] * tc[None, :]


def prepare_variant(y, t, variant):
    """Return (values, times, post-transform) for a robustness variant.
    The SAME post-transform is applied to observed and simulated data, so
    the null always matches what was done to the data."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    if variant == "drop_first":
        y, t = y[1:], t[1:]
    elif variant not in ("all", "detrended"):
        raise ValueError(f"unknown variant {variant}")
    post = detrend_rows if variant == "detrended" else identity_rows
    return y, t, post


def mc_p_greater(obs, null):
    """One-sided Monte Carlo p-value P(null >= obs), with the +1
    correction so p is never 0. Ties count against the claim."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan
    return float((1 + np.sum(null >= obs - 1e-9)) / (1 + null.size))


def noise_null_test(y_v, t_v, post, sigma, n_sim, rng, step_size):
    """H0: the series is a constant level plus rating noise of SD sigma.
    Simulate H0 series of the same length at the series' own mean (this
    plug-in level is conservative near the floor: the observed mean sits
    above a clipped true level, so the simulated series are clipped less
    and vary more), make them ratings, apply the same transform, compare
    SDs."""
    obs = sample_sd(post(y_v[None, :], t_v)[0])
    sims = quantize_clip(float(np.mean(y_v))
                         + sigma * rng.standard_normal((n_sim, y_v.size)),
                         step_size)
    sd_null = np.std(post(sims, t_v), axis=1, ddof=1)
    return obs, mc_p_greater(obs, sd_null), float(np.median(sd_null))


def dissoc_null_test(s_v, f_v, t_v, post, sigma, n_sim, rng, step_size):
    """H0: Sadness and Fear share ALL their non-noise movement (a common
    component c_t), each scale adds only its own rating noise (SD sigma).
    Statistic: SD of D_t = Sadness_t - Fear_t, in which c_t cancels.
    c_t is simulated with the LARGEST SD consistent with the data, which
    maximises leakage through the scale bounds and so makes the null
    conservative."""
    S = post(s_v[None, :], t_v)[0]
    F = post(f_v[None, :], t_v)[0]
    obs = sample_sd(S - F)
    var_s, var_f = np.var(S, ddof=1), np.var(F, ddof=1)
    sig_c = float(np.sqrt(max(max(var_s, var_f) - sigma ** 2, 0.0)))
    n = s_v.size
    c = sig_c * rng.standard_normal((n_sim, n))
    s_sim = quantize_clip(np.mean(s_v) + c + sigma * rng.standard_normal((n_sim, n)), step_size)
    f_sim = quantize_clip(np.mean(f_v) + c + sigma * rng.standard_normal((n_sim, n)), step_size)
    sd_null = np.std(post(s_sim, t_v) - post(f_sim, t_v), axis=1, ddof=1)
    return obs, mc_p_greater(obs, sd_null), float(np.median(sd_null)), sig_c


def asymmetry_signflip_test(S, F, n_sim, rng):
    """Do Sadness and Fear vary by DIFFERENT amounts at the same probes?
    Uses the person's own other scale as the noise reference - no external
    noise estimate needed.

    H0: at every probe, (Sadness, Fear) deviations from their own means are
    exchangeable - shared movement loads equally on both and generic rating
    noise is equally large on both scales. Then swapping the two ratings
    within any probe leaves the data distribution unchanged, even if the
    shared movement is autocorrelated. Swapping probe t flips the sign of
    u_t = dS_t^2 - dF_t^2, so the test is a sign-flip test on mean(u),
    two-sided (either scale may be the more variable one).
    mean(u) = (n-1)/n * (Var(Sadness) - Var(Fear))."""
    dS = np.asarray(S, float) - np.mean(S)
    dF = np.asarray(F, float) - np.mean(F)
    u = dS ** 2 - dF ** 2
    obs = float(np.mean(u))
    if np.all(np.abs(u) < 1e-12):
        return obs, 1.0
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_sim, u.size))
    null = (signs * u).mean(axis=1)
    return obs, float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-9)) / (1 + n_sim))


def bayes_prevalence(k, n, alpha=ALPHA, beta=PREV_BETA, cred=0.95,
                     grid_n=20001):
    """Bayesian population prevalence (Ince et al., 2021, eLife).
    k of n participants show a significant within-participant test at
    level alpha. With prevalence gamma, P(significant) = alpha +
    gamma * (beta - alpha). Uniform prior on gamma; posterior on a grid."""
    out = dict(k=int(k), n=int(n), prev_map=np.nan, prev_mean=np.nan,
               prev_lower95=np.nan, prev_hpdi_low=np.nan,
               prev_hpdi_high=np.nan, p_prev_gt_half=np.nan)
    if n <= 0:
        return out
    g = np.linspace(0.0, 1.0, grid_n)
    th = alpha + g * (beta - alpha)
    logp = xlogy(k, th) + xlogy(n - k, 1.0 - th)
    w = np.exp(logp - logsumexp(logp))
    cdf = np.cumsum(w)
    order = np.argsort(w)[::-1]
    inside = order[: int(np.searchsorted(np.cumsum(w[order]), cred)) + 1]
    out.update(
        prev_map=float(np.clip((k / n - alpha) / (beta - alpha), 0.0, 1.0)),
        prev_mean=float(np.sum(w * g)),
        prev_lower95=float(g[min(int(np.searchsorted(cdf, 1.0 - cred)), grid_n - 1)]),
        prev_hpdi_low=float(g[inside].min()),
        prev_hpdi_high=float(g[inside].max()),
        p_prev_gt_half=float(np.sum(w[g > 0.5])))
    return out


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values; NaN stays NaN."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def wilcoxon_greater(d):
    """One-sample Wilcoxon signed-rank test, H1: median(d) > 0, plus the
    matched-pairs rank-biserial correlation as effect size (-1..1)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    nz = d[np.abs(d) > 1e-12]
    res = dict(n=int(d.size), n_nonzero=int(nz.size), statistic=np.nan,
               p_value=np.nan, effect_size=np.nan, median=np.nan)
    if d.size:
        res["median"] = float(np.median(d))
    if nz.size < 1:
        return res
    w = stats.wilcoxon(nz, alternative="greater")
    r = stats.rankdata(np.abs(nz))
    rp, rn = r[nz > 0].sum(), r[nz < 0].sum()
    res.update(statistic=float(w.statistic), p_value=float(w.pvalue),
               effect_size=float((rp - rn) / (rp + rn)))
    return res


def clopper_pearson(k, n, conf=0.95):
    if n == 0:
        return np.nan, np.nan
    a = 1 - conf
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


# ===================================================================
# STEP 0 - LOAD + AUDIT
# ===================================================================

def load_probes(perblock_dir, subj_max):
    files = sorted(Path(perblock_dir).glob("*.csv"))
    if not files:
        sys.exit(f"ERROR: no CSV files in {perblock_dir}")
    frames, faudit = [], []
    for f in files:
        subj = subject_from_filename(f.name)
        row = dict(file=f.name, subject=subj)
        if subj is None:
            faudit.append({**row, "status": "skipped: no subject number in filename"})
            continue
        if subj >= subj_max:
            faudit.append({**row, "status": f"skipped: subject >= SUBJ_MAX ({subj_max})"})
            continue
        d = pd.read_csv(f)
        role_c = _find_col(d, ["block_role"])
        emo_c = _find_col(d, ["emotion"])
        rat_c = _find_col(d, ["rating", "response", "vas", "value"])
        if None in (role_c, emo_c, rat_c):
            faudit.append({**row, "status": "skipped: no block_role/emotion/rating column"})
            continue
        time_c = _find_col(d, ["elapsed_s", "elapsed"])
        ctx_c = _find_col(d, ["context"])
        n_file = len(d)
        if ctx_c is not None:
            d = d[d[ctx_c].astype(str).str.strip() == "within_rest"]
        n_wr = len(d)
        frames.append(pd.DataFrame({
            "subject": np.full(n_wr, subj, dtype=int),
            "block_role": d[role_c].astype(str).str.strip().str.lower().to_numpy(),
            "emotion": d[emo_c].astype(str).str.strip().str.capitalize().to_numpy(),
            "rating": pd.to_numeric(d[rat_c], errors="coerce").to_numpy(dtype=float),
            "elapsed_s": (pd.to_numeric(d[time_c], errors="coerce").to_numpy(dtype=float)
                          if time_c else np.full(n_wr, np.nan)),
            "row_in_file": np.arange(n_wr),
            "source_file": f.name}))
        faudit.append({**row, "status": "loaded", "rows_in_file": n_file,
                       "rows_within_rest": n_wr, "rating_column": rat_c,
                       "time_column": time_c or "(none: file order)",
                       "context_column": ctx_c or "(none: all rows kept)"})
    faudit = pd.DataFrame(faudit)
    if not frames:
        sys.exit("ERROR: no usable per-block files (see column requirements)")
    p = pd.concat(frames, ignore_index=True)

    # ---- unexpected labels ------------------------------------------------
    bad = ~p["block_role"].isin(COND_ORDER) | ~p["emotion"].isin(RATING_ORDER)
    if bad.any():
        print(f"  [warn] {int(bad.sum())} rows with unexpected labels dropped: "
              f"block_role {sorted(set(map(str, p.loc[bad, 'block_role'])))}, "
              f"emotion {sorted(set(map(str, p.loc[bad, 'emotion'])))}")
        p = p[~bad].copy()

    # ---- the same block coming from two files (e.g. a rerun) --------------
    nf = p.groupby(["subject", "block_role"])["source_file"].nunique()
    dup = nf[nf > 1]
    if len(dup):
        print(f"  *** {len(dup)} subject x block cells come from >1 file - "
              f"EXCLUDED (resolve by hand): ***")
        for (s, b) in dup.index:
            fl = sorted(p.loc[(p.subject == s) & (p.block_role == b), "source_file"].unique())
            print(f"      subject {s:03d} {b}: {fl}")
        bad_cells = set(dup.index)
        keep = [k not in bad_cells for k in zip(p["subject"], p["block_role"])]
        p = p[keep].copy()

    # ---- order within each series: time if usable, else file order -------
    p = p.sort_values(["subject", "block_role", "emotion", "row_in_file"]).reset_index(drop=True)
    p["time_source"] = "elapsed_s"
    for (s, b, e), g in p.groupby(["subject", "block_role", "emotion"], sort=False):
        t = g["elapsed_s"]
        if not (t.notna().all() and t.is_unique):
            p.loc[g.index, "time_source"] = "file_order"
    use_t = p["time_source"] == "elapsed_s"
    p["_sortkey"] = np.where(use_t, p["elapsed_s"], p["row_in_file"])
    p = p.sort_values(["subject", "block_role", "emotion", "_sortkey"]).reset_index(drop=True)
    # probe index is assigned BEFORE dropping missing ratings, so Sadness
    # and Fear stay aligned when one of them timed out
    p["probe_idx"] = p.groupby(["subject", "block_role", "emotion"]).cumcount()
    p["t"] = np.where(p["time_source"] == "elapsed_s",
                      p["elapsed_s"], p["probe_idx"].astype(float))
    p = p.drop(columns="_sortkey")
    return p, faudit


def check_scale(p):
    """Verify the rating grid and range on the real data."""
    r = p["rating"].dropna().to_numpy()
    on_grid = float(np.mean(np.abs(r / SLIDER_STEP - np.round(r / SLIDER_STEP)) < 1e-6))
    out_range = int(np.sum((r < SCALE_MIN) | (r > SCALE_MAX)))
    step_used = SLIDER_STEP if on_grid >= 0.99 else None
    print(f"  ratings on the {SLIDER_STEP:g}-point slider grid: {100 * on_grid:.2f}%  "
          f"-> simulated ratings {'rounded to the grid' if step_used else 'NOT rounded'}")
    if out_range:
        print(f"  *** {out_range} ratings outside [{SCALE_MIN:g}, {SCALE_MAX:g}] - check the input ***")
    return step_used, on_grid, out_range


def build_series(p):
    """dict (subject, block, emotion) -> ratings and times, NaN removed."""
    series, audit = {}, []
    for (s, b, e), g in p.groupby(["subject", "block_role", "emotion"], sort=True):
        ok = g["rating"].notna()
        y = g.loc[ok, "rating"].to_numpy(float)
        t = g.loc[ok, "t"].to_numpy(float)
        included = y.size >= MIN_PROBES
        audit.append(dict(subject=s, block_role=b, emotion=e, n_rows=len(g),
                          n_valid=int(y.size), n_missing=int((~ok).sum()),
                          time_source=g["time_source"].iloc[0],
                          included=included,
                          reason="" if included else f"fewer than {MIN_PROBES} valid probes"))
        if included:
            series[(s, b, e)] = dict(y=y, t=t)
    return series, pd.DataFrame(audit)


def pair_emotions(p):
    """Sadness and Fear given at the same probe. Matched on elapsed_s
    where both carry the same timestamp, otherwise on probe index."""
    pairs, audit = {}, []
    for (s, b), g in p.groupby(["subject", "block_role"], sort=True):
        gs = g[g["emotion"] == "Sadness"]
        gf = g[g["emotion"] == "Fear"]
        row = dict(subject=s, block_role=b, n_sadness_rows=len(gs), n_fear_rows=len(gf))
        if gs.empty or gf.empty:
            audit.append({**row, "pairing": "none", "n_pairs": 0, "included": False,
                          "reason": "one emotion missing"})
            continue
        m, method = None, None
        if gs["elapsed_s"].notna().all() and gf["elapsed_s"].notna().all():
            ks, kf = gs["elapsed_s"].round(2), gf["elapsed_s"].round(2)
            if ks.is_unique and kf.is_unique:
                mm = pd.merge(gs.assign(key=ks.to_numpy()), gf.assign(key=kf.to_numpy()),
                              on="key", suffixes=("_s", "_f"))
                if len(mm) >= 0.9 * max(len(gs), len(gf)):
                    m, method = mm, "elapsed_s"
        if m is None:
            m = pd.merge(gs, gf, on="probe_idx", suffixes=("_s", "_f"))
            method = "probe_index"
        m = m[m["rating_s"].notna() & m["rating_f"].notna()].sort_values("t_s")
        n_pairs = len(m)
        included = n_pairs >= MIN_PROBES
        audit.append({**row, "pairing": method, "n_pairs": n_pairs, "included": included,
                      "reason": "" if included else f"fewer than {MIN_PROBES} pairs"})
        if included:
            pairs[(s, b)] = dict(s=m["rating_s"].to_numpy(float),
                                 f=m["rating_f"].to_numpy(float),
                                 t=m["t_s"].to_numpy(float))
    return pairs, pd.DataFrame(audit)


def crosscheck_battery(series, analysis_dir):
    """Compare mean/SD here with the 'mean'/'std' measures of the existing
    dynamic-measure battery (dyn{NNN}_vas_analysis.csv). A mismatch means
    the two pipelines read different data."""
    files = sorted(Path(analysis_dir).glob("dyn*_vas_analysis.csv"))
    if not files:
        print(f"  [skip] battery cross-check: no dyn*_vas_analysis.csv in {analysis_dir}")
        return pd.DataFrame()
    rows = []
    for f in files:
        s = subject_from_filename(f.name)
        if s is None or s >= SUBJ_MAX:
            continue
        d = pd.read_csv(f)
        if not {"block_role", "emotion", "measure", "value"} <= set(d.columns):
            continue
        if "transform" in d.columns:
            d = d[d["transform"].astype(str) == "raw"]
        d = d[d["measure"].isin(["mean", "std"])]
        for _, r in d.iterrows():
            key = (s, str(r["block_role"]).lower(), str(r["emotion"]).capitalize())
            v = pd.to_numeric(r["value"], errors="coerce")
            if key not in series or not np.isfinite(v):
                continue
            y = series[key]["y"]
            if r["measure"] == "mean":
                rows.append(dict(subject=s, block_role=key[1], emotion=key[2],
                                 measure="mean", battery=v, here=float(np.mean(y)),
                                 abs_diff=abs(v - np.mean(y))))
            else:
                d0, d1 = abs(v - np.std(y, ddof=0)), abs(v - np.std(y, ddof=1))
                rows.append(dict(subject=s, block_role=key[1], emotion=key[2],
                                 measure="std", battery=v, here=float(np.std(y, ddof=1)),
                                 abs_diff=min(d0, d1),
                                 ddof_match=0 if d0 <= d1 else 1))
    cc = pd.DataFrame(rows)
    if cc.empty:
        print("  [skip] battery cross-check: nothing comparable")
        return cc
    for meas, g in cc.groupby("measure"):
        worst = g["abs_diff"].max()
        tag = "MATCH" if worst < 1e-6 else "*** MISMATCH - investigate ***"
        extra = ""
        if meas == "std":
            extra = f" (battery uses ddof={int(g['ddof_match'].mode().iloc[0])})"
        print(f"  battery cross-check {meas:4s}: {len(g)} series, max |diff| "
              f"{worst:.3g}  {tag}{extra}")
    return cc


# ===================================================================
# STEP 1-4 - PER-SERIES ANALYSES
# ===================================================================

def describe(y):
    n = y.size
    m = float(np.mean(y))
    sd = sample_sd(y)
    msd = max_sd_given_mean(m, n)
    nd = int(np.unique(y).size)
    return dict(n=n, mean=m, median=float(np.median(y)), sd=sd,
                min=float(np.min(y)), max=float(np.max(y)),
                range=float(np.ptp(y)), n_distinct=nd, is_flat=nd == 1,
                frac_zero=float(np.mean(y <= SCALE_MIN)),
                frac_nonzero=float(np.mean(y > SCALE_MIN)),
                frac_at_max=float(np.mean(y >= SCALE_MAX)),
                n_at_cursor_start=int(np.sum(y == CURSOR_START)),
                frac_at_cursor_start=float(np.mean(y == CURSOR_START)),
                max_sd_given_mean=msd,
                rvi_sd=(sd / msd) if (msd and np.isfinite(msd) and msd > 0) else np.nan)


def ids(key):
    s, b, e = key
    return dict(subject=s, block_role=b, emotion=e)


def run_descriptives(series):
    rows = []
    for key, d in series.items():
        for v in VARIANTS_LEVEL:
            y, _, _ = prepare_variant(d["y"], d["t"], v)
            rows.append({**ids(key), "variant": v, **describe(y)})
    return pd.DataFrame(rows)


def run_nonnull(series):
    rows = []
    for key, d in series.items():
        for v in VARIANTS_LEVEL:
            y, _, _ = prepare_variant(d["y"], d["t"], v)
            k, n = int(np.sum(y > SCALE_MIN)), y.size
            for p0 in LAPSE_GRID:
                p = float(stats.binom.sf(k - 1, n, p0))
                rows.append({**ids(key), "variant": v, "lapse_rate": p0,
                             "n": n, "n_nonzero": k, "frac_nonzero": k / n,
                             "p_value": p, "significant": p < ALPHA})
    return pd.DataFrame(rows)


def run_fluctuation(series, step_size, n_sim, n_boot):
    rows = []
    bi = {b: i for i, b in enumerate(COND_ORDER)}
    ei = {e: i for i, e in enumerate(RATING_ORDER)}
    for key, d in series.items():
        s, b, e = key
        for v in VARIANTS_FLUCT:
            y_v, t_v, post = prepare_variant(d["y"], d["t"], v)
            vals = post(y_v[None, :], t_v)[0]
            n = vals.size
            sd_obs = sample_sd(vals)
            rng_b = make_rng(1, s, bi[b], ei[e], VARIANT_CODE[v])
            sd_boot = np.std(vals[rng_b.integers(0, n, size=(n_boot, n))], axis=1, ddof=1)
            sd_lo, sd_hi = np.percentile(sd_boot, [2.5, 97.5])
            for sig in NOISE_SD_GRID:
                rng = make_rng(2, s, bi[b], ei[e], VARIANT_CODE[v], sigma_code(sig))
                obs, p, sd_null_med = noise_null_test(y_v, t_v, post, sig, n_sim, rng, step_size)
                lat = float(np.sqrt(max(obs ** 2 - sig ** 2, 0.0)))
                lat_boot = np.sqrt(np.clip(sd_boot ** 2 - sig ** 2, 0.0, None))
                lat_lo, lat_hi = np.percentile(lat_boot, [2.5, 97.5])
                if lat_lo > ROPE_SD:
                    verdict = "fluctuating"
                elif sd_hi < ROPE_SD:
                    verdict = "flat"
                else:
                    verdict = "inconclusive"
                rows.append({**ids(key), "variant": v, "noise_sd": sig, "n": n,
                             "sd_obs": obs, "sd_obs_ci_low": float(sd_lo),
                             "sd_obs_ci_high": float(sd_hi),
                             "sd_null_median": sd_null_med,
                             "sd_excess": obs - sd_null_med,
                             "p_value": p, "significant": p < ALPHA,
                             "latent_sd": lat, "latent_sd_ci_low": float(lat_lo),
                             "latent_sd_ci_high": float(lat_hi),
                             "verdict": verdict})
    return pd.DataFrame(rows)


def run_dissociation(pairs, step_size, n_sim):
    rows = []
    bi = {b: i for i, b in enumerate(COND_ORDER)}
    for (s, b), d in pairs.items():
        for v in VARIANTS_FLUCT:
            s_v, t_v, post = prepare_variant(d["s"], d["t"], v)
            f_v, _, _ = prepare_variant(d["f"], d["t"], v)
            S, F = post(s_v[None, :], t_v)[0], post(f_v[None, :], t_v)[0]
            var_s, var_f = float(np.var(S, ddof=1)), float(np.var(F, ddof=1))
            cov = float(np.cov(S, F, ddof=1)[0, 1])
            corr = (float(np.corrcoef(S, F)[0, 1]) if var_s > 0 and var_f > 0 else np.nan)
            bound_s = float(np.mean((s_v <= SCALE_MIN) | (s_v >= SCALE_MAX)))
            bound_f = float(np.mean((f_v <= SCALE_MIN) | (f_v >= SCALE_MAX)))
            eligible = bound_s <= MAX_BOUND_FRAC and bound_f <= MAX_BOUND_FRAC

            def stable(x):
                return (sample_sd(x) <= STABLE_SD
                        and OFFFLOOR_MARGIN <= np.mean(x) <= SCALE_MAX - OFFFLOOR_MARGIN)
            st = [nm for nm, x in (("Sadness", s_v), ("Fear", f_v)) if stable(x)]
            stable_ref = "both" if len(st) == 2 else (st[0] if st else "none")

            # internal-reference test: does not depend on the noise SD
            if eligible:
                a_obs, a_p = asymmetry_signflip_test(
                    S, F, n_sim, make_rng(6, s, bi[b], VARIANT_CODE[v]))
            else:
                a_obs, a_p = np.nan, np.nan
            sd_s, sd_f = sample_sd(S), sample_sd(F)
            more_var = ("equal" if abs(sd_s - sd_f) < 1e-12
                        else ("Sadness" if sd_s > sd_f else "Fear"))

            base_row = dict(subject=s, block_role=b, variant=v, n_pairs=int(s_v.size),
                            mean_sadness=float(np.mean(s_v)), mean_fear=float(np.mean(f_v)),
                            sd_sadness=sample_sd(S), sd_fear=sample_sd(F),
                            cov_sad_fear=cov, corr_sad_fear=corr,
                            spec_var_sadness=var_s - max(cov, 0.0),
                            spec_var_fear=var_f - max(cov, 0.0),
                            bound_frac_sadness=bound_s, bound_frac_fear=bound_f,
                            eligible=eligible, stable_offfloor_reference=stable_ref,
                            more_variable_scale=more_var,
                            sd_difference_sad_minus_fear=sd_s - sd_f,
                            asym_var_difference=a_obs, asym_p_value=a_p,
                            asym_significant=bool(np.isfinite(a_p) and a_p < ALPHA))
            for sig in NOISE_SD_GRID:
                row = {**base_row, "noise_sd": sig}
                if eligible:
                    rng = make_rng(3, s, bi[b], VARIANT_CODE[v], sigma_code(sig))
                    obs, p, med, sig_c = dissoc_null_test(s_v, f_v, t_v, post, sig,
                                                          n_sim, rng, step_size)
                    row.update(sd_diff=obs, sd_diff_null_median=med,
                               sd_diff_excess=obs - med, shared_sd_simulated=sig_c,
                               specific_sd=float(np.sqrt(max(obs ** 2 - 2 * sig ** 2, 0.0))),
                               p_value=p, significant=p < ALPHA,
                               status="tested")
                else:
                    row.update(sd_diff=sample_sd(S - F), sd_diff_null_median=np.nan,
                               sd_diff_excess=np.nan, shared_sd_simulated=np.nan,
                               specific_sd=np.nan, p_value=np.nan, significant=False,
                               status="untestable: floor/ceiling")
                rows.append(row)
    return pd.DataFrame(rows)


# ===================================================================
# STEP 5 - POPULATION LEVEL
# ===================================================================

def cells():
    for b in COND_ORDER:
        for e in RATING_ORDER:
            yield b, e


def group_level(series, desc, nonnull, fluct, dissoc):
    G, P, C = [], [], []

    def grow(**kw):
        base = dict(analysis="", block_role="", emotion="", variant="", parameter="",
                    parameter_value=np.nan, test="", n=np.nan, k=np.nan,
                    statistic=np.nan, p_value=np.nan, effect_size=np.nan,
                    effect_size_name="", estimate=np.nan, estimate_name="",
                    ci_low=np.nan, ci_high=np.nan, note="")
        base.update(kw)
        G.append(base)

    # ---------------- PART A: non-null ----------------
    for v in VARIANTS_LEVEL:
        for b, e in cells():
            dd = desc[(desc.variant == v) & (desc.block_role == b) & (desc.emotion == e)]
            n = len(dd)
            if n == 0:
                continue
            k = int((dd["frac_nonzero"] > 0).sum())
            lo, hi = clopper_pearson(k, n)
            grow(analysis="nonnull", block_role=b, emotion=e, variant=v,
                 test="share_of_people_with_any_nonzero_rating", n=n, k=k,
                 estimate=k / n, estimate_name="proportion", ci_low=lo, ci_high=hi,
                 note="descriptive; exact (Clopper-Pearson) 95% CI")
            for p0 in LAPSE_GRID:
                w = wilcoxon_greater(dd["frac_nonzero"].to_numpy() - p0)
                grow(analysis="nonnull", block_role=b, emotion=e, variant=v,
                     parameter="lapse_rate", parameter_value=p0,
                     test="wilcoxon_frac_nonzero_above_lapse", n=w["n"],
                     statistic=w["statistic"], p_value=w["p_value"],
                     effect_size=w["effect_size"], effect_size_name="rank_biserial_r",
                     estimate=w["median"], estimate_name="median(frac_nonzero - lapse)")
                nn = nonnull[(nonnull.variant == v) & (nonnull.block_role == b)
                             & (nonnull.emotion == e) & (nonnull.lapse_rate == p0)]
                pr = bayes_prevalence(int(nn["significant"].sum()), len(nn))
                P.append(dict(analysis="nonnull", block_role=b, emotion=e, variant=v,
                              parameter="lapse_rate", parameter_value=p0, **pr))

    # ---------------- PART B: fluctuation ----------------
    for v in VARIANTS_FLUCT:
        for b, e in cells():
            keys = [k for k in series if k[1] == b and k[2] == e]
            if not keys:
                continue
            ff = fluct[(fluct.variant == v) & (fluct.block_role == b) & (fluct.emotion == e)]
            sd_first = ff.drop_duplicates("subject")
            n = len(sd_first)
            k = int((sd_first["sd_obs"] > 1e-12).sum())
            lo, hi = clopper_pearson(k, n)
            grow(analysis="fluctuation", block_role=b, emotion=e, variant=v,
                 test="share_of_people_with_nonflat_series", n=n, k=k,
                 estimate=k / n if n else np.nan, estimate_name="proportion",
                 ci_low=lo, ci_high=hi, note="strict flatness (SD = 0); descriptive")
            groups = []
            for key in keys:
                y_v, t_v, post = prepare_variant(series[key]["y"], series[key]["t"], v)
                groups.append(post(y_v[None, :], t_v)[0])
            if len(groups) >= 2 and any(np.std(gx) > 0 for gx in groups):
                bf = stats.levene(*groups, center="median")
                grow(analysis="fluctuation", block_role=b, emotion=e, variant=v,
                     test="brown_forsythe_people_differ_in_variability", n=len(groups),
                     statistic=float(bf.statistic), p_value=float(bf.pvalue),
                     note="H1: within-person variability differs between people")
            for sig in NOISE_SD_GRID:
                fs = ff[ff.noise_sd == sig]
                w = wilcoxon_greater(fs["sd_excess"].to_numpy())
                grow(analysis="fluctuation", block_role=b, emotion=e, variant=v,
                     parameter="noise_sd", parameter_value=sig,
                     test="wilcoxon_sd_above_noise_null", n=w["n"],
                     statistic=w["statistic"], p_value=w["p_value"],
                     effect_size=w["effect_size"], effect_size_name="rank_biserial_r",
                     estimate=w["median"], estimate_name="median(SD_obs - SD_null)")
                pr = bayes_prevalence(int(fs["significant"].sum()), len(fs))
                P.append(dict(analysis="fluctuation", block_role=b, emotion=e, variant=v,
                              parameter="noise_sd", parameter_value=sig, **pr))
                vc = fs["verdict"].value_counts()
                nt = len(fs)
                C.append(dict(analysis="fluctuation", block_role=b, emotion=e, variant=v,
                              noise_sd=sig, n=nt, n_significant=int(fs["significant"].sum()),
                              n_fluctuating=int(vc.get("fluctuating", 0)),
                              n_flat=int(vc.get("flat", 0)),
                              n_inconclusive=int(vc.get("inconclusive", 0)),
                              frac_fluctuating=vc.get("fluctuating", 0) / nt if nt else np.nan,
                              frac_flat=vc.get("flat", 0) / nt if nt else np.nan,
                              frac_inconclusive=vc.get("inconclusive", 0) / nt if nt else np.nan))

    # ---------------- PART B: dissociation ----------------
    for v in VARIANTS_FLUCT:
        for b in COND_ORDER:
            dd = dissoc[(dissoc.variant == v) & (dissoc.block_role == b)]
            if dd.empty:
                continue
            # asymmetry test: one row per subject (it does not depend on noise_sd)
            d1 = dd.drop_duplicates("subject")
            e1 = d1[d1["eligible"]]
            pr = bayes_prevalence(int(e1["asym_significant"].sum()), len(e1))
            P.append(dict(analysis="asymmetry", block_role=b, emotion="Sadness-Fear",
                          variant=v, parameter="", parameter_value=np.nan, **pr))
            sig1 = e1[e1["asym_significant"]]
            C.append(dict(analysis="asymmetry", block_role=b, emotion="Sadness-Fear",
                          variant=v, n=len(d1), n_eligible=len(e1),
                          n_significant=len(sig1),
                          n_sadness_more_variable=int((sig1.more_variable_scale == "Sadness").sum()),
                          n_fear_more_variable=int((sig1.more_variable_scale == "Fear").sum())))
            for sig in NOISE_SD_GRID:
                ds = dd[dd.noise_sd == sig]
                el = ds[ds["eligible"]]
                w = wilcoxon_greater(el["sd_diff_excess"].to_numpy())
                grow(analysis="dissociation", block_role=b, emotion="Sadness-Fear",
                     variant=v, parameter="noise_sd", parameter_value=sig,
                     test="wilcoxon_sd_difference_above_noise_null", n=w["n"],
                     statistic=w["statistic"], p_value=w["p_value"],
                     effect_size=w["effect_size"], effect_size_name="rank_biserial_r",
                     estimate=w["median"], estimate_name="median(SD_D_obs - SD_D_null)",
                     note=f"eligible pairs only ({len(el)} of {len(ds)})")
                pr = bayes_prevalence(int(el["significant"].sum()), len(el))
                P.append(dict(analysis="dissociation", block_role=b, emotion="Sadness-Fear",
                              variant=v, parameter="noise_sd", parameter_value=sig, **pr))
                C.append(dict(analysis="dissociation", block_role=b, emotion="Sadness-Fear",
                              variant=v, noise_sd=sig, n=len(ds), n_eligible=len(el),
                              n_significant=int(el["significant"].sum()),
                              n_stable_reference_sadness=int((ds.stable_offfloor_reference
                                                              .isin(["Sadness", "both"])).sum()),
                              n_stable_reference_fear=int((ds.stable_offfloor_reference
                                                           .isin(["Fear", "both"])).sum())))

    G = pd.DataFrame(G)
    # BH-FDR within each family = same analysis/test/variant/parameter,
    # across the block x emotion cells
    G["p_fdr"] = np.nan
    # plain-Python str() so missing values become "nan" in every pandas
    # version (pandas 3 keeps NaN missing under .astype(str))
    fam = ["analysis", "test", "variant", "parameter", "parameter_value"]
    famkey = pd.Series(["|".join(str(x) for x in r) for r in G[fam].itertuples(index=False)],
                       index=G.index)
    for key in famkey.unique():
        idx = famkey == key
        G.loc[idx, "p_fdr"] = bh_fdr(G.loc[idx, "p_value"].to_numpy())
    G["is_primary"] = ((G.block_role == BLOCK_PRIMARY) & (G.variant == VARIANT_PRIMARY)
                       & (G.parameter_value.isna()
                          | ((G.parameter == "noise_sd") & (G.parameter_value == NOISE_SD_PRIMARY))
                          | ((G.parameter == "lapse_rate") & (G.parameter_value == LAPSE_PRIMARY))))
    P = pd.DataFrame(P)
    P["is_primary"] = ((P.block_role == BLOCK_PRIMARY) & (P.variant == VARIANT_PRIMARY)
                       & (P.parameter_value.isna()
                          | ((P.parameter == "noise_sd") & (P.parameter_value == NOISE_SD_PRIMARY))
                          | ((P.parameter == "lapse_rate") & (P.parameter_value == LAPSE_PRIMARY))))
    return G, P, pd.DataFrame(C)


# ===================================================================
# STEP 6 - MIXTURE TYPOLOGY
# ===================================================================

def _sample_gmm(gm, n, rng):
    comp = rng.choice(gm.n_components, size=n, p=gm.weights_)
    X = np.empty((n, gm.means_.shape[1]))
    for c in range(gm.n_components):
        idx = np.where(comp == c)[0]
        if idx.size == 0:
            continue
        if gm.covariance_type == "diag":
            X[idx] = gm.means_[c] + rng.standard_normal((idx.size, X.shape[1])) * np.sqrt(gm.covariances_[c])
        elif gm.covariance_type == "full":
            X[idx] = rng.multivariate_normal(gm.means_[c], gm.covariances_[c], size=idx.size)
        else:
            raise ValueError("bootstrap sampling implemented for diag/full only")
    return X


def run_mixture(desc, n_blrt, n_stab):
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        print("  [skip] scikit-learn not available in this environment - typology skipped")
        return pd.DataFrame(), pd.DataFrame()

    def fit(X, k, n_init, seed):
        gm = GaussianMixture(n_components=k, covariance_type=MIX_COV_TYPE,
                             reg_covar=MIX_REG_COVAR, n_init=n_init,
                             max_iter=1000, random_state=seed)
        return gm.fit(X)

    def ll(gm, X):
        return float(gm.score(X) * X.shape[0])

    sel_rows, asg_rows = [], []
    for ci, (b, e) in enumerate(cells()):
        dd = desc[(desc.variant == MIX_VARIANT) & (desc.block_role == b)
                  & (desc.emotion == e)].sort_values("subject")
        n = len(dd)
        if n < MIX_MIN_SUBJECTS:
            print(f"  [skip] {b}/{e}: {n} subjects < MIX_MIN_SUBJECTS")
            continue
        feats = {"log1p_sd": np.log1p(dd["sd"].to_numpy(float)),
                 "frac_nonzero": dd["frac_nonzero"].to_numpy(float)}
        used = [k for k, x in feats.items() if np.std(x) > 0]
        if not used:
            print(f"  [skip] {b}/{e}: features have no variance")
            continue
        raw = np.column_stack([feats[k] for k in used])
        X = (raw - raw.mean(0)) / raw.std(0)
        models, metrics = {}, []
        for k in range(1, MIX_K_MAX + 1):
            if k > n // 2:
                break
            gm = fit(X, k, MIX_N_INIT, SEED + k)
            models[k] = gm
            resp = gm.predict_proba(X)
            lab = resp.argmax(1)
            sizes = np.bincount(lab, minlength=k)
            ent = (1.0 if k == 1 else
                   float(1 - (-np.sum(xlogy(resp, resp))) / (n * np.log(k))))
            row = dict(block_role=b, emotion=e, n=n, features="+".join(used), k=k,
                       log_likelihood=ll(gm, X), bic=float(gm.bic(X)), aic=float(gm.aic(X)),
                       min_class_n=int(sizes.min()), class_sizes="/".join(map(str, sizes)),
                       entropy=ent, blrt_p=np.nan, stability_ari=1.0 if k == 1 else np.nan)
            if k >= 2:
                g0 = models[k - 1]
                lr_obs = 2 * (ll(gm, X) - ll(g0, X))
                rng = make_rng(4, ci, k)
                lr_null = np.empty(n_blrt)
                for i in range(n_blrt):
                    Xb = _sample_gmm(g0, n, rng)
                    seed_b = int(rng.integers(0, 2 ** 31 - 1))
                    lr_null[i] = 2 * (ll(fit(Xb, k, MIX_N_INIT_BOOT, seed_b), Xb)
                                      - ll(fit(Xb, k - 1, MIX_N_INIT_BOOT, seed_b), Xb))
                row["blrt_p"] = float((1 + np.sum(lr_null >= lr_obs)) / (1 + n_blrt))
                rng_s = make_rng(5, ci, k)
                aris = []
                for _ in range(n_stab):
                    idx = rng_s.integers(0, n, n)
                    gb = fit(X[idx], k, MIX_N_INIT_BOOT, int(rng_s.integers(0, 2 ** 31 - 1)))
                    aris.append(adjusted_rand_score(lab, gb.predict(X)))
                row["stability_ari"] = float(np.mean(aris))
            row["passes_all_criteria"] = bool(
                k == 1 or (row["blrt_p"] < ALPHA and row["min_class_n"] >= MIN_CLASS_N
                           and row["entropy"] >= MIN_ENTROPY
                           and row["stability_ari"] >= MIN_STABILITY_ARI))
            metrics.append(row)
        met = pd.DataFrame(metrics)
        k_bic = int(met.loc[met["bic"].idxmin(), "k"])
        k_sel = 1
        for _, r in met.sort_values("k").iterrows():
            if r["k"] == 1:
                continue
            if r["passes_all_criteria"]:
                k_sel = int(r["k"])
            else:
                break
        met["k_best_bic"] = k_bic
        met["k_selected"] = k_sel
        sel_rows.append(met)
        print(f"  {b:4s}/{e:7s}: n={n}, BIC prefers k={k_bic}, all criteria -> k={k_sel}"
              f"{'  (continuum)' if k_sel == 1 else ''}")

        for kind, kk in (("selected", k_sel), ("best_bic", k_bic)):
            gm = models[kk]
            resp = gm.predict_proba(X)
            lab = resp.argmax(1)
            # relabel: class 1 = lowest mean fluctuation amplitude
            order = np.argsort([raw[lab == c, 0].mean() if np.any(lab == c) else np.inf
                                for c in range(kk)])
            remap = {old: new + 1 for new, old in enumerate(order)}
            for i, (_, r) in enumerate(dd.iterrows()):
                asg_rows.append(dict(block_role=b, emotion=e, solution=kind, k=kk,
                                     subject=int(r["subject"]), mean=r["mean"], sd=r["sd"],
                                     frac_nonzero=r["frac_nonzero"],
                                     class_label=remap[int(lab[i])],
                                     posterior_max=float(resp[i].max())))
    sel = pd.concat(sel_rows, ignore_index=True) if sel_rows else pd.DataFrame()
    return sel, pd.DataFrame(asg_rows)


# ===================================================================
# DOCUMENTATION TABLES
# ===================================================================

def config_table(step_size, on_grid, n_sim, n_boot, n_blrt, n_stab):
    rows = [
        ("SUBJ_MAX", SUBJ_MAX, "Subjects with ID below this value are analysed."),
        ("MIN_PROBES", MIN_PROBES, "Minimum valid probes for a series (or Sadness-Fear pairs) to be analysed."),
        ("SLIDER_STEP", SLIDER_STEP, "Rating resolution: the cursor moves in steps of this size."),
        ("ratings_on_grid_pct", round(100 * on_grid, 2), "Share of ratings that lie on the slider grid (data check)."),
        ("simulated_ratings_rounded", step_size is not None, "Whether simulated ratings are rounded to the grid like real ones."),
        ("CURSOR_START", CURSOR_START, "Cursor start value; ratings at this value are counted (reported, not removed)."),
        ("LAPSE_GRID", " / ".join(map(str, LAPSE_GRID)), "Lapse rates tested in the non-null test."),
        ("LAPSE_PRIMARY", LAPSE_PRIMARY, "Lapse rate of the main result."),
        ("NOISE_SD_GRID", " / ".join(map(str, NOISE_SD_GRID)), "Rating-noise SDs tested (sensitivity analysis)."),
        ("NOISE_SD_PRIMARY", NOISE_SD_PRIMARY, "Noise SD of the main result (~1/10 of the scale, Dejonckheere et al., 2022)."),
        ("ROPE_SD", ROPE_SD, "Smallest resolvable change, used for the fluctuating / flat / inconclusive verdict."),
        ("MAX_BOUND_FRAC", MAX_BOUND_FRAC, "Dissociation test: max share of ratings at 0 or 100 per scale."),
        ("ALPHA", ALPHA, "Significance level of every single test."),
        ("N_SIM", n_sim, "Simulated null series per Monte Carlo test."),
        ("N_BOOT", n_boot, "Bootstrap resamples per series."),
        ("MIX_K_MAX", MIX_K_MAX, "Largest number of classes tried in the mixture model."),
        ("MIX_N_BLRT", n_blrt, "Bootstrap draws for the likelihood-ratio test between k-1 and k classes."),
        ("MIX_N_STAB", n_stab, "Bootstrap resamples to test class stability."),
        ("MIN_CLASS_N / MIN_ENTROPY / MIN_STABILITY_ARI",
         f"{MIN_CLASS_N} / {MIN_ENTROPY} / {MIN_STABILITY_ARI}",
         "Acceptance criteria for an extra class."),
        ("PRIMARY", f"block={BLOCK_PRIMARY}, variant={VARIANT_PRIMARY}",
         "Main result: baseline block, all probes."),
        ("SEED", SEED, "Random seed; all results are exactly reproducible."),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "explanation"])


GLOSSARY = [
    ("VAS rating", "A rating on a visual analogue slider from 0 (not at all) to 100 (extremely). Here: how sad and how afraid the person feels right now."),
    ("probe", "One moment in the rest block when the slider appears and the person rates Sadness and Fear (about every 20-60 s)."),
    ("series", "All ratings of one emotion by one person in one rest block (~25 probes)."),
    ("block (base / sad / fear)", "The three 20-minute rest periods: base = before any video (no external emotional input at all); sad / fear = after the sad / fear video."),
    ("non-null", "The person reports some emotion (rating above 0) - a statement about the LEVEL of the ratings."),
    ("fluctuation", "The ratings change over the block - a statement about CHANGE, independent of the level."),
    ("lapse rate", "The share of non-zero ratings a person with no emotion could still produce by accident (mis-press, unmoved cursor). Tested at 5%, 10%, 20%."),
    ("exact binomial test", "Asks whether k non-zero ratings out of n are more than the lapse rate would produce by chance."),
    ("flat series", "All ratings in the series are identical (SD = 0)."),
    ("SD (standard deviation)", "Typical distance of the ratings from their own mean, in rating points."),
    ("relative variability index (rvi_sd)", "SD divided by the largest SD that is possible given the series' mean on a 0-100 scale. Near the floor (e.g. mean 2) little variation is possible; this index corrects for that (Mestdagh et al., 2018)."),
    ("rating noise / noise SD", "Even with an unchanged feeling, people do not give exactly the same number each time. Studies that ask the same question twice within seconds find this noise is about 10 points on a 0-100 scale (Dejonckheere et al., 2022). We test 5, 7.5, 10 and 12.5."),
    ("Monte Carlo null test", "We simulate thousands of rating series from a person whose feeling never changes (constant level + rating noise, rounded to the slider steps and limited to 0-100) and ask how often such a person shows as much variation as the real one. That share is the p-value."),
    ("latent SD", "Estimated variation of the feeling itself after removing rating noise: sqrt(SD_observed^2 - noise_SD^2), in rating points."),
    ("bootstrap CI", "A 95% confidence interval made by re-drawing the person's own ratings thousands of times."),
    ("smallest resolvable change (ROPE)", "One slider step (5 points): the smallest change the instrument can show."),
    ("verdict: fluctuating", "Even the low end of the latent-SD interval is above one slider step: the feeling varies beyond noise and beyond the instrument's resolution."),
    ("verdict: flat", "Even the high end of the OBSERVED SD interval is below one slider step: no relevant variation, even counting all variation as feeling."),
    ("verdict: inconclusive", "Neither can be claimed with this person's data - typically floor series with rare blips or variation close to the noise level."),
    ("difference test (dissociation)", "At each probe, Sadness minus Fear. Anything that pushes both ratings together (attention, response habits, general bad mood) cancels out. If this difference still varies more than rating noise, the two emotions moved separately - which response noise cannot explain."),
    ("asymmetry test", "Do Sadness and Fear vary by different amounts at the same probes? Generic rating noise (same slider, same hand, same moment) would make both scales wobble equally. If one scale moves while the other stays put, that cannot be generic noise. Needs no assumed noise size: each person's other scale is their own noise reference. Tested by randomly swapping the two ratings within probes."),
    ("more_variable_scale", "Which emotion varied more at the same probes (shown for significant asymmetry)."),
    ("eligible (dissociation)", "Both scales have at most 20% of ratings at 0 or 100. A scale stuck at 0 cannot show shared movement, so the test would be unfair."),
    ("stable off-floor reference", "One scale is nearly constant (SD at most 2.5) at a level between 10 and 90 - direct proof that this person CAN give stable ratings."),
    ("Bayesian prevalence", "From how many people individually pass a test, estimates the share of the whole population in which the effect is present, with uncertainty (Ince et al., 2021, eLife)."),
    ("prev_map", "Most likely population share."),
    ("prev_lower95", "With 95% probability, the population share is at least this value."),
    ("prev_hpdi_low / high", "95% most-credible range of the population share."),
    ("p_prev_gt_half", "Probability that more than half of the population shows the effect."),
    ("Wilcoxon signed-rank test", "Group test of whether, across people, values are typically above a reference (e.g. observed SD above the noise-only SD). Robust to outliers; no normality assumed."),
    ("rank-biserial r", "Effect size of the Wilcoxon test from -1 to 1; 1 = every person above the reference."),
    ("Brown-Forsythe test", "Tests whether people differ in how much their ratings vary (H1: some vary more than others)."),
    ("FDR (Benjamini-Hochberg)", "Correction for testing the six block x emotion cells: p_fdr controls the expected share of false positives among the significant results."),
    ("variant: all", "All probes of the block (main analysis)."),
    ("variant: drop_first", "First probe of each block removed, in case the first rating is a settling-in reaction."),
    ("variant: detrended", "A straight-line trend over the block removed, to show that variation is not only a slow drift (fatigue, boredom)."),
    ("Gaussian mixture model", "Tests whether people fall into distinct groups ('emotion styles') or form one continuum, based on how much and how often they report emotion."),
    ("BIC", "Model-fit score that penalises extra groups; lower is better."),
    ("bootstrap LRT (blrt_p)", "Tests whether k groups describe the data better than k-1 groups (Nylund et al., 2007)."),
    ("entropy", "How clearly people are assigned to groups (0-1; above 0.8 = clear)."),
    ("stability (ARI)", "Whether the same groups re-appear when the sample is re-drawn (1 = identical)."),
    ("k_selected", "Largest number of groups passing all criteria (LRT, group size >= 5, entropy >= .8, stability >= .7). k = 1 means a continuum, not types."),
]


# ===================================================================
# MAIN
# ===================================================================

def main(args):
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_sim, n_boot = args.n_sim, args.n_boot
    n_blrt, n_stab = args.n_blrt, args.n_stab

    step(f"STEP 0 - load within-rest VAS probes (subjects < {SUBJ_MAX})")
    probes, faudit = load_probes(args.perblock_dir, SUBJ_MAX)
    print(f"  files: {int((faudit.status == 'loaded').sum())} loaded, "
          f"{int((faudit.status != 'loaded').sum())} skipped")
    step_size, on_grid, out_range = check_scale(probes)
    series, saudit = build_series(probes)
    pairs, paudit = pair_emotions(probes)
    subj = sorted({k[0] for k in series})
    print(f"  subjects analysed: {len(subj)}  ({', '.join(f'{s:03d}' for s in subj)})")
    print(f"  series analysed : {len(series)} of {len(saudit)} "
          f"(excluded: {int((~saudit.included).sum())}, < {MIN_PROBES} probes)")
    print(f"  Sadness-Fear pairs: {len(pairs)} subject x block "
          f"(pairing: {paudit['pairing'].value_counts().to_dict()})")
    print(f"  time source: {saudit['time_source'].value_counts().to_dict()}")
    if not series:
        sys.exit("ERROR: no series left to analyse")
    cc = crosscheck_battery(series, args.analysis_dir)
    faudit.to_csv(out / "00_file_audit.csv", index=False)
    saudit.to_csv(out / "00_series_audit.csv", index=False)
    paudit.to_csv(out / "00_pair_audit.csv", index=False)
    if len(cc):
        cc.to_csv(out / "00_battery_crosscheck.csv", index=False)
    config_table(step_size, on_grid, n_sim, n_boot, n_blrt, n_stab).to_csv(
        out / "00_run_config.csv", index=False)

    step("STEP 1 - descriptives per series")
    desc = run_descriptives(series)
    desc.to_csv(out / "01_series_descriptives.csv", index=False)
    da = desc[desc.variant == "all"]
    print(f"  flat series (SD = 0): {int(da.is_flat.sum())} of {len(da)}")
    print(f"  ratings exactly at the cursor start ({CURSOR_START:g}): "
          f"{int(da.n_at_cursor_start.sum())} of {int(da.n.sum())} probes")

    step("STEP 2 - non-null test per series (exact binomial vs lapse rate)")
    nonnull = run_nonnull(series)
    nonnull.to_csv(out / "02_nonnull_per_series.csv", index=False)

    step(f"STEP 3 - fluctuation test per series ({n_sim} null draws, {n_boot} bootstrap)")
    fluct = run_fluctuation(series, step_size, n_sim, n_boot)
    fluct.to_csv(out / "03_fluctuation_per_series.csv", index=False)

    step("STEP 4 - Sadness/Fear dissociation test per subject x block")
    dissoc = run_dissociation(pairs, step_size, n_sim)
    dissoc.to_csv(out / "04_dissociation_per_subject_block.csv", index=False)

    step("STEP 5 - population level (prevalence, group tests, FDR)")
    G, P, C = group_level(series, desc, nonnull, fluct, dissoc)
    G.to_csv(out / "05_group_tests.csv", index=False)
    P.to_csv(out / "06_prevalence.csv", index=False)
    C.to_csv(out / "07_counts.csv", index=False)

    step("STEP 6 - typology (Gaussian mixture, 1..%d classes)" % MIX_K_MAX)
    sel, asg = run_mixture(desc, n_blrt, n_stab)
    if len(sel):
        sel.to_csv(out / "08_mixture_model_selection.csv", index=False)
        asg.to_csv(out / "09_mixture_assignments.csv", index=False)
    pd.DataFrame(GLOSSARY, columns=["term", "plain_explanation"]).to_csv(
        out / "10_glossary.csv", index=False)

    step("SUMMARY - primary settings (baseline block, all probes)")
    for _, r in P[P.is_primary].iterrows():
        par = (f"{r.parameter}={r.parameter_value:g}" if r.parameter
               else "(no noise SD)")
        print(f"  {r.analysis:12s} {r.emotion:12s} {par:15s}: {r.k:2d}/{r.n:2d} individually "
              f"significant -> prevalence {r.prev_map:.2f} "
              f"[95% HPDI {r.prev_hpdi_low:.2f}-{r.prev_hpdi_high:.2f}], "
              f"at least {r.prev_lower95:.2f} with 95% probability")
    vc = C[(C.analysis == "fluctuation") & (C.block_role == BLOCK_PRIMARY)
           & (C.variant == VARIANT_PRIMARY) & (C.noise_sd == NOISE_SD_PRIMARY)]
    for _, r in vc.iterrows():
        print(f"  verdicts {r.emotion:8s}: fluctuating {int(r.n_fluctuating)}, "
              f"flat {int(r.n_flat)}, inconclusive {int(r.n_inconclusive)} (n={int(r.n)})")
    print(f"\nAll CSVs in {out}   ({time.time() - t0:.0f} s)")


# ===================================================================
# SELF-TEST - verifies the statistical machinery on synthetic data
# ===================================================================

def selftest():
    results = []

    def check(name, cond, detail=""):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    rng = np.random.default_rng(1)

    # 1 max SD given mean: compare with exhaustive vertex enumeration
    import itertools
    worst = 0.0
    for _ in range(300):
        n = int(rng.integers(2, 7))
        m = float(rng.uniform(0, 100))
        best = 0.0
        for free in range(n):
            for combo in itertools.product((0.0, 100.0), repeat=n - 1):
                x = list(combo)
                rest = n * m - sum(x)
                if -1e-9 <= rest <= 100 + 1e-9:
                    x.insert(free, rest)
                    best = max(best, float(np.std(x, ddof=1)))
        worst = max(worst, abs(best - max_sd_given_mean(m, n)))
    check("max SD given mean = exhaustive search", worst < 1e-8, f"(max diff {worst:.1e})")

    # 2 detrending removes a pure line and keeps the mean
    t = np.sort(rng.uniform(0, 1200, 25))
    x = 30 + 0.02 * t + rng.normal(0, 3, 25)
    dtr = detrend_rows(x[None, :], t)[0]
    slope_after = np.polyfit(t, dtr, 1)[0]
    check("detrend: slope removed, mean kept",
          abs(slope_after) < 1e-12 and abs(dtr.mean() - x.mean()) < 1e-9)

    # 3 Monte Carlo null calibration (false positives under H0)
    def fpr(mu_fn, sigma=10.0, reps=2000, n=25, n_sim=2000, specific=0.0):
        hits = 0
        r = np.random.default_rng(11)
        for _ in range(reps):
            mu = mu_fn(r)
            lat = mu + specific * r.standard_normal(n)
            y = quantize_clip(lat + sigma * r.standard_normal(n), SLIDER_STEP)
            yv, tv, post = prepare_variant(y, np.arange(n, dtype=float), "all")
            _, p, _ = noise_null_test(yv, tv, post, sigma, n_sim, r, SLIDER_STEP)
            hits += p < ALPHA
        return hits / reps
    se = np.sqrt(ALPHA * (1 - ALPHA) / 2000)
    f_mid = fpr(lambda r: r.uniform(35, 65))
    check("H0 mid-scale: false-positive rate ~ alpha",
          abs(f_mid - ALPHA) < 3 * se + 0.005, f"(FPR {f_mid:.3f})")
    f_floor = fpr(lambda r: r.uniform(0, 5))
    check("H0 at the floor: false-positive rate <= alpha (conservative)",
          f_floor <= ALPHA + 3 * se, f"(FPR {f_floor:.3f})")
    pw = fpr(lambda r: r.uniform(35, 65), specific=15.0)
    check("power: latent SD 15 on top of noise 10 is detected", pw > 0.8, f"(power {pw:.3f})")

    # 4 dissociation null calibration and power
    def dis(spec_sd, reps=1500, n=25, sigma=10.0, sig_c=15.0):
        r = np.random.default_rng(22)
        hits = 0
        for _ in range(reps):
            c = sig_c * r.standard_normal(n)
            s = quantize_clip(45 + c + spec_sd * r.standard_normal(n) + sigma * r.standard_normal(n), SLIDER_STEP)
            f = quantize_clip(40 + c + sigma * r.standard_normal(n), SLIDER_STEP)
            sv, tv, post = prepare_variant(s, np.arange(n, dtype=float), "all")
            fv, _, _ = prepare_variant(f, np.arange(n, dtype=float), "all")
            _, p, _, _ = dissoc_null_test(sv, fv, tv, post, sigma, 2000, r, SLIDER_STEP)
            hits += p < ALPHA
        return hits / reps
    d0 = dis(0.0)
    check("dissociation H0 (large shared movement only): FPR <= alpha",
          d0 <= ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / 1500), f"(FPR {d0:.3f})")
    d1 = dis(20.0)
    check("dissociation power: emotion-specific SD 20 detected", d1 > 0.8, f"(power {d1:.3f})")

    # 4b asymmetry test: H0 with AUTOCORRELATED shared movement, and Person A
    r = np.random.default_rng(33)
    hits, reps = 0, 1500
    for _ in range(reps):
        c = np.zeros(25)
        for i in range(1, 25):
            c[i] = 0.8 * c[i - 1] + 9 * r.standard_normal()
        s = quantize_clip(50 + c + 10 * r.standard_normal(25), SLIDER_STEP)
        f = quantize_clip(45 + c + 10 * r.standard_normal(25), SLIDER_STEP)
        hits += asymmetry_signflip_test(s, f, 2000, r)[1] < ALPHA
    fa = hits / reps
    check("asymmetry H0 (autocorrelated shared movement): FPR ~ alpha",
          fa <= ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / reps), f"(FPR {fa:.3f})")
    pa = np.mean([asymmetry_signflip_test(quantize_clip(45 + 15 * r.standard_normal(25), SLIDER_STEP),
                                          np.full(25, 30.0), 2000, r)[1] < ALPHA
                  for _ in range(300)])
    check("asymmetry power: Person A (Sadness SD 15, Fear fixed at 30) detected",
          pa > 0.95, f"(power {pa:.3f})")

    # 5 Bayesian prevalence: grid vs analytic (truncated Beta) results
    ok = True
    for k, n in ((0, 30), (5, 30), (15, 30), (30, 30), (12, 40)):
        pr = bayes_prevalence(k, n)
        B = stats.beta(k + 1, n - k + 1)
        Fa, Fb = B.cdf(ALPHA), B.cdf(PREV_BETA)
        th_lb = B.ppf(Fa + 0.05 * (Fb - Fa))
        lb = (th_lb - ALPHA) / (PREV_BETA - ALPHA)
        mp = max(min((k / n - ALPHA) / (PREV_BETA - ALPHA), 1), 0)
        ok &= abs(pr["prev_lower95"] - lb) < 1e-3 and abs(pr["prev_map"] - mp) < 1e-12
        ok &= pr["prev_hpdi_low"] <= pr["prev_map"] + 1e-4 <= pr["prev_hpdi_high"] + 2e-4
    check("Bayesian prevalence: grid = analytic MAP / lower bound", ok)

    # 6 BH-FDR against a hand-computed example
    adj = bh_fdr([0.01, 0.04, 0.03, 0.005, np.nan])
    check("BH-FDR matches hand calculation",
          np.allclose(adj[:4], [0.02, 0.04, 0.04, 0.02]) and np.isnan(adj[4]))

    # 7 Wilcoxon wrapper: all positive -> r = 1
    w = wilcoxon_greater(np.arange(1, 11, dtype=float))
    check("Wilcoxon wrapper: all above reference -> r = 1, small p",
          abs(w["effect_size"] - 1) < 1e-12 and w["p_value"] < 0.01)

    print(f"\n  {sum(results)}/{len(results)} checks passed")
    return all(results)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="verify the statistics and exit")
    ap.add_argument("--quick", action="store_true", help="fewer simulations (testing only)")
    ap.add_argument("--perblock-dir", default=str(VAS_PERBLOCK))
    ap.add_argument("--analysis-dir", default=str(VAS_ANALYSIS))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()
    a.n_sim, a.n_boot, a.n_blrt, a.n_stab = ((2000, 1000, 49, 50) if a.quick
                                             else (N_SIM, N_BOOT, MIX_N_BLRT, MIX_N_STAB))
    return a


if __name__ == "__main__":
    args = parse_args()
    if args.selftest:
        step("SELF-TEST")
        sys.exit(0 if selftest() else 1)
    main(args)
