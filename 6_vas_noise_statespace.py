# -*- coding: utf-8 -*-
"""
6_vas_noise_statespace.py
=========================
Estimate the rating noise sigma of the within-rest VAS ratings from the
data, instead of assuming it (5_vas_fluctuation_proof.py uses sigma = 10
from the literature).

MODEL (per series = participant x block x emotion)
    rating(t) = mu + x(t) + e(t)
    e(t)  ~ N(0, sigma^2), independent across probes        rating noise
    x(t)  latent feeling, Ornstein-Uhlenbeck (continuous-time AR(1)):
          corr(x(t), x(t')) = exp(-|t - t'| / tau), SD(x) = s
So the ratings of one series are multivariate normal with covariance
    s^2 * exp(-|t_j - t_k| / tau) + sigma^2 * I
Noise and feeling are told apart by their time structure: noise is
uncorrelated from probe to probe, the feeling is correlated over tau
seconds. Irregular probe times are handled exactly (elapsed_s).

    sigma, tau : shared by all series of a fit (pooled)
    s          : per series (people differ in how much they fluctuate)
    mu         : per series, removed by REML (restricted likelihood)

FIT
    Profile likelihood on a grid of (sigma, tau). For every grid point each
    series' s is maximised separately on a fine grid.
    95% CI of sigma: profile likelihood ratio, chi2(1) cut-off 3.84.
    Fits: all eligible series pooled, and each block x emotion cell.

IMPORTANT LIMITS (printed with the result)
  - The normal model ignores the 0/100 bounds, so only series with at most
    MAX_BOUND_FRAC of ratings at 0 or 100 are used (same rule as Test 3).
  - Rounding to the 5-point grid adds variance 5^2/12 = 2.08: the
    estimated sigma includes it. sigma_noise_only = sqrt(sigma^2 - 2.08)
    is reported too; the null of script 5 rounds after adding sigma, so
    sigma_noise_only is the value to compare with NOISE_SD_GRID there.
  - If the feeling changes faster than the probe interval (tau well below
    ~45 s) it looks like noise and sigma is overestimated; the profile
    over tau shows this (flat or maximal at the smallest tau).
  - Noise shared by Sadness and Fear is part of sigma here (each scale is
    modelled on its own).

Usage
    python3 6_vas_noise_statespace.py               # real data
    python3 6_vas_noise_statespace.py --selftest    # recover a known sigma
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ===================================================================
# CONFIG
# ===================================================================

SUBJ_MAX = 200
BASE = Path("/home/ZhangJianfeng_04/emo_study/Behav")
VAS_PERBLOCK = BASE / "analysis/behav_per_block"
OUT_DIR = BASE / "analysis/vas_noise_statespace"

MIN_PROBES = 15            # same as 5_vas_fluctuation_proof.py
MAX_BOUND_FRAC = 0.20      # share of ratings at 0 or 100 allowed per series
SLIDER_STEP = 5.0
SIGMA_GRID = np.arange(0.5, 20.001, 0.1)                # rating points
TAU_GRID = np.exp(np.linspace(np.log(10), np.log(3000), 25))  # seconds
S_GRID = np.concatenate([[0.0], np.exp(np.linspace(np.log(0.2), np.log(60), 150))])
                           # per-series SD of the feeling, searched on this grid
CHI2_95 = 3.841

COND_ORDER = ["base", "sad", "fear"]
RATING_ORDER = ["Sadness", "Fear"]


# ===================================================================
# DATA
# ===================================================================

def load_series(perblock_dir, subj_max):
    """dict (subject, block, emotion) -> (t, y) of within-rest probes."""
    out = {}
    for f in sorted(Path(perblock_dir).glob("*.csv")):
        m = re.match(r"dyn(\d+)_", f.name)
        if not m or int(m.group(1)) >= subj_max:
            continue
        d = pd.read_csv(f)
        d.columns = d.columns.str.strip().str.lower()
        if "context" in d.columns:
            d = d[d["context"].astype(str).str.strip() == "within_rest"]
        for (b, e), g in d.groupby([d["block_role"].str.strip().str.lower(),
                                    d["emotion"].str.strip().str.capitalize()]):
            if b not in COND_ORDER or e not in RATING_ORDER:
                continue
            g = g.assign(rating=pd.to_numeric(g["rating"], errors="coerce"),
                         elapsed_s=pd.to_numeric(g["elapsed_s"], errors="coerce"))
            g = g.dropna(subset=["rating", "elapsed_s"]).sort_values("elapsed_s")
            out[(int(m.group(1)), b, e)] = (g["elapsed_s"].to_numpy(float),
                                            g["rating"].to_numpy(float))
    return out


def eligible(y):
    return (y.size >= MIN_PROBES
            and np.mean((y <= 0) | (y >= 100)) <= MAX_BOUND_FRAC
            and np.std(y) > 0)


# ===================================================================
# LIKELIHOOD
# ===================================================================

def series_profile(y, R):
    """REML log-likelihood of one series on SIGMA_GRID, maximised over the
    SD of the feeling s (on S_GRID). mu is removed by REML, which avoids the
    downward bias of variance estimates that plain ML has when each short
    series gets its own mean.

    With R = Q diag(lam) Q', the covariance s^2 R + sigma^2 I has the same
    eigenvectors and eigenvalues D = s^2 lam + sigma^2, so every (sigma, s)
    costs O(n) after one eigendecomposition."""
    n = y.size
    lam, Q = np.linalg.eigh(R)
    lam = np.clip(lam, 0.0, None)
    yt, ot = Q.T @ y, Q.T @ np.ones(n)
    D = (S_GRID[None, :, None] ** 2 * lam[None, None, :]
         + SIGMA_GRID[:, None, None] ** 2)                    # sigma x s x n
    a = np.sum(ot ** 2 / D, axis=2)                           # 1' C^-1 1
    b = np.sum(ot * yt / D, axis=2)                           # 1' C^-1 y
    c = np.sum(yt ** 2 / D, axis=2)                           # y' C^-1 y
    ll = -0.5 * ((n - 1) * np.log(2 * np.pi) + np.sum(np.log(D), axis=2)
                 + np.log(a) + c - b * b / a)
    return ll.max(axis=1)


def profile(series):
    """Profile log-likelihood over SIGMA_GRID x TAU_GRID (sum over series)."""
    dt = [np.abs(t[:, None] - t[None, :]) for t, _ in series]
    ll = np.zeros((SIGMA_GRID.size, TAU_GRID.size))
    for j, tau in enumerate(TAU_GRID):
        for (_, y), d in zip(series, dt):
            ll[:, j] += series_profile(y, np.exp(-d / tau))
    return ll


def ci_from_profile(x, prof, cut):
    """Interval where the profile is above cut, ends linearly interpolated."""
    above = np.flatnonzero(prof >= cut)
    lo_i, hi_i = above.min(), above.max()
    lo = x[lo_i] if lo_i == 0 else np.interp(cut, [prof[lo_i - 1], prof[lo_i]],
                                             [x[lo_i - 1], x[lo_i]])
    hi = x[hi_i] if hi_i == x.size - 1 else np.interp(cut, [prof[hi_i + 1], prof[hi_i]],
                                                      [x[hi_i + 1], x[hi_i]])
    return float(lo), float(hi)


def summarise(ll, label, n_series):
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    prof_sigma = ll.max(axis=1)                   # maximised over tau
    lo, hi = ci_from_profile(SIGMA_GRID, prof_sigma, ll.max() - CHI2_95 / 2)
    sig = SIGMA_GRID[i]
    noise_only = np.sqrt(max(sig ** 2 - SLIDER_STEP ** 2 / 12, 0.0))
    return dict(fit=label, n_series=n_series,
                sigma_hat=round(float(sig), 2), sigma_ci_low=round(lo, 2),
                sigma_ci_high=round(hi, 2),
                sigma_noise_only=round(float(noise_only), 2),
                tau_hat_s=round(float(TAU_GRID[j]), 1),
                tau_at_grid_edge=bool(j in (0, TAU_GRID.size - 1)),
                sigma_at_grid_edge=bool(i in (0, SIGMA_GRID.size - 1)),
                max_loglik=round(float(ll.max()), 2))


def run(series_dict, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    use = {k: v for k, v in series_dict.items() if eligible(v[1])}
    print(f"series loaded: {len(series_dict)}, eligible (>= {MIN_PROBES} probes, "
          f"<= {MAX_BOUND_FRAC:.0%} at 0/100): {len(use)}")
    fits = [("all", list(use.values()))]
    for b in COND_ORDER:
        for e in RATING_ORDER:
            fits.append((f"{b} {e}", [v for k, v in use.items() if k[1:] == (b, e)]))
    rows, prof_rows = [], []
    for label, ser in fits:
        if len(ser) < 5:
            print(f"  {label:14s}: only {len(ser)} series - skipped")
            continue
        ll = profile(ser)
        r = summarise(ll, label, len(ser))
        rows.append(r)
        for i, sig in enumerate(SIGMA_GRID):
            prof_rows.append(dict(fit=label, sigma=sig, profile_loglik=ll[i].max(),
                                  tau_at_max=TAU_GRID[int(np.argmax(ll[i]))]))
        print(f"  {label:14s}: n = {r['n_series']:3d}  sigma = {r['sigma_hat']:4.1f} "
              f"[{r['sigma_ci_low']:.1f}, {r['sigma_ci_high']:.1f}]  "
              f"noise only = {r['sigma_noise_only']:4.1f}  tau = {r['tau_hat_s']:.0f} s"
              + ("  (tau at grid edge)" if r["tau_at_grid_edge"] else ""))
    pd.DataFrame(rows).to_csv(out_dir / "sigma_estimates.csv", index=False)
    pd.DataFrame(prof_rows).to_csv(out_dir / "sigma_profile.csv", index=False)
    print(f"\n-> {out_dir}/sigma_estimates.csv, sigma_profile.csv")
    return pd.DataFrame(rows)


# ===================================================================
# SELF-TEST
# ===================================================================

def selftest():
    """Simulate series with known sigma and check that it is recovered."""
    rng = np.random.default_rng(1)
    true_sigma, true_tau = 6.0, 300.0
    sim = {}
    for i in range(80):
        t = np.cumsum(rng.uniform(20, 60, 25))
        s = rng.uniform(4, 15)
        R = np.exp(-np.abs(t[:, None] - t[None, :]) / true_tau)
        x = np.linalg.cholesky(s * s * R + 1e-9 * np.eye(25)) @ rng.standard_normal(25)
        y = 50 + x + true_sigma * rng.standard_normal(25)
        y = np.clip(np.round(y / SLIDER_STEP) * SLIDER_STEP, 0, 100)
        sim[(i, "base", "Sadness")] = (t, y)
    print(f"SELF-TEST: 80 series, true sigma = {true_sigma} (+ rounding), "
          f"true tau = {true_tau:.0f} s")
    out = Path("selftest_noise_statespace")
    r = run(sim, out).iloc[0]
    expect = np.sqrt(true_sigma ** 2 + SLIDER_STEP ** 2 / 12)
    ok = r.sigma_ci_low <= expect <= r.sigma_ci_high
    print(f"expected sigma incl. rounding = {expect:.2f}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    run(load_series(VAS_PERBLOCK, SUBJ_MAX), OUT_DIR)
