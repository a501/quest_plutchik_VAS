# -*- coding: utf-8 -*-
"""
7_vas_first_probe_test.py
=========================
Group-level test of systematic change within the rest blocks: is the
first rating after the video higher (or lower) than the rest of the block?

This complements Test 2 of 5_vas_fluctuation_proof.py. Test 2 asks, per
series, whether the SD exceeds rating noise; with ~25 probes it is weak
against a change that happens at one or two probes. A paired test across
participants pools that change. If the first probe differs from the rest
after the videos but not at baseline, the rating does change within the
block, even where Test 2 does not detect it per series.

PER SERIES (participant x block x emotion, within-rest probes ordered by
elapsed_s, at least MIN_PROBES probes)
    first      rating at the first probe
    rest_mean  mean of all later probes (the first probe excluded, so the
               two sides of the comparison do not share data)
    last       rating at the last probe
    slope      least-squares slope over time, rating points per minute

TESTS PER BLOCK x EMOTION CELL (across participants)
    A  first vs rest_mean   paired t-test, Wilcoxon signed-rank, d_z
    B  first vs last        paired t-test, Wilcoxon, d_z
    C  slope vs 0           one-sample t-test, Wilcoxon
    Holm correction across the 6 cells within each test (A, B, C).
    The baseline cells are the control: no video precedes them.

Note: analyze_behav_vas.py already removes a first probe that is the
training press (50/50, timed out), so "first" is the first real rating.

Outputs (OUT_DIR): first_probe_per_series.csv, first_probe_tests.csv

Usage
    python3 7_vas_first_probe_test.py
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
OUT_DIR = BASE / "analysis/vas_first_probe"
MIN_PROBES = 15            # same as 5_vas_fluctuation_proof.py

COND_ORDER = ["base", "sad", "fear"]
RATING_ORDER = ["Sadness", "Fear"]


# ===================================================================
# DATA
# ===================================================================

def per_series(perblock_dir, subj_max):
    rows = []
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
            y, t = g["rating"].to_numpy(float), g["elapsed_s"].to_numpy(float)
            if y.size < MIN_PROBES:
                continue
            slope = (np.polyfit(t / 60.0, y, 1)[0] if np.ptp(t) > 0 else np.nan)
            rows.append(dict(subject=int(m.group(1)), block_role=b, emotion=e,
                             n=y.size, first=y[0], first_time_s=t[0],
                             rest_mean=y[1:].mean(), last=y[-1],
                             slope_per_min=slope))
    return pd.DataFrame(rows)


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


def paired(x, y):
    diff = x - y
    t = stats.ttest_rel(x, y)
    w = (stats.wilcoxon(diff) if np.any(diff != 0) else None)
    return dict(n=int(diff.size), mean_a=x.mean(), mean_b=y.mean(),
                mean_diff=diff.mean(), sd_diff=diff.std(ddof=1),
                t=t.statistic, df=diff.size - 1, p_t=t.pvalue,
                p_wilcoxon=(w.pvalue if w is not None else 1.0),
                d_z=diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan)


def run_tests(s):
    rows = []
    for b in COND_ORDER:
        for e in RATING_ORDER:
            g = s[(s.block_role == b) & (s.emotion == e)]
            if len(g) < 3:
                continue
            cell = dict(block_role=b, emotion=e)
            rows.append({**cell, "test": "A first vs rest_mean",
                         **paired(g["first"].to_numpy(), g["rest_mean"].to_numpy())})
            rows.append({**cell, "test": "B first vs last",
                         **paired(g["first"].to_numpy(), g["last"].to_numpy())})
            sl = g["slope_per_min"].dropna().to_numpy()
            t = stats.ttest_1samp(sl, 0.0)
            w = stats.wilcoxon(sl) if np.any(sl != 0) else None
            rows.append({**cell, "test": "C slope vs 0", "n": int(sl.size),
                         "mean_a": sl.mean(), "mean_diff": sl.mean(),
                         "sd_diff": sl.std(ddof=1), "t": t.statistic, "df": sl.size - 1,
                         "p_t": t.pvalue, "p_wilcoxon": (w.pvalue if w is not None else 1.0),
                         "d_z": sl.mean() / sl.std(ddof=1) if sl.std(ddof=1) > 0 else np.nan})
    r = pd.DataFrame(rows)
    for test, idx in r.groupby("test").groups.items():
        r.loc[idx, "p_t_holm"] = holm(r.loc[idx, "p_t"])
        r.loc[idx, "p_wilcoxon_holm"] = holm(r.loc[idx, "p_wilcoxon"])
    return r


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = per_series(VAS_PERBLOCK, SUBJ_MAX)
    print(f"series with >= {MIN_PROBES} probes: {len(s)} "
          f"({s.subject.nunique()} participants)")
    s.to_csv(OUT_DIR / "first_probe_per_series.csv", index=False)
    r = run_tests(s)
    r.to_csv(OUT_DIR / "first_probe_tests.csv", index=False)
    show = r[["test", "block_role", "emotion", "n", "mean_a", "mean_b", "mean_diff",
              "t", "df", "p_t", "p_t_holm", "p_wilcoxon_holm", "d_z"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        for test, g in show.groupby("test", sort=False):
            print(f"\n{test}  (mean_a = first or slope, mean_b = comparison)")
            print(g.drop(columns="test").to_string(index=False))
    print(f"\n-> {OUT_DIR}/first_probe_per_series.csv, first_probe_tests.csv")


if __name__ == "__main__":
    main()
