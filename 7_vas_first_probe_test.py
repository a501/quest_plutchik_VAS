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
    slope_wo_first  the same slope without the first probe
    block_num  position of the block in the session (1 = baseline, always
               first; 2/3 = the two video blocks, counterbalanced)

TESTS PER BLOCK x EMOTION CELL (across participants)
    A  first vs rest_mean   paired t-test, Wilcoxon signed-rank, d_z
    B  first vs last        paired t-test, Wilcoxon, d_z
    C  slope vs 0           one-sample t-test, Wilcoxon
    D  slope_wo_first vs 0  as C, without the first probe
    Holm correction across the 6 cells within each test (A-D).

CONTROLS FOR THE START-OF-REST EFFECT (first rating raised in every block
that starts a rest period, also at baseline, where no video precedes it)
    E  within-block control: induced minus non-induced scale in the same
       video block (Sadness - Fear after sad, Fear - Sadness after fear),
       for d = first - rest_mean and for both slopes. Both scales share the
       block, its position and the start of the rest; only the induced
       emotion differs. Paired t-test, Wilcoxon.
    F  video vs baseline: induced scale in the video block minus the same
       scale at baseline (paired). Confounded with block order, because
       baseline is always the first block.
    G  block position: induced scale when its video block was 2nd vs 3rd
       (between participants, Welch t). Tests whether the effect depends
       on order; baseline (always 1st) cannot be separated from order.
    H  probe profile: probes 1..PROFILE_K each vs the late mean of the
       block; shows how many early probes are raised (e.g. at baseline),
       i.e. how many to drop if early baseline ratings are treated as
       familiarisation rather than emotion.

Note: analyze_behav_vas.py already removes a first probe that is the
training press (50/50, timed out), so "first" is the first real rating.

Outputs (OUT_DIR): first_probe_per_series.csv, first_probe_tests.csv,
first_probe_controls.csv, first_probe_profile.csv

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
PROFILE_K = 5              # probe profile: probes 1..PROFILE_K, each compared
PROFILE_LATE_FROM = 7      # with the mean of probes PROFILE_LATE_FROM..end

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
            num = (pd.to_numeric(g["block_num"], errors="coerce").dropna()
                   if "block_num" in g.columns else pd.Series(dtype=float))
            rows.append(dict(subject=int(m.group(1)), block_role=b, emotion=e,
                             block_num=(int(num.iloc[0]) if num.size else np.nan),
                             n=y.size, first=y[0], first_time_s=t[0],
                             rest_mean=y[1:].mean(), last=y[-1],
                             slope_per_min=slope_per_min(t, y),
                             slope_wo_first=slope_per_min(t[1:], y[1:]),
                             late_mean=y[PROFILE_LATE_FROM - 1:].mean(),
                             **{f"probe{k}": y[k - 1] for k in range(1, PROFILE_K + 1)}))
    return pd.DataFrame(rows)


def slope_per_min(t, y):
    return np.polyfit(t / 60.0, y, 1)[0] if np.ptp(t) > 0 else np.nan


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
            for test, col in (("C slope vs 0", "slope_per_min"),
                              ("D slope_wo_first vs 0", "slope_wo_first")):
                sl = g[col].dropna().to_numpy()
                rows.append({**cell, "test": test,
                             **paired(sl, np.zeros(sl.size)), "mean_b": np.nan})
    r = pd.DataFrame(rows)
    for test, idx in r.groupby("test").groups.items():
        r.loc[idx, "p_t_holm"] = holm(r.loc[idx, "p_t"])
        r.loc[idx, "p_wilcoxon_holm"] = holm(r.loc[idx, "p_wilcoxon"])
    return r


INDUCED = {"sad": ("Sadness", "Fear"), "fear": ("Fear", "Sadness")}
MEASURES = ("first_minus_rest", "slope_per_min", "slope_wo_first")


def run_controls(s):
    """Tests E-G: control the start-of-rest effect (see header)."""
    s = s.assign(first_minus_rest=s["first"] - s["rest_mean"])
    w = s.pivot_table(index="subject", columns=["block_role", "emotion"],
                      values=list(MEASURES))
    rows = []
    for b, (ind, other) in INDUCED.items():
        for meas in MEASURES:
            # E: induced vs non-induced scale, same block
            x, y = w.get((meas, b, ind)), w.get((meas, b, other))
            if x is not None and y is not None:
                m = x.notna() & y.notna()
                rows.append(dict(test="E induced vs other scale, same block", block_role=b,
                                 contrast=f"{ind} - {other}", measure=meas,
                                 **paired(x[m].to_numpy(), y[m].to_numpy())))
            # F: induced scale, video block vs baseline
            y = w.get((meas, "base", ind))
            if x is not None and y is not None:
                m = x.notna() & y.notna()
                rows.append(dict(test="F video block vs baseline (order-confounded)",
                                 block_role=b, contrast=f"{ind}: {b} - base", measure=meas,
                                 **paired(x[m].to_numpy(), y[m].to_numpy())))
            # G: block position 2 vs 3 for the induced scale (between participants)
            g = s[(s.block_role == b) & (s.emotion == ind)]
            p2 = g.loc[g.block_num == 2, meas].dropna().to_numpy()
            p3 = g.loc[g.block_num == 3, meas].dropna().to_numpy()
            if p2.size >= 3 and p3.size >= 3:
                t = stats.ttest_ind(p2, p3, equal_var=False)
                u = stats.mannwhitneyu(p2, p3)
                rows.append(dict(test="G block position 2 vs 3", block_role=b,
                                 contrast=f"{ind}: 2nd - 3rd", measure=meas,
                                 n=int(p2.size + p3.size), mean_a=p2.mean(), mean_b=p3.mean(),
                                 mean_diff=p2.mean() - p3.mean(), t=t.statistic,
                                 df=t.df, p_t=t.pvalue, p_wilcoxon=u.pvalue))
    return pd.DataFrame(rows)


def run_profile(s):
    """H: how many early probes are raised? Probe k vs the late mean
    (probes PROFILE_LATE_FROM..end), paired, per cell, Holm over the 6 cells
    within each k. Decides how many baseline probes a start effect spans."""
    rows = []
    for k in range(1, PROFILE_K + 1):
        for b in COND_ORDER:
            for e in RATING_ORDER:
                g = s[(s.block_role == b) & (s.emotion == e)]
                if len(g) >= 3:
                    rows.append(dict(test=f"H probe {k} vs late mean", probe=k,
                                     block_role=b, emotion=e,
                                     **paired(g[f"probe{k}"].to_numpy(),
                                              g["late_mean"].to_numpy())))
    r = pd.DataFrame(rows)
    for k, idx in r.groupby("probe").groups.items():
        r.loc[idx, "p_t_holm"] = holm(r.loc[idx, "p_t"])
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
    c = run_controls(s)
    c.to_csv(OUT_DIR / "first_probe_controls.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print("\nCONTROLS (p uncorrected; p_wilcoxon = Mann-Whitney for G)")
        print(c[["test", "contrast", "measure", "n", "mean_a", "mean_b", "mean_diff",
                 "t", "df", "p_t", "p_wilcoxon"]].to_string(index=False))
    pr = run_profile(s)
    pr.to_csv(OUT_DIR / "first_probe_profile.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(f"\nPROBE PROFILE: mean rating at probe k minus mean of probes "
              f"{PROFILE_LATE_FROM}..end (p_Holm in brackets)")
        tab = pr.assign(cell=pr.block_role + " " + pr.emotion,
                        txt=[f"{d:+.1f} ({p:.3f})" for d, p in zip(pr.mean_diff, pr.p_t_holm)])
        print(tab.pivot(index="cell", columns="probe", values="txt").to_string())
    print(f"\n-> {OUT_DIR}/first_probe_per_series.csv, first_probe_tests.csv, "
          f"first_probe_controls.csv, first_probe_profile.csv")


if __name__ == "__main__":
    main()
