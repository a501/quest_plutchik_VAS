#!/usr/bin/env python3
"""
11_meg_screen_select.py
=======================
Two-step reading of the MEG-behaviour result tables, instead of one FDR
correction over everything:

STEP 1  SCREEN (uncorrected)
        Which neural measures come back again and again? For every network x
        MEG measure (pooled over methods, and per method) count the tests with
        p < ALPHA across all contexts (behavioural variable, block, emotion,
        hypothesis), compare with the number expected by chance, and show in
        which contexts the hits fall - a mix of significant and
        non-significant contexts can show which conditions a measure tracks.
        -> S1_recurrence.csv        one row per network x measure (x method)
           S1_hits_by_context.csv   network x measure rows, one column per
                                    context: signed statistic, '*' if p < ALPHA

STEP 2  SELECTED SET (FDR only here)
        You pick the network x measure pairs (by hand with --select, or the
        top K of the screen with --auto-select K). BH-FDR is then applied
        only to the tests of the selected pairs.
        -> S2_selected_fdr.csv

CAUTION  Step 2 uses the same data that drove the selection, so its q-values
         are optimistic (the selection already favoured small p-values).
         Report the selected results as exploratory, or select with
         something independent of the behaviour link (e.g. reliability of
         the MEG measure, or one block / one half of the subjects) and test
         on the rest. The binomial p in step 1 treats tests as independent;
         they are not (same subjects, correlated measures) - use it to rank,
         not as a test.

INPUT   any of: M1_between_subject.csv, M2_within_subject.csv (4_),
        V2_exploratory.csv (10_vas_meg_networks.py). The table type is detected
        from its columns.

USAGE
  python 11_meg_screen_select.py M2_within_subject.csv
  python 11_meg_screen_select.py M2_within_subject.csv --select "DMN:MSE_costa_m2_AUC,DMN:LZC"
  python 11_meg_screen_select.py M1_between_subject.csv --auto-select 5
  options: --alpha 0.05 --out DIR (default: next to the input file)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# columns per table type: p-value, signed statistic, context columns
TYPES = {
    "M1": dict(key="behav_param", p="p_spearman", stat="spearman_rho",
               context=["behav_condition", "emotion", "meg_block", "behav_param"]),
    "M2": dict(key="beta_meg", p="p_meg", stat="beta_meg", context=["emotion"]),
    "V2": dict(key="hyp", p="p", stat="stat", context=["hyp"]),
}
FEATURE = ["group", "measure"]


def detect(df):
    for name, t in TYPES.items():
        if t["key"] in df.columns and t["p"] in df.columns:
            return name, t
    raise ValueError(f"unknown table; columns: {list(df.columns)}")


def bh(p):
    p = np.asarray(p, float)
    out = np.full(p.size, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size:
        q = p[ok]
        o = np.argsort(q)
        r = q[o] * ok.size / (np.arange(ok.size) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        tmp = np.empty_like(q)
        tmp[o] = np.minimum(r, 1)
        out[ok] = tmp
    return out


def recurrence(d, t, alpha, by):
    rows = []
    for k, g in d.groupby(by):
        g = g[np.isfinite(g[t["p"]])]
        n = len(g)
        if n == 0:
            continue
        sig = g[g[t["p"]] < alpha]
        ctx = pd.Series([" | ".join(map(str, r)) for r in sig[t["context"]].itertuples(index=False)],
                        dtype=object)
        both = np.nan
        if "method" in d.columns and "method" not in by and len(sig):
            per = sig.groupby(t["context"])["method"].nunique()
            both = int((per >= 2).sum())
        rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,)),
                         n_tests=n, n_sig=len(sig), share_sig=len(sig) / n,
                         expected_sig=alpha * n,
                         binom_p_rank_only=stats.binomtest(len(sig), n, alpha,
                                                           alternative="greater").pvalue,
                         n_contexts_with_hit=ctx.nunique(),
                         contexts_sig_in_both_methods=both,
                         share_positive_among_sig=(np.mean(sig[t["stat"]] > 0) if len(sig) else np.nan),
                         min_p=g[t["p"]].min(),
                         contexts_with_hit="; ".join(sorted(ctx.unique()))))
    return pd.DataFrame(rows).sort_values(["n_sig", "min_p"], ascending=[False, True])


def hits_by_context(d, t, alpha):
    d = d.copy()
    d["context"] = [" | ".join(map(str, r)) for r in d[t["context"]].itertuples(index=False)]
    if "method" in d.columns:
        d["context"] = d["context"] + " | " + d["method"]
    d["cell"] = d[t["stat"]].map(lambda v: f"{v:+.2f}") + np.where(d[t["p"]] < alpha, "*", "")
    return d.pivot_table(index=FEATURE, columns="context", values="cell", aggfunc="first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--select", default=None, help='"group:measure,group:measure"')
    ap.add_argument("--auto-select", type=int, default=None, help="top K network x measure pairs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    f = Path(a.table)
    d = pd.read_csv(f)
    name, t = detect(d)
    out = Path(a.out) if a.out else f.parent / f"screen_{f.stem}"
    out.mkdir(parents=True, exist_ok=True)
    n_all = int(np.isfinite(d[t["p"]]).sum())
    print(f"{f.name}: type {name}, {n_all} tests, p column {t['p']}, contexts {t['context']}")
    print(f"  uncorrected p < {a.alpha}: {int((d[t['p']] < a.alpha).sum())} "
          f"(expected by chance {a.alpha * n_all:.0f})")

    # ---- step 1 ----
    R = recurrence(d, t, a.alpha, FEATURE)
    R["level"] = "pooled over methods"
    parts = [R]
    if "method" in d.columns:
        Rm = recurrence(d, t, a.alpha, FEATURE + ["method"])
        Rm["level"] = "per method"
        parts.append(Rm)
    pd.concat(parts, ignore_index=True).to_csv(out / "S1_recurrence.csv", index=False)
    hits_by_context(d, t, a.alpha).to_csv(out / "S1_hits_by_context.csv")
    show = R.head(15)[FEATURE + ["n_tests", "n_sig", "expected_sig", "n_contexts_with_hit",
                                 "share_positive_among_sig", "min_p"]]
    print("\nmost recurrent network x measure pairs (pooled over methods):")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # ---- step 2 ----
    sel = None
    if a.select:
        sel = [tuple(s.strip().split(":", 1)) for s in a.select.split(",") if s.strip()]
        src = "by hand"
    elif a.auto_select:
        sel = list(R.head(a.auto_select)[FEATURE].itertuples(index=False, name=None))
        src = f"top {a.auto_select} of the screen (same data: q-values optimistic)"
    if sel:
        S = d.merge(pd.DataFrame(sel, columns=FEATURE), on=FEATURE, how="inner")
        missing = set(sel) - set(S[FEATURE].itertuples(index=False, name=None))
        if missing:
            print(f"  !! not in the table: {sorted(missing)}")
        S["q_selected_set"] = bh(S[t["p"]].values)
        S["selection"] = src
        S["n_tests_in_fdr"] = int(np.isfinite(S[t["p"]]).sum())
        S.sort_values(t["p"]).to_csv(out / "S2_selected_fdr.csv", index=False)
        print(f"\nselected {len(sel)} pairs ({src}): FDR over {S['n_tests_in_fdr'].iloc[0]} tests "
              f"instead of {n_all}; q < .05: {int((S['q_selected_set'] < .05).sum())}")
    print(f"\noutputs in {out}")


if __name__ == "__main__":
    main()
