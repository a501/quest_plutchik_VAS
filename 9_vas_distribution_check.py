# -*- coding: utf-8 -*-
"""
9_vas_distribution_check.py
===========================
Which distribution do the per-series statistics follow, and do the
between-condition comparisons change when tests respect it?

The comparisons on the raincloud slides used paired t-tests (and the
pipeline Wilcoxon tests), which ignore the shape of the data:
    series mean, series SD   >= 0, right-skewed  -> Gamma or Weibull?
    share of probes > 0      in [0, 1]          -> Beta? (or binomial)
All three also have point masses that these distributions do not allow:
mean = SD = 0 (all-zero series) and share = 0 or 1.

PART 1 - DISTRIBUTION FIT, per block x emotion cell
    mean, SD: count zeros; fit Gamma, Weibull and lognormal (location 0,
        maximum likelihood) to the positive values; AIC; Kolmogorov-Smirnov
        D with a parametric-bootstrap p-value (parameters are estimated, so
        the plain KS p-value would be too lenient)
    share > 0: count exact 0 and 1; fit Beta (a) to the values strictly
        inside (0, 1) and (b) to all values after squeezing 0 -> SQUEEZE_LO
        and 1 -> SQUEEZE_HI; AIC, KS as above

PART 2 - TESTS THAT RESPECT THE DISTRIBUTION
    The same 9 comparisons as the slides (3 block pairs per scale, Sadness
    vs Fear per block), repeated measures handled by generalised estimating
    equations (GEE, exchangeable correlation within participant, robust
    standard errors):
    mean, SD
      hurdle model, part 1  P(value = 0)      logistic GEE  -> odds ratio
      hurdle model, part 2  value | value > 0 Gamma GEE, log link
                                                -> ratio of means
      one-part model        Tweedie GEE (compound Poisson-Gamma, power 1.5,
                            allows exact zeros), log link -> ratio of means
    share > 0
      binomial: the share is k non-zero probes out of n, so logistic GEE on
      the probes (k ones, n - k zeros per series) -> odds ratio
      Beta regression (logit link, cluster-robust SEs by participant) on the
      share with 0 -> SQUEEZE_LO and 1 -> SQUEEZE_HI -> odds ratio of the
      mean share; sensitivity: Smithson & Verkuilen (2006) squeeze
      (y (N - 1) + 0.5) / N, N = number of series
    For comparison, on the same pairs: paired t-test and Wilcoxon
    signed-rank (participants with both cells). Holm over the 9
    comparisons within each test and measure.

Outputs (OUT_DIR): dist_fit.csv, dist_tests.csv (one row per comparison
and test, with a column saying whether the conclusion at alpha = .05
differs from the paired t-test)

Requires statsmodels (pip install --user statsmodels).

Usage
    python3 9_vas_distribution_check.py
    python3 9_vas_distribution_check.py --desc path/to/01_series_descriptives.csv
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

BASE = Path("/home/ZhangJianfeng_04/emo_study/Behav/analysis")
DESC = BASE / "vas_fluctuation_proof/01_series_descriptives.csv"
OUT_DIR = BASE / "vas_distribution_check"
N_BOOT_KS = 500
TWEEDIE_POWER = 1.5
SQUEEZE_LO, SQUEEZE_HI = 0.001, 0.999   # Beta: share of 0 and 1 moved inside
ALPHA = 0.05
SEED = 20261028

CELLS = [("base", "Sadness"), ("base", "Fear"), ("sad", "Sadness"),
         ("sad", "Fear"), ("fear", "Sadness"), ("fear", "Fear")]
PAIRS = [(0, 2), (0, 4), (2, 4), (1, 3), (1, 5), (3, 5), (0, 1), (2, 3), (4, 5)]
CELL_NAME = [f"{b} {e}" for b, e in CELLS]


# ===================================================================
# PART 1 - DISTRIBUTION FIT
# ===================================================================

def fit_dist(name, x):
    """MLE fit with location 0 (Beta: on (0, 1)); returns params, loglik, AIC."""
    if name == "gamma":
        par = stats.gamma.fit(x, floc=0)
        ll = stats.gamma.logpdf(x, *par).sum()
    elif name == "weibull":
        par = stats.weibull_min.fit(x, floc=0)
        ll = stats.weibull_min.logpdf(x, *par).sum()
    elif name == "lognormal":
        par = stats.lognorm.fit(x, floc=0)
        ll = stats.lognorm.logpdf(x, *par).sum()
    elif name == "beta":
        par = stats.beta.fit(x, floc=0, fscale=1)
        ll = stats.beta.logpdf(x, *par).sum()
    return par, float(ll), float(2 * 2 - 2 * ll)


DIST = {"gamma": stats.gamma, "weibull": stats.weibull_min,
        "lognormal": stats.lognorm, "beta": stats.beta}


def ks_boot(name, x, par, rng):
    """KS statistic and parametric-bootstrap p-value (refit on each sample)."""
    d = DIST[name]
    D = stats.kstest(x, d.cdf, args=par).statistic
    exceed = 0
    for _ in range(N_BOOT_KS):
        xb = d.rvs(*par, size=x.size, random_state=rng)
        xb = xb[(xb > 0) & ((xb < 1) if name == "beta" else True)]
        if xb.size < 5:
            continue
        pb, _, _ = fit_dist(name, xb)
        exceed += stats.kstest(xb, d.cdf, args=pb).statistic >= D
    return float(D), (exceed + 1) / (N_BOOT_KS + 1)


def part1(d, rng):
    rows = []
    for b, e in CELLS:
        g = d[(d.block_role == b) & (d.emotion == e)]
        for meas in ("mean", "sd", "frac_nonzero"):
            x = g[meas].to_numpy(float)
            x = x[np.isfinite(x)]
            if meas == "frac_nonzero":
                cnt = dict(n=x.size, n_zero=int((x == 0).sum()), n_one=int((x == 1).sum()))
                for label, vals in (("share > 0 (inside 0-1 only)", x[(x > 0) & (x < 1)]),
                                    ("share > 0 (0/1 squeezed)",
                                     np.clip(x, SQUEEZE_LO, SQUEEZE_HI))):
                    if vals.size < 8:
                        continue
                    par, ll, aic = fit_dist("beta", vals)
                    D, p = ks_boot("beta", vals, par, rng)
                    rows.append(dict(cell=f"{b} {e}", measure=label, **cnt,
                                     n_fitted=vals.size, distribution="beta", aic=aic,
                                     ks_D=D, ks_p_boot=p, delta_aic=0.0, best_by_aic=True,
                                     params=" ".join(f"{v:.3g}" for v in par)))
                continue
            else:
                pos = x[x > 0]
                base = dict(cell=f"{b} {e}", measure=meas, n=x.size,
                            n_zero=int((x == 0).sum()), n_one=np.nan, n_fitted=pos.size)
            if pos.size < 8:
                continue
            names = ["gamma", "weibull", "lognormal"]
            fits = {}
            for nm in names:
                par, ll, aic = fit_dist(nm, pos)
                D, p = ks_boot(nm, pos, par, rng)
                fits[nm] = aic
                rows.append({**base, "distribution": nm, "aic": aic,
                             "ks_D": D, "ks_p_boot": p,
                             "params": " ".join(f"{v:.3g}" for v in par)})
            best = min(fits, key=fits.get)
            for r in rows[-len(names):]:
                r["delta_aic"] = r["aic"] - fits[best]
                r["best_by_aic"] = r["distribution"] == best
    return pd.DataFrame(rows)


# ===================================================================
# PART 2 - TESTS
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


def gee_contrasts(df, y, family, label, measure):
    """GEE y ~ 0 + cell, exchangeable within subject; Wald tests of the 9
    pairwise cell differences on the link scale (log or logit)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    df = df.copy()
    df["cell"] = pd.Categorical(df["cell"], categories=CELL_NAME)
    present = [c for c in CELL_NAME if (df["cell"] == c).any()]
    m = smf.gee(f"{y} ~ 0 + C(cell, levels=present)", "subject", data=df,
                family=family, cov_struct=sm.cov_struct.Exchangeable())
    try:
        r = m.fit(maxiter=200)
    except Exception as ex:          # noqa: BLE001 - report and continue
        print(f"  ! {label} {measure}: GEE failed ({type(ex).__name__}: {ex})")
        return []
    names = list(r.params.index)
    rows = []
    for i, j in PAIRS:
        a, b = CELL_NAME[i], CELL_NAME[j]
        if a not in present or b not in present:
            continue
        c = np.zeros(len(names))
        c[present.index(a)], c[present.index(b)] = 1, -1
        est = float(c @ r.params.values)
        se = float(np.sqrt(c @ r.cov_params().values @ c))
        z = est / se if se > 0 else np.nan
        rows.append(dict(measure=measure, test=label, comparison=f"{a} vs {b}",
                         estimate=float(np.exp(est)), estimate_name=(
                             "odds ratio" if "logistic" in label or "binomial" in label
                             else "ratio of means"),
                         stat=z, p=float(2 * stats.norm.sf(abs(z)))))
    return rows


def beta_contrasts(df, y, label, measure):
    """Beta regression y ~ 0 + cell (logit link for the mean, constant
    precision), cluster-robust SEs by participant; Wald tests of the 9
    pairwise differences -> odds ratio of the mean share."""
    from statsmodels.othermod.betareg import BetaModel
    df = df.copy()
    present = [c for c in CELL_NAME if (df["cell"] == c).any()]
    df["cell"] = pd.Categorical(df["cell"], categories=present)
    m = BetaModel.from_formula(f"{y} ~ 0 + C(cell)", df)
    try:
        r = m.fit(cov_type="cluster", cov_kwds={"groups": df["subject"].to_numpy()},
                  disp=0, maxiter=500)
    except Exception as ex:          # noqa: BLE001 - report and continue
        print(f"  ! {label} {measure}: Beta regression failed ({type(ex).__name__}: {ex})")
        return []
    k = len(present)                 # mean parameters; the last one is precision
    beta, cov = r.params.values[:k], r.cov_params().values[:k, :k]
    rows = []
    for i, j in PAIRS:
        a, b = CELL_NAME[i], CELL_NAME[j]
        if a not in present or b not in present:
            continue
        c = np.zeros(k)
        c[present.index(a)], c[present.index(b)] = 1, -1
        est, se = float(c @ beta), float(np.sqrt(c @ cov @ c))
        z = est / se if se > 0 else np.nan
        rows.append(dict(measure=measure, test=label, comparison=f"{a} vs {b}",
                         estimate=float(np.exp(est)), estimate_name="odds ratio of mean",
                         stat=z, p=float(2 * stats.norm.sf(abs(z)))))
    return rows


def paired_rows(w, measure):
    rows = []
    for i, j in PAIRS:
        a, b = CELL_NAME[i], CELL_NAME[j]
        if a not in w or b not in w:
            continue
        m = w[a].notna() & w[b].notna()
        x, y = w.loc[m, a], w.loc[m, b]
        t = stats.ttest_rel(x, y)
        diff = x - y
        wil = stats.wilcoxon(diff) if np.any(diff != 0) else None
        rows.append(dict(measure=measure, test="paired t", comparison=f"{a} vs {b}",
                         estimate=float(diff.mean()), estimate_name="mean difference",
                         stat=float(t.statistic), p=float(t.pvalue)))
        rows.append(dict(measure=measure, test="Wilcoxon", comparison=f"{a} vs {b}",
                         estimate=float(diff.median()), estimate_name="median difference",
                         stat=float(wil.statistic) if wil else np.nan,
                         p=float(wil.pvalue) if wil else 1.0))
    return rows


def part2(d):
    import statsmodels.api as sm
    d = d.assign(cell=d.block_role + " " + d.emotion)
    rows = []
    for meas in ("mean", "sd"):
        w = d.pivot_table(index="subject", columns="cell", values=meas)
        rows += paired_rows(w, meas)
        z = d.assign(is_zero=(d[meas] == 0).astype(float))
        rows += gee_contrasts(z, "is_zero", sm.families.Binomial(),
                              "hurdle part 1: logistic P(zero)", meas)
        pos = d[d[meas] > 0]
        rows += gee_contrasts(pos, meas, sm.families.Gamma(sm.families.links.Log()),
                              "hurdle part 2: Gamma | > 0", meas)
        rows += gee_contrasts(d, meas,
                              sm.families.Tweedie(var_power=TWEEDIE_POWER,
                                                  link=sm.families.links.Log()),
                              "Tweedie (zeros included)", meas)
    w = d.pivot_table(index="subject", columns="cell", values="frac_nonzero")
    rows += paired_rows(w, "share > 0")
    k = np.rint(d["frac_nonzero"] * d["n"]).astype(int)
    probes = pd.DataFrame({
        "subject": np.repeat(d["subject"].to_numpy(), d["n"].to_numpy()),
        "cell": np.repeat(d["cell"].to_numpy(), d["n"].to_numpy()),
        "nonzero": np.concatenate([np.r_[np.ones(kk), np.zeros(nn - kk)]
                                   for kk, nn in zip(k, d["n"].astype(int))])})
    rows += gee_contrasts(probes, "nonzero", sm.families.Binomial(),
                          "binomial (logistic GEE on probes)", "share > 0")
    n_ser = len(d)
    sh = d.assign(share_sq=d["frac_nonzero"].clip(SQUEEZE_LO, SQUEEZE_HI),
                  share_sv=(d["frac_nonzero"] * (n_ser - 1) + 0.5) / n_ser)
    rows += beta_contrasts(sh, "share_sq",
                           f"Beta regression (0/1 -> {SQUEEZE_LO}/{SQUEEZE_HI})", "share > 0")
    rows += beta_contrasts(sh, "share_sv",
                           "Beta regression (Smithson-Verkuilen squeeze)", "share > 0")
    r = pd.DataFrame(rows)
    for (meas, test), idx in r.groupby(["measure", "test"]).groups.items():
        r.loc[idx, "p_holm"] = holm(r.loc[idx, "p"])
    ref = r[r.test == "paired t"].set_index(["measure", "comparison"])["p_holm"]
    r["sig"] = r["p_holm"] < ALPHA
    r["differs_from_t"] = [
        (row.sig != (ref.get((row.measure, row.comparison), np.nan) < ALPHA))
        if row.test != "paired t" else False for row in r.itertuples()]
    return r


# ===================================================================
# MAIN
# ===================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--desc", default=str(DESC))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.desc)
    if "variant" in d.columns:
        d = d[d["variant"] == "all"]
    rng = np.random.default_rng(SEED)

    print("PART 1 - distribution fit (positive values; share > 0: values in (0, 1))")
    f = part1(d, rng)
    f.to_csv(out / "dist_fit.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200,
                           "display.float_format", "{:.3f}".format):
        print(f[["cell", "measure", "n", "n_zero", "n_one", "n_fitted", "distribution",
                 "delta_aic", "ks_D", "ks_p_boot"]].to_string(index=False))
        best = (f[f.best_by_aic].groupby(["measure", "distribution"]).size()
                .rename("cells best by AIC"))
        print("\n" + best.to_string())

    print("\nPART 2 - tests (Holm over the 9 comparisons per test and measure)")
    r = part2(d)
    r.to_csv(out / "dist_tests.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300,
                           "display.float_format", "{:.3f}".format):
        tab = r.assign(txt=[f"{e:.2f} ({p:.3f}){' *' if s else ''}"
                            for e, p, s in zip(r.estimate, r.p_holm, r.sig)])
        for meas, g in tab.groupby("measure", sort=False):
            print(f"\n{meas}: estimate (p_Holm), * = p_Holm < .05")
            print(g.pivot(index="comparison", columns="test", values="txt").to_string())
        n_diff = r[r.differs_from_t].groupby(["measure", "test"]).size()
        print("\nconclusions that differ from the paired t-test:")
        print(n_diff.to_string() if len(n_diff) else "  none")
    print(f"\n-> {out}/dist_fit.csv, dist_tests.csv")


if __name__ == "__main__":
    main()
