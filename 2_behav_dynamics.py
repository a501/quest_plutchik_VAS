#!/usr/bin/env python3
"""
2_behav_dynamics.py
===================
All behavioural VAS dynamics analyses that do not need mixed models.
(Mixed models -> 3_behav_models.R; MEG link -> 4_meg_behav_link.py)

Input:  OUT_DIR/vas_probes_long.csv, OUT_DIR/vas_postvideo.csv  (from 1_)
A "series" = one subject x condition (baseline/sad/fear) x emotion
(Sadness/Fear), all ratings incl. timeouts ('valid' from 1_ = rating present;
timeouts are always used), time in MINUTES from rest onset.

A  INDUCTION CHECK  (are the two videos equally specific?)
   Post-video ratings (raw, single ratings) + first in-block rating.
   Tests: target vs non-target per video; cross-contamination asymmetry
   (Fear after sad video vs Sadness after fear video); specificity
   (target - non-target) sad video vs fear video.
   -> A_induction_subject.csv, A_induction_tests.csv

B  RECOVERY CURVES  (affective chronometry)
   Per series: constant, linear, exponential  y = c + A*exp(-t/tau),
   power  y = c + A*(1 + t/1min)^-beta.  AICc per model.
   Group: same 4 models on the pooled data per condition x emotion,
   bootstrap over subjects (tau CI, how often exp beats power).
   -> B_recovery_subject.csv, B_recovery_group.csv

C  VARIOGRAM  (uneven intervals as a built-in experiment)
   All pairs of ratings within a series: semivariance 0.5*(r_j - r_i)^2 vs
   time between them. Done on residuals after removing the recovery trend
   ('detrended') and on the demeaned raw series ('raw', sensitivity).
   Group fits: nugget only / linear (free drift) / OU (regulated: levels off)
   / power.  Per-subject OU fit gives tau_var, sill, nugget.
   NOTE: detrending with a fitted curve removes slow variance, so tau from the
   'detrended' variogram is biased SHORT. Compare with the 'raw' variant; read
   the model comparison (levels off vs keeps rising) more than the tau value.
   -> C_variogram_subject.csv, C_variogram_group_binned.csv,
      C_variogram_group_fits.csv

D  INTERVAL RANDOMNESS  (prerequisite for C and for probe reactivity in 3_)
   Is the fixation interval unrelated to time in block? Is the interval
   sequence the same for everyone (fixed schedule)?
   -> D_interval_subject.csv, D_interval_summary.csv

E  SADNESS-FEAR COUPLING  (differentiation)
   Per subject x condition: Pearson/Spearman on raw ratings AND on
   detrended residuals (shared recovery trend removed).
   -> E_coupling_subject.csv, E_coupling_group.csv

F  CARRY-OVER  (exploratory: confounded with video sequence + fatigue)
   Start level (mean of first 3 probes) of each block depending on what came
   before; end of block 2 vs start of block 3.
   -> F_carryover_subject.csv, F_carryover_tests.csv

G  HIDDEN MARKOV MODEL  on the joint (Sadness, Fear) ratings
   One HMM over all subjects and conditions (states comparable),
   K = 2..5 chosen by BIC. Ratings are de-quantised (uniform jitter within the
   5-point step) so the model cannot create zero-variance states on one value. Per subject x condition: fractional occupancy,
   switches, dwell time.
   -> G_hmm_model_selection.csv, G_hmm_states.csv, G_hmm_transmat.csv,
      G_hmm_subject.csv

H  RATING BEHAVIOUR  (back-and-forth on the VAS; no RT)
   Per series: mean excess presses, reversals, overshoot.
   -> H_rating_behaviour_subject.csv

I  RELIABILITY  odd vs even probes, Spearman-Brown corrected, per parameter
   -> I_reliability.csv

J  MASTER TABLE + REDUNDANCY  (Dejonckheere et al. 2019 check)
   One row per series with all parameters; for each dynamic parameter the
   R^2 explained by the series mean + SD.
   -> J_params_subject.csv, J_redundancy.csv

USAGE   python 2_behav_dynamics.py [--out DIR] [--boot 500] [--skip-hmm]
"""
import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = Path("/home/ZhangJianfeng_04/emo_study/Behav/dynamics_models")
CONDS = ["baseline", "sad", "fear"]
EMOS = ["Sadness", "Fear"]
TARGET = {"sad": "Sadness", "fear": "Fear"}
MIN_VALID = 18                       # same exclusion floor as the RT analysis
TAU_BOUNDS = (0.1, 60.0)             # minutes
VG_BINS_S = [0, 40, 60, 90, 120, 180, 240, 360, 480, 720, 1200]
RNG = np.random.default_rng(42)


# ==========================================================================
# helpers
# ==========================================================================
def aicc(sse, n, k):
    if n - k - 1 <= 0 or sse <= 0:
        return np.nan
    return n * np.log(sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def f_exp(t, c, A, tau):
    return c + A * np.exp(-t / tau)


def f_pow(t, c, A, beta):
    return c + A * (1.0 + t) ** (-beta)


def fit_curves(t, y):
    """Fit const/linear/exp/power. Returns dict of params, AICc, predictions."""
    n = len(y)
    out = {"n": n}
    sse_c = np.sum((y - y.mean()) ** 2)
    out["aicc_const"] = aicc(sse_c, n, 1)
    b, a = np.polyfit(t, y, 1)
    pred_lin = a + b * t
    out.update(lin_slope_per_min=b, aicc_linear=aicc(np.sum((y - pred_lin) ** 2), n, 2))
    out["pred_linear"] = pred_lin
    # exponential: several starts
    best = None
    for tau0 in (1, 5, 15):
        try:
            p, _ = curve_fit(f_exp, t, y, p0=[y[-3:].mean(), y[0] - y[-3:].mean(), tau0],
                             bounds=([-10, -110, TAU_BOUNDS[0]], [110, 110, TAU_BOUNDS[1]]),
                             maxfev=20000)
            sse = np.sum((y - f_exp(t, *p)) ** 2)
            if best is None or sse < best[1]:
                best = (p, sse)
        except Exception:
            pass
    if best:
        c, A, tau = best[0]
        out.update(exp_c=c, exp_A=A, exp_tau_min=tau, exp_half_life_min=tau * np.log(2),
                   exp_tau_at_bound=bool(np.isclose(tau, TAU_BOUNDS, rtol=1e-3).any()),
                   aicc_exp=aicc(best[1], n, 3),
                   exp_r2=1 - best[1] / sse_c if sse_c > 0 else np.nan)
        out["pred_exp"] = f_exp(t, *best[0])
    try:
        p, _ = curve_fit(f_pow, t, y, p0=[y[-3:].mean(), y[0] - y[-3:].mean(), 0.5],
                         bounds=([-10, -110, 0.01], [110, 110, 10]), maxfev=20000)
        out.update(pow_c=p[0], pow_A=p[1], pow_beta=p[2],
                   aicc_power=aicc(np.sum((y - f_pow(t, *p)) ** 2), n, 3))
    except Exception:
        pass
    aics = {m: out.get(f"aicc_{m}", np.nan) for m in ("const", "linear", "exp", "power")}
    aics = {m: v for m, v in aics.items() if np.isfinite(v)}
    out["best_model"] = min(aics, key=aics.get) if aics else ""
    out["d_aicc_exp_minus_power"] = out.get("aicc_exp", np.nan) - out.get("aicc_power", np.nan)
    return out


def detrend(t, y, fit):
    """Residuals after the recovery trend: exp if it fitted and beat linear, else linear."""
    if "pred_exp" in fit and fit.get("aicc_exp", np.inf) < fit.get("aicc_linear", np.inf):
        return y - fit["pred_exp"], "exp"
    return y - fit["pred_linear"], "linear"


def paired_test(a, b, label):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    r = dict(comparison=label, n=int(m.sum()))
    if r["n"] < 3:
        return r
    d = a - b
    t, pt = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(a, b)
    except ValueError:
        w, pw = np.nan, np.nan
    r.update(mean_a=a.mean(), mean_b=b.mean(), mean_diff=d.mean(),
             dz=d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
             t=t, p_t=pt, W=w, p_wilcoxon=pw)
    return r


def two_group_test(a, b, label):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    r = dict(comparison=label, n_a=len(a), n_b=len(b))
    if len(a) < 3 or len(b) < 3:
        return r
    t, pt = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    r.update(mean_a=a.mean(), mean_b=b.mean(), hedges_d=(a.mean() - b.mean()) / sp if sp else np.nan,
             t_welch=t, p_welch=pt, U=u, p_mannwhitney=pu)
    return r


# ==========================================================================
# data
# ==========================================================================
def load(out):
    d = pd.read_csv(out / "vas_probes_long.csv", dtype={"subj": str, "root_subj": str})
    pv = pd.read_csv(out / "vas_postvideo.csv", dtype={"subj": str})
    d["t_min"] = d["elapsed_s"] / 60.0
    d, pv = d[d["subj"].astype(int) < 200], pv[pv["subj"].astype(int) < 200]
    return d, pv


def series_iter(d):
    # Keep ALL probes so probe_idx matches snippet_idx.
    # Timeout (non-valid) ratings become NaN; fitters drop them but spacing is preserved.
    v = d.copy().sort_values("elapsed_s")
    v["rating"] = v["rating"].where(v["valid"], other=np.nan)
    for (s, c, e), g in v.groupby(["subj", "condition", "emotion"]):
        t = g["t_min"].values
        y = g["rating"].values.astype(float)
        finite = np.isfinite(y)
        if finite.sum() < 5:
            continue
        yield s, c, e, t[finite], y[finite], g


# ==========================================================================
# A induction
# ==========================================================================
def section_A(d, pv, out):
    w = (pv[pv["valid"]].assign(col=lambda x: x["video"] + "_" + x["emotion"])
         .pivot_table(index="subj", columns="col", values="rating", aggfunc="first"))
    first = (d[d["valid"]].sort_values("probe_idx").groupby(["subj", "condition", "emotion"])
             ["rating"].first().reset_index())
    first["col"] = "first_" + first["condition"] + "_" + first["emotion"]
    w = w.join(first.pivot(index="subj", columns="col", values="rating"), how="outer")
    w.to_csv(out / "A_induction_subject.csv")

    g = lambda c: w[c].values if c in w else np.full(len(w), np.nan)
    rows = []
    for src, fmt in (("postvideo", "{v}_{e}"), ("first_probe", "first_{v}_{e}")):
        S = {(v, e): g(fmt.format(v=v, e=e)) for v in ("sad", "fear") for e in EMOS}
        for v in ("sad", "fear"):
            tgt, non = TARGET[v], [e for e in EMOS if e != TARGET[v]][0]
            rows.append(dict(source=src, **paired_test(S[(v, tgt)], S[(v, non)],
                                                       f"{v} video: {tgt} vs {non}")))
        rows.append(dict(source=src, **paired_test(
            S[("sad", "Fear")], S[("fear", "Sadness")],
            "cross-contamination: Fear after sad video vs Sadness after fear video")))
        rows.append(dict(source=src, **paired_test(
            S[("sad", "Sadness")] - S[("sad", "Fear")], S[("fear", "Fear")] - S[("fear", "Sadness")],
            "specificity (target-nontarget): sad video vs fear video")))
        rows.append(dict(source=src, **paired_test(
            S[("sad", "Sadness")], S[("fear", "Fear")],
            "target intensity: Sadness after sad vs Fear after fear")))
    pd.DataFrame(rows).to_csv(out / "A_induction_tests.csv", index=False)
    print("A induction check done")


# ==========================================================================
# B recovery + C variogram (share the fits)
# ==========================================================================
def section_BC(d, out, n_boot):
    rec, vg_rows, vg_sub, resid_store = [], [], [], {}
    for s, c, e, t, y, g in series_iter(d):
        base = dict(subj=s, condition=c, emotion=e, n_valid=len(y),
                    mean=y.mean(), sd=y.std(ddof=1) if len(y) > 1 else np.nan,
                    median=np.median(y), n_nonzero=int((y > 0).sum()),
                    frac_zero=float((y == 0).mean()),
                    first=y[0], last=y[-1], start3=y[:3].mean(), end3=y[-3:].mean())
        flat = np.ptp(y) == 0
        base["flat"] = bool(flat)
        base["usable"] = bool(len(y) >= MIN_VALID and not flat)
        if flat or len(y) < 5:
            rec.append(base)
            continue
        fit = fit_curves(t, y)
        res, trend = detrend(t, y, fit)
        # Store only the finite-probe subset so residuals align with probe_idx
        finite_g = g[g["valid"] & g["rating"].notna()]
        resid_store[(s, c, e)] = (t, res, finite_g)
        base.update({k: v for k, v in fit.items() if not k.startswith("pred_")})
        base["detrend_model"] = trend
        base["resid_sd"] = res.std(ddof=1)
        # lag-1 autocorrelation of residuals (inertia, ignores spacing)
        base["resid_ac1"] = np.corrcoef(res[:-1], res[1:])[0, 1] if len(res) > 3 else np.nan
        # variogram
        for variant, z in (("detrended", res), ("raw", y - y.mean())):
            i, j = np.triu_indices(len(t), 1)
            dt = (t[j] - t[i]) * 60.0
            gam = 0.5 * (z[j] - z[i]) ** 2
            b = np.digitize(dt, VG_BINS_S) - 1
            for k in range(len(VG_BINS_S) - 1):
                m = b == k
                if m.sum():
                    vg_sub.append(dict(subj=s, condition=c, emotion=e, variant=variant,
                                       bin_lo_s=VG_BINS_S[k], bin_hi_s=VG_BINS_S[k + 1],
                                       dt_mean_s=dt[m].mean(), gamma=gam[m].mean(),
                                       n_pairs=int(m.sum())))
            if variant == "detrended":
                ou = fit_variogram(dt / 60.0, gam)
                base.update({f"vg_{k}": v for k, v in ou.items()})
        rec.append(base)
    R = pd.DataFrame(rec)
    R.to_csv(out / "B_recovery_subject.csv", index=False)
    V = pd.DataFrame(vg_sub)
    V.to_csv(out / "C_variogram_subject.csv", index=False)
    print("B per-series recovery fits done")

    # ---- B group: pooled fits + bootstrap over subjects ----
    grp = []
    v = d[d["valid"]]
    for c, e in itertools.product(CONDS, EMOS):
        gg = v[(v["condition"] == c) & (v["emotion"] == e)]
        subs = gg["subj"].unique()
        if len(subs) < 3:
            continue
        f = fit_curves(gg["t_min"].values, gg["rating"].values.astype(float))
        row = dict(condition=c, emotion=e, n_subjects=len(subs), n_points=len(gg),
                   **{k: val for k, val in f.items() if not k.startswith("pred_")})
        taus, wins = [], []
        by = {s: x for s, x in gg.groupby("subj")}
        for _ in range(n_boot):
            bs = pd.concat([by[s] for s in RNG.choice(subs, len(subs))])
            fb = fit_curves(bs["t_min"].values, bs["rating"].values.astype(float))
            taus.append(fb.get("exp_tau_min", np.nan))
            wins.append(fb.get("d_aicc_exp_minus_power", np.nan) < 0)
        taus = np.array(taus, float)
        row.update(boot_tau_ci_lo=np.nanpercentile(taus, 2.5), boot_tau_ci_hi=np.nanpercentile(taus, 97.5),
                   boot_frac_exp_beats_power=float(np.mean(wins)), n_boot=n_boot)
        grp.append(row)
    pd.DataFrame(grp).to_csv(out / "B_recovery_group.csv", index=False)
    print("B group recovery fits done")

    # ---- C group ----
    if len(V):
        sb = V.groupby(["subj", "condition", "emotion", "variant", "bin_lo_s", "bin_hi_s"]).agg(
            gamma=("gamma", "mean"), dt=("dt_mean_s", "mean")).reset_index()
        gb = sb.groupby(["condition", "emotion", "variant", "bin_lo_s", "bin_hi_s"]).agg(
            gamma_mean=("gamma", "mean"), gamma_sem=("gamma", "sem"),
            dt_mean_s=("dt", "mean"), n_subjects=("subj", "nunique")).reset_index()
        gb.to_csv(out / "C_variogram_group_binned.csv", index=False)
        fits = []
        for (c, e, var), x in gb.groupby(["condition", "emotion", "variant"]):
            for model, res in fit_variogram_models(x["dt_mean_s"].values / 60.0,
                                                   x["gamma_mean"].values,
                                                   x["n_subjects"].values).items():
                fits.append(dict(condition=c, emotion=e, variant=var, model=model, **res))
        F = pd.DataFrame(fits)
        if len(F):
            F["best"] = F.groupby(["condition", "emotion", "variant"])["aicc"].transform(
                lambda a: a == a.min())
        F.to_csv(out / "C_variogram_group_fits.csv", index=False)
    print("C variogram done")
    return R, resid_store


def fit_variogram(dt_min, gam):
    """Per-series OU variogram: gamma = nugget + sill*(1-exp(-dt/tau))."""
    try:
        p, _ = curve_fit(lambda x, n, s, tau: n + s * (1 - np.exp(-x / tau)), dt_min, gam,
                         p0=[gam.min(), gam.mean(), 3],
                         bounds=([0, 0, TAU_BOUNDS[0]], [np.inf, np.inf, TAU_BOUNDS[1]]),
                         maxfev=20000)
        return dict(nugget=p[0], sill=p[1], tau_min=p[2],
                    tau_at_bound=bool(np.isclose(p[2], TAU_BOUNDS, rtol=1e-3).any()))
    except Exception:
        return {}


def fit_variogram_models(x, y, w):
    models = {
        "nugget_only": (lambda x, n: n + 0 * x, [y.mean()], ([0], [np.inf])),
        "linear_free_drift": (lambda x, n, a: n + a * x, [y.min(), 1], ([0, 0], [np.inf, np.inf])),
        "OU_regulated": (lambda x, n, s, tau: n + s * (1 - np.exp(-x / tau)), [y.min(), y.max(), 3],
                         ([0, 0, TAU_BOUNDS[0]], [np.inf, np.inf, TAU_BOUNDS[1]])),
        "power": (lambda x, n, a, H: n + a * x ** H, [y.min(), 1, 0.5], ([0, 0, 0.01], [np.inf, np.inf, 1.99])),
    }
    sigma = 1 / np.sqrt(np.maximum(w, 1))
    out = {}
    for name, (f, p0, bnd) in models.items():
        try:
            p, _ = curve_fit(f, x, y, p0=p0, bounds=bnd, sigma=sigma, maxfev=20000)
            sse = np.sum(((y - f(x, *p)) / sigma) ** 2)
            out[name] = dict(params=";".join(f"{v:.4g}" for v in p),
                             aicc=aicc(sse, len(y), len(p)), n_bins=len(y))
            if name == "OU_regulated":
                out[name].update(tau_min=p[2], sill=p[1], nugget=p[0])
            if name == "power":
                out[name].update(H_exponent=p[2])
        except Exception as ex:
            out[name] = dict(error=type(ex).__name__)
    return out


# ==========================================================================
# D interval randomness
# ==========================================================================
def section_D(d, out):
    p = d.drop_duplicates(["subj", "block", "probe_idx"]).copy()
    p["interval"] = p["fix_interval_s"].fillna(p["gap_s"])
    p["interval_source"] = np.where(p["fix_interval_s"].notna(), "triggers", "elapsed_gap")
    rows, seqs = [], []
    for (s, c), g in p.groupby(["subj", "condition"]):
        g = g.dropna(subset=["interval"])
        if len(g) < 5:
            continue
        r1 = stats.spearmanr(g["interval"], g["probe_idx"])[0]
        r2 = stats.spearmanr(g["interval"], g["elapsed_s"])[0]
        rows.append(dict(subj=s, condition=c, n=len(g), interval_source=g["interval_source"].iloc[0],
                         mean_interval_s=g["interval"].mean(), sd_interval_s=g["interval"].std(),
                         rho_interval_probeidx=r1, rho_interval_elapsed=r2))
        seqs.append(tuple((g["interval"] / 10).round().astype(int)))
    D = pd.DataFrame(rows)
    D.to_csv(out / "D_interval_subject.csv", index=False)
    summ = dict(n_series=len(seqs), n_unique_sequences=len(set(seqs)))
    if len(D) > 2:
        w = stats.wilcoxon(D["rho_interval_elapsed"].dropna())
        summ.update(mean_rho_interval_elapsed=D["rho_interval_elapsed"].mean(),
                    wilcoxon_p_rho_vs_0=w.pvalue)
    vals = (p["interval"].dropna() / 10).round() * 10
    for k, v in vals.value_counts().sort_index().items():
        summ[f"count_{int(k)}s"] = int(v)
    pd.DataFrame([summ]).to_csv(out / "D_interval_summary.csv", index=False)
    print(f"D interval check done: {summ['n_unique_sequences']} unique sequences in {summ['n_series']} series")


# ==========================================================================
# E coupling
# ==========================================================================
def section_E(d, resid, out):
    # Use all probes; timeout ratings become NaN and dropna removes incomplete pairs only.
    all_d = d.copy()
    all_d["rating"] = all_d["rating"].where(all_d["valid"], other=np.nan)
    w = all_d.pivot_table(index=["subj", "condition", "probe_idx"], columns="emotion",
                          values="rating").dropna().reset_index()
    rows = []
    for (s, c), g in w.groupby(["subj", "condition"]):
        S, F = g["Sadness"].values, g["Fear"].values
        r = dict(subj=s, condition=c, n=len(g), n_nonzero_both=int(((S > 0) & (F > 0)).sum()),
                 sad_flat=bool(np.ptp(S) == 0), fear_flat=bool(np.ptp(F) == 0))
        if len(g) >= 5 and not r["sad_flat"] and not r["fear_flat"]:
            r["pearson_raw"] = stats.pearsonr(S, F)[0]
            r["spearman_raw"] = stats.spearmanr(S, F)[0]
        rs, rf = resid.get((s, c, "Sadness")), resid.get((s, c, "Fear"))
        if rs is not None and rf is not None:
            a = pd.Series(rs[1], index=rs[2]["probe_idx"].values)
            b = pd.Series(rf[1], index=rf[2]["probe_idx"].values)
            j = pd.concat([a, b], axis=1, join="inner").dropna()
            if len(j) >= 5 and j.std().min() > 0:
                r["pearson_resid"] = stats.pearsonr(j[0], j[1])[0]
                r["spearman_resid"] = stats.spearmanr(j[0], j[1])[0]
        rows.append(r)
    E = pd.DataFrame(rows)
    E.to_csv(out / "E_coupling_subject.csv", index=False)
    grp = []
    for col in ("pearson_raw", "pearson_resid", "spearman_raw", "spearman_resid"):
        if col not in E:
            continue
        for c in CONDS:
            z = np.arctanh(E.loc[E["condition"] == c, col].clip(-0.999, 0.999).dropna())
            if len(z) >= 3:
                t, p = stats.ttest_1samp(z, 0)
                grp.append(dict(measure=col, test=f"{c}: mean coupling vs 0", n=len(z),
                                mean_r=np.tanh(z.mean()), t=t, p=p))
        wide = E.pivot(index="subj", columns="condition", values=col)
        for a, b in itertools.combinations(CONDS, 2):
            if a in wide and b in wide:
                za = np.arctanh(wide[a].clip(-0.999, 0.999)).values
                zb = np.arctanh(wide[b].clip(-0.999, 0.999)).values
                grp.append(dict(measure=col, **paired_test(za, zb, f"Fisher z {a} vs {b}")))
    pd.DataFrame(grp).to_csv(out / "E_coupling_group.csv", index=False)
    print("E coupling done")
    return E


# ==========================================================================
# F carry-over
# ==========================================================================
def section_F(R, d, out):
    order = d.drop_duplicates("subj").set_index("subj")
    first_video = (d[d["block"] == 2].drop_duplicates("subj").set_index("subj")["condition"])
    rows = []
    for s, fv in first_video.items():
        r = dict(subj=s, first_video=fv)
        for c in ("sad", "fear"):
            for e in EMOS:
                x = R[(R["subj"] == s) & (R["condition"] == c) & (R["emotion"] == e)]
                if len(x):
                    r[f"{c}_{e}_start3"] = x["start3"].iloc[0]
                    r[f"{c}_{e}_end3"] = x["end3"].iloc[0]
        rows.append(r)
    F = pd.DataFrame(rows)
    F.to_csv(out / "F_carryover_subject.csv", index=False)
    tests = []
    for c in ("sad", "fear"):
        other = "fear" if c == "sad" else "sad"
        for e in EMOS:
            col = f"{c}_{e}_start3"
            if col not in F:
                continue
            a = F.loc[F["first_video"] == other, col].values  # block came 2nd (after other video)
            b = F.loc[F["first_video"] == c, col].values      # block came 1st (after baseline)
            tests.append(two_group_test(a.astype(float), b.astype(float),
                                        f"{c} block, {e} start: after {other} block vs after baseline"))
    for e in EMOS:
        x, y = [], []
        for _, r in F.iterrows():
            fv = r["first_video"]
            sv = "fear" if fv == "sad" else "sad"
            x.append(r.get(f"{fv}_{e}_end3", np.nan))
            y.append(r.get(f"{sv}_{e}_start3", np.nan))
        x, y = np.array(x, float), np.array(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 5:
            rho, p = stats.spearmanr(x[m], y[m])
            tests.append(dict(comparison=f"{e}: end of block 2 vs start of block 3 (spearman)",
                              n_a=int(m.sum()), rho=rho, p=p))
    pd.DataFrame(tests).to_csv(out / "F_carryover_tests.csv", index=False)
    print("F carry-over done")


# ==========================================================================
# G HMM
# ==========================================================================
def section_G(d, out, kmax=5, restarts=10):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("G skipped: pip install hmmlearn")
        return None
    all_d = d.copy()
    all_d["rating"] = all_d["rating"].where(all_d["valid"], other=np.nan)
    w = all_d.pivot_table(index=["subj", "condition", "probe_idx"], columns="emotion",
                          values=["rating", "elapsed_s"]).dropna().reset_index()
    seqs, keys = [], []
    for (s, c), g in w.groupby(["subj", "condition"]):
        g = g.sort_values("probe_idx")
        if len(g) >= 5:
            seqs.append(g)
            keys.append((s, c))
    X = np.vstack([g[[("rating", "Sadness"), ("rating", "Fear")]].values for g in seqs]).astype(float)
    # ratings are on a 5-point grid with many exact zeros: a Gaussian HMM then
    # builds zero-variance states at single grid values. De-quantise with
    # uniform jitter within the grid step (fixed seed) before fitting.
    X = X + np.random.default_rng(0).uniform(-2.5, 2.5, X.shape)
    L = [len(g) for g in seqs]
    sel, models = [], {}
    for K in range(2, kmax + 1):
        best = None
        for r in range(restarts):
            m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=300,
                            min_covar=1.0, random_state=r)
            try:
                m.fit(X, L)
                ll = m.score(X, L)
            except Exception:
                continue
            if best is None or ll > best[1]:
                best = (m, ll)
        if best is None:
            continue
        npar = K * (K - 1) + (K - 1) + 2 * K + 2 * K
        bic = -2 * best[1] + npar * np.log(len(X))
        sel.append(dict(K=K, loglik=best[1], n_params=npar, BIC=bic))
        models[K] = best[0]
    S = pd.DataFrame(sel)
    S.to_csv(out / "G_hmm_model_selection.csv", index=False)
    K = int(S.loc[S["BIC"].idxmin(), "K"])
    m = models[K]
    order = np.argsort(m.means_.sum(1))           # state 0 = lowest overall intensity
    remap = {old: new for new, old in enumerate(order)}
    st = pd.DataFrame(dict(state=range(K), mean_sad=m.means_[order, 0], mean_fear=m.means_[order, 1],
                           var_sad=np.array([np.diag(c) for c in m.covars_])[order, 0],
                           var_fear=np.array([np.diag(c) for c in m.covars_])[order, 1]))
    st["K_selected"] = K
    st.to_csv(out / "G_hmm_states.csv", index=False)
    pd.DataFrame(m.transmat_[np.ix_(order, order)], columns=[f"to_{i}" for i in range(K)]).rename_axis(
        "from_state").to_csv(out / "G_hmm_transmat.csv")
    rows = []
    for (s, c), g in zip(keys, seqs):
        x = X[sum(L[:keys.index((s, c))]):sum(L[:keys.index((s, c)) + 1])]
        z = np.array([remap[k] for k in m.predict(x)])
        tm = g[("elapsed_s", "Sadness")].values / 60
        r = dict(subj=s, condition=c, n=len(z), n_switches=int((np.diff(z) != 0).sum()))
        r["switch_rate_per_probe"] = r["n_switches"] / max(len(z) - 1, 1)
        for k in range(K):
            r[f"occ_state{k}"] = float((z == k).mean())
        runs, starts = [], [0]
        for i in range(1, len(z) + 1):
            if i == len(z) or z[i] != z[i - 1]:
                runs.append((i - starts[-1], tm[min(i, len(z) - 1)] - tm[starts[-1]]))
                starts.append(i)
        r["mean_dwell_probes"] = np.mean([a for a, _ in runs])
        r["mean_dwell_min"] = np.mean([b for _, b in runs])
        rows.append(r)
    G = pd.DataFrame(rows)
    G.to_csv(out / "G_hmm_subject.csv", index=False)
    print(f"G HMM done: K = {K}")
    if K == kmax:
        print(f"   !! best K is the largest tried ({kmax}); rerun with a larger kmax to check")
    return G


# ==========================================================================
# H rating behaviour
# ==========================================================================
def section_H(d, out):
    v = d[d["valid"] & d["screen_match"].isin(["matched", "equal_ratings"])]
    if v.empty:
        print("H skipped: no screen data (trigger files missing?)")
        return None
    H = v.groupby(["subj", "condition", "emotion"]).agg(
        n_screens=("n_press", "size"), excess_press_mean=("excess_press", "mean"),
        reversals_mean=("n_reversals", "mean"), overshoot_mean=("overshoot", "mean"),
        frac_any_reversal=("n_reversals", lambda x: float((x > 0).mean())),
        n_press_mean=("n_press", "mean")).reset_index()
    H.to_csv(out / "H_rating_behaviour_subject.csv", index=False)
    print("H rating behaviour done")
    return H


# ==========================================================================
# I reliability
# ==========================================================================
def section_I(d, out):
    rows = []
    v = d[d["valid"]].sort_values("elapsed_s")
    for (c, e), g in v.groupby(["condition", "emotion"]):
        vals = {k: {} for k in ("mean", "sd", "exp_tau_min", "resid_sd", "excess_press")}
        for s, x in g.groupby("subj"):
            for half, xx in (("odd", x[x["probe_idx"] % 2 == 1]), ("even", x[x["probe_idx"] % 2 == 0])):
                xx = xx[xx["valid"]]   # odd/even split on all probe_idx, fit on all ratings (timeouts included)
                y, t = xx["rating"].values.astype(float), xx["t_min"].values
                if len(y) < MIN_VALID // 2 or np.ptp(y) == 0:
                    continue
                f = fit_curves(t, y)
                res, _ = detrend(t, y, f)
                vals["mean"].setdefault(s, {})[half] = y.mean()
                vals["sd"].setdefault(s, {})[half] = y.std(ddof=1)
                vals["resid_sd"].setdefault(s, {})[half] = res.std(ddof=1)
                if not f.get("exp_tau_at_bound", True):
                    vals["exp_tau_min"].setdefault(s, {})[half] = f["exp_tau_min"]
                vals["excess_press"].setdefault(s, {})[half] = xx["excess_press"].mean()
        for k, dct in vals.items():
            pairs = np.array([(h["odd"], h["even"]) for h in dct.values()
                              if "odd" in h and "even" in h and np.isfinite([h["odd"], h["even"]]).all()])
            if len(pairs) >= 5 and pairs.std(0).min() > 0:
                r = stats.pearsonr(pairs[:, 0], pairs[:, 1])[0]
                rows.append(dict(condition=c, emotion=e, parameter=k, n_subjects=len(pairs),
                                 r_split_half=r, spearman_brown=2 * r / (1 + r) if r > -1 else np.nan))
    pd.DataFrame(rows).to_csv(out / "I_reliability.csv", index=False)
    print("I reliability done")


# ==========================================================================
# J master + redundancy
# ==========================================================================
def section_J(R, E, G, H, out):
    P = R.copy()
    if E is not None:
        P = P.merge(E[["subj", "condition"] + [c for c in E if c.startswith(("pearson", "spearman"))]],
                    on=["subj", "condition"], how="left")
    if G is not None:
        P = P.merge(G, on=["subj", "condition"], how="left", suffixes=("", "_hmm"))
    if H is not None:
        P = P.merge(H, on=["subj", "condition", "emotion"], how="left")
    P.to_csv(out / "J_params_subject.csv", index=False)
    dyn = [c for c in ("exp_tau_min", "exp_A", "lin_slope_per_min", "resid_sd", "resid_ac1",
                       "vg_tau_min", "vg_sill", "vg_nugget", "pearson_resid", "switch_rate_per_probe",
                       "mean_dwell_min", "excess_press_mean", "reversals_mean") if c in P]
    rows = []
    for (c, e), g in P[P["usable"]].groupby(["condition", "emotion"]):
        for col in dyn:
            x = g[[col, "mean", "sd"]].apply(pd.to_numeric, errors="coerce").dropna()
            if col in ("exp_tau_min",):
                x = x[~g.loc[x.index, "exp_tau_at_bound"].astype(bool)]
            if len(x) < 8 or x[col].std() == 0:
                continue
            Xm = np.column_stack([np.ones(len(x)), x["mean"], x["sd"]])
            beta, *_ = np.linalg.lstsq(Xm, x[col].values, rcond=None)
            pred = Xm @ beta
            r2 = 1 - np.sum((x[col] - pred) ** 2) / np.sum((x[col] - x[col].mean()) ** 2)
            rows.append(dict(condition=c, emotion=e, parameter=col, n=len(x),
                             r_with_mean=np.corrcoef(x[col], x["mean"])[0, 1],
                             r_with_sd=np.corrcoef(x[col], x["sd"])[0, 1],
                             R2_explained_by_mean_sd=r2))
    pd.DataFrame(rows).to_csv(out / "J_redundancy.csv", index=False)
    print("J master table + redundancy done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--skip-hmm", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    d, pv = load(out)
    print(f"{d['subj'].nunique()} subjects, {int(d['valid'].sum())} valid ratings")
    section_A(d, pv, out)
    R, resid = section_BC(d, out, a.boot)
    section_D(d, out)
    E = section_E(d, resid, out)
    section_F(R, d, out)
    G = None if a.skip_hmm else section_G(d, out)
    H = section_H(d, out)
    section_I(d, out)
    section_J(R, E, G, H, out)
    print(f"\nall outputs in {out}")


if __name__ == "__main__":
    main()
