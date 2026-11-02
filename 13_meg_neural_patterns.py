#!/usr/bin/env python3
"""
13_meg_neural_patterns.py
=========================
Describe the neural patterns in the source-space MEG dynamics and relate
them to the VAS behaviour. Four parts, each run separately per method
(mean_flip, pca_flip):

PART 1  PATTERNS AMONG THE MEASURES  (rotated factor analysis)
        Baseline block, subject x 12 MEG measures, each measure averaged over
        the 4 networks (47 subjects are enough for 12 measures, not for 48).
        Number of factors from parallel analysis (eigenvalues above the 95th
        percentile of random data), varimax rotation. Sad and fear blocks are
        scored with the BASELINE model, so a factor means the same in every
        block. Stability: the factor analysis is refitted in the sad and fear
        blocks and in the other method, and compared with Tucker's
        congruence (> .95 = same factor, .85-.94 = fair, < .85 = different).
        -> P1_parallel_analysis.csv, P1_loadings.csv, P1_congruence.csv,
           P1_factor_scores.csv, P1_factor_vs_behaviour.csv

PART 2  NETWORK-SPECIFIC PATTERNS
        Each network's value = whole-brain level + network deviation.
        Deviation of a network on a factor = factor score computed from that
        network's measures minus the whole-brain factor score.
        Tests: (a) how much variance is network-specific, per measure;
        (b) is a network deviation a stable trait (correlation of the
        deviation between blocks; near 0 = noise); (c) does it relate to
        behaviour beyond the whole-brain score (partial correlation
        controlling for the whole-brain factor score).
        -> P2_network_specific_variance.csv, P2_network_deviation_stability.csv,
           P2_network_deviation_vs_behaviour.csv

PART 3  TYPES OF PEOPLE  (latent profile analysis)
        Gaussian mixtures with 1..MAX_PROFILES classes on the baseline factor
        scores. BIC per k; a k > 1 is accepted only if every class has at
        least MIN_CLASS_N people and the solution is stable (mean adjusted
        Rand index of bootstrap refits >= MIN_ARI); otherwise k = 1
        (a continuum, no types). Compared with the behavioural fluctuation
        types of script 5 (class_label in V0_behaviour_table) and with the
        behavioural variables (Kruskal-Wallis).
        -> P3_profile_selection.csv, P3_profiles.csv, P3_profile_means.csv,
           P3_profiles_vs_behaviour.csv

PART 4  BRAIN STATES OVER TIME  (hidden Markov model)
        Snippet-level factor scores (the 19 s before each VAS probe; network
        average of the 12 measures, baseline model of part 1), one sequence
        per subject x block, centred within subject x block (--hmm-center
        block, default: states then describe changes WITHIN a block, not
        offsets between blocks or recordings; 'subject' or 'none' keep
        them; --hmm-detrend also removes a linear drift per block).
        Gaussian HMM, K = 2..--max-states. BIC keeps falling with thousands
        of non-normal snippets, so K is chosen by held-out log-likelihood
        (subjects split into CV_FOLDS folds) among solutions in which every
        state holds >= MIN_STATE_OCC of the snippets; BIC is reported too.
        Per subject x block: fractional occupancy of each state, switch rate,
        mean dwell (snippets). Related to the behaviour per block, and
        state-locked ratings: is the rating that follows a snippet in state s
        higher than the person's own block mean? (per-subject difference,
        Wilcoxon across subjects; ratings and rating changes).
        Snippet k of a block precedes probe k+1 (same alignment and QC as
        4_meg_behav_link.py; blocks failing the QC are left out).
        Needs hmmlearn (pip install hmmlearn in MEG_env); skipped otherwise.
        -> P4_hmm_selection.csv, P4_hmm_states.csv, P4_hmm_transmat.csv,
           P4_hmm_subject_block.csv, P4_hmm_vs_behaviour.csv,
           P4_state_locked_ratings.csv

DISTRIBUTIONS  The MEG measures are skewed (|skew| up to ~6, most fail
        Shapiro-Wilk), and factor analysis, Gaussian mixtures and the HMM
        assume normal data. They are therefore transformed first
        (--transform rankint, default: rank-based inverse normal, pooled over
        subjects, blocks and networks per method x measure; or log / raw).
        P0_distributions.csv shows skew, kurtosis, Shapiro p and the largest
        |z| before and after.
All behaviour tests are rank-based (Spearman / partial Spearman, Wilcoxon,
Kruskal-Wallis), because the behavioural statistics are skewed with exact
zeros (script 9). All tests are UNCORRECTED (description). Behaviour: mean of
the two scales per subject x block from V0_behaviour_table.csv (script 10);
latent_sd is always partialled on the mean level, as in 10_ and 12_.

USAGE (cluster)
  cd /home/ZhangJianfeng_04/emo_study/Behav/
  export PATH="/home/ZhangJianfeng_04/.conda/envs/MEG_env/bin:$PATH"
  python3 13_meg_neural_patterns.py
  options: --transform {rankint,log,raw} --n-factors K --skip-hmm
           --hmm-center {block,subject,none} --hmm-detrend --max-states 8
           --n-boot 200
"""
import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
BEHAV = Path("/home/ZhangJianfeng_04/emo_study/Behav")
NET_DIR = BEHAV / "analysis" / "vas_meg_networks"          # outputs of 10_
BEHAV_DM = BEHAV / "dynamics_models"                       # vas_probes_long.csv
OUT_DIR = BEHAV / "analysis" / "vas_meg_patterns"
DEFAULT_STATIC_GLOB = "/home/ZhangJianfeng_04/emo_study/MEG/SourceSpace/dynamics/*static*.csv"

MEASURES = ["ACW0", "ACW50", "ACF_area_to_zero", "IRASA_slope", "SpectralSlope",
            "LZC", "SampEn_m2", "MSE_costa_m2_AUC", "Spec_lowhigh_8Hz", "MedianFreq",
            "IRASA_oscpower", "SpectralEntropy"]                 # as in 10_
BLOCKS = ["baseline", "sad", "fear"]
BEHAV_VARS = ["latent_sd", "mean", "frac_nonzero", "sd", "first_minus_rest"]
N_PARALLEL = 1000
MAX_PROFILES = 5
MIN_CLASS_N = 5
MIN_ARI = 0.60
MAX_STATES = 8
CV_FOLDS = 5                 # HMM: number of states chosen on held-out subjects
MIN_STATE_OCC = 0.05         # every state must hold >= 5% of the snippets
HMM_RESTARTS = 10
MIN_SEQ = 10
MIN_N = 12
SEED = 20261102


def load_module4():
    spec = importlib.util.spec_from_file_location("meg4", HERE / "4_meg_behav_link.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==========================================================================
# statistics helpers
# ==========================================================================
def partial_rank(x, y, Z=None):
    """Spearman-type (partial) correlation; ranks residualised on ranks of Z."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    k = 0
    if Z is not None:
        Z = np.asarray(Z, float).reshape(len(x), -1)
        m &= np.isfinite(Z).all(1)
        k = Z.shape[1]
    n = int(m.sum())
    if n < max(MIN_N, k + 5):
        return np.nan, np.nan, n
    rx, ry = stats.rankdata(x[m]), stats.rankdata(y[m])
    if k:
        R = np.column_stack([np.ones(n)] + [stats.rankdata(z) for z in Z[m].T])
        rx = rx - R @ np.linalg.lstsq(R, rx, rcond=None)[0]
        ry = ry - R @ np.linalg.lstsq(R, ry, rcond=None)[0]
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan, np.nan, n
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = n - 2 - k
    t = r * np.sqrt(df / max(1 - r ** 2, 1e-12))
    return r, float(2 * stats.t.sf(abs(t), df)), n


def behaviour_test(x, g, bv):
    """x: MEG score aligned with frame g; latent_sd partialled on mean."""
    cov = g[["mean"]].values if (bv == "latent_sd" and "mean" in g) else None
    return partial_rank(x, g[bv].values, cov)


def tucker(a, b):
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


# ==========================================================================
# distributions
# ==========================================================================
def transform_values(df, how, value="level", by=("method", "measure")):
    """Make the MEG measures closer to normal before the Gaussian models
    (factor analysis, mixtures, HMM). Pooled over subjects, blocks and
    networks within method x measure, so block differences are kept.
      rankint  rank-based inverse normal transform (handles skew of either
               sign and outliers; default)
      log      log of positive measures (helps right skew only)
      raw      unchanged"""
    df = df.copy()
    if how == "raw":
        return df
    g = df.groupby(list(by))[value]
    if how == "log":
        pos = g.transform("min") > 0
        df.loc[pos, value] = np.log(df.loc[pos, value])
        return df
    r = g.rank(method="average")
    n = g.transform("count")
    df[value] = stats.norm.ppf((r - 0.5) / n)
    return df


def distribution_report(T_raw, T_tr, out):
    rows = []
    for lab, T in (("raw", T_raw), ("transformed", T_tr)):
        x = T[(T["condition"] == "baseline") & T["measure"].isin(MEASURES)]
        for (meth, meas), g in x.groupby(["method", "measure"]):
            v = g.groupby("subj")["level"].mean().dropna()
            if len(v) < 8:
                continue
            z = (v - v.mean()) / v.std()
            rows.append(dict(data=lab, method=meth, measure=meas, n=len(v), skew=stats.skew(v),
                             excess_kurtosis=stats.kurtosis(v), shapiro_p=stats.shapiro(v).pvalue,
                             max_abs_z=float(z.abs().max())))
    D = pd.DataFrame(rows)
    D.to_csv(out / "P0_distributions.csv", index=False)
    s_ = D.groupby("data").agg(median_abs_skew=("skew", lambda v: np.median(np.abs(v))),
                               n_shapiro_p05=("shapiro_p", lambda v: int((v < .05).sum())),
                               n=("skew", "size"))
    print("PART 0 distributions of the baseline MEG measures (subject means over networks):")
    print(s_.to_string())


# ==========================================================================
# PART 1 - factor analysis
# ==========================================================================
def parallel_analysis(Z, rng):
    n, p = Z.shape
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.T)))[::-1]
    sim = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.normal(size=(n, p)).T)))[::-1]
                    for _ in range(N_PARALLEL)])
    thr = np.percentile(sim, 95, axis=0)
    k = 0
    while k < p and ev[k] > thr[k]:
        k += 1
    return ev, thr, max(k, 1)


def fit_fa(Z, k):
    from sklearn.decomposition import FactorAnalysis
    fa = FactorAnalysis(n_components=k, rotation="varimax" if k > 1 else None,
                        random_state=0).fit(Z)
    L = fa.components_.T.copy()                       # measures x factors
    sign = np.sign(L[np.argmax(np.abs(L), axis=0), np.arange(k)])
    L *= sign
    order = np.argsort(-np.sum(L ** 2, axis=0))       # factor 1 = most variance
    return fa, L[:, order], sign, order


def fa_scores(fa, sign, order, Z):
    return (fa.transform(Z) * sign)[:, order]


def factor_name(L, measures, j, n_top=3):
    idx = np.argsort(-np.abs(L[:, j]))[:n_top]
    return " ".join(f"{'+' if L[i, j] > 0 else '-'}{measures[i]}" for i in idx)


def block_matrix(T, method, block, net_average=True):
    x = T[(T["method"] == method) & (T["condition"] == block) & T["measure"].isin(MEASURES)]
    if net_average:
        return x.groupby(["subj", "measure"])["level"].mean().unstack().reindex(columns=MEASURES)
    return x.pivot_table(index="subj", columns=["group", "measure"], values="level")


def part1(T, Bm, n_factors, rng, out):
    models, par_rows, load_rows, cong_rows, score_rows, beh_rows = {}, [], [], [], [], []
    for meth in sorted(T["method"].unique()):
        X = block_matrix(T, meth, "baseline").dropna()
        mu, sd = X.mean(), X.std()
        Z = ((X - mu) / sd).values
        ev, thr, k = parallel_analysis(Z, rng)
        if n_factors:
            k = n_factors
        for i, (e, t) in enumerate(zip(ev, thr)):
            par_rows.append(dict(method=meth, component=i + 1, eigenvalue=e, random_95=t, kept=i < k))
        fa, L, sign, order = fit_fa(Z, k)
        names = [factor_name(L, MEASURES, j) for j in range(k)]
        models[meth] = dict(fa=fa, L=L, sign=sign, order=order, mu=mu, sd=sd, k=k, names=names)
        print(f"PART 1 {meth}: n = {len(X)}, eigenvalues {np.round(ev[:4], 2)}, "
              f"random 95% {np.round(thr[:4], 2)} -> {k} factor(s)")
        for j in range(k):
            print(f"   F{j + 1}: {names[j]}  ({np.sum(L[:, j] ** 2) / len(MEASURES):.0%} of variance)")
            for i, m in enumerate(MEASURES):
                load_rows.append(dict(method=meth, factor=f"F{j + 1}", factor_name=names[j],
                                      measure=m, loading=L[i, j]))
        # stability: refit in the other blocks
        for blk in ("sad", "fear"):
            Y = block_matrix(T, meth, blk).dropna()
            if len(Y) < MIN_N:
                continue
            _, Lb, _, _ = fit_fa(((Y - Y.mean()) / Y.std()).values, k)
            for j in range(k):
                c = [abs(tucker(L[:, j], Lb[:, jj])) for jj in range(k)]
                cong_rows.append(dict(method=meth, factor=f"F{j + 1}", compared_with=f"{blk} block refit",
                                      congruence=max(c)))
        # scores in every block with the baseline model
        for blk in BLOCKS:
            Y = block_matrix(T, meth, blk).dropna()
            if not len(Y):
                continue
            S = fa_scores(fa, sign, order, ((Y - mu) / sd).values)
            for s_, row in zip(Y.index, S):
                score_rows.append(dict(method=meth, condition=blk, subj=s_,
                                       **{f"F{j + 1}": row[j] for j in range(k)}))
    meths = sorted(models)
    if len(meths) == 2 and models[meths[0]]["k"] == models[meths[1]]["k"]:
        a, b = models[meths[0]]["L"], models[meths[1]]["L"]
        for j in range(a.shape[1]):
            cong_rows.append(dict(method=meths[0], factor=f"F{j + 1}", compared_with=f"{meths[1]}",
                                  congruence=max(abs(tucker(a[:, j], b[:, jj])) for jj in range(b.shape[1]))))
    S = pd.DataFrame(score_rows)
    W = S.merge(Bm, on=["subj", "condition"], how="left")
    for (meth, blk), g in W.groupby(["method", "condition"]):
        for f in [c for c in g.columns if c.startswith("F") and c[1:].isdigit()]:
            for bv in [v for v in BEHAV_VARS if v in g]:
                r, p, n = behaviour_test(g[f].values, g, bv)
                beh_rows.append(dict(method=meth, condition=blk, factor=f,
                                     factor_name=models[meth]["names"][int(f[1:]) - 1],
                                     behaviour=bv, partial_on_mean=bv == "latent_sd", n=n, rho=r, p=p))
    pd.DataFrame(par_rows).to_csv(out / "P1_parallel_analysis.csv", index=False)
    pd.DataFrame(load_rows).to_csv(out / "P1_loadings.csv", index=False)
    pd.DataFrame(cong_rows).to_csv(out / "P1_congruence.csv", index=False)
    S.to_csv(out / "P1_factor_scores.csv", index=False)
    R = pd.DataFrame(beh_rows)
    R.to_csv(out / "P1_factor_vs_behaviour.csv", index=False)
    show(R[R["p"] < .05], "PART 1 factor x behaviour, p < .05")
    return models, S


# ==========================================================================
# PART 2 - network-specific patterns
# ==========================================================================
def part2(T, Bm, models, out):
    var_rows, stab_rows, beh_rows = [], [], []
    dev_all = []
    for meth, md in models.items():
        k = md["k"]
        for blk in BLOCKS:
            N = block_matrix(T, meth, blk, net_average=False).dropna()
            if len(N) < MIN_N:
                continue
            groups = sorted({g for g, _ in N.columns})
            # z-score each network x measure across subjects, whole-brain = mean of networks
            Zn = (N - N.mean()) / N.std()
            whole = sum(Zn[g].reindex(columns=MEASURES) for g in groups) / len(groups)
            if blk == "baseline":
                for m in MEASURES:
                    tot = np.nanmean([np.nanvar(Zn[(g, m)]) for g in groups])
                    spec = np.nanmean([np.nanvar(Zn[(g, m)] - whole[m]) for g in groups])
                    var_rows.append(dict(method=meth, measure=m, network_specific_share=spec / tot))
            # factor-score coefficients of the baseline model, applied to z-scored inputs
            fa, sign, order = md["fa"], md["sign"], md["order"]
            ws = fa_scores(fa, sign, order, whole.fillna(0).values)
            for g in groups:
                Gm = Zn[g].reindex(columns=MEASURES).fillna(0).values
                dv = fa_scores(fa, sign, order, Gm) - ws
                for s_, row, wrow in zip(N.index, dv, ws):
                    dev_all.append(dict(method=meth, condition=blk, subj=s_, network=g,
                                        **{f"F{j + 1}_dev": row[j] for j in range(k)},
                                        **{f"F{j + 1}_whole": wrow[j] for j in range(k)}))
    D = pd.DataFrame(dev_all)
    pd.DataFrame(var_rows).to_csv(out / "P2_network_specific_variance.csv", index=False)
    if not len(D):
        return
    fcols = sorted(c for c in D.columns if c.endswith("_dev"))
    # (b) stability of the deviations across blocks
    for (meth, net), g in D.groupby(["method", "network"]):
        for f in fcols:
            wide = g.pivot_table(index="subj", columns="condition", values=f)
            for a, b in (("baseline", "sad"), ("baseline", "fear"), ("sad", "fear")):
                if a in wide and b in wide:
                    r, p, n = partial_rank(wide[a].values, wide[b].values)
                    stab_rows.append(dict(method=meth, network=net, factor=f.replace("_dev", ""),
                                          blocks=f"{a} vs {b}", n=n, rho=r, p=p))
    # (c) deviation vs behaviour beyond the whole-brain score
    W = D.merge(Bm, on=["subj", "condition"], how="left")
    for (meth, blk, net), g in W.groupby(["method", "condition", "network"]):
        for f in fcols:
            fw = f.replace("_dev", "_whole")
            for bv in [v for v in BEHAV_VARS if v in g]:
                covs = [fw] + (["mean"] if bv == "latent_sd" else [])
                r, p, n = partial_rank(g[f].values, g[bv].values, g[covs].values)
                beh_rows.append(dict(method=meth, condition=blk, network=net,
                                     factor=f.replace("_dev", ""), behaviour=bv,
                                     controls=" + ".join(["whole-brain score"] + covs[1:]),
                                     n=n, rho=r, p=p))
    St, R = pd.DataFrame(stab_rows), pd.DataFrame(beh_rows)
    St.to_csv(out / "P2_network_deviation_stability.csv", index=False)
    R.to_csv(out / "P2_network_deviation_vs_behaviour.csv", index=False)
    V = pd.DataFrame(var_rows)
    print("\nPART 2 network-specific share of variance (baseline, median over measures): " +
          ", ".join(f"{m} {v:.0%}" for m, v in V.groupby("method")["network_specific_share"].median().items()))
    if len(St):
        print("PART 2 stability of network deviations between blocks (median rho): " +
              ", ".join(f"{k[0]} {k[1]} {v:.2f}" for k, v in
                        St.groupby(["method", "network"])["rho"].median().items()))
    show(R[R["p"] < .05], "PART 2 network deviation x behaviour (beyond whole brain), p < .05")


# ==========================================================================
# PART 3 - latent profiles
# ==========================================================================
def part3(S, B, Bm, n_boot, rng, out):
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score
    sel_rows, prof_rows, mean_rows, beh_rows = [], [], [], []
    for meth, s in S[S["condition"] == "baseline"].groupby("method"):
        fcols = [c for c in s.columns if c.startswith("F") and c[1:].isdigit() and s[c].notna().any()]
        s = s.dropna(subset=fcols)
        X = s[fcols].values
        n = len(X)
        fits = {}
        for k in range(1, MAX_PROFILES + 1):
            if n < k * MIN_CLASS_N:
                break
            gm = GaussianMixture(k, covariance_type="full", n_init=20, random_state=0).fit(X)
            lab = gm.predict(X)
            sizes = np.bincount(lab, minlength=k)
            aris = []
            for _ in range(n_boot if k > 1 else 0):
                i = rng.integers(0, n, n)
                gb = GaussianMixture(k, covariance_type="full", n_init=3,
                                     random_state=int(rng.integers(1e9))).fit(X[i])
                aris.append(adjusted_rand_score(lab, gb.predict(X)))
            ari = float(np.mean(aris)) if aris else np.nan
            ok = k == 1 or (sizes.min() >= MIN_CLASS_N and ari >= MIN_ARI)
            fits[k] = (gm, lab)
            sel_rows.append(dict(method=meth, k=k, n=n, bic=gm.bic(X), min_class_n=int(sizes.min()),
                                 stability_ari=ari, passes=ok))
        sel = pd.DataFrame([r for r in sel_rows if r["method"] == meth])
        k_bic = int(sel.loc[sel["bic"].idxmin(), "k"])
        # same rule as script 5: step up from k = 2 while every criterion passes,
        # never beyond the k that BIC prefers
        k_sel = 1
        for _, r in sel.sort_values("k").iterrows():
            if r["k"] == 1:
                continue
            if r["passes"] and r["k"] <= k_bic:
                k_sel = int(r["k"])
            else:
                break
        print(f"\nPART 3 {meth}: BIC prefers k = {k_bic}; with size and stability criteria k = {k_sel}"
              + ("  (continuum: no neural types)" if k_sel == 1 else ""))
        gm, lab = fits[k_sel]
        order = np.argsort(gm.means_[:, 0])                     # profile 1 = lowest F1
        remap = {old: new + 1 for new, old in enumerate(order)}
        prof = s[["subj"]].assign(method=meth, k=k_sel, profile=[remap[l] for l in lab],
                                  posterior_max=gm.predict_proba(X).max(1))
        prof_rows.append(prof)
        for old, new in remap.items():
            mean_rows.append(dict(method=meth, k=k_sel, profile=new, n=int(np.sum(lab == old)),
                                  **{f"{f}_mean": gm.means_[old, i] for i, f in enumerate(fcols)}))
        if k_sel > 1:
            b = Bm[Bm["condition"] == "baseline"].merge(prof, on="subj")
            for bv in [v for v in BEHAV_VARS if v in b]:
                grp = [g[bv].dropna().values for _, g in b.groupby("profile")]
                if all(len(x) >= 3 for x in grp):
                    h, p = stats.kruskal(*grp)
                    beh_rows.append(dict(method=meth, test="Kruskal-Wallis", behaviour=bv,
                                         stat=h, p=p, **{f"median_profile{i + 1}": np.median(x)
                                                         for i, x in enumerate(grp)}))
            if "class_label" in B.columns:
                c = (B[(B["condition"] == "baseline")].dropna(subset=["class_label"])
                     .groupby("subj")["class_label"].max().rename("behav_class").reset_index()
                     .merge(prof, on="subj"))
                if len(c) >= MIN_N and c["behav_class"].nunique() > 1:
                    ct = pd.crosstab(c["profile"], c["behav_class"])
                    chi2, p, _, _ = stats.chi2_contingency(ct)
                    beh_rows.append(dict(method=meth, test="neural profile x behavioural class (chi2)",
                                         behaviour="class_label (script 5)", stat=chi2, p=p,
                                         table=ct.to_dict()))
    pd.DataFrame(sel_rows).to_csv(out / "P3_profile_selection.csv", index=False)
    if prof_rows:
        pd.concat(prof_rows).to_csv(out / "P3_profiles.csv", index=False)
    pd.DataFrame(mean_rows).to_csv(out / "P3_profile_means.csv", index=False)
    R = pd.DataFrame(beh_rows)
    R.to_csv(out / "P3_profiles_vs_behaviour.csv", index=False)
    if len(R):
        show(R[R["p"] < .05], "PART 3 profiles x behaviour, p < .05")


# ==========================================================================
# PART 4 - HMM brain states
# ==========================================================================
def snippet_factor_scores(m4, models, pattern, version, transform, out):
    m = m4.localize_snippets(m4.load_meg(pattern, version))
    m = m[m["measure"].isin(MEASURES)]
    G = m4.roi_groups(m, out)
    mg = m4.group_values(m, G)                      # network mean per snippet
    # same transform as the subject-level data, pooled over all snippets
    mg = transform_values(mg, transform, value="value")
    wb = (mg.groupby(["subj", "condition", "snippet_idx", "method", "measure"])["value"]
          .mean().unstack("measure").reindex(columns=MEASURES))
    rows = []
    for meth, md in models.items():
        if meth not in wb.index.get_level_values("method"):
            continue
        x = wb.xs(meth, level="method").dropna()
        Z = ((x - x.mean()) / x.std()).values      # snippet scale differs from subject means
        Sc = fa_scores(md["fa"], md["sign"], md["order"], Z)
        f = pd.DataFrame(Sc, index=x.index, columns=[f"F{j + 1}" for j in range(md["k"])]).reset_index()
        f["method"] = meth
        rows.append(f)
    return pd.concat(rows, ignore_index=True), m


def _best_hmm(seqs, K, restarts, rng):
    from hmmlearn.hmm import GaussianHMM
    X, L = np.vstack(seqs), [len(s) for s in seqs]
    best = None
    for _ in range(restarts):
        mdl = GaussianHMM(K, covariance_type="full", n_iter=500, min_covar=1e-3,
                          random_state=int(rng.integers(1e9)))
        try:
            mdl.fit(X, L)
            ll = mdl.score(X, L)
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best[1]):
            best = (mdl, ll)
    return best


def fit_hmm(seqs, subj_of_seq, max_states, rng):
    """For each K: model on all data (BIC, occupancy) and the held-out
    log-likelihood per snippet with subjects split into CV_FOLDS folds.
    BIC keeps falling with thousands of snippets and non-normal data, so K is
    chosen on the held-out likelihood."""
    X = np.vstack(seqs)
    d = X.shape[1]
    subs = np.unique(subj_of_seq)
    folds = np.array_split(rng.permutation(subs), CV_FOLDS)
    res = []
    for K in range(2, max_states + 1):
        best = _best_hmm(seqs, K, HMM_RESTARTS, rng)
        if best is None:
            continue
        mdl, ll = best
        states = mdl.predict(X, [len(s) for s in seqs])
        occ = np.bincount(states, minlength=K) / len(states)
        cv_ll, cv_n = 0.0, 0
        for f in folds:
            test = [s for s, sb in zip(seqs, subj_of_seq) if sb in set(f)]
            train = [s for s, sb in zip(seqs, subj_of_seq) if sb not in set(f)]
            b = _best_hmm(train, K, max(2, HMM_RESTARTS // 3), rng)
            if b is None or not test:
                cv_ll = np.nan
                break
            cv_ll += b[0].score(np.vstack(test), [len(s) for s in test])
            cv_n += sum(len(s) for s in test)
        npar = (K - 1) + K * (K - 1) + K * d + K * d * (d + 1) / 2
        res.append(dict(K=K, model=mdl, loglik=ll, n_params=npar,
                        bic=-2 * ll + npar * np.log(len(X)),
                        cv_loglik_per_snippet=cv_ll / cv_n if cv_n else np.nan,
                        min_state_occupancy=float(occ.min())))
    return res


def choose_k(res):
    ok = [r for r in res if r["min_state_occupancy"] >= MIN_STATE_OCC and np.isfinite(r["cv_loglik_per_snippet"])]
    if not ok:
        return min(res, key=lambda r: r["bic"])
    return max(ok, key=lambda r: r["cv_loglik_per_snippet"])


def part4(m4, models, Bm, pattern, version, center, detrend, transform, max_states, rng, out):
    try:
        import hmmlearn  # noqa: F401
    except ImportError:
        print("\nPART 4 skipped: hmmlearn missing (pip install hmmlearn in MEG_env)")
        return
    F, m = snippet_factor_scores(m4, models, pattern, version, transform, out)
    d = m4.probe_table(BEHAV_DM)
    cells = m4.alignment(m, d, 1, out)                 # also writes M0_alignment_qc.csv here
    F = F.merge(cells, on=["subj", "condition"], how="inner")
    sel_rows, st_rows, tm_rows, sb_rows, beh_rows, lock_rows = [], [], [], [], [], []
    for meth, f in F.groupby("method"):
        fcols = [c for c in f.columns if c.startswith("F") and c[1:].isdigit() and f[c].notna().any()]
        f = f.dropna(subset=fcols).sort_values(["subj", "condition", "snippet_idx"]).copy()
        if center == "subject":
            f[fcols] = f[fcols] - f.groupby("subj")[fcols].transform("mean")
        elif center == "block":
            f[fcols] = f[fcols] - f.groupby(["subj", "condition"])[fcols].transform("mean")
        if detrend:       # remove a linear drift within each block
            for c in fcols:
                f[c] = f.groupby(["subj", "condition"], group_keys=False)[[c, "snippet_idx"]].apply(
                    lambda g: g[c] - np.polyval(np.polyfit(g["snippet_idx"], g[c], 1), g["snippet_idx"])
                    if len(g) > 2 else g[c] - g[c].mean())
        keys, seqs = [], []
        for (s_, c), g in f.groupby(["subj", "condition"]):
            if len(g) >= MIN_SEQ:
                keys.append((s_, c))
                seqs.append(g[fcols].values)
        if len(seqs) < MIN_N:
            continue
        res = fit_hmm(seqs, np.array([k_[0] for k_ in keys]), max_states, rng)
        best = choose_k(res)
        for r in res:
            sel_rows.append(dict(method=meth, K=r["K"], loglik=r["loglik"], n_params=r["n_params"], bic=r["bic"],
                                 cv_loglik_per_snippet=r["cv_loglik_per_snippet"],
                                 min_state_occupancy=r["min_state_occupancy"], selected=r is best))
        mdl, K = best["model"], best["K"]
        order = np.argsort(mdl.means_[:, 0])            # state 1 = lowest F1
        remap = {old: new + 1 for new, old in enumerate(order)}
        k_bic = min(res, key=lambda r: r["bic"])["K"]
        print(f"\nPART 4 {meth}: {len(seqs)} sequences, centring {center}"
              f"{' + detrend' if detrend else ''}; BIC prefers K = {k_bic}, held-out likelihood "
              f"(states >= {MIN_STATE_OCC:.0%} occupancy) -> K = {K}"
              + ("  (largest tried - consider a larger --max-states)" if K == max_states else ""))
        for old, new in remap.items():
            st_rows.append(dict(method=meth, K=K, state=new,
                                **{f"{c}_mean": mdl.means_[old, i] for i, c in enumerate(fcols)}))
            for old2, new2 in remap.items():
                tm_rows.append(dict(method=meth, from_state=new, to_state=new2,
                                    p=mdl.transmat_[old, old2]))
        lab_rows = []
        for (s_, c), x in zip(keys, seqs):
            z = np.array([remap[v] for v in mdl.predict(x)])
            g = f[(f["subj"] == s_) & (f["condition"] == c)]
            lab_rows.append(pd.DataFrame(dict(subj=s_, condition=c, snippet_idx=g["snippet_idx"].values, state=z)))
            runs = np.diff(np.flatnonzero(np.diff(np.r_[-1, z, -1]) != 0))
            sb_rows.append(dict(method=meth, subj=s_, condition=c, n=len(z),
                                switch_rate=float(np.mean(np.diff(z) != 0)),
                                mean_dwell_snippets=float(np.mean(runs)),
                                **{f"occ_state{k}": float(np.mean(z == k)) for k in range(1, K + 1)}))
        SB = pd.DataFrame([r for r in sb_rows if r["method"] == meth])
        W = SB.merge(Bm, on=["subj", "condition"], how="left")
        for blk, g in W.groupby("condition"):
            for mcol in ["switch_rate", "mean_dwell_snippets"] + [f"occ_state{k}" for k in range(1, K + 1)]:
                for bv in [v for v in BEHAV_VARS if v in g]:
                    r, p, n = behaviour_test(g[mcol].values, g, bv)
                    beh_rows.append(dict(method=meth, condition=blk, hmm_measure=mcol, behaviour=bv,
                                         partial_on_mean=bv == "latent_sd", n=n, rho=r, p=p))
        # state-locked ratings: snippet k -> probe k+1
        lab = pd.concat(lab_rows)
        lab["probe_idx"] = lab["snippet_idx"] + 1
        pr = d[["subj", "condition", "emotion", "probe_idx", "rating_for_delta", "delta"]].merge(
            lab, on=["subj", "condition", "probe_idx"], how="inner")
        pr["rating_c"] = pr["rating_for_delta"] - pr.groupby(["subj", "condition", "emotion"])[
            "rating_for_delta"].transform("mean")
        for emo, e in pr.groupby("emotion"):
            for outcome in ("rating_c", "delta"):
                for k in range(1, K + 1):
                    diffs = []
                    for s_, g in e.groupby("subj"):
                        a, b = g.loc[g["state"] == k, outcome].dropna(), g.loc[g["state"] != k, outcome].dropna()
                        if len(a) >= 2 and len(b) >= 2:
                            diffs.append(a.mean() - b.mean())
                    diffs = np.array(diffs)
                    if len(diffs) >= MIN_N and np.any(diffs != 0):
                        w = stats.wilcoxon(diffs)
                        lock_rows.append(dict(method=meth, emotion=emo,
                                              outcome="rating (centred)" if outcome == "rating_c" else "rating change",
                                              state=k, n_subjects=len(diffs),
                                              median_diff_state_vs_other=float(np.median(diffs)),
                                              p_wilcoxon=w.pvalue))
    pd.DataFrame(sel_rows).to_csv(out / "P4_hmm_selection.csv", index=False)
    pd.DataFrame(st_rows).to_csv(out / "P4_hmm_states.csv", index=False)
    pd.DataFrame(tm_rows).to_csv(out / "P4_hmm_transmat.csv", index=False)
    pd.DataFrame(sb_rows).to_csv(out / "P4_hmm_subject_block.csv", index=False)
    R, Lk = pd.DataFrame(beh_rows), pd.DataFrame(lock_rows)
    R.to_csv(out / "P4_hmm_vs_behaviour.csv", index=False)
    Lk.to_csv(out / "P4_state_locked_ratings.csv", index=False)
    if len(R):
        show(R[R["p"] < .05], "PART 4 HMM measures x behaviour, p < .05")
    if len(Lk):
        show(Lk[Lk["p_wilcoxon"] < .05], "PART 4 state-locked ratings, p < .05")


# ==========================================================================
def show(df, title):
    print(f"\n{title}:")
    if not len(df):
        print("  none")
        return
    sort = [c for c in ("p", "p_wilcoxon") if c in df][0]
    print(df.sort_values(sort).head(25).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-dir", default=str(NET_DIR))
    ap.add_argument("--static-glob", default=DEFAULT_STATIC_GLOB)
    ap.add_argument("--version", default="ecg")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--n-factors", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--skip-hmm", action="store_true")
    ap.add_argument("--hmm-center", choices=["block", "subject", "none"], default="block")
    ap.add_argument("--hmm-detrend", action="store_true",
                    help="also remove a linear drift within each block before the HMM")
    ap.add_argument("--max-states", type=int, default=MAX_STATES)
    ap.add_argument("--transform", choices=["rankint", "log", "raw"], default="rankint")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net = Path(a.net_dir)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 220)

    T_raw = pd.read_csv(net / "V0_meg_table.csv", dtype={"subj": str})
    T = transform_values(T_raw, a.transform)
    print(f"MEG measures transformed: {a.transform}")
    B = pd.read_csv(net / "V0_behaviour_table.csv", dtype={"subj": str})
    bvars = [v for v in BEHAV_VARS if v in B.columns]
    Bm = B.groupby(["subj", "condition"])[bvars].mean().reset_index()   # mean of the two scales
    missing = [m for m in MEASURES if m not in set(T["measure"])]
    if missing:
        print(f"note: measures not in V0_meg_table: {missing} - dropped")
        MEASURES[:] = [m for m in MEASURES if m not in missing]

    distribution_report(T_raw, T, out)
    models, S = part1(T, Bm, a.n_factors, rng, out)
    part2(T, Bm, models, out)
    part3(S, B, Bm, a.n_boot, rng, out)
    if not a.skip_hmm:
        part4(load_module4(), models, Bm, a.static_glob, a.version, a.hmm_center, a.hmm_detrend,
              a.transform, a.max_states, rng, out)
    print(f"\noutputs in {out}")


if __name__ == "__main__":
    main()
