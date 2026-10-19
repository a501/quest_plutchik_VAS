#!/usr/bin/env python3
"""
4_meg_behav_link.py
===================
Link the behavioural VAS dynamics (1_, 2_, 3_) to the source-space MEG
dynamics (SS_dynamics static CSVs: one value per subject x condition x
pre-VAS snippet x unit x method x measure).

Expected static-CSV columns (checked at start, fails loudly otherwise):
    subject, version, condition, unit_name, method, measure, value, snippet_idx
    optional: flag (kept when empty), level (roi/label)
Only one cleaning version is used (default ecg, as decided for SS_emo).

STEP 0  ROI GROUPS + ALIGNMENT QC
        Units are assigned to networks by name patterns (ROI_GROUPS below).
        The membership is written out -> CHECK IT before trusting anything.
        Snippet k of a block is aligned to probe k+offset of the same block
        (snippets are the 19 s before each VAS onset). The number of snippets
        vs probes per subject x condition is reported; mismatches are flagged.
        -> M0_roi_groups.csv, M0_alignment_qc.csv

STEP 1  BETWEEN-SUBJECT: neural timescale -> emotional dynamics
        MEG predictor = subject mean over snippets, per network, from
        (a) the BASELINE block (trait-like intrinsic timescale) and
        (b) the same block as the behaviour.
        Behaviour = per-series parameters from J_params_subject.csv
        (log recovery tau, log variogram tau, residual inertia, residual SD,
        amplitude) and, if present, ctsem subject drifts (R6_ctsem_subject_*).
        Pearson + Spearman, Fisher-z 95% CI, BH-FDR over the whole table.
        The pre-registered PRIMARY test is flagged:
            DMN x ACW0 (baseline) x log recovery tau of the target emotion.
        NOTE: n = subjects with source data (those with a FreeSurfer recon),
        likely < 40 - check n_subjects in the output.
        -> M1_between_subject.csv

STEP 2  WITHIN-SUBJECT: pre-probe MEG -> next rating change
        delta = rating_k - rating_{k-1} (consecutive valid probes).
        MEG value z-scored within subject x condition (within-person effect).
        OLS with subject-clustered standard errors:
            delta ~ meg_z + prev_rating_c + log_t + C(condition)
            (+ second model with meg_z x condition)
        per network x measure x method x emotion, BH-FDR across all.
        -> M2_within_subject.csv

STEP 3  WITHIN-SUBJECT MULTIVARIATE: one test per emotion x method
        Ridge regression from ALL units x measures (pre-probe snippet) to the
        next rating change (within-person centred). GroupKFold by subject,
        out-of-sample r between predicted and observed change. Permutation
        null: change shuffled within subject x condition (N_PERM).
        -> M3_decoding.csv, M3_decoding_weights.csv

USAGE
    python 4_meg_behav_link.py --static-glob "/path/to/dynamics/*static*.csv"
    options: --version ecg --behav-dir DIR --snippet-offset 1 --n-perm 200
Heavy (many fits + permutations): submit via SLURM or run on fox-c18.
"""
import argparse
import glob
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BEHAV_DIR = Path("/home/ZhangJianfeng_04/emo_study/Behav/dynamics_models")
DEFAULT_STATIC_GLOB = "/home/ZhangJianfeng_04/emo_study/MEG/SourceSpace/dynamics/*static*.csv"
REQUIRED = ["subject", "version", "condition", "unit_name", "method", "measure", "value", "snippet_idx"]
COND_MAP = {"base": "baseline", "baseline": "baseline", "sad": "sad", "fear": "fear"}
TARGET = {"sad": "Sadness", "fear": "Fear"}

# (include, exclude) regex on unit_name -- verify with M0_roi_groups.csv
ROI_GROUPS = {
    "DMN": (r"vmPFC|PCC|mPFC|[Pp]recuneus", r"$^"),
    "Salience_Interoceptive": (r"[Ii]nsula|Intero_|dACC|aMCC|ACC", r"PCC"),
    "Frontoparietal": (r"DLPFC|dlPFC|VLPFC|IFG|IPS|IPL", r"$^"),
    "Sensory_control": (r"Extero_|V1|[Vv]isual|Occipital|Postcentral|Somato|Motor|Auditory", r"$^"),
}
TIMESCALE_MEASURES = r"^(ACW0|ACW50|ACF_area_to_zero|ACF_lag1)$|[Ss]lope"
PRIMARY = dict(group="DMN", measure="ACW0", meg_block="baseline", behav="log_exp_tau")


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    m = np.isfinite(p)
    if m.sum():
        q = p[m]
        o = np.argsort(q)
        r = q[o] * m.sum() / (np.arange(m.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        tmp = np.empty_like(q); tmp[o] = np.minimum(r, 1)
        out[m] = tmp
    return out


def norm_subj(x):
    m = re.search(r"(\d+)", str(x))
    return m.group(1).zfill(3) if m else str(x)


# ==========================================================================
# load
# ==========================================================================
def load_meg(pattern, version):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no static CSVs match {pattern}")
    parts = []
    for f in files:
        x = pd.read_csv(f, low_memory=False)
        miss = [c for c in REQUIRED if c not in x.columns]
        if miss:
            raise KeyError(f"{Path(f).name}: missing {miss}; columns are {list(x.columns)}")
        x = x[x["version"].astype(str) == version]
        if "flag" in x.columns:
            x = x[x["flag"].isna() | (x["flag"].astype(str).str.strip().isin(["", "nan"]))]
        if "level" in x.columns:
            x = x[x["level"].astype(str) == "roi"]
        parts.append(x[REQUIRED])
    m = pd.concat(parts, ignore_index=True)
    m["value"] = pd.to_numeric(m["value"], errors="coerce")       # 'sat' -> NaN
    m = m.dropna(subset=["value"])
    m["subj"] = m["subject"].map(norm_subj)
    m = m[m["subj"].astype(int) < 200]
    m["condition"] = m["condition"].astype(str).str.lower().map(COND_MAP)
    m = m.dropna(subset=["condition"])
    print(f"MEG: {len(files)} files, {m['subj'].nunique()} subjects, {m['unit_name'].nunique()} units, "
          f"{m['measure'].nunique()} measures, version={version}")
    return m


def roi_groups(m, out):
    rows = []
    for u in sorted(m["unit_name"].unique()):
        for g, (inc, exc) in ROI_GROUPS.items():
            if re.search(inc, u) and not re.search(exc, u):
                rows.append(dict(group=g, unit_name=u))
    G = pd.DataFrame(rows)
    G.to_csv(out / "M0_roi_groups.csv", index=False)
    for g in ROI_GROUPS:
        print(f"  {g}: {list(G.loc[G['group'] == g, 'unit_name'])}")
    return G


def group_values(m, G):
    x = m.merge(G, on="unit_name")
    return (x.groupby(["subj", "condition", "snippet_idx", "method", "measure", "group"])["value"]
            .mean().reset_index())


def probe_table(behav_dir):
    d = pd.read_csv(behav_dir / "vas_probes_long.csv", dtype={"subj": str})
    d = d[d["subj"].astype(int) < 200]
    # Keep ALL probes (including timeouts) so probe_idx aligns with snippet_idx.
    # Timeouts count as valid ratings; "valid" is False only where the rating is missing.
    d = d.sort_values(["subj", "block", "emotion", "probe_idx"])
    # NaN-out ratings for non-valid probes so they don't contribute to delta,
    # but keep the row so the index sequence stays intact.
    d["rating_for_delta"] = d["rating"].where(d["valid"], other=np.nan)
    g = d.groupby(["subj", "block", "emotion"])
    d["prev_rating"] = g["rating_for_delta"].shift(1)
    d["prev_idx"] = g["probe_idx"].shift(1)
    # Delta is only meaningful between truly adjacent probes (no gap in index).
    # A missing rating at probe k leaves prev_rating NaN for probe k+1, propagating
    # the gap correctly without dropping any rows.
    d.loc[d["probe_idx"] - d["prev_idx"] != 1, "prev_rating"] = np.nan
    d["delta"] = d["rating_for_delta"] - d["prev_rating"]
    d["log_t"] = np.log(d["elapsed_s"] / 60)
    return d


# ==========================================================================
# step 0 alignment
# ==========================================================================
def alignment(mg, d, offset, out):
    sn = mg.groupby(["subj", "condition"])["snippet_idx"].agg(["min", "max", "nunique"]).reset_index()
    pr = (d.drop_duplicates(["subj", "condition", "probe_idx"]).groupby(["subj", "condition"])
          ["probe_idx"].max().rename("n_probes_valid_max").reset_index())
    q = sn.merge(pr, how="outer", on=["subj", "condition"])
    q["snippet_offset_used"] = offset
    q["flag_count_mismatch"] = (q["max"] + offset) > q["n_probes_valid_max"] + 2
    q.to_csv(out / "M0_alignment_qc.csv", index=False)
    print(f"alignment: {int(q['flag_count_mismatch'].sum())} subject x condition cells flagged")


# ==========================================================================
# step 1 between-subject
# ==========================================================================
def between(mg, behav_dir, out):
    P = pd.read_csv(behav_dir / "J_params_subject.csv", dtype={"subj": str})
    P = P[P["usable"]]
    P["log_exp_tau"] = np.where(P["exp_tau_at_bound"].astype(bool), np.nan, np.log(P["exp_tau_min"]))
    if "vg_tau_min" in P:
        P["log_vg_tau"] = np.where(P["vg_tau_at_bound"].fillna(True).astype(bool), np.nan,
                                   np.log(P["vg_tau_min"]))
    behav_cols = [c for c in ("log_exp_tau", "log_vg_tau", "resid_ac1", "resid_sd", "exp_A") if c in P]
    # optional ctsem subject drifts
    for f in sorted(behav_dir.glob("R6_ctsem_subject_*.csv")):
        c = pd.read_csv(f, dtype={"subj": str})
        dcols = [k for k in c.columns if k.startswith("drift")]
        for e, col in (("Sadness", "drift_sad_sad"), ("Fear", "drift_fear_fear")):
            if col in c:
                tmp = c[["subj", "condition", col]].rename(columns={col: "ctsem_drift_self"})
                tmp["emotion"] = e
                P = P.merge(tmp, on=["subj", "condition", "emotion"], how="left", suffixes=("", "_x"))
        if "ctsem_drift_self" in P and "ctsem_drift_self" not in behav_cols:
            behav_cols.append("ctsem_drift_self")

    ts = mg[mg["measure"].str.contains(TIMESCALE_MEASURES, regex=True)]
    print(f"  timescale measures used: {sorted(ts['measure'].unique())}")
    meg_subj = ts.groupby(["subj", "condition", "method", "measure", "group"])["value"].mean().reset_index()
    rows = []
    for cond in ("sad", "fear"):
        for emo in ("Sadness", "Fear"):
            B = P[(P["condition"] == cond) & (P["emotion"] == emo)].set_index("subj")
            for meg_block in ("baseline", cond):
                M = meg_subj[meg_subj["condition"] == meg_block]
                for (meth, meas, grp), x in M.groupby(["method", "measure", "group"]):
                    x = x.set_index("subj")["value"]
                    for bc in behav_cols:
                        j = pd.concat([x, B[bc]], axis=1, join="inner").dropna()
                        n = len(j)
                        if n < 8:
                            continue
                        r, p = stats.pearsonr(j.iloc[:, 0], j.iloc[:, 1])
                        rho, prho = stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])
                        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
                        rows.append(dict(
                            behav_condition=cond, emotion=emo, target_emotion=emo == TARGET[cond],
                            meg_block=meg_block, method=meth, measure=meas, group=grp, behav_param=bc,
                            n_subjects=n, pearson_r=r, r_ci_lo=np.tanh(z - 1.96 * se),
                            r_ci_hi=np.tanh(z + 1.96 * se), p_pearson=p, spearman_rho=rho,
                            p_spearman=prho,
                            primary=(grp == PRIMARY["group"] and meas == PRIMARY["measure"]
                                     and meg_block == PRIMARY["meg_block"]
                                     and bc == PRIMARY["behav"] and emo == TARGET[cond])))
    R = pd.DataFrame(rows)
    if len(R):
        R["p_pearson_fdr"] = bh(R["p_pearson"])
        R["p_spearman_fdr"] = bh(R["p_spearman"])
    R.to_csv(out / "M1_between_subject.csv", index=False)
    print(f"step 1 done: {len(R)} tests, {int(R['primary'].sum()) if len(R) else 0} flagged primary")


# ==========================================================================
# step 2 within-subject (network level)
# ==========================================================================
def ols_cluster(y, X, groups):
    import statsmodels.api as sm
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def within(mg, d, offset, out):
    import statsmodels.api as sm
    mg = mg.copy()
    mg["probe_idx"] = mg["snippet_idx"] + offset
    rows = []
    for emo in ("Sadness", "Fear"):
        b = d[(d["emotion"] == emo) & d["delta"].notna()][
            ["subj", "condition", "probe_idx", "delta", "prev_rating", "log_t"]]
        b["prev_rating_c"] = b["prev_rating"] - b.groupby(["subj", "condition"])["prev_rating"].transform("mean")
        for (meth, meas, grp), x in mg.groupby(["method", "measure", "group"]):
            j = b.merge(x[["subj", "condition", "probe_idx", "value"]], on=["subj", "condition", "probe_idx"])
            if j["subj"].nunique() < 8 or len(j) < 100:
                continue
            j["meg_z"] = j.groupby(["subj", "condition"])["value"].transform(
                lambda v: (v - v.mean()) / v.std() if v.std() > 0 else v * 0)
            dm = pd.get_dummies(j["condition"], drop_first=False).astype(float)
            dm = dm.drop(columns=[c for c in ("baseline",) if c in dm])
            X = pd.concat([j[["meg_z", "prev_rating_c", "log_t"]], dm], axis=1)
            X = sm.add_constant(X)
            try:
                f = ols_cluster(j["delta"].values, X, j["subj"].values)
            except Exception:
                continue
            r = dict(emotion=emo, method=meth, measure=meas, group=grp, n_obs=len(j),
                     n_subjects=j["subj"].nunique(), beta_meg=f.params["meg_z"],
                     se_meg=f.bse["meg_z"], p_meg=f.pvalues["meg_z"])
            Xi = X.copy()
            for c in dm.columns:
                Xi[f"meg_z:{c}"] = X["meg_z"] * X[c]
            try:
                fi = ols_cluster(j["delta"].values, Xi, j["subj"].values)
                for c in dm.columns:
                    r[f"beta_meg_x_{c}"] = fi.params[f"meg_z:{c}"]
                    r[f"p_meg_x_{c}"] = fi.pvalues[f"meg_z:{c}"]
                r["beta_meg_in_baseline"] = fi.params["meg_z"]
            except Exception:
                pass
            rows.append(r)
    W = pd.DataFrame(rows)
    if len(W):
        W["p_meg_fdr"] = bh(W["p_meg"])
    W.to_csv(out / "M2_within_subject.csv", index=False)
    print(f"step 2 done: {len(W)} models")


# ==========================================================================
# step 3 multivariate decoding
# ==========================================================================
def decoding(m, d, offset, out, n_perm, n_folds=10, max_missing=0.2):
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(0)
    rows, wrows = [], []
    m = m.copy()
    m["probe_idx"] = m["snippet_idx"] + offset
    m["feat"] = m["unit_name"] + "|" + m["measure"]
    for meth, mm in m.groupby("method"):
        F = mm.pivot_table(index=["subj", "condition", "probe_idx"], columns="feat", values="value")
        F = F.loc[:, F.isna().mean() <= max_missing]
        # z-score within subject x condition, impute 0
        F = F.groupby(level=["subj", "condition"]).transform(
            lambda v: (v - v.mean()) / v.std() if v.std() > 0 else v * 0).fillna(0.0)
        for emo in ("Sadness", "Fear"):
            b = d[(d["emotion"] == emo) & d["delta"].notna()].set_index(["subj", "condition", "probe_idx"])["delta"]
            j = F.join(b, how="inner")
            if j.index.get_level_values("subj").nunique() < 10:
                continue
            y = j.pop("delta")
            y = (y - y.groupby(level=["subj", "condition"]).transform("mean")).values
            X = j.values
            groups = j.index.get_level_values("subj").values
            cell = pd.Index(list(zip(groups, j.index.get_level_values("condition"))))
            alpha = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(X, y).alpha_
            k = min(n_folds, len(np.unique(groups)))

            def cv_r(yv):
                pred = np.zeros_like(yv)
                for tr, te in GroupKFold(n_splits=k).split(X, yv, groups):
                    pred[te] = Ridge(alpha=alpha).fit(X[tr], yv[tr]).predict(X[te])
                return np.corrcoef(pred, yv)[0, 1]

            r_obs = cv_r(y)
            null = []
            codes = pd.factorize(cell)[0]
            for _ in range(n_perm):
                yp = y.copy()
                for c in np.unique(codes):
                    idx = np.flatnonzero(codes == c)
                    yp[idx] = rng.permutation(yp[idx])
                null.append(cv_r(yp))
            null = np.array(null)
            rows.append(dict(method=meth, emotion=emo, n_obs=len(y), n_subjects=len(np.unique(groups)),
                             n_features=X.shape[1], alpha=alpha, cv_folds=k, r_out_of_sample=r_obs,
                             null_mean=null.mean(), null_p95=np.percentile(null, 95),
                             p_perm=(np.sum(null >= r_obs) + 1) / (n_perm + 1), n_perm=n_perm))
            coef = Ridge(alpha=alpha).fit(X, y).coef_
            for f, c in zip(j.columns, coef):
                u, ms = f.split("|", 1)
                wrows.append(dict(method=meth, emotion=emo, unit_name=u, measure=ms, weight=c))
            print(f"  decoding {meth} {emo}: r = {r_obs:.3f}, p = {rows[-1]['p_perm']:.3f}")
    pd.DataFrame(rows).to_csv(out / "M3_decoding.csv", index=False)
    W = pd.DataFrame(wrows)
    if len(W):
        W["abs_weight"] = W["weight"].abs()
        W.sort_values(["method", "emotion", "abs_weight"], ascending=[True, True, False]).to_csv(
            out / "M3_decoding_weights.csv", index=False)
    print("step 3 done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-glob", default=DEFAULT_STATIC_GLOB)
    ap.add_argument("--version", default="ecg")
    ap.add_argument("--behav-dir", default=str(BEHAV_DIR))
    ap.add_argument("--out", default=None, help="default: behav-dir")
    ap.add_argument("--snippet-offset", type=int, default=None,
                    help="probe_idx = snippet_idx + offset (default: 1 if snippets start at 0, else 0)")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--skip", nargs="*", default=[], choices=["between", "within", "decoding"])
    a = ap.parse_args()
    bdir = Path(a.behav_dir)
    out = Path(a.out) if a.out else bdir
    out.mkdir(parents=True, exist_ok=True)

    m = load_meg(a.static_glob, a.version)
    G = roi_groups(m, out)
    mg = group_values(m, G)
    d = probe_table(bdir)
    offset = a.snippet_offset if a.snippet_offset is not None else (1 if m["snippet_idx"].min() == 0 else 0)
    print(f"snippet -> probe offset: {offset}")
    alignment(mg, d, offset, out)
    if "between" not in a.skip:
        between(mg, bdir, out)
    if "within" not in a.skip:
        within(mg, d, offset, out)
    if "decoding" not in a.skip:
        decoding(m, d, offset, out, a.n_perm)


if __name__ == "__main__":
    main()
