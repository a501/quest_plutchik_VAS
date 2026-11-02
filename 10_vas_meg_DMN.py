#!/usr/bin/env python3
"""
10_vas_meg_DMN.py
================
Relate the behavioural VAS results of scripts 5-9 to the source-space MEG
dynamics (same static CSVs, ROI groups and snippet alignment as
4_meg_behav_link.py, which is imported for loading).

BEHAVIOUR (one row per subject x block x emotion; block = baseline/sad/fear)
  5_vas_fluctuation_proof   01  mean, sd, frac_nonzero, rvi_sd
                            03  latent_sd (noise-corrected fluctuation SD),
                                verdict; 'all' and 'detrended' variants
                            04  corr_sad_fear, specific_sd (per subject x block)
                            09  mixture class (selected solution)
  6_vas_noise_statespace        pooled rating-noise sigma -> chooses which
                                noise_sd row of 03/04 is used (nearest grid
                                value to sigma_noise_only of the 'all' fit)
  7_vas_first_probe_test        first, rest_mean, first_minus_rest,
                                slope_per_min, slope_wo_first
  8_vas_induction_model         start, end, decline (basedrop1 version)
  9_vas_distribution_check      no per-subject output; its result (skewed
                                values, exact zeros) is why all tests here
                                are rank-based (Spearman / Mann-Whitney)

MEG (per subject x block x network x method x measure)
  level     mean over the pre-probe snippets of the block
  delta     video block minus baseline (same subject)
  first     first snippet minus mean of snippets >= LATE_FROM (within block;
            needs the within-block snippet index from 4_)

HYPOTHESES (one PRIMARY test each; Holm across the primaries)
  H1 presence      baseline frac_nonzero (mean of the two scales)
                   ~ Salience_Interoceptive ACW0, baseline
  H2 amplitude     baseline latent_sd (mean of scales), partial on mean level
                   ~ DMN LZC, baseline
  H3 typology      baseline fluctuation class (highest vs lowest class of the
                   selected mixture solution) -> DMN LZC, Mann-Whitney
  H4 induction     decline of the induced scale (8), mean of sad+fear blocks
                   ~ delta (video - baseline) Salience_Interoceptive ACW0
  H5 start-of-rest first_minus_rest (7), baseline, mean of scales
                   ~ first-snippet contrast, Sensory_control Spec_lowhigh_8Hz
                   (+ group test: is there a first-snippet MEG effect at all?)
  H6 differentiation baseline corr_sad_fear (5/04)
                   ~ DMN MSE_costa_m2_AUC, baseline
  H7 state coupling across the 3 blocks within person (repeated-measures
                   correlation): latent_sd ~ DMN LZC
  Each hypothesis also runs an EXPLORATORY grid over all networks x
  MEG_MEASURES x methods, BH-FDR within the hypothesis.

OUTPUTS (OUT_DIR)
  V0_behaviour_table.csv   merged behavioural variables (check it first)
  V0_meg_table.csv         network-level MEG summaries
  V1_primary.csv           7 primary tests, Holm-corrected
  V2_exploratory.csv       all grid tests, BH-FDR within hypothesis
  V3_first_snippet_group.csv  H5 group test of the MEG first-snippet effect

USAGE
  python 10_vas_meg_DMN.py --static-glob "/path/*static*.csv"
  options: --behav-dir DIR --out DIR --version ecg --n-boot 2000
           --noise-sd 2.0 (override the value taken from script 6)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
BEHAV = Path("/home/ZhangJianfeng_04/emo_study/Behav/analysis")
DEFAULT_STATIC_GLOB = "/home/ZhangJianfeng_04/emo_study/MEG/SourceSpace/dynamics/*static*.csv"
SUBDIRS = dict(
    fluct="vas_fluctuation_proof_basedrop1",       # script 5 (BASE_DROP_FIRST = 1)
    noise="vas_noise_statespace",                  # script 6
    first="vas_first_probe",                       # script 7
    induction="vas_induction_model/basedrop1",     # script 8
)
NOISE_SD_FALLBACK = 2.0      # script 6 estimate 1.7 [1.4, 1.9] -> nearest grid value
NOISE_SD_GRID = (2.0, 3.0, 5.0, 7.5, 10.0, 12.5)   # as in script 5
VARIANT = "all"
LATE_FROM = 6                # snippets >= 6 precede probes >= 7 (late part, as in script 7)
MIN_N = 12
PRIMARY_METHOD = "mean_flip"
COND = {"base": "baseline", "baseline": "baseline", "sad": "sad", "fear": "fear"}
INDUCED = {"sad": "Sadness", "fear": "Fear"}
MEG_MEASURES = {
    "timescale": ["ACW0", "ACW50", "ACF_area_to_zero"],
    "aperiodic": ["IRASA_slope", "SpectralSlope"],
    "complexity": ["LZC", "SampEn_m2", "MSE_costa_m2_AUC"],
    "spectral": ["Spec_lowhigh_8Hz", "MedianFreq", "IRASA_oscpower", "SpectralEntropy"],
}
ALL_MEASURES = [m for v in MEG_MEASURES.values() for m in v]

HYP = [
    dict(id="H1", label="presence: baseline share of non-zero ratings", kind="corr",
         behav="frac_nonzero", block="baseline", emotion="mean", meg="level",
         group="Salience_Interoceptive", measure="ACW0", covar=None),
    dict(id="H2", label="amplitude: baseline latent SD (partial on mean)", kind="corr",
         behav="latent_sd", block="baseline", emotion="mean", meg="level",
         group="DMN", measure="LZC", covar="mean"),
    dict(id="H3", label="typology: highest vs lowest fluctuation class at baseline", kind="class",
         behav="class_label", block="baseline", emotion="mean", meg="level",
         group="DMN", measure="LZC", covar=None),
    dict(id="H4", label="induction: decline of induced scale vs MEG change from baseline", kind="corr",
         behav="decline", block="video", emotion="induced", meg="delta",
         group="Salience_Interoceptive", measure="ACW0", covar=None),
    dict(id="H5", label="start of rest: first minus rest vs first-snippet MEG contrast", kind="corr",
         behav="first_minus_rest", block="baseline", emotion="mean", meg="first",
         group="Sensory_control", measure="Spec_lowhigh_8Hz", covar=None),
    dict(id="H6", label="differentiation: Sadness-Fear coupling at baseline", kind="corr",
         behav="corr_sad_fear", block="baseline", emotion="pair", meg="level",
         group="DMN", measure="MSE_costa_m2_AUC", covar=None),
    dict(id="H7", label="state coupling: latent SD vs MEG across blocks within person", kind="rmcorr",
         behav="latent_sd", block="all", emotion="mean", meg="level",
         group="DMN", measure="LZC", covar=None),
]


def load_module4():
    spec = importlib.util.spec_from_file_location("meg4", HERE / "4_meg_behav_link.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==========================================================================
# behaviour
# ==========================================================================
def _ids(df):
    df = df.copy()
    df["subj"] = df["subject"].astype(int).astype(str).str.zfill(3)
    df["condition"] = df["block_role"].astype(str).str.strip().str.lower().map(COND)
    return df.dropna(subset=["condition"])


def pick_noise_sd(bdir, override):
    if override is not None:
        return override
    f = bdir / SUBDIRS["noise"] / "sigma_estimates.csv"
    if f.exists():
        s = pd.read_csv(f)
        row = s[s["fit"] == "all"]
        if len(row):
            sig = float(row["sigma_noise_only"].iloc[0])
            pick = min(NOISE_SD_GRID, key=lambda g: abs(g - sig))
            print(f"noise SD from script 6: {sig:.2f} -> grid value {pick}")
            return pick
    print(f"script 6 output not found -> noise SD {NOISE_SD_FALLBACK}")
    return NOISE_SD_FALLBACK


def load_behaviour(bdir, noise_sd):
    fd = bdir / SUBDIRS["fluct"]
    keys = ["subj", "condition", "emotion"]
    desc = _ids(pd.read_csv(fd / "01_series_descriptives.csv"))
    desc = desc[desc["variant"] == VARIANT][keys + ["mean", "sd", "frac_nonzero", "rvi_sd"]]

    fl = _ids(pd.read_csv(fd / "03_fluctuation_per_series.csv"))
    fl = fl[np.isclose(fl["noise_sd"], noise_sd)]
    lat = fl[fl["variant"] == VARIANT][keys + ["latent_sd", "verdict"]]
    lat_d = (fl[fl["variant"] == "detrended"][keys + ["latent_sd"]]
             .rename(columns={"latent_sd": "latent_sd_detrended"}))
    B = desc.merge(lat, on=keys, how="outer").merge(lat_d, on=keys, how="outer")

    f9 = fd / "09_mixture_assignments.csv"
    if f9.exists():
        mx = _ids(pd.read_csv(f9))
        mx = mx[mx["solution"] == "selected"]
        mx["n_classes"] = mx["k"]
        B = B.merge(mx[keys + ["class_label", "n_classes"]], on=keys, how="left")

    f7 = bdir / SUBDIRS["first"] / "first_probe_per_series.csv"
    if f7.exists():
        s7 = _ids(pd.read_csv(f7))
        s7["first_minus_rest"] = s7["first"] - s7["rest_mean"]
        B = B.merge(s7[keys + ["block_num", "first", "rest_mean", "first_minus_rest",
                               "slope_per_min", "slope_wo_first"]], on=keys, how="outer")

    f8 = bdir / SUBDIRS["induction"] / "induction_per_series.csv"
    if f8.exists():
        s8 = _ids(pd.read_csv(f8))
        B = B.merge(s8[keys + ["start", "end", "decline"]], on=keys, how="outer")

    # per subject x block (pair) measures from 04
    dis = _ids(pd.read_csv(fd / "04_dissociation_per_subject_block.csv"))
    dis = dis[(dis["variant"] == VARIANT) & np.isclose(dis["noise_sd"], noise_sd)]
    pair = dis[["subj", "condition", "corr_sad_fear", "specific_sd"]].drop_duplicates(["subj", "condition"])
    return B, pair


def behaviour_vector(B, pair, h):
    """Subject-indexed Series (or subject x condition for rmcorr) for one hypothesis."""
    var = h["behav"]
    if h["emotion"] == "pair":
        x = pair[pair["condition"] == h["block"]].set_index("subj")[var]
        return x.dropna()
    if h["kind"] == "rmcorr":
        return B.groupby(["subj", "condition"])[var].mean().dropna()
    if h["emotion"] == "induced":
        rows = [B[(B["condition"] == c) & (B["emotion"] == e)].set_index("subj")[var]
                for c, e in INDUCED.items()]
        return pd.concat(rows, axis=1).mean(axis=1, skipna=False).dropna()
    b = B[B["condition"] == h["block"]]
    if h["kind"] == "class":
        # per scale: 1 = highest class, 0 = lowest class (others / one-class solutions dropped)
        b = b.dropna(subset=["class_label"])
        b = b[b["n_classes"] > 1]
        top = b["class_label"] == b["n_classes"]
        low = b["class_label"] == 1
        b = b[top | low].assign(high=top[top | low].astype(float))
        return b.groupby("subj")["high"].max()          # high in either scale
    return b.groupby("subj")[var].mean().dropna()


def covariate_vector(B, h):
    if not h["covar"]:
        return None
    return B[B["condition"] == h["block"]].groupby("subj")[h["covar"]].mean()


# ==========================================================================
# MEG
# ==========================================================================
def meg_tables(m4, pattern, version):
    m = m4.localize_snippets(m4.load_meg(pattern, version))
    m = m[m["measure"].isin(ALL_MEASURES)]
    rows = []
    for g, (inc, exc) in m4.ROI_GROUPS.items():
        u = m["unit_name"]
        sel = u.str.contains(inc, regex=True) & ~u.str.contains(exc, regex=True)
        rows.append(m[sel].assign(group=g))
    x = pd.concat(rows, ignore_index=True)
    net = (x.groupby(["subj", "condition", "snippet_idx", "method", "measure", "group"])["value"]
           .mean().reset_index())
    k = ["subj", "condition", "method", "measure", "group"]
    level = net.groupby(k)["value"].mean().rename("level").reset_index()
    first = net[net["snippet_idx"] == 0].set_index(k)["value"]
    late = net[net["snippet_idx"] >= LATE_FROM].groupby(k)["value"].mean()
    fc = (first - late).rename("first").reset_index()
    T = level.merge(fc, on=k, how="left")
    base = T[T["condition"] == "baseline"].set_index(["subj", "method", "measure", "group"])["level"]
    T["delta"] = T["level"].values - base.reindex(
        pd.MultiIndex.from_frame(T[["subj", "method", "measure", "group"]])).values
    return T


def meg_vector(T, h, method, group, measure):
    t = T[(T["method"] == method) & (T["group"] == group) & (T["measure"] == measure)]
    if h["kind"] == "rmcorr":
        return t.set_index(["subj", "condition"])[h["meg"]].dropna()
    if h["block"] == "video":
        return t[t["condition"].isin(INDUCED)].groupby("subj")[h["meg"]].mean().dropna()
    return t[t["condition"] == h["block"]].set_index("subj")[h["meg"]].dropna()


# ==========================================================================
# statistics
# ==========================================================================
def _rank_resid(y, z):
    ry, rz = stats.rankdata(y), stats.rankdata(z)
    b = np.polyfit(rz, ry, 1)
    return ry - np.polyval(b, rz)


def spearman_test(x, y, z=None, n_boot=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if z is not None:                      # partial Spearman: ranks residualised on z
        z = np.asarray(z, float)
        x, y = _rank_resid(x, z), _rank_resid(y, z)
        r, p = stats.pearsonr(x, y)
    else:
        r, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            bs.append(stats.spearmanr(x[i], y[i])[0] if z is None else np.corrcoef(x[i], y[i])[0, 1])
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return dict(n=n, stat=r, ci_lo=lo, ci_hi=hi, p=p)


def class_test(cls, meg):
    hi, lo = meg[cls == 1], meg[cls == 0]
    if len(hi) < 3 or len(lo) < 3:
        return dict(n=len(cls), p=np.nan)
    u, p = stats.mannwhitneyu(hi, lo)
    return dict(n=len(cls), n_high=len(hi), n_low=len(lo), stat=u / (len(hi) * len(lo)),  # AUC-type
                median_high=np.median(hi), median_low=np.median(lo), p=p)


def rmcorr(xy):
    """Repeated-measures correlation (Bakdash & Marusich 2017) on a frame with
    columns subj, x, y; ranks within the whole sample first (rank-based)."""
    d = xy.dropna().copy()
    d = d[d.groupby("subj")["x"].transform("size") >= 2]
    d["x"], d["y"] = stats.rankdata(d["x"]), stats.rankdata(d["y"])
    xc = d["x"] - d.groupby("subj")["x"].transform("mean")
    yc = d["y"] - d.groupby("subj")["y"].transform("mean")
    n_s, n = d["subj"].nunique(), len(d)
    if n_s < MIN_N or (xc ** 2).sum() == 0 or (yc ** 2).sum() == 0:
        return dict(n=n_s, p=np.nan)
    r = float((xc * yc).sum() / np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    df = n - n_s - 1
    t = r * np.sqrt(df / max(1 - r ** 2, 1e-12))
    return dict(n=n_s, n_obs=n, stat=r, p=2 * stats.t.sf(abs(t), df))


def run_one(h, bvec, cvec, T, method, group, measure, n_boot):
    mv = meg_vector(T, h, method, group, measure)
    if h["kind"] == "rmcorr":
        j = pd.concat([bvec.rename("x"), mv.rename("y")], axis=1, join="inner").reset_index()
        return rmcorr(j)
    j = pd.concat([bvec.rename("b"), mv.rename("m")], axis=1, join="inner")
    if cvec is not None:
        j = j.join(cvec.rename("c"), how="inner")
    j = j.dropna()
    if len(j) < MIN_N:
        return dict(n=len(j), p=np.nan)
    if h["kind"] == "class":
        return class_test(j["b"].values, j["m"].values)
    return spearman_test(j["b"].values, j["m"].values,
                         j["c"].values if cvec is not None else None, n_boot)


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


def run(B, pair, T, bh, out, n_boot):
    out.mkdir(parents=True, exist_ok=True)
    prim, expl = [], []
    for h in HYP:
        if h["behav"] not in B.columns and h["behav"] not in pair.columns:
            print(f"{h['id']}: behavioural variable {h['behav']} missing - skipped")
            continue
        bvec = behaviour_vector(B, pair, h)
        cvec = covariate_vector(B, h)
        meta = dict(hyp=h["id"], label=h["label"], behaviour=h["behav"], block=h["block"],
                    meg_summary=h["meg"], covariate=h["covar"] or "")
        r = run_one(h, bvec, cvec, T, PRIMARY_METHOD, h["group"], h["measure"], n_boot)
        prim.append({**meta, "method": PRIMARY_METHOD, "group": h["group"], "measure": h["measure"], **r})
        for method in sorted(T["method"].unique()):
            for group in sorted(T["group"].unique()):
                for measure in ALL_MEASURES:
                    r = run_one(h, bvec, cvec, T, method, group, measure, n_boot=0)
                    expl.append({**meta, "method": method, "group": group, "measure": measure,
                                 "primary": (method, group, measure) == (PRIMARY_METHOD, h["group"], h["measure"]),
                                 **{k: v for k, v in r.items() if k not in ("ci_lo", "ci_hi")}})
        print(f"{h['id']} primary: n = {prim[-1].get('n')}, stat = {prim[-1].get('stat', np.nan):.3f}, "
              f"p = {prim[-1].get('p', np.nan):.4f}")
    P = pd.DataFrame(prim)
    P["p_holm"] = holm(P["p"])
    P.to_csv(out / "V1_primary.csv", index=False)
    E = pd.DataFrame(expl)
    E["p_fdr_within_hyp"] = E.groupby("hyp")["p"].transform(lambda p: pd.Series(bh(p.values), index=p.index))
    E.to_csv(out / "V2_exploratory.csv", index=False)

    # H5 group check: is there a first-snippet effect in MEG at all (any block)?
    g = []
    for (c, meth, meas, grp), x in T.groupby(["condition", "method", "measure", "group"]):
        v = x["first"].dropna()
        if len(v) >= MIN_N and np.any(v != 0):
            w = stats.wilcoxon(v)
            g.append(dict(condition=c, method=meth, measure=meas, group=grp, n=len(v),
                          median_first_minus_late=v.median(), p_wilcoxon=w.pvalue))
    Gt = pd.DataFrame(g)
    if len(Gt):
        Gt["p_fdr"] = bh(Gt["p_wilcoxon"].values)
    Gt.to_csv(out / "V3_first_snippet_group.csv", index=False)
    return P, E


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-glob", default=DEFAULT_STATIC_GLOB)
    ap.add_argument("--version", default="ecg")
    ap.add_argument("--behav-dir", default=str(BEHAV))
    ap.add_argument("--out", default=None, help="default: <behav-dir>/vas_meg_DMN")
    ap.add_argument("--noise-sd", type=float, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    bdir = Path(a.behav_dir)
    out = Path(a.out) if a.out else bdir / "vas_meg_DMN"
    out.mkdir(parents=True, exist_ok=True)
    m4 = load_module4()
    noise = pick_noise_sd(bdir, a.noise_sd)
    B, pair = load_behaviour(bdir, noise)
    B.to_csv(out / "V0_behaviour_table.csv", index=False)
    print(f"behaviour: {B['subj'].nunique()} subjects, {len(B)} series, columns {list(B.columns)}")
    T = meg_tables(m4, a.static_glob, a.version)
    T.to_csv(out / "V0_meg_table.csv", index=False)
    print(f"MEG: {T['subj'].nunique()} subjects; overlap with behaviour: "
          f"{len(set(T['subj']) & set(B['subj']))}")
    run(B, pair, T, m4.bh, out, a.n_boot)
    print(f"outputs in {out}")


if __name__ == "__main__":
    sys.exit(main())
