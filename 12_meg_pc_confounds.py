#!/usr/bin/env python3
"""
12_meg_pc_confounds.py
======================
Are the MEG principal components (PCs) that relate to the VAS behaviour
explained by non-emotional factors - head position, noise / data quality,
heart rate, drowsiness?

The 12 MEG measures x 4 networks of 10_vas_meg_networks.py are highly
correlated (the same measure correlates r = .75-.99 across networks), so they
are reduced to PCs first. The PCA uses MEG only, so choosing a PC is not
biased by the behaviour.

STEP 1  PCA (per method) on the baseline block, subject x (measure, network)
        matrix of V0_meg_table.csv (level = mean over snippets), z-scored.
        All PCs with >= MIN_VAR of the variance (max MAX_PC). Sad and fear
        block scores use the BASELINE weights, so a PC means the same thing
        in every block.
STEP 2  CONFOUNDS per subject x block (cached in C0_confounds.csv)
        from the preprocessed MEG file ({stem}_preproc_ecg.fif, 3_preproc/):
          head_x/y/z_mm, head_dist_mm  head origin in device coordinates
                                       (dev_head_t): position in the helmet,
                                       distance to the sensors
          meg_mag_rms_fT, meg_grad_rms  broadband signal size (noise / SNR)
          heart_rate_bpm               ECG channel, R-peaks (arousal)
          blink_rate_per_min           EOG channels (drowsiness / arousal)
        from {stem}_ica.fif: ica_n_components, ica_n_excluded (data quality)
        from vas_probes_long.csv: timeout_rate (missed ratings: drowsiness /
                                  disengagement)
STEP 3  TESTS (Spearman / rank-based partial correlations)
        C1  every PC x every confound, per block
        C2  every PC x every behavioural variable, per block
        C3  PC-behaviour link before and after removing each confound alone
            and all confounds together (latent_sd always also partialled on
            the mean level, as in 10_)
        If a PC-behaviour link survives the confound adjustment, it is not
        explained by these factors; if it disappears, the confound that
        removes it is named in C3.

OUTPUTS (OUT_DIR)
  C0_confounds.csv, C1_pc_vs_confounds.csv, C2_pc_vs_behaviour.csv,
  C3_pc_behaviour_adjusted.csv, C4_pca_variance.csv, C4_pca_loadings.csv

USAGE (cluster)
  cd /home/ZhangJianfeng_04/emo_study/Behav/
  export PATH="/home/ZhangJianfeng_04/.conda/envs/MEG_env/bin:$PATH"
  python3 12_meg_pc_confounds.py              # confounds read once, then cached
  python3 12_meg_pc_confounds.py --refresh    # re-read the MEG files
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

BEHAV = Path("/home/ZhangJianfeng_04/emo_study/Behav")
NET_DIR = BEHAV / "analysis" / "vas_meg_networks"          # outputs of 10_
PROBES = BEHAV / "dynamics_models" / "vas_probes_long.csv"
PREPROC = Path("/home/ZhangJianfeng_04/emo_study/MEG/3_preproc")
OUT_DIR = BEHAV / "analysis" / "vas_meg_pc_confounds"

MIN_VAR = 0.02
MAX_PC = 10
MIN_N = 12
COND = {"base": "baseline", "sad": "sad", "fear": "fear"}
BEHAV_VARS = ["latent_sd", "mean", "frac_nonzero", "sd", "first_minus_rest"]
CONFOUNDS = ["head_x_mm", "head_y_mm", "head_z_mm", "head_dist_mm",
             "meg_mag_rms_fT", "meg_grad_rms", "heart_rate_bpm", "blink_rate_per_min",
             "ica_n_components", "ica_n_excluded", "timeout_rate"]
RMS_WINDOW_S = 60.0


# ==========================================================================
# confounds
# ==========================================================================
def parse_stem(name):
    m = re.match(r"^(\d+)_.*_\db_(base|sad|fear)\d*_preproc_ecg$", name)
    return (m.group(1).zfill(3), COND[m.group(2)]) if m else (None, None)


def meg_confounds(path):
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
    info, sf = raw.info, raw.info["sfreq"]
    r = {}
    t = info.get("dev_head_t")
    if t is not None:
        xyz = np.asarray(t["trans"])[:3, 3] * 1000.0
        r.update(head_x_mm=xyz[0], head_y_mm=xyz[1], head_z_mm=xyz[2],
                 head_dist_mm=float(np.linalg.norm(xyz)))
    # broadband signal size on a middle window
    mid = raw.times[-1] / 2
    seg = raw.copy().crop(max(0, mid - RMS_WINDOW_S / 2), min(raw.times[-1], mid + RMS_WINDOW_S / 2))
    for kind, key, scale in (("mag", "meg_mag_rms_fT", 1e15), ("grad", "meg_grad_rms", 1e13)):
        picks = mne.pick_types(info, meg=kind, exclude="bads")
        if len(picks):
            x = seg.get_data(picks=picks)
            r[key] = float(np.median(np.sqrt(np.mean((x - x.mean(1, keepdims=True)) ** 2, 1)))) * scale
    dur_min = raw.times[-1] / 60.0
    ecg = [c for c in info["ch_names"] if "ECG" in c.upper()]
    if ecg:
        try:
            e = raw.copy().pick(ecg[:1]).load_data()
            ev, _, _ = mne.preprocessing.find_ecg_events(e, ch_name=ecg[0], verbose=False)
            r["heart_rate_bpm"] = len(ev) / dur_min
        except Exception as ex:
            print(f"    ECG failed: {ex}")
    eog = [c for c in info["ch_names"] if "EOG" in c.upper()]
    if eog:
        try:
            e = raw.copy().pick(eog).load_data()
            ev = mne.preprocessing.find_eog_events(e, ch_name=eog, verbose=False)
            r["blink_rate_per_min"] = len(ev) / dur_min
        except Exception as ex:
            print(f"    EOG failed: {ex}")
    ica_f = str(path).replace("_preproc_ecg.fif", "_ica.fif")
    if Path(ica_f).exists():
        ica = mne.preprocessing.read_ica(ica_f, verbose=False)
        r.update(ica_n_components=ica.n_components_, ica_n_excluded=len(ica.exclude))
    return r


def load_confounds(out, refresh):
    cache = out / "C0_confounds.csv"
    if cache.exists() and not refresh:
        print(f"confounds from cache {cache}")
        return pd.read_csv(cache, dtype={"subj": str})
    rows = []
    files = sorted(f for f in glob.glob(str(PREPROC / "*_preproc_ecg.fif"))
                   if not re.search(r"-\d+\.fif$", f))
    print(f"reading {len(files)} preprocessed MEG files ...")
    for f in files:
        subj, cond = parse_stem(Path(f).stem)
        if subj is None or int(subj) >= 200:
            continue
        try:
            r = meg_confounds(f)
        except Exception as ex:
            print(f"  {Path(f).name}: failed ({ex})")
            continue
        rows.append(dict(subj=subj, condition=cond, file=Path(f).name, **r))
        print(f"  {Path(f).name}: ok")
    C = pd.DataFrame(rows)
    if len(C):   # one row per subject x block (first file if a block was cut twice)
        C = C.drop_duplicates(["subj", "condition"])
    d = pd.read_csv(PROBES, dtype={"subj": str})
    d = d[d["subj"].astype(int) < 200]
    to = (d.drop_duplicates(["subj", "condition", "probe_idx"])
          .groupby(["subj", "condition"])["timeout"].mean().rename("timeout_rate").reset_index())
    C = to.merge(C, on=["subj", "condition"], how="outer") if len(C) else to
    C.to_csv(cache, index=False)
    return C


# ==========================================================================
# PCA
# ==========================================================================
def pca_scores(T):
    scores, var_rows, load_rows = [], [], []
    for meth, x in T.groupby("method"):
        def mat(c):
            return x[x["condition"] == c].pivot_table(index="subj", columns=["measure", "group"], values="level")
        X = mat("baseline").dropna()
        mu, sd = X.mean(), X.std()
        Z = (X - mu) / sd
        _, s, vt = np.linalg.svd(Z.values, full_matrices=False)
        ev = s ** 2 / np.sum(s ** 2)
        k = int(min(MAX_PC, max(1, np.sum(ev >= MIN_VAR))))
        W = vt[:k].T
        W = W * np.sign(W[np.argmax(np.abs(W), axis=0), np.arange(k)])   # largest loading positive
        for i in range(k):
            var_rows.append(dict(method=meth, pc=f"PC{i + 1}", var_explained=ev[i],
                                 cum_var=ev[:i + 1].sum(), n_subjects=len(Z)))
            for (meas, grp), w in zip(Z.columns, W[:, i]):
                load_rows.append(dict(method=meth, pc=f"PC{i + 1}", measure=meas, group=grp, loading=w))
        for c in ("baseline", "sad", "fear"):
            Y = mat(c).reindex(columns=X.columns).dropna()
            S = ((Y - mu) / sd).values @ W
            for subj, row in zip(Y.index, S):
                scores += [dict(method=meth, condition=c, subj=subj, pc=f"PC{i + 1}", score=row[i])
                           for i in range(k)]
    return pd.DataFrame(scores), pd.DataFrame(var_rows), pd.DataFrame(load_rows)


# ==========================================================================
# statistics
# ==========================================================================
def partial_rank(x, y, Z=None):
    """Spearman-type partial correlation: ranks of x and y residualised on the
    ranks of the covariates Z (n x k). Returns r, p, n."""
    m = np.isfinite(x) & np.isfinite(y)
    if Z is not None:
        Z = np.asarray(Z, float).reshape(len(x), -1)
        m &= np.isfinite(Z).all(1)
    n = int(m.sum())
    k = 0 if Z is None else Z.shape[1]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-dir", default=str(NET_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net = Path(a.net_dir)

    T = pd.read_csv(net / "V0_meg_table.csv", dtype={"subj": str})
    B = pd.read_csv(net / "V0_behaviour_table.csv", dtype={"subj": str})
    S, Var, Load = pca_scores(T)
    Var.to_csv(out / "C4_pca_variance.csv", index=False)
    Load.to_csv(out / "C4_pca_loadings.csv", index=False)
    for meth, v in Var.groupby("method"):
        print(f"PCA {meth}: {len(v)} PCs, variance " +
              ", ".join(f"{p} {x:.0%}" for p, x in zip(v["pc"], v["var_explained"])))

    C = load_confounds(out, a.refresh)
    conf = [c for c in CONFOUNDS if c in C.columns and C[c].notna().sum() >= MIN_N]
    print(f"confounds available: {conf}")
    bvars = [v for v in BEHAV_VARS if v in B.columns]
    Bm = B.groupby(["subj", "condition"])[bvars].mean().reset_index()     # mean of the two scales

    W = (S.pivot_table(index=["method", "condition", "subj"], columns="pc", values="score")
         .reset_index().merge(Bm, on=["subj", "condition"], how="left")
         .merge(C[["subj", "condition"] + conf], on=["subj", "condition"], how="left"))
    pcs = sorted([c for c in W.columns if re.fullmatch(r"PC\d+", c)], key=lambda s: int(s[2:]))

    r1, r2, r3 = [], [], []
    for (meth, cond), g in W.groupby(["method", "condition"]):
        for pc in pcs:
            x = g[pc].values.astype(float)
            for cf in conf:
                r, p, n = partial_rank(x, g[cf].values.astype(float))
                r1.append(dict(method=meth, condition=cond, pc=pc, confound=cf, n=n, rho=r, p=p))
            for bv in bvars:
                y = g[bv].values.astype(float)
                base_cov = g[["mean"]].values.astype(float) if bv == "latent_sd" else None
                r, p, n = partial_rank(x, y, base_cov)
                r2.append(dict(method=meth, condition=cond, pc=pc, behaviour=bv,
                               partial_on_mean=bv == "latent_sd", n=n, rho=r, p=p))
                adj = [("none", [])] + [(cf, [cf]) for cf in conf] + [("all confounds", conf)]
                for lab, cols in adj:
                    cov_cols = (["mean"] if bv == "latent_sd" else []) + cols
                    Z = g[cov_cols].values.astype(float) if cov_cols else None
                    r, p, n = partial_rank(x, y, Z)
                    # unadjusted link on the SAME subjects (confounds can be missing
                    # for some), so a change in rho is due to the adjustment only
                    keep = g[cols].notna().all(1).values if cols else np.ones(len(g), bool)
                    Zb = base_cov[keep] if base_cov is not None else None
                    r0, _, _ = partial_rank(x[keep], y[keep], Zb)
                    r3.append(dict(method=meth, condition=cond, pc=pc, behaviour=bv,
                                   adjusted_for=lab, n=n, rho=r, p=p,
                                   rho_unadjusted_same_subjects=r0))
    R1, R2, R3 = pd.DataFrame(r1), pd.DataFrame(r2), pd.DataFrame(r3)
    R1.to_csv(out / "C1_pc_vs_confounds.csv", index=False)
    R2.to_csv(out / "C2_pc_vs_behaviour.csv", index=False)
    R3.to_csv(out / "C3_pc_behaviour_adjusted.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nPC x confound, baseline, p < .05:")
    s = R1[(R1["condition"] == "baseline") & (R1["p"] < .05)].sort_values("p")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}") if len(s) else "  none")
    print("\nPC x behaviour, p < .05 (latent_sd partialled on mean):")
    s = R2[R2["p"] < .05].sort_values(["condition", "p"])
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}") if len(s) else "  none")
    if len(s):
        keys = s[["method", "condition", "pc", "behaviour"]].drop_duplicates()
        adj = R3.merge(keys, on=["method", "condition", "pc", "behaviour"])
        tab = adj.pivot_table(index=["method", "condition", "pc", "behaviour"],
                              columns="adjusted_for", values="rho")
        print("\nthose links after confound adjustment (rho):")
        print(tab.round(2).to_string())
    print(f"\noutputs in {out}")


if __name__ == "__main__":
    main()
