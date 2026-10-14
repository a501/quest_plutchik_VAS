# -*- coding: utf-8 -*-
"""
correlate_quest_dynamics.py
===========================
Exploratory correlations between the questionnaire scores
(score_questionnaires.py) and the Plutchik EMA dynamic measures
(extract_plutchik_dynamics.py).

NO multiple-comparison correction, per request. This is explicitly a
hypothesis-GENERATING pass, not a confirmatory one.

--------------------------------------------------------------------
READ THIS BEFORE INTERPRETING ANY r
--------------------------------------------------------------------
The number of tests here is very large: (n questionnaire scores) x
(n measures) x (n emotions). With ~55 scores, ~19 measures and 52
emotions that is on the order of 50,000 uncorrected tests, of which
~5% (about 2,700) will reach p<.05 by chance alone even if nothing
is related. Two consequences:

  * The count of "significant" results is meaningless on its own.
    The script therefore prints the OBSERVED vs CHANCE-EXPECTED count
    of p<.05 hits, so you can see whether the yield exceeds noise.
  * With small n, a correlation needs to be very large to reach
    p<.05 at all (n=7 needs |r|>=.75). Such estimates are extremely
    unstable - a single subject moves them substantially. Effect
    sizes here should be read as "worth a closer look later", never
    as findings.

By default Spearman is used rather than Pearson: with a handful of
subjects, a single outlier dominates a Pearson r, and several of the
dynamic measures (ACW lags, frequencies) are not plausibly linear-
normal. Set METHOD = "pearson" to switch.

Usage:
    python3 correlate_quest_dynamics.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ===================================================================
# CONFIG
# ===================================================================

DYN_CSV = Path("/home/ZhangJianfeng_04/emo_study/Behav/Pluchik/analysis/"
               "plutchik_dynamic_measures.csv")
QUEST_CSV = Path("/home/ZhangJianfeng_04/emo_study/Behav/quest/analysis/"
                 "questionnaire_scores_wide.csv")
OUT_DIR = Path("/home/ZhangJianfeng_04/emo_study/Behav/quest/analysis/"
               "corr_with_dynamics")

METHOD = "spearman"          # or "pearson"
LENGTH_MODE = "full"         # must match PRIMARY_LENGTH_MODE in script 1
MIN_N = 5                    # minimum complete pairs to attempt a correlation
PRIMARY_EMOTIONS = ["joy", "trust", "fear", "sadness",
                    "disgust", "anger", "anticipation", "surprise"]


# --- heatmap geometry (kept identical to 4_correlate_all_modalities.py) ---
# Cells are square; the factor with MORE variables goes on the X axis and
# the smaller one on Y, with paging along X. Cell size is fixed so label
# legibility does not change with matrix shape.
CELL_IN = 0.46
FIG_W, FIG_H_MAX = 13.33, 7.5
_Y_LABEL_IN, _CBAR_IN, _X_LABEL_IN, _TITLE_IN = 2.8, 1.1, 2.3, 0.8
FS_TITLE, FS_XTICK, FS_YTICK, FS_ANNOT, FS_CBAR = 16, 11, 13, 10, 11


def _draw_heatmap_pages(mat, pmat, title, stem, out_dir, nmat=None):
    """Square-celled, paged heatmap. Larger factor on X, smaller on Y."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import textwrap

    if mat.shape[0] > mat.shape[1]:
        mat, pmat = mat.T, pmat.T
        if nmat is not None:
            nmat = nmat.T          # must follow mat, not be guessed from shape

    n_rows = mat.shape[0]
    plot_w = FIG_W - _Y_LABEL_IN - _CBAR_IN
    per_page = max(6, int(plot_w / CELL_IN))
    cell = min(CELL_IN, (FIG_H_MAX - _X_LABEL_IN - _TITLE_IN) / max(n_rows, 1))
    fig_h = min(FIG_H_MAX, n_rows * cell + _X_LABEL_IN + _TITLE_IN)

    # per-cell N alongside the figure: n is NOT constant across a
    # heatmap (pairwise deletion). Two things shrink it: a subject whose
    # series was flat for that emotion has no LZC/SampEn/ACW/spectral
    # value at all, and scales added to the battery later (EAQ, TSQ)
    # exist only for subjects run after the change.
    if nmat is not None:
        try:
            nmat.reindex(index=mat.index, columns=mat.columns).to_csv(
                Path(out_dir) / f"{stem}_n.csv")
        except Exception as e:
            print(f"  [warn] could not write {stem}_n.csv: {e}")

    pages = max(1, int(np.ceil(mat.shape[1] / per_page)))
    for pg in range(pages):
        sl = slice(pg * per_page, (pg + 1) * per_page)
        sub, psub = mat.iloc[:, sl], pmat.iloc[:, sl]
        if sub.shape[1] == 0:
            continue
        fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
        sns.heatmap(sub, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    annot=_stars(psub), fmt="", annot_kws={"size": FS_ANNOT},
                    cbar_kws={"label": f"{METHOD} r", "shrink": .85},
                    linewidths=0.4, linecolor="white", square=True)
        t = title + (f"   [page {pg+1}/{pages}]" if pages > 1 else "")
        ax.set_title(textwrap.fill(t, width=64), fontsize=FS_TITLE,
                     fontweight="bold", pad=14, loc="left")
        ax.tick_params(axis="x", labelrotation=90, labelsize=FS_XTICK)
        ax.tick_params(axis="y", labelrotation=0, labelsize=FS_YTICK)
        ax.set_xlabel(""); ax.set_ylabel("")
        cb = ax.collections[0].colorbar
        cb.ax.tick_params(labelsize=FS_CBAR); cb.set_label(f"{METHOD} r",
                                                           fontsize=FS_CBAR + 1)
        fig.tight_layout()
        p = Path(out_dir) / f"{stem}_p{pg+1}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {p}")


def _stars(pmat):
    """Significance markers as a plain ndarray. Avoids DataFrame.map,
    which only exists in pandas >= 2.1."""
    return np.where(pmat.notna().to_numpy() & (pmat.to_numpy() < .05), "*", "")




def load_inputs():
    dyn = pd.read_csv(DYN_CSV)
    if "length_mode" in dyn.columns:
        dyn = dyn[dyn["length_mode"] == LENGTH_MODE]
    dyn["value_num"] = pd.to_numeric(dyn["value"], errors="coerce")
    dyn = dyn.dropna(subset=["value_num"])

    quest = pd.read_csv(QUEST_CSV)
    id_col = quest.columns[0]
    quest = quest.rename(columns={id_col: "subject"})
    quest["subject"] = quest["subject"].astype(int)   # 032 -> 32, matches others
    quest = quest.set_index("subject")

    shared = sorted(set(dyn["participant"]) & set(quest.index))
    print(f"Dynamics: {dyn['participant'].nunique()} subject(s); "
          f"questionnaires: {len(quest)} subject(s); "
          f"OVERLAP USED: {len(shared)} -> {shared}")
    if not shared:
        raise RuntimeError(
            "No subjects in common between the dynamics and questionnaire "
            "files. Check that participant IDs use the same numbering in "
            "both (Plutchik uses the experiment-ID field; questionnaires "
            "use the filename prefix).")
    if len(shared) < MIN_N:
        print(f"  WARNING: only {len(shared)} overlapping subject(s). "
              f"Correlations on this n are not interpretable; treat every "
              f"result below as descriptive only.")
    return dyn, quest, shared


def run_correlations(dyn, quest, shared):
    from scipy import stats
    fn = stats.spearmanr if METHOD == "spearman" else stats.pearsonr

    rows = []
    q_cols = [c for c in quest.columns if quest[c].notna().sum() >= MIN_N]
    for (emo, meas), g in dyn.groupby(["emotion", "measure"]):
        series = g.set_index("participant")["value_num"]
        for qc in q_cols:
            pair = pd.DataFrame({"d": series, "q": quest[qc]}).loc[
                [s for s in shared if s in series.index]].dropna()
            if len(pair) < MIN_N or pair["d"].std() == 0 or pair["q"].std() == 0:
                continue
            r, p = fn(pair["d"], pair["q"])
            rows.append(dict(emotion=emo, measure=meas, quest_score=qc,
                             r=float(r), p=float(p), n=len(pair)))
    res = pd.DataFrame(rows)
    return res


def report_yield(res):
    n_tests = len(res)
    n_sig = int((res["p"] < .05).sum())
    expected = 0.05 * n_tests
    print(f"\n  {n_tests} correlations run.")
    print(f"  p<.05: {n_sig} observed vs {expected:.0f} expected by chance "
          f"({n_sig/expected:.2f}x chance)" if expected else "")
    if expected and n_sig <= expected * 1.2:
        print(f"  -> This yield is at or near chance. Treat the whole table "
              f"as null; do not pick individual 'hits' out of it.")


def plot_corr_heatmaps(res, out_dir, emotions=None):
    """One figure per emotion: questionnaire scores x dynamic measures,
    using the same square-cell, large-tick layout as script 4."""
    emotions = emotions or sorted(res["emotion"].unique())
    for emo in emotions:
        sub = res[res["emotion"] == emo]
        if sub.empty:
            continue
        mat = sub.pivot_table(index="quest_score", columns="measure", values="r")
        pmat = sub.pivot_table(index="quest_score", columns="measure",
                               values="p").reindex(index=mat.index,
                                                   columns=mat.columns)
        nmat = sub.pivot_table(index="quest_score", columns="measure",
                               values="n").reindex(index=mat.index,
                                                   columns=mat.columns)
        title = (f"Questionnaires x {emo} dynamics   ({METHOD}, "
                 f"n={int(sub['n'].min())}-{int(sub['n'].max())}, "
                 f"* p<.05 uncorrected)")
        _draw_heatmap_pages(mat, pmat, title, f"corr_{emo}_{METHOD}",
                            out_dir, nmat=nmat)


def plot_top_scatters(res, dyn, quest, out_dir, top_n=20):
    """Scatter panels for the strongest |r|, PPT-sized grids. Included so
    every 'hit' can be eyeballed - with small n most will visibly hinge
    on one or two points, which is exactly the thing to notice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    top = res.reindex(res["r"].abs().sort_values(ascending=False).index).head(top_n)
    ncols, nrows = 5, int(np.ceil(len(top) / 5))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.33, 7.5), squeeze=False)
    fig.suptitle(f"Strongest {len(top)} correlations "
                 f"({METHOD}, UNCORRECTED - inspect for single-point leverage)",
                 fontsize=12, fontweight="bold")
    for i, (_, r) in enumerate(top.iterrows()):
        ax = axes[i // ncols][i % ncols]
        d = dyn[(dyn["emotion"] == r.emotion) & (dyn["measure"] == r.measure)]
        d = d.set_index("participant")["value_num"]
        pair = pd.DataFrame({"d": d, "q": quest[r.quest_score]}).dropna()
        ax.scatter(pair["q"], pair["d"], s=22, color="#2c3e50")
        ax.set_title(f"{r.emotion}/{r.measure}\n{r.quest_score}\n"
                     f"r={r.r:.2f} p={r.p:.3f}", fontsize=5.5)
        ax.tick_params(labelsize=5)
        sns.despine(ax=ax)
    for i in range(len(top), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = Path(out_dir) / f"corr_top_scatters_{METHOD}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def main():
    dyn, quest, shared = load_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    res = run_correlations(dyn, quest, shared)
    if res.empty:
        print("No correlations could be computed (too few overlapping "
              "subjects, or no variance).")
        return
    out = OUT_DIR / f"corr_quest_x_dynamics_{METHOD}.csv"
    res.to_csv(out, index=False)
    print(f"[OK] {len(res)} correlations -> {out}")
    report_yield(res)

    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_corr_heatmaps(res, fig_dir)
    plot_top_scatters(res, dyn, quest, fig_dir)


if __name__ == "__main__":
    main()
