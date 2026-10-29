# -*- coding: utf-8 -*-
"""
10b_paper_figures.py
====================
Figures for the manuscript on intrinsic emotions (called by
10_paper_baseline_results.py, which loads the data and passes it to
make_all). Main figures use the baseline block only; Extended Data figures
show all blocks. Style: 180 mm wide, 7 pt Arial, panel letters a, b, c.

Fig. 1  design and test logic (schematic; simulated illustrations)
Fig. 2  emotions are present at rest (Test 1)
Fig. 3  rating error is small (state-space sigma, Sadness - Fear bound)
Fig. 4  baseline ratings vary beyond rating error (Test 2)
Fig. 5  not a start effect or drift; partly scale-specific; stable
        individual differences; types
Extended Data Fig. 1  positive control: condition comparisons, all blocks
Extended Data Fig. 2  Test 2 in all blocks
Extended Data Fig. 3  distribution fits and all tests of the comparisons
Extended Data Fig. 4  simulation: false-positive rate and power
Extended Data Fig. 5  Test 3 asymmetry, prevalence and types, all blocks
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch
import logging
import numpy as np
import pandas as pd
from scipy import stats

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
MM = 1 / 25.4
W = 180 * MM
SIGMA_MAIN = 3.0
SAD, FEAR = "#2F6EBA", "#D1495B"
COL = {"Sadness": SAD, "Fear": FEAR}
GREY, LIGHT, DARK = "#9A9A9A", "#DADADA", "#333333"
BLOCKS = ["base", "sad", "fear"]
EMOS = ["Sadness", "Fear"]
CELLS = [(b, e) for b in BLOCKS for e in EMOS]
BNAME = {"base": "Baseline", "sad": "After sad", "fear": "After fear"}
SHORT = {"base": "Base", "sad": "Sad", "fear": "Fear"}
PAIRS = [(0, 2), (0, 4), (2, 4), (1, 3), (1, 5), (3, 5), (0, 1), (2, 3), (4, 5)]

plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
})
RNG = np.random.default_rng(7)


# ===================================================================
# HELPERS
# ===================================================================

def letter(ax, s, dx=-0.13, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def kde(v, grid):
    v = v[np.isfinite(v)]
    if v.size < 2 or np.std(v) == 0:
        return np.zeros_like(grid)
    bw = 1.06 * np.std(v) * v.size ** (-1 / 5)
    z = (grid[:, None] - v[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(1) / (v.size * bw * np.sqrt(2 * np.pi))


def raincloud(ax, groups, colors, pos=None, width=0.32, filled=None, s=7):
    """Half violin (left, within the data range), box, mean diamond, jittered points.
    filled: list of boolean arrays, True = filled point, False = open point."""
    pos = list(range(len(groups))) if pos is None else pos
    for k, (v, c, x0) in enumerate(zip(groups, colors, pos)):
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        v = v[ok]
        if v.size == 0:
            continue
        grid = np.linspace(v.min(), v.max(), 300)
        d = kde(v, grid)
        if d.max() > 0:
            d = d / d.max() * width
            ax.fill_betweenx(grid, x0 - 0.05 - d, x0 - 0.05, color=c, alpha=0.30, lw=0)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(Rectangle((x0 - 0.03, q1), 0.06, q3 - q1, fill=False, ec=DARK, lw=0.6))
        ax.plot([x0 - 0.03, x0 + 0.03], [med, med], color=DARK, lw=1.0)
        ax.plot(x0, v.mean(), marker="D", color=DARK, ms=2.2)
        jit = x0 + 0.09 + RNG.uniform(0, 0.2, v.size)
        if filled is None:
            ax.scatter(jit, v, s=s, color=c, alpha=0.8, lw=0)
        else:
            f = np.asarray(filled[k], bool)[ok]
            ax.scatter(jit[f], v[f], s=s, color=c, alpha=0.9, lw=0)
            ax.scatter(jit[~f], v[~f], s=s, facecolor="white", edgecolor=c, lw=0.5)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def brackets(ax, sig_pairs, top, step):
    """sig_pairs: list of (i, j, p). Brackets stacked from top upwards."""
    sig_pairs = sorted(sig_pairs, key=lambda r: (r[1] - r[0], r[0]))
    for k, (i, j, p) in enumerate(sig_pairs):
        y = top + step * (k + 0.5)
        ax.plot([i, i, j, j], [y - step * 0.25, y, y, y - step * 0.25], color=DARK, lw=0.6)
        ax.text((i + j) / 2, y - step * 0.05, stars(p), ha="center", va="bottom", fontsize=7)
    return top + step * (len(sig_pairs) + 0.6)


def cell(df, b, e, variant="all"):
    q = df[(df.block_role == b) & (df.emotion == e)]
    return q[q.variant == variant] if "variant" in q.columns else q


# ===================================================================
# FIGURE 1 - design and logic (schematic)
# ===================================================================

def fig1(D, out):
    fig = plt.figure(figsize=(W, 80 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.85, 1.15], hspace=0.62, wspace=0.42,
                          left=0.06, right=0.99, top=0.96, bottom=0.14)
    # a  timeline
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    letter(ax, "a", dx=-0.045, dy=0.95)
    segs = [(0, 27, "Baseline rest, ~20 min", "no preceding stimulus", True),
            (29, 38, "Video 1", "sad or fear", False),
            (40, 67, "Rest, ~20 min", "after video 1", True),
            (69, 78, "Video 2", "the other video", False),
            (80, 100, "Rest, ~20 min", "after video 2", True)]
    for x0, x1, name, sub, rest in segs:
        fc = SAD if False else ("#EAF1FA" if rest else "#F4F4F4")
        ec = SAD if (rest and x0 == 0) else GREY
        lw = 1.2 if (rest and x0 == 0) else 0.6
        ax.add_patch(FancyBboxPatch((x0, 4.3), x1 - x0, 3.2, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=fc if x0 == 0 else ("#F7F7F7" if rest else "#EFEFEF"),
                                    ec=ec, lw=lw))
        ax.text((x0 + x1) / 2, 6.6, name, ha="center", va="center", fontsize=6.6,
                fontweight="bold" if x0 == 0 else "normal")
        ax.text((x0 + x1) / 2, 5.1, sub, ha="center", va="center", fontsize=5.8, color=DARK)
        if rest:
            t = np.cumsum(np.r_[1.0, RNG.uniform(0.75, 1.45, 30)])
            t = t[t < (x1 - x0) - 0.6] + x0 + 0.3
            ax.vlines(t, 3.2, 3.9, color=SAD if x0 == 0 else GREY, lw=0.6)
    ax.text(0, 1.4, "Ticks: probes every 20–60 s (median 25 per rest block): “How sad do you feel "
            "right now?” and “How afraid do you feel right now?”\nVAS 0 (not at all) to 100 "
            "(extremely), steps of 5, cursor starts at 50. Video order counterbalanced; MEG "
            "recorded throughout.", fontsize=6.3, va="center", color=DARK, linespacing=1.5)
    ax.annotate("", xy=(100, 8.6), xytext=(0, 8.6),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.6))
    ax.text(50, 9.2, "Session time", ha="center", fontsize=6.3, color=GREY)

    # b  three null models, simulated illustrations
    t = np.arange(25)
    rng = np.random.default_rng(3)
    titles = ["Test 1  Is the emotion present?", "Test 2  Does it vary beyond rating error?",
              "Test 3  Do the two scales move apart?"]
    h0 = ["H0: non-zero ratings are lapses\n(slips, probability 10% per probe)",
          "H0: constant feeling + rating error σ,\nrounded to steps of 5, clipped at 0",
          "H0: both scales share all movement;\neach adds independent error σ"]
    for k in range(3):
        ax = fig.add_subplot(gs[1, k])
        if k == 0:
            y = np.zeros(25)
            y[[6, 17]] = [15, 10]
            ax.plot(t, y, "o-", color=GREY, ms=1.8, lw=0.7)
            ax.set_ylim(-3, 40)
        elif k == 1:
            y = np.clip(np.round((20 + rng.normal(0, 3, 25)) / 5) * 5, 0, 100)
            ax.axhspan(20 - 6, 20 + 6, color=LIGHT, lw=0)
            ax.axhline(20, color=DARK, lw=0.6, ls="--")
            ax.plot(t, y, "o-", color=GREY, ms=1.8, lw=0.7)
            ax.set_ylim(-3, 40)
        else:
            c = 20 + 8 * np.sin(t / 4)
            ys = np.clip(np.round((c + rng.normal(0, 2, 25)) / 5) * 5, 0, 100)
            yf = np.clip(np.round((c - 6 + rng.normal(0, 2, 25)) / 5) * 5, 0, 100)
            ax.plot(t, ys, "o-", color=SAD, ms=1.8, lw=0.7, label="Sadness")
            ax.plot(t, yf, "o-", color=FEAR, ms=1.8, lw=0.7, label="Fear")
            ax.legend(loc="upper right", fontsize=5.8, handlelength=1.2, borderaxespad=0)
            ax.set_ylim(-3, 40)
        ax.set_title(titles[k], loc="left", fontsize=6.8, fontweight="bold", pad=3)
        ax.text(0.0, -0.42, h0[k], transform=ax.transAxes, fontsize=6, va="top", color=DARK)
        ax.set_xlabel("Probe", labelpad=1)
        ax.set_ylabel("Rating", labelpad=1)
        ax.set_xticks([0, 12, 24])
        if k == 0:
            letter(ax, "b", dx=-0.22, dy=1.16)
            ax.text(0.0, 1.2, "Simulated series that each null hypothesis (H0) allows; a real "
                    "series passes a test only if it departs from them", transform=ax.transAxes,
                    fontsize=6.3, color=DARK, va="bottom")
    save(fig, out, "Fig1_design_and_tests")


# ===================================================================
# FIGURE 2 - presence
# ===================================================================

def fig2(desc, nonnull, P, out):
    fig, axs = plt.subplots(1, 3, figsize=(W, 58 * MM),
                            gridspec_kw={"width_ratios": [0.9, 1.5, 0.9], "wspace": 0.45})
    # a raincloud of share > 0, filled = significant at lapse 10%
    ax = axs[0]
    vals, fills = [], []
    for e in EMOS:
        d = cell(desc, "base", e).set_index("subject")
        n = cell(nonnull, "base", e)
        n = n[np.isclose(n.lapse_rate, 0.1)].set_index("subject").reindex(d.index)
        vals.append(100 * d.frac_nonzero.to_numpy())
        fills.append(n.significant.to_numpy(bool))
    raincloud(ax, vals, [SAD, FEAR], filled=fills)
    ax.axhline(24, color=DARK, lw=0.6, ls="--")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(EMOS)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-4, 104)
    ax.set_ylabel("Probes rated above 0 (%)")
    ax.set_title("Dashed: lapse threshold (6 of 25 probes)", fontsize=6.3, loc="left")
    letter(ax, "a", dx=-0.32)
    # b participant tiles
    ax = axs[1]
    st = {}
    for e in EMOS:
        d = cell(desc, "base", e).set_index("subject")
        n = cell(nonnull, "base", e)
        n = n[np.isclose(n.lapse_rate, 0.1)].set_index("subject")
        st[e] = pd.Series(np.where(n.reindex(d.index).significant, 2,
                                   np.where(d.frac_nonzero > 0, 1, 0)), index=d.index)
    S = pd.DataFrame(st).dropna().astype(int)
    S = S.assign(score=S.Sadness + S.Fear + 0.01 * S.Sadness).sort_values("score", ascending=False)
    for col, e in enumerate(EMOS):
        for i, v in enumerate(S[e]):
            c = COL[e]
            fc = c if v == 2 else ("white" if v == 0 else c)
            al = 1 if v == 2 else (1 if v == 0 else 0.30)
            ax.add_patch(Rectangle((i, 1 - col), 0.86, 0.86, fc=fc, alpha=al,
                                   ec=c if v else LIGHT, lw=0.4))
    ax.set_xlim(-0.5, len(S) + 0.5)
    ax.set_ylim(-0.25, 2.1)
    ax.set_yticks([1.43, 0.43])
    ax.set_yticklabels(EMOS)
    ax.set_xticks([0, 9, 19, 29, 39, 48])
    ax.set_xticklabels(["1", "10", "20", "30", "40", "49"])
    ax.set_xlabel("Participant (sorted)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(fc=GREY, ec=GREY, label="present (above lapses, p < .05)"),
                       Patch(fc=GREY, alpha=0.3, ec=GREY, label="rated > 0, within lapse rate"),
                       Patch(fc="white", ec=LIGHT, label="always 0")],
              loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=6,
              handlelength=1.0, columnspacing=1.0)
    for e, yy in (("Sadness", 1.43), ("Fear", 0.43)):
        ax.text(len(S) + 0.8, yy, f"{(S[e] == 2).sum()} present\n{(S[e] == 1).sum()} within lapses"
                f"\n{(S[e] == 0).sum()} always 0", fontsize=5.6, va="center", color=COL[e])
    ax.set_xlim(-0.5, len(S) + 9)
    ax.set_title("Each tile: one participant, baseline block", fontsize=6.3, loc="left")
    letter(ax, "b", dx=-0.1)
    # c prevalence
    ax = axs[2]
    for k, e in enumerate(EMOS):
        for j, lapse in enumerate((0.05, 0.10, 0.20)):
            r = P[(P.analysis == "nonnull") & (P.block_role == "base") & (P.emotion == e) &
                  (P.variant == "all") & np.isclose(P.parameter_value, lapse)].iloc[0]
            y = k * 4 + j
            ax.plot([100 * r.prev_hpdi_low, 100 * r.prev_hpdi_high], [y, y], color=COL[e],
                    lw=1.6 if lapse == 0.1 else 0.8, alpha=1 if lapse == 0.1 else 0.6)
            ax.plot(100 * r.prev_map, y, "o", color=COL[e], ms=3.5 if lapse == 0.1 else 2.5)
            ax.text(101, y, f"{int(lapse * 100)}%", fontsize=5.6, va="center", color=DARK)
    ax.axvline(50, color=GREY, lw=0.6, ls=":")
    ax.set_yticks([1, 5])
    ax.set_yticklabels(EMOS)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Population prevalence (%)")
    ax.set_title("MAP, 95% HPDI; lapse rate at right", fontsize=6.3, loc="left")
    ax.tick_params(axis="y", length=0)
    letter(ax, "c", dx=-0.32)
    save(fig, out, "Fig2_presence")


# ===================================================================
# FIGURE 3 - rating error
# ===================================================================

def fig3(prof, sig, dissoc, out):
    fig = plt.figure(figsize=(W, 100 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.8], width_ratios=[1.3, 1],
                          hspace=0.55, wspace=0.3, left=0.08, right=0.98, top=0.93, bottom=0.09)
    ax = fig.add_subplot(gs[0, 0])
    for fit, c, lw, lab in (("all", DARK, 1.4, "All off-floor series (n = 150)"),
                            ("base Sadness", SAD, 1.0, "Baseline Sadness (n = 24)"),
                            ("base Fear", FEAR, 1.0, "Baseline Fear (n = 25)")):
        g = prof[prof.fit == fit].sort_values("sigma")
        ax.plot(g.sigma, g.profile_loglik - g.profile_loglik.max(), color=c, lw=lw, label=lab)
    ax.axhline(-1.92, color=GREY, lw=0.6, ls="--")
    ax.text(11.9, -1.92 - 3.5, "95% CI cut-off", fontsize=5.8, color=GREY, ha="right")
    for sv, lab in ((3, "main"), (5, "5"), (10, "literature")):
        ax.axvline(sv, color=GREY, lw=0.5, ls=":")
        ax.text(sv, 4.5, lab, fontsize=5.8, color=GREY, ha="center", va="bottom", clip_on=False)
    ax.set_xlim(0, 12)
    ax.set_ylim(-60, 3)
    ax.set_xlabel("Rating error \u03c3 (points)")
    ax.set_ylabel("Profile log-likelihood \u2212 maximum")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.18), fontsize=6, handlelength=1.6)
    letter(ax, "a", dx=-0.14, dy=1.06)
    ax = fig.add_subplot(gs[0, 1])
    x = dissoc[(dissoc.variant == "all") & dissoc.eligible].drop_duplicates(["subject", "block_role"])
    xb = (x[x.block_role == "base"].sd_diff / np.sqrt(2)).to_numpy()
    xa = (x.sd_diff / np.sqrt(2)).to_numpy()
    raincloud(ax, [xb, xa], [DARK, GREY])
    for sv in (3, 5, 10):
        ax.axhline(sv, color=GREY, lw=0.5, ls=":")
        ax.text(1.62, sv + 0.3, f"\u03c3 = {sv}", fontsize=5.8, va="bottom", color=GREY)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Baseline blocks\n(n = {len(xb)})", f"All blocks\n(n = {len(xa)})"])
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylabel("SD(Sadness \u2212 Fear) / \u221a2 (points)")
    ax.set_title("Model-free upper bound on independent error", fontsize=6.3, loc="left")
    letter(ax, "b", dx=-0.2, dy=1.06)
    ax = fig.add_subplot(gs[1, :])
    g = sig.set_index("fit")
    rows = [("State-space model, all off-floor series", g.loc["all", "sigma_hat"],
             g.loc["all", "sigma_ci_low"], g.loc["all", "sigma_ci_high"], DARK, "95% CI"),
            ("State-space model, baseline Sadness", g.loc["base Sadness", "sigma_hat"],
             g.loc["base Sadness", "sigma_ci_low"], g.loc["base Sadness", "sigma_ci_high"], SAD, "95% CI"),
            ("State-space model, baseline Fear", g.loc["base Fear", "sigma_hat"],
             g.loc["base Fear", "sigma_ci_low"], g.loc["base Fear", "sigma_ci_high"], FEAR, "95% CI"),
            ("Sadness \u2212 Fear bound, baseline blocks", np.median(xb), np.percentile(xb, 25),
             np.percentile(xb, 75), DARK, "median, IQR"),
            ("Sadness \u2212 Fear bound, all blocks", np.median(xa), np.percentile(xa, 25),
             np.percentile(xa, 75), GREY, "median, IQR"),
            ("Single items in daily life (Dejonckheere et al., 2022)", 10, np.nan, np.nan, GREY,
             "about 1/10 of the scale")]
    for i, (lab, m, lo, hi, c, note) in enumerate(rows):
        if np.isfinite(lo):
            ax.plot([lo, hi], [i, i], color=c, lw=1.4)
        ax.plot(m, i, "o" if i < 5 else "s", color=c, ms=3.5, mfc=c if i < 5 else "white")
        ax.text(12.2, i, f"{m:.1f}" + (f" [{lo:.1f}\u2013{hi:.1f}]" if np.isfinite(lo) else "")
                + f"  ({note})", fontsize=5.8, color=c, va="center")
    for sv, lab in ((3, "main analysis"), (5, "sensitivity"), (10, "sensitivity")):
        ax.axvline(sv, color=SAD if sv == 3 else GREY, lw=0.8 if sv == 3 else 0.5,
                   ls="-" if sv == 3 else ":")
        ax.text(sv, -0.95, f"\u03c3 = {sv}: {lab}", fontsize=5.8, ha="center",
                color=SAD if sv == 3 else GREY, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    ax.set_ylim(len(rows) - 0.4, -1.3)
    ax.set_xlim(0, 12)
    ax.set_xlabel("Rating error \u03c3 (points)")
    ax.tick_params(axis="y", length=0)
    letter(ax, "c", dx=-0.37, dy=1.0)
    save(fig, out, "Fig3_rating_error")


# ===================================================================
# FIGURE 4 - beyond noise
# ===================================================================

def fig4(fl, P, nonnull, out):
    fig = plt.figure(figsize=(W, 112 * MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.55, wspace=0.42,
                          left=0.07, right=0.99, top=0.95, bottom=0.08)
    f3 = fl[(fl.block_role == "base") & (fl.variant == "all") & np.isclose(fl.noise_sd, SIGMA_MAIN)]
    # a observed vs null SD
    ax = fig.add_subplot(gs[0, 0])
    for e in EMOS:
        g = f3[f3.emotion == e]
        s = g.significant.to_numpy(bool)
        ax.scatter(g.sd_null_median[s], g.sd_obs[s], s=9, color=COL[e], lw=0, alpha=0.9)
        ax.scatter(g.sd_null_median[~s], g.sd_obs[~s], s=9, facecolor="white",
                   edgecolor=COL[e], lw=0.5)
    m = 25
    ax.plot([0, m], [0, m], color=GREY, lw=0.6, ls="--")
    ax.set_xlim(1.8, 3.7)
    ax.set_ylim(-0.5, m)
    ax.set_xlabel("SD expected from rating error alone\n(median of 10,000 simulations, σ = 3)")
    ax.set_ylabel("Observed series SD (points)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SAD, ms=3, label="Sadness"),
                       Line2D([], [], marker="o", ls="", color=FEAR, ms=3, label="Fear"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=DARK, ms=3,
                              label="not significant")],
              loc="upper left", fontsize=6, handletextpad=0.2)
    ax.text(3.68, 0.3, "dashed: observed = expected", fontsize=5.6, color=GREY, ha="right")
    letter(ax, "a", dx=-0.3)
    # b % significant vs sigma
    ax = fig.add_subplot(gs[0, 1])
    grid = sorted(fl.noise_sd.unique())
    ax.axvspan(1.4, 5.7, color="#EAF1FA", lw=0)
    for e in EMOS:
        y = [100 * fl[(fl.block_role == "base") & (fl.variant == "all") & (fl.emotion == e) &
                      np.isclose(fl.noise_sd, s)].significant.mean() for s in grid]
        ax.plot(grid, y, "o-", color=COL[e], ms=3, label=e)
    ax.axhline(5, color=GREY, lw=0.6, ls="--")
    ax.text(12.4, 6.5, "5% expected", fontsize=5.6, color=GREY, ha="right")
    ax.axvline(3, color=SAD, lw=0.8)
    ax.text(1.55, 97, "σ range\nfrom data", fontsize=5.6, color=SAD, va="top")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Assumed rating error σ (points)")
    ax.set_ylabel("Series varying beyond error (%)")
    ax.legend(loc="upper right", fontsize=6)
    letter(ax, "b", dx=-0.3)
    # c verdicts
    ax = fig.add_subplot(gs[0, 2])
    cats = [("fluctuating", 1.0), ("inconclusive", 0.45), ("flat", 0.12)]
    x = 0
    ticks, labs = [], []
    for e in EMOS:
        for s in (3, 5, 10):
            g = fl[(fl.block_role == "base") & (fl.variant == "all") & (fl.emotion == e) &
                   np.isclose(fl.noise_sd, s)]
            bottom = 0
            for c, a in cats:
                v = 100 * (g.verdict == c).mean()
                ax.bar(x, v, bottom=bottom, color=COL[e], alpha=a, width=0.8, ec="white", lw=0.4)
                bottom += v
            ticks.append(x)
            labs.append(f"{s:g}")
            x += 1
        x += 0.6
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs)
    ax.text(1, -17, "Sadness", ha="center", fontsize=6.5, color=SAD)
    ax.text(4.6, -17, "Fear", ha="center", fontsize=6.5, color=FEAR)
    ax.set_xlabel("σ", labelpad=1)
    ax.set_ylabel("Series (%)")
    ax.set_ylim(0, 100)
    ax.legend(handles=[Patch(fc=DARK, alpha=a, label=c) for c, a in cats], loc="upper center",
              bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=6, handlelength=1)
    letter(ax, "c", dx=-0.3, dy=1.1)
    # d prevalence vs sigma
    ax = fig.add_subplot(gs[1, 0:2])
    for k, e in enumerate(EMOS):
        for s in grid:
            r = P[(P.analysis == "fluctuation") & (P.block_role == "base") & (P.emotion == e) &
                  (P.variant == "all") & np.isclose(P.parameter_value, s)].iloc[0]
            xx = np.log(s) + (k - 0.5) * 0.045
            ax.plot([xx, xx], [100 * r.prev_hpdi_low, 100 * r.prev_hpdi_high], color=COL[e], lw=1.2)
            ax.plot(xx, 100 * r.prev_map, "o", color=COL[e], ms=3.5,
                    label=e if s == grid[0] else None)
    ax.axhline(50, color=GREY, lw=0.6, ls=":")
    ax.axvspan(np.log(3) - 0.09, np.log(3) + 0.09, color="#EAF1FA", lw=0, zorder=0)
    ax.text(np.log(3), 101, "main", fontsize=5.8, color=SAD, ha="center", va="bottom")
    ax.set_xticks(np.log(grid))
    ax.set_xticklabels([f"{s:g}" for s in grid])
    ax.set_xlabel("Assumed rating error σ (points, log scale)")
    ax.set_ylabel("Population prevalence of\nfluctuation (%), MAP and 95% HPDI")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=6)
    letter(ax, "d", dx=-0.13)
    # e presence x fluctuation (sigma = 3)
    ax = fig.add_subplot(gs[1, 2])
    n = nonnull[(nonnull.block_role == "base") & (nonnull.variant == "all") &
                np.isclose(nonnull.lapse_rate, 0.1)][["subject", "emotion", "significant"]]
    m = f3.merge(n.rename(columns={"significant": "present"}), on=["subject", "emotion"])
    y = 0
    yt, yl = [], []
    for e in EMOS:
        for pres, lab in ((True, "present"), (False, "not present")):
            g = m[(m.emotion == e) & (m.present == pres)]
            k = int(g.significant.sum())
            ax.barh(y, k, color=COL[e], height=0.7)
            ax.barh(y, len(g) - k, left=k, color=COL[e], alpha=0.25, height=0.7)
            ax.text(len(g) + 0.6, y, f"{k}/{len(g)}", va="center", fontsize=5.8)
            yt.append(y)
            yl.append(f"{e}, {lab}")
            y += 1
        y += 0.5
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 40)
    ax.set_xlabel("Series (dark: vary beyond error, σ = 3)")
    ax.tick_params(axis="y", length=0)
    ax.set_title("Present emotions move", fontsize=6.3, loc="left")
    letter(ax, "e", dx=-0.62)
    save(fig, out, "Fig4_beyond_noise")


# ===================================================================
# FIGURE 5 - robustness, specificity, individual differences
# ===================================================================

def fig5(fps, fl, dissoc, T, asg, out):
    fig = plt.figure(figsize=(W, 118 * MM))
    gs = fig.add_gridspec(2, 12, hspace=0.62, wspace=2.2, left=0.07, right=0.99,
                          top=0.95, bottom=0.08)
    # a first - rest
    ax = fig.add_subplot(gs[0, 0:4])
    fp = fps[fps.block_role == "base"].copy()
    fp["d"] = fp["first"] - fp["rest_mean"]
    raincloud(ax, [fp[fp.emotion == e].d.to_numpy() for e in EMOS], [SAD, FEAR])
    ax.axhline(0, color=GREY, lw=0.6, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(EMOS)
    ax.set_xlim(-0.5, 1.4)
    ax.set_ylabel("First rating − mean of later ratings")
    ax.set_title("Start of rest (first probe ~40 s)", fontsize=6.3, loc="left")
    letter(ax, "a", dx=-0.35)
    # b robustness variants
    ax = fig.add_subplot(gs[0, 4:8])
    vars_ = [("all", "all probes", "o"), ("drop_first", "first probe removed", "s"),
             ("detrended", "linear trend removed", "^")]
    for k, e in enumerate(EMOS):
        for j, (v, lab, mk) in enumerate(vars_):
            y = [100 * fl[(fl.block_role == "base") & (fl.variant == v) & (fl.emotion == e) &
                          np.isclose(fl.noise_sd, s)].significant.mean() for s in (3, 5, 10)]
            xs = np.arange(3) + k * 3.6 + (j - 1) * 0.22
            ax.plot(xs, y, mk, color=COL[e], ms=3, mfc=COL[e] if j == 0 else "white",
                    label=lab if k == 0 else None, ls="")
    ax.axhline(5, color=GREY, lw=0.6, ls="--")
    ax.set_xticks(list(np.arange(3)) + list(np.arange(3) + 3.6))
    ax.set_xticklabels(["3", "5", "10"] * 2)
    ax.text(1, -20, "Sadness", ha="center", fontsize=6.5, color=SAD)
    ax.text(4.6, -20, "Fear", ha="center", fontsize=6.5, color=FEAR)
    ax.set_xlabel("σ", labelpad=1)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Series varying beyond error (%)")
    leg = ax.legend(loc="lower left", fontsize=5.8, handletextpad=0.2)
    for h in leg.legend_handles:
        h.set_color(DARK)
        h.set_markerfacecolor(DARK if h.get_marker() == "o" else "white")
    letter(ax, "b", dx=-0.3)
    # c Sadness-Fear correlation and difference test
    ax = fig.add_subplot(gs[0, 8:12])
    x = dissoc[(dissoc.variant == "all") & (dissoc.block_role == "base") & dissoc.eligible &
               np.isclose(dissoc.noise_sd, SIGMA_MAIN)]
    raincloud(ax, [x.corr_sad_fear.to_numpy()], [DARK], filled=[x.significant.to_numpy(bool)], s=10)
    ax.axhline(0, color=GREY, lw=0.6, ls="--")
    ax.set_xticks([0])
    ax.set_xticklabels([f"Baseline blocks\noff the floor (n = {len(x)})"])
    ax.set_xlim(-0.5, 0.6)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("r(Sadness, Fear) across probes")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=DARK, ms=3,
                              label="difference test p < .05 (σ = 3)"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=DARK, ms=3,
                              label="n.s.")], loc="lower left", fontsize=5.8, handletextpad=0.2)
    letter(ax, "c", dx=-0.4)
    # d cross-block consistency
    ax = fig.add_subplot(gs[1, 0:6])
    c = T["ST08B_cross_block_consistency"]
    labels = []
    for i, r in enumerate(c.itertuples()):
        y = i + (0.6 if r.measure.endswith("mean") else 0)
        col = COL[r.scale]
        ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=1.2)
        ax.plot(r.spearman_rho, y, "o", color=col, ms=3.5,
                mfc=col if r.p_holm < .05 else "white")
        labels.append((y, f"{'SD' if 'SD' in r.measure else 'Mean'}, {r.scale}: vs "
                          f"{r.baseline_vs.replace(' video', '').lower()}"))
    ax.axvline(0, color=GREY, lw=0.6, ls="--")
    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([l for _, l in labels], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(-0.4, 1)
    ax.set_xlabel("Spearman ρ with baseline (95% CI); filled: p_Holm < .05")
    ax.tick_params(axis="y", length=0)
    letter(ax, "d", dx=-0.42)
    # e typology at baseline
    s = asg[(asg.solution == "selected") & (asg.block_role == "base")]
    for k, e in enumerate(EMOS):
        ax = fig.add_subplot(gs[1, 6 + 3 * k:9 + 3 * k])
        g = s[s.emotion == e]
        order = g.groupby("class_label").frac_nonzero.median().sort_values().index
        alphas = np.linspace(0.25, 1, len(order)) if len(order) > 1 else [1]
        for ci, cl in enumerate(order):
            h = g[g.class_label == cl]
            ax.scatter(100 * h.frac_nonzero + RNG.uniform(-1.5, 1.5, len(h)), np.log1p(h.sd),
                       s=9, color=COL[e], alpha=alphas[ci], lw=0.3, edgecolor="white",
                       label=f"class {ci + 1}, n = {len(h)}")
        ax.set_xlim(-5, 105)
        ax.set_ylim(-0.1, 3.6)
        ax.set_xlabel("Probes > 0 (%)")
        ax.set_ylabel("log(1 + SD)" if k == 0 else "")
        ax.set_title(f"{e}: {len(order)} class{'es' if len(order) > 1 else ''}", fontsize=6.3,
                     loc="left", color=COL[e])
        ax.legend(loc="upper left", fontsize=5.4, handletextpad=0.1, borderaxespad=0.1)
        if k == 0:
            letter(ax, "e", dx=-0.45)
    save(fig, out, "Fig5_robustness_specificity_individuals")


# ===================================================================
# EXTENDED DATA
# ===================================================================

def best_test_pairs(dt, measure, test):
    key = {f"{b} {e}": k for k, (b, e) in enumerate(CELLS)}
    g = dt[(dt.measure == measure) & (dt.test == test)]
    out = []
    for c, p in zip(g.comparison, g.p_holm):
        a, b = c.split(" vs ")
        i, j = sorted((key[a], key[b]))
        if p < .05:
            out.append((i, j, p))
    return out


def ed1(desc, dt, fps, out):
    fig = plt.figure(figsize=(W, 150 * MM))
    gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.25, left=0.07, right=0.99,
                          top=0.97, bottom=0.06)
    d = desc[desc.variant == "all"]
    labs = [f"{BNAME[b]}\n{e}" for b, e in CELLS]
    cols = [COL[e] for _, e in CELLS]
    specs = [("mean", "Series mean (0–100)", "mean", "Tweedie (zeros included)", 1, "a"),
             ("sd", "Series SD (points)", "sd", "Tweedie (zeros included)", 1, "b"),
             ("frac_nonzero", "Probes rated > 0 (%)", "share > 0",
              "binomial (logistic GEE on probes)", 100, "c")]
    for k, (col, ylab, meas, test, sc, let) in enumerate(specs):
        ax = fig.add_subplot(gs[k // 2, k % 2]) if k < 2 else fig.add_subplot(gs[1, 0])
        vals = [sc * d[(d.block_role == b) & (d.emotion == e)][col].to_numpy() for b, e in CELLS]
        raincloud(ax, vals, cols)
        top = max(np.nanmax(v) for v in vals)
        hi = brackets(ax, best_test_pairs(dt, meas, test), top * 1.03, top * 0.09)
        ax.set_ylim(min(0, min(np.nanmin(v) for v in vals)) - top * 0.03, hi)
        ax.set_xticks(range(6))
        ax.set_xticklabels(labs, fontsize=6)
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.6, 5.5)
        letter(ax, let, dx=-0.1)
    fp = fps.copy()
    fp["d"] = fp["first"] - fp["rest_mean"]
    for k, (col, ylab, let) in enumerate((("d", "First rating − mean of later", "d"),
                                          ("slope_per_min", "Slope (points per min)", "e"))):
        ax = fig.add_subplot(gs[1, 1]) if k == 0 else fig.add_subplot(gs[2, 0])
        vals = [fp[(fp.block_role == b) & (fp.emotion == e)][col].to_numpy() for b, e in CELLS]
        raincloud(ax, vals, cols)
        ax.axhline(0, color=GREY, lw=0.6, ls="--")
        ax.set_xticks(range(6))
        ax.set_xticklabels(labs, fontsize=6)
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.6, 5.5)
        letter(ax, let, dx=-0.1)
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    save(fig, out, "ExtData_Fig1_positive_control")


def ed2(fl, out):
    fig, axs = plt.subplots(1, 3, figsize=(W, 62 * MM),
                            gridspec_kw={"width_ratios": [1, 1.25, 1.25], "wspace": 0.35})
    grid = sorted(fl.noise_sd.unique())
    ls = {"base": "-", "sad": "--", "fear": ":"}
    ax = axs[0]
    for b, e in CELLS:
        y = [100 * fl[(fl.block_role == b) & (fl.variant == "all") & (fl.emotion == e) &
                      np.isclose(fl.noise_sd, s)].significant.mean() for s in grid]
        ax.plot(grid, y, ls[b], marker="o", ms=2.5, color=COL[e], label=f"{BNAME[b]} · {e}",
                lw=1.3 if b == "base" else 0.8)
    ax.axhline(5, color=GREY, lw=0.6, ls="--")
    ax.axvline(3, color=SAD, lw=0.6)
    ax.set_ylim(0, 100)
    ax.set_xlabel("σ (points)")
    ax.set_ylabel("Series varying beyond error (%)")
    ax.legend(fontsize=5.6, loc="lower left", handlelength=2)
    letter(ax, "a", dx=-0.3)
    ax = axs[1]
    cats = [("fluctuating", 1.0), ("inconclusive", 0.45), ("flat", 0.12)]
    x, ticks, labs = 0, [], []
    for b, e in CELLS:
        for s in (3, 5, 10):
            g = fl[(fl.block_role == b) & (fl.variant == "all") & (fl.emotion == e) &
                   np.isclose(fl.noise_sd, s)]
            bottom = 0
            for c, a in cats:
                v = 100 * (g.verdict == c).mean()
                ax.bar(x, v, bottom=bottom, color=COL[e], alpha=a, width=0.85, ec="white", lw=0.3)
                bottom += v
            x += 1
        ticks.append(x - 2)
        labs.append(f"{SHORT[b]}\n{e[:3]}")
        x += 0.7
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, fontsize=6)
    ax.set_ylabel("Series (%) at σ = 3, 5, 10")
    ax.legend(handles=[Patch(fc=DARK, alpha=a, label=c) for c, a in cats], fontsize=5.8,
              loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, handlelength=1)
    letter(ax, "b", dx=-0.2, dy=1.08)
    ax = axs[2]
    f3 = fl[(fl.variant == "all") & np.isclose(fl.noise_sd, SIGMA_MAIN)]
    vals = [f3[(f3.block_role == b) & (f3.emotion == e)].sd_excess.to_numpy() for b, e in CELLS]
    fills = [f3[(f3.block_role == b) & (f3.emotion == e)].significant.to_numpy(bool) for b, e in CELLS]
    raincloud(ax, vals, [COL[e] for _, e in CELLS], filled=fills, s=5)
    ax.axhline(0, color=GREY, lw=0.6, ls="--")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{SHORT[b]}\n{e[:3]}" for b, e in CELLS], fontsize=6)
    ax.set_xlim(-0.6, 5.5)
    ax.set_ylabel("Observed SD − noise-only SD (σ = 3)")
    letter(ax, "c", dx=-0.2)
    save(fig, out, "ExtData_Fig2_test2_all_blocks")


def ed3(desc, dt, out):
    fig = plt.figure(figsize=(W, 120 * MM))
    gs = fig.add_gridspec(2, 6, height_ratios=[0.8, 1.2], hspace=0.6, wspace=0.6,
                          left=0.12, right=0.99, top=0.96, bottom=0.12)
    d = desc[desc.variant == "all"]
    for k, (e, col, xlab) in enumerate((("Sadness", "mean", "Baseline Sadness: series mean"),
                                       ("Fear", "mean", "Baseline Fear: series mean"),
                                       ("Sadness", "sd", "Baseline Sadness: series SD"))):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        x = d[(d.block_role == "base") & (d.emotion == e)][col].to_numpy(float)
        pos = x[x > 0]
        bins = np.linspace(0, pos.max() * 1.05, 14)
        ax.hist(pos, bins=bins, density=True, color=LIGHT, ec="white", lw=0.4)
        g = np.linspace(0.05, bins[-1], 300)
        for dist, lab, lsty in ((stats.gamma, "Gamma", "-"), (stats.weibull_min, "Weibull", "--"),
                                (stats.lognorm, "lognormal", ":")):
            par = dist.fit(pos, floc=0)
            ax.plot(g, dist.pdf(g, *par), lsty, color=COL[e], lw=1.0, label=lab)
        ax.set_xlabel(xlab + " (> 0)")
        ax.set_ylabel("Density")
        ax.text(0.97, 0.55, f"{int((x == 0).sum())} series = 0", transform=ax.transAxes,
                ha="right", fontsize=5.8, color=DARK)
        ax.legend(fontsize=5.6, loc="upper right")
        if k == 0:
            letter(ax, "a", dx=-0.35)
    T = dt.copy()
    T["lo"] = np.where(T.estimate > 0, np.exp(np.log(T.estimate.clip(lower=1e-9)) -
                       1.96 * np.abs(np.log(T.estimate.clip(lower=1e-9)) / T.stat)), np.nan)
    T["hi"] = np.where(T.estimate > 0, np.exp(np.log(T.estimate.clip(lower=1e-9)) +
                       1.96 * np.abs(np.log(T.estimate.clip(lower=1e-9)) / T.stat)), np.nan)
    comps = list(dict.fromkeys(T[T.test == "paired t"].comparison))

    def nice(c):
        a, b = c.split(" vs ")
        (ab, ae), (bb, be) = a.split(), b.split()
        return f"{BNAME[ab]}: {ae} vs {be}" if ab == bb else f"{ae}: {BNAME[ab].lower()} vs {BNAME[bb].lower()}"

    msd = [("hurdle part 2: Gamma | > 0", "Gamma > 0", "s", "#4C78A8"),
           ("hurdle part 2: Weibull | > 0", "Weibull > 0", "^", "#9ECAE9"),
           ("Tweedie (zeros included)", "Tweedie", "o", DARK)]
    shr = [("binomial (logistic GEE on probes)", "Binomial", "o", DARK),
           ("Beta regression (0/1 -> 0.001/0.999)", "Beta", "s", "#B07AA1")]
    panels = [("mean", msd, "Ratio of means", 1, "Series mean"),
              ("sd", msd, "Ratio of SDs", 1, "Series SD"),
              ("share > 0", shr, "Odds ratio", 100, "Share of probes > 0")]
    for k, (meas, tests, xl, sc, title) in enumerate(panels):
        ax = fig.add_subplot(gs[1, 2 * k])
        for j, (tn, lab, mk, c) in enumerate(tests):
            g = T[(T.measure == meas) & (T.test == tn)].set_index("comparison")
            for i, cmp in enumerate(comps):
                r = g.loc[cmp]
                y = i + (j - (len(tests) - 1) / 2) * 0.25
                ax.plot([r.lo, r.hi], [y, y], color=c, lw=0.9)
                ax.plot(r.estimate, y, mk, color=c, ms=2.8, mfc=c if r.p_holm < .05 else "white",
                        label=lab if i == 0 else None)
        ax.axvline(1, color=GREY, lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(0.3, 3.3)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks([0.5, 1, 2])
        ax.set_xticklabels(["0.5", "1", "2"])
        ax.set_xlabel(xl)
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels([nice(c) for c in comps] if k == 0 else [], fontsize=5.8)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=6.5, x=0)
        ax.legend(fontsize=5.4, loc="upper center", bbox_to_anchor=(1.0, -0.16), ncol=3,
                  handletextpad=0.1, columnspacing=0.6)
        if k == 0:
            letter(ax, "b", dx=-1.25)
        ax = fig.add_subplot(gs[1, 2 * k + 1])
        g = T[(T.measure == meas) & (T.test == "paired t")].set_index("comparison")
        for i, cmp in enumerate(comps):
            r = g.loc[cmp]
            se = abs(r.estimate / r.stat) * sc
            ax.plot([r.estimate * sc - 2.01 * se, r.estimate * sc + 2.01 * se], [i, i],
                    color=GREY, lw=0.9)
            ax.plot(r.estimate * sc, i, "D", color=GREY, ms=2.6,
                    mfc=GREY if r.p_holm < .05 else "white")
        ax.axvline(0, color=GREY, lw=0.6, ls="--")
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_xlabel("Paired t: difference" + (" (pp)" if sc == 100 else " (points)"))
    fig.text(0.12, 0.015, "Filled: p_Holm < .05 (9 comparisons per test and measure). "
             "Lines: 95% CI (GEE/cluster-robust SE; t-test: t with df ≈ 49).",
             fontsize=5.8, color=DARK)
    save(fig, out, "ExtData_Fig3_distributions_and_tests")


def ed4(dsim, out):
    name = {"paired t": "Paired t", "Wilcoxon": "Wilcoxon",
            "hurdle part 2: Gamma | > 0": "Gamma > 0", "hurdle part 2: Weibull | > 0": "Weibull > 0",
            "Tweedie (zeros included)": "Tweedie", "binomial (logistic GEE on probes)": "Binomial",
            "Beta regression (0/1 -> 0.001/0.999)": "Beta .001/.999"}
    meas = [("mean", "Series mean"), ("sd", "Series SD"), ("share > 0", "Share > 0")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 70 * MM), sharey=True,
                            gridspec_kw={"wspace": 0.08})
    labels, ys, y = [], [], 0
    for m, ml in meas:
        g = dsim[dsim.measure == m]
        for t in name:
            if t not in set(g.test):
                continue
            for ax, scen in zip(axs, ("H0 (no effect)", "effect added")):
                r = g[(g.test == t) & (g.scenario == scen)].iloc[0]
                lo, hi = r.rate_p_lt_05 - 1.96 * r.mc_se, r.rate_p_lt_05 + 1.96 * r.mc_se
                bad = scen.startswith("H0") and lo > 0.05
                c = FEAR if bad else (DARK if scen.startswith("H0") else SAD)
                ax.plot([100 * lo, 100 * hi], [y, y], color=c, lw=1.2)
                ax.plot(100 * r.rate_p_lt_05, y, "o", color=c, ms=3.2)
            labels.append(f"{ml}: {name[t]}")
            ys.append(y)
            y += 1
        y += 0.6
    axs[0].axvline(5, color=GREY, lw=0.6, ls="--")
    axs[0].set_xlim(0, 10)
    axs[0].set_xlabel("False-positive rate (%), no true effect")
    axs[1].set_xlim(0, 100)
    axs[1].set_xlabel("Power (%), known effect added")
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels(labels, fontsize=6)
    axs[0].invert_yaxis()
    axs[0].text(9.9, -0.6, "red: 95% interval above 5%", ha="right", fontsize=5.8, color=FEAR)
    letter(axs[0], "a", dx=-0.62)
    letter(axs[1], "b", dx=-0.05)
    save(fig, out, "ExtData_Fig4_test_calibration")


def ed5(dissoc, P, asg, out):
    fig = plt.figure(figsize=(W, 125 * MM))
    gs = fig.add_gridspec(2, 12, hspace=0.75, wspace=1.2, left=0.08, right=0.99,
                          top=0.96, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0:5])
    x = dissoc[(dissoc.variant == "all") & dissoc.eligible].drop_duplicates(["subject", "block_role"]).copy()
    x["lr"] = np.log2(x.sd_sadness / x.sd_fear).replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["lr"])
    vals = [x[x.block_role == b].lr.to_numpy() for b in BLOCKS]
    fills = [x[x.block_role == b].asym_significant.to_numpy(bool) for b in BLOCKS]
    raincloud(ax, vals, [DARK] * 3, filled=fills)
    ax.axhline(0, color=GREY, lw=0.6, ls="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{BNAME[b]}\nn = {len(v)}" for b, v in zip(BLOCKS, vals)])
    ax.set_xlim(-0.6, 2.5)
    ax.set_ylabel("log2(SD Sadness / SD Fear)")
    ax.text(-0.55, 1.25, "↑ Sadness varies more", fontsize=5.8, color=SAD)
    ax.text(-0.55, -1.85, "↓ Fear varies more", fontsize=5.8, color=FEAR)
    ax.set_title("Filled: asymmetry test p < .05", fontsize=6.3, loc="left")
    letter(ax, "a", dx=-0.17)
    ax = fig.add_subplot(gs[0, 7:12])
    rows = [("present", "nonnull", 0.1, "o", 1.0), ("fluctuating, σ = 3", "fluctuation", 3.0, "s", 0.9),
            ("σ = 5", "fluctuation", 5.0, "^", 0.6), ("σ = 10", "fluctuation", 10.0, "D", 0.35)]
    for k, (b, e) in enumerate(CELLS):
        for j, (lab, an, par, mk, a) in enumerate(rows):
            r = P[(P.analysis == an) & (P.block_role == b) & (P.emotion == e) & (P.variant == "all") &
                  np.isclose(P.parameter_value, par)].iloc[0]
            yy = k * 5 + j
            ax.plot([100 * r.prev_hpdi_low, 100 * r.prev_hpdi_high], [yy, yy], color=COL[e], lw=1, alpha=a)
            ax.plot(100 * r.prev_map, yy, mk, color=COL[e], ms=2.8, alpha=a,
                    label=lab if k == 1 else None)
    ax.axvline(50, color=GREY, lw=0.6, ls=":")
    ax.set_yticks([k * 5 + 1.5 for k in range(6)])
    ax.set_yticklabels([f"{SHORT[b]} · {e}" for b, e in CELLS], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Population prevalence (%), MAP and 95% HPDI")
    ax.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=4,
              handletextpad=0.2)
    ax.tick_params(axis="y", length=0)
    letter(ax, "b", dx=-0.3)
    s = asg[asg.solution == "selected"]
    for k, (b, e) in enumerate(CELLS):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        g = s[(s.block_role == b) & (s.emotion == e)]
        order = g.groupby("class_label").frac_nonzero.median().sort_values().index
        alphas = np.linspace(0.25, 1, len(order)) if len(order) > 1 else [1]
        for ci, cl in enumerate(order):
            h = g[g.class_label == cl]
            ax.scatter(100 * h.frac_nonzero + RNG.uniform(-1.5, 1.5, len(h)), np.log1p(h.sd), s=6,
                       color=COL[e], alpha=alphas[ci], lw=0.2, edgecolor="white")
        ax.set_title(f"{SHORT[b]} · {e[:3]}: k = {len(order)}", fontsize=6, loc="left")
        ax.set_xlim(-5, 105)
        ax.set_ylim(-0.1, 3.8)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("Probes > 0 (%)", fontsize=6)
        if k == 0:
            ax.set_ylabel("log(1 + SD)")
            letter(ax, "c", dx=-0.6)
        else:
            ax.set_yticklabels([])
    save(fig, out, "ExtData_Fig5_specificity_prevalence_types")


# ===================================================================
# ENTRY
# ===================================================================

def make_all(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig1(data["desc"], out)
    fig2(data["desc"], data["nonnull"], data["P"], out)
    fig3(data["prof"], data["sig"], data["dissoc"], out)
    fig4(data["fl"], data["P"], data["nonnull"], out)
    fig5(data["fps"], data["fl"], data["dissoc"], data["T"], data["asg"], out)
    ed1(data["desc"], data["dt"], data["fps"], out)
    ed2(data["fl"], out)
    ed3(data["desc"], data["dt"], out)
    ed4(data["dsim"], out)
    ed5(data["dissoc"], data["P"], data["asg"], out)
