# -*- coding: utf-8 -*-
"""
plot_vas_timeline.py
====================
Plot every subject's Sadness and Fear ratings over the WHOLE emo_rest
session: base rest -> video 1 -> post-video rating -> rest -> video 2 ->
post-video rating -> rest, on one real-time axis.

Sources
  Behav/raw/dyn{NNN}_results.csv                     the ratings
  Behav/MEG_trigger_files/{NNN}_emo_rest_all_triggers.csv   the timing

Blocks and their sad/fear labels come from analyze_behav_vas.py (same
block definition, so the plot shows exactly the blocks that are analysed).
Timeouts are plotted like every other rating.

Timing
  Trigger codes: 1 rest onset, 2 rest offset, 3 probe onset, 5 post-video
  rating onset, 7 sad video, 8 fear video, 9 video offset.
  - Rest windows = rest_onset -> rest_offset lasting 18-22 min (as in
    cut_dyn_blocks_python.py). A window is labelled sad / fear if a sad /
    fear video started after the previous window, else base. EVERY such
    window is drawn, also one without ratings ("no ratings"), so the plot
    shows what cut_dyn_blocks_python.py cuts.
  - Each rating block goes to the MEG window whose probe-onset triggers fit
    its probe times best (label only breaks ties). If the label of that
    window differs from the block's label, both are shown and flagged.
  - Each probe (elapsed_s, block-relative) is placed at the matching probe
    onset (3) of its window: nearest onset within MATCH_TOL_S of
    rest_onset + elapsed_s + median lag. Unmatched probes are placed at
    rest_onset + elapsed_s + median lag.
  - Post-video ratings are placed at the post-video rating onset (5) after
    their video.
  - No usable trigger file, or a block that fits no MEG window: drawn on its
    own elapsed_s clock AFTER everything else (never on top of a real
    block), labelled "(block clock)", and the title says so.

Outputs (OUT_DIR = Behav/analysis/vas_timeline)
  dyn{NNN}_timeline.png          one figure per subject
  all_subjects_timeline.pdf      all subjects, one page each
  timeline_qc.csv                per subject: timing source, matched probes, lag

Plain Python, no SLURM. Usage:
    python3 plot_vas_timeline.py              # all subjects (ID < SUBJ_MAX)
    python3 plot_vas_timeline.py 014 019      # these subjects only
"""

import argparse
import glob
import os
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend (Linux server)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import analyze_behav_vas as abv       # block definition + loading

# ===================================================================
# CONFIG
# ===================================================================

BEHAV_DIR = abv.BEHAV_DIR
TRIG_DIR = abv.TRIG_DIR
OUT_DIR = os.path.join(abv.OUT_DIR, "vas_timeline")
SUBJ_MAX = abv.SUBJ_MAX

MATCH_TOL_S = 5.0          # probe <-> probe-onset trigger matching tolerance
MIN_MATCH_FRAC = 0.5       # a rating block is placed in a MEG rest window only
                           # if at least this share of its probes match there
FALLBACK_GAP_MIN = 3.0     # gap between blocks when there are no triggers

# trigger codes
C_REST_ON, C_REST_OFF, C_PROBE_ON, C_POST_ON = 1, 2, 3, 5
C_VSAD, C_VFEAR, C_VOFF = 7, 8, 9

# colours: emotions = series; block / video spans are neutral backgrounds
COL = {"Sadness": "#2a78d6", "Fear": "#eb6834"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REST_BG, VIDEO_BG = "#f0efec", "#d9d8d3"


# ===================================================================
# TRIGGERS
# ===================================================================

def load_trigger_events(subject_id):
    """Rest windows (labelled base/sad/fear), videos and post-video rating
    onsets from the trigger file. Returns None if not usable."""
    path = os.path.join(TRIG_DIR, f"{subject_id}_emo_rest_all_triggers.csv")
    if not os.path.exists(path):
        return None
    t = pd.read_csv(path)
    if "trigger_code" not in t.columns or "timestamp_s" not in t.columns:
        return None
    codes = t["trigger_code"].to_numpy(dtype=int)
    times = t["timestamp_s"].to_numpy(dtype=float)
    idx = np.arange(len(codes))

    videos = []
    for i in np.flatnonzero(np.isin(codes, [C_VSAD, C_VFEAR])):
        off = np.flatnonzero((codes == C_VOFF) & (idx > i))
        post = np.flatnonzero((codes == C_POST_ON) & (idx > i))
        videos.append(dict(i=i, label="sad" if codes[i] == C_VSAD else "fear",
                           t_on=times[i],
                           t_off=times[off[0]] if len(off) else np.nan,
                           t_post=times[post[0]] if len(post) else np.nan))

    windows, prev_end = [], -1
    for i in np.flatnonzero(codes == C_REST_ON):
        off = np.flatnonzero((codes == C_REST_OFF) & (idx > i))
        if len(off) == 0:
            continue
        j = off[0]
        dur = times[j] - times[i]
        if not (abv.REST_DUR_MIN_S <= dur <= abv.REST_DUR_MAX_S):
            continue
        before = [v for v in videos if prev_end < v["i"] < i]
        role = before[-1]["label"] if before else "base"
        probes = times[(codes == C_PROBE_ON) & (idx > i) & (idx < j)]
        windows.append(dict(role=role, t_on=times[i], t_off=times[j],
                            probes=probes, video=before[-1] if before else None))
        prev_end = j
    if not windows:
        return None
    return dict(windows=windows, videos=videos)


def place_block(elapsed, win):
    """Session time of each probe of one block, matched to the probe-onset
    triggers of its rest window. Returns (times, n_matched, lag_s)."""
    rel = win["probes"] - win["t_on"]
    if len(rel) == 0:
        return win["t_on"] + elapsed, 0, np.nan
    # lag between the task's block clock and rest onset: median offset of
    # nearest pairs, robust to a few missing triggers
    near = np.array([rel[np.argmin(np.abs(rel - e))] - e for e in elapsed])
    lag = float(np.median(near))
    out, n_match = [], 0
    for e in elapsed:
        d = np.abs(rel - (e + lag))
        k = int(np.argmin(d))
        if d[k] <= MATCH_TOL_S:
            out.append(win["t_on"] + rel[k])
            n_match += 1
        else:
            out.append(win["t_on"] + e + lag)
    return np.array(out), n_match, lag


# ===================================================================
# ONE SUBJECT
# ===================================================================

def build_timeline(subject_id):
    """Long table of every rating with its session time, plus the spans to
    shade and a QC row."""
    df = abv.load_results(subject_id)
    qc, log = dict(subject=subject_id), []
    blocks = abv.define_blocks(df, subject_id, qc, log)
    trig = load_trigger_events(subject_id)

    rows, spans, flags = [], [], []
    wins = trig["windows"] if trig else []
    qc.update(timing="triggers" if trig else "no trigger file: block clock only",
              n_probes=0, n_matched=0, n_meg_rest_windows=len(wins),
              meg_rest_windows=",".join(w["role"] for w in wins))
    if trig and len(wins) != 3:
        flags.append(f"{len(wins)} MEG rest windows (expected 3)")

    # each rating block goes to the unused MEG rest window whose probe-onset
    # triggers fit its probe times best (the window label only breaks ties),
    # so a wrong or missing video trigger cannot move a block
    used = {}
    fallback = []
    for k in sorted(blocks):
        b = blocks[k]
        role = b["block_role"].iloc[0]
        el = pd.to_numeric(b["elapsed_s"], errors="coerce").to_numpy(dtype=float)
        uniq = np.unique(el[np.isfinite(el)])
        best = None
        for wi, w in enumerate(wins):
            if wi in used:
                continue
            tt, n_m, lag = place_block(uniq, w)
            score = (n_m, w["role"] == role)
            if best is None or score > best[0]:
                best = (score, wi, tt, n_m, lag)
        if best is not None and best[3] >= MIN_MATCH_FRAC * len(uniq):
            _, wi, tt, n_m, lag = best
            used[wi] = role
            qc[f"lag_{role}_s"] = round(lag, 2) if np.isfinite(lag) else np.nan
            if wins[wi]["role"] != role:
                flags.append(f"{role} ratings fit the MEG '{wins[wi]['role']}' window")
        else:
            fallback.append((role, uniq))
            tt, n_m = None, 0
        qc["n_probes"] += len(uniq)
        qc["n_matched"] += n_m
        if tt is not None:
            tmap = dict(zip(uniq, tt))
            for _, r in b.iterrows():
                if np.isfinite(r["elapsed_s"]):
                    rows.append(dict(kind="rest", block=role, emotion=r["emotion"],
                                     rating=r["rating"], t=tmap[r["elapsed_s"]]))

    # MEG view: every valid rest window and its video, with or without ratings
    for wi, w in enumerate(wins):
        label = w["role"] if wi in used else f"{w['role']} (no ratings)"
        spans.append(("rest", label, w["t_on"], w["t_off"]))
        v = w["video"]
        if v is not None and np.isfinite(v["t_off"]):
            spans.append(("video", v["label"], v["t_on"], v["t_off"]))
    for wi in sorted(used):
        if used[wi] != wins[wi]["role"]:
            i = [n for n, sp in enumerate(spans) if sp[0] == "rest" and sp[2] == wins[wi]["t_on"]][0]
            spans[i] = ("rest", f"MEG {wins[wi]['role']} / ratings {used[wi]}", *spans[i][2:])

    # blocks without a matching MEG window: on their own clock, AFTER
    # everything else, so they never overlap a real block
    if fallback:
        if trig:
            qc["timing"] = "triggers (some blocks on block clock)"
            flags.append(f"no MEG window for: {', '.join(r for r, _ in fallback)}")
        start = max((sp[3] for sp in spans), default=-FALLBACK_GAP_MIN * 60) \
            + FALLBACK_GAP_MIN * 60
        for role, uniq in fallback:
            b = [blocks[k] for k in blocks if blocks[k]["block_role"].iloc[0] == role][0]
            spans.append(("rest", f"{role} (block clock)", start,
                          start + max(uniq.max(), 1)))
            for _, r in b.iterrows():
                if np.isfinite(r["elapsed_s"]):
                    rows.append(dict(kind="rest", block=role, emotion=r["emotion"],
                                     rating=r["rating"], t=start + r["elapsed_s"]))
            start += uniq.max() + FALLBACK_GAP_MIN * 60
        spans.sort(key=lambda sp: sp[2])

    # post-video ratings: k-th post_video run belongs to the k-th video
    ctx = df["context"].astype(str).str.strip().str.lower()
    post = df[ctx == "post_video"].copy()
    post["run"] = (ctx.ne(ctx.shift()) & ctx.eq("post_video")).cumsum()[ctx == "post_video"]
    vids = [v for v in (trig["videos"] if trig else []) if np.isfinite(v["t_post"])]
    if not vids:
        # no video timing: put each post-video rating in the gap before the
        # rest block it precedes
        rest = [s for s in spans if s[0] == "rest"]
        vids = [dict(label=b[1], t_post=(a[3] + b[2]) / 2) for a, b in zip(rest, rest[1:])]
    for n, (_, g) in enumerate(post.groupby("run")):
        if n < len(vids):
            for _, r in g.iterrows():
                rows.append(dict(kind="post_video", block=vids[n]["label"],
                                 emotion=r["emotion"], rating=r["rating"],
                                 t=vids[n]["t_post"]))

    tl = pd.DataFrame(rows)
    t0 = min(s[2] for s in spans) if spans else 0.0
    if len(tl):
        tl["t_min"] = (tl["t"] - t0) / 60
    spans = [(a, b, (s - t0) / 60, (e - t0) / 60) for a, b, s, e in spans]
    qc["blocks"] = ",".join(blocks[k]["block_role"].iloc[0] for k in sorted(blocks))
    qc["flags"] = "; ".join(x for x in [qc.get("flags", "")] + flags if x)
    return tl, spans, qc


def plot_subject(subject_id, tl, spans, qc):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=.6)
    for kind, label, s, e in spans:
        if kind == "rest":
            ax.axvspan(s, e, color=REST_BG, lw=0, zorder=0)
            ax.text((s + e) / 2, 107, label.replace(" (", "\n(").replace(" / ", "\n") + " rest",
                    ha="center", va="bottom", fontsize=8, color=INK2)
        else:
            ax.axvspan(s, e, color=VIDEO_BG, lw=0, zorder=0)
            ax.text((s + e) / 2, 107, f"{label}\nvideo", ha="center", va="bottom",
                    fontsize=8, color=INK)
    for emo in abv.EMOTIONS:
        r = tl[(tl.kind == "rest") & (tl.emotion == emo)].sort_values("t_min")
        for _, g in r.groupby("block"):        # no line across blocks
            ax.plot(g.t_min, g.rating, color=COL[emo], lw=2, zorder=2)
        ax.scatter(r.t_min, r.rating, s=22, color=COL[emo], edgecolor=SURF,
                   linewidth=1, zorder=3, label=emo)
        p = tl[(tl.kind == "post_video") & (tl.emotion == emo)]
        ax.scatter(p.t_min, p.rating, s=64, marker="D", color=COL[emo],
                   edgecolor=INK, linewidth=.8, zorder=4)
    ax.scatter([], [], s=64, marker="D", color="white", edgecolor=INK,
               label="post-video rating")
    ax.set_ylim(-3, 104)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(left=min(-0.5, *(s for _, _, s, _ in spans)) if spans else None)
    ax.set_xlabel("session time (min from first rest onset)")
    ax.set_ylabel("VAS rating (0-100)")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.16))
    match = (f"{qc['n_matched']}/{qc['n_probes']} probes matched to triggers"
             if qc["timing"].startswith("triggers") else qc["timing"])
    flags = textwrap.fill(qc["flags"], 150) if qc.get("flags") else ""
    ax.set_title(f"Subject {subject_id}  -  {match}" + (f"\n{flags}" if flags else ""),
                 loc="left", color=INK, fontsize=10, pad=34)
    fig.tight_layout()
    return fig


# ===================================================================
# MAIN
# ===================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("subjects", nargs="*", help="subject IDs, e.g. 014 019 (default: all)")
    a = ap.parse_args()

    found = sorted(m.group(1) for p in glob.glob(os.path.join(BEHAV_DIR, "dyn*_results.csv"))
                   if (m := re.match(r"dyn(\d+)_results\.csv$", os.path.basename(p))))
    subs = ([s for s in found if s in {x.zfill(3) for x in a.subjects}] if a.subjects
            else [s for s in found if int(s) < SUBJ_MAX])
    os.makedirs(OUT_DIR, exist_ok=True)

    qcs = []
    with PdfPages(os.path.join(OUT_DIR, "all_subjects_timeline.pdf")) as pdf:
        for s in subs:
            try:
                tl, spans, qc = build_timeline(s)
                fig = plot_subject(s, tl, spans, qc)
                fig.savefig(os.path.join(OUT_DIR, f"dyn{s}_timeline.png"), dpi=150)
                pdf.savefig(fig)
                plt.close(fig)
                qc["status"] = "ok"
                print(f"[{s}] {qc['timing']}: {qc['n_matched']}/{qc['n_probes']} probes "
                      f"matched  blocks {qc['blocks']}")
            except Exception as e:
                qc = dict(subject=s, status=f"error: {type(e).__name__}: {e}")
                print(f"[{s}] FAILED {type(e).__name__}: {e}")
            qcs.append(qc)

    pd.DataFrame(qcs).to_csv(os.path.join(OUT_DIR, "timeline_qc.csv"), index=False)
    print(f"\nFigures + all_subjects_timeline.pdf + timeline_qc.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
