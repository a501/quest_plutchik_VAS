#!/usr/bin/env python3
"""
1_build_vas_dataset.py
======================
Build the probe-level VAS dataset for the emo_rest / dyn task. Every later
script (2_, 3_, 4_) reads the CSVs written here.

INPUTS (per subject)
    Behav/raw/dyn{subj}_results.csv                     behavioural ratings
    Behav/MEG_trigger_files/{subj}_emo_rest_all_triggers.csv   (optional, but
        needed for condition order from the video triggers, the exact fixation
        interval, and the button-press measures)

STEP 1  Load behavioural file
        - column names lower-cased
        - the first two rows (timeout trial before block 1) are dropped
          (DROP_FIRST_PAIR, same convention as the earlier scripts)
        - 'post_video' rows (elapsed_s = NaN) separate the 3 rest blocks:
          rows before the 1st post_video run = block 1, between = block 2,
          after the 2nd run = block 3
        - elapsed_s is already block-relative (GetSecs() - block_t): not reset
        - a probe = the 2 rows (Sadness, Fear) sharing one elapsed_s
        - valid = status == 'confirmed' and rating not NaN

STEP 2  Condition labels
        - preferred: order of the video triggers (7 = sad, 8 = fear) in the
          trigger file -> block 2 = first video, block 3 = second video
        - fallback: subject parity (odd -> block 2 = sad). Root subject 000
          (and its repeat visits 200/300) is treated like 001 (sad first).
        - both sources are recorded; disagreements are flagged in qc_build.csv

STEP 3  Trigger-side probe timing (if trigger file exists)
        - false-start trim: > 3 rest_onset (1) -> drop everything before the 2nd
        - rest windows = rest_onset (1) -> next rest_offset (2), kept only if
          18-22 min long
        - each behavioural probe is matched to the nearest vas_probe_onset (3)
          in its block (tolerance MATCH_TOL_S)
        - fix_interval_s = probe onset (3) - previous probe offset (14)
          = the pure fixation interval, without the rating time

STEP 4  Button-press measures per rating screen (if trigger file exists)
        - screen = vas_emotion_onset (4) followed by presses 11 (left) /
          12 (right) up to confirm 13
        - the cursor is simulated: starts at 50, step 5, clipped 0-100
          (CFG.vas_step = 5, start 50 from the MATLAB task)
        - which emotion a screen belongs to is decided by matching the
          simulated final position to the two behavioural ratings of that
          probe; probe_order is used only when both ratings are equal or the
          match fails. Button direction (12 = up or 11 = up) is chosen per
          subject by whichever gives more matches.
        - n_press, n_reversals (direction changes), excess_press
          (= presses beyond the minimum needed to reach the final value,
          i.e. back-and-forth), overshoot (furthest point from 50 beyond the
          final value, VAS units)
        - RT is deliberately NOT computed (participants use the VAS screen to
          rest their eyes, so RT has no clear meaning)

OUTPUTS  (OUT_DIR)
    vas_probes_long.csv   one row per subject x block x probe x emotion
    vas_postvideo.csv     post-video ratings + last base-block rating
    qc_build.csv          one row per subject: counts, sources, flags

USAGE
    python 1_build_vas_dataset.py
    python 1_build_vas_dataset.py --subjects 001 002
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ZhangJianfeng_04/emo_study/Behav")
RAW_DIR = ROOT / "raw"
TRIG_DIR = ROOT / "MEG_trigger_files"
OUT_DIR = ROOT / "dynamics_models"

DROP_FIRST_PAIR = True
REST_MIN_S, REST_MAX_S = 18 * 60, 22 * 60
MATCH_TOL_S = 5.0
VAS_START, VAS_STEP, VAS_MIN, VAS_MAX = 50, 5, 0, 100

C_REST_ON, C_REST_OFF, C_PROBE_ON, C_SCREEN = 1, 2, 3, 4
C_VSAD, C_VFEAR, C_LEFT, C_RIGHT, C_CONF, C_PROBE_OFF = 7, 8, 11, 12, 13, 14


# --------------------------------------------------------------------------
# behavioural file
# --------------------------------------------------------------------------
def norm_emotion(x):
    s = str(x).strip().lower()
    if s.startswith("sad"):
        return "Sadness"
    if s.startswith("fear"):
        return "Fear"
    return np.nan


def load_behav(subj):
    path = RAW_DIR / f"dyn{subj}_results.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    for col in ("context", "elapsed_s", "emotion", "rating"):
        if col not in df.columns:
            raise KeyError(f"{path.name}: column '{col}' missing; has {list(df.columns)}")
    if DROP_FIRST_PAIR:
        df = df.iloc[2:].reset_index(drop=True)
    df["emotion"] = df["emotion"].map(norm_emotion)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip().str.lower()
    else:
        df["status"] = "confirmed"
    if "probe_order" not in df.columns:
        df["probe_order"] = ""
    df["valid"] = (df["status"] == "confirmed") & df["rating"].notna()

    # segment: runs of post_video rows split the rest blocks
    is_post = df["context"].astype(str).str.strip().str.lower().eq("post_video")
    block, run, prev_post = 1, 0, False
    blocks, runs = [], []
    for p in is_post:
        if p and not prev_post:
            run += 1
        if not p and prev_post:
            block += 1
        blocks.append(block if not p else np.nan)
        runs.append(run if p else np.nan)
        prev_post = p
    df["block"] = blocks
    df["post_run"] = runs
    return df


# --------------------------------------------------------------------------
# trigger file
# --------------------------------------------------------------------------
def load_triggers(subj):
    path = TRIG_DIR / f"{subj}_emo_rest_all_triggers.csv"
    if not path.exists():
        return None, "no trigger file"
    t = pd.read_csv(path)
    code_col = "trigger_code" if "trigger_code" in t.columns else None
    time_col = next((c for c in ("timestamp_s", "time_from_start_s") if c in t.columns), None)
    if code_col is None or time_col is None:
        return None, f"trigger columns not found: {list(t.columns)}"
    t = t.rename(columns={code_col: "code", time_col: "t"})[["code", "t"]].copy()
    t["code"] = t["code"].astype(int)
    on = np.flatnonzero(t["code"].values == C_REST_ON)
    note = ""
    if len(on) > 3:
        t = t.iloc[on[1]:].reset_index(drop=True)
        note = f"false start trimmed ({len(on)} rest_onsets)"
    return t, note


def rest_windows(t):
    wins = []
    codes, times = t["code"].values, t["t"].values
    for i in np.flatnonzero(codes == C_REST_ON):
        nxt = np.flatnonzero((codes == C_REST_OFF) & (np.arange(len(codes)) > i))
        if len(nxt) == 0:
            continue
        j = nxt[0]
        dur = times[j] - times[i]
        if REST_MIN_S <= dur <= REST_MAX_S:
            wins.append((i, j))
    return wins


def simulate(presses, up_code):
    pos, path = VAS_START, [VAS_START]
    for c in presses:
        pos += VAS_STEP if c == up_code else -VAS_STEP
        pos = min(max(pos, VAS_MIN), VAS_MAX)
        path.append(pos)
    return pos, path


def screen_features(presses, up_code):
    final, path = simulate(presses, up_code)
    n = len(presses)
    rev = int(np.sum(np.diff(presses) != 0)) if n > 1 else 0
    net = abs(final - VAS_START) / VAS_STEP
    overshoot = max(abs(p - VAS_START) for p in path) - abs(final - VAS_START)
    return dict(implied=final, n_press=n, n_reversals=rev,
                excess_press=n - net, overshoot=overshoot)


def first_emotion(probe_order):
    s = str(probe_order).lower()
    i_s, i_f = s.find("sad"), s.find("fear")
    if i_s < 0 and i_f < 0:
        return None
    if i_f < 0 or (0 <= i_s < i_f):
        return "Sadness"
    return "Fear"


def trigger_probes(t, i0, i1):
    """Probes inside one rest window: onset time (rel), fixation interval, screens."""
    sub = t.iloc[i0:i1 + 1].reset_index(drop=True)
    t0 = sub["t"].iloc[0]
    codes, times = sub["code"].values, sub["t"].values
    probes, last_off = [], None
    on_idx = np.flatnonzero(codes == C_PROBE_ON)
    for k, i in enumerate(on_idx):
        end = on_idx[k + 1] if k + 1 < len(on_idx) else len(codes)
        seg = codes[i:end]
        segt = times[i:end]
        off = np.flatnonzero(seg == C_PROBE_OFF)
        screens = []
        sidx = np.flatnonzero(seg == C_SCREEN)
        for m, s in enumerate(sidx):
            s_end = sidx[m + 1] if m + 1 < len(sidx) else (off[0] if len(off) else len(seg))
            ss = seg[s + 1:s_end]
            conf = np.flatnonzero(ss == C_CONF)
            ss = ss[:conf[0]] if len(conf) else ss
            screens.append(dict(presses=[c for c in ss if c in (C_LEFT, C_RIGHT)],
                                confirmed=len(conf) > 0))
        fix = (times[i] - last_off) if last_off is not None else np.nan
        probes.append(dict(rel_s=times[i] - t0, fix_interval_s=fix, screens=screens))
        if len(off):
            last_off = segt[off[0]]
    return probes


# --------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------
def parity_order(subj):
    root = int(subj) % 100
    sad_first = (root % 2 == 1) or root == 0
    return ["sad", "fear"] if sad_first else ["fear", "sad"]


def trigger_order(t):
    if t is None:
        return None
    vids = [("sad" if c == C_VSAD else "fear") for c in t["code"].values if c in (C_VSAD, C_VFEAR)]
    seq = []
    for v in vids:
        if not seq or seq[-1] != v:
            seq.append(v)
    return seq[:2] if len(seq) >= 2 and set(seq[:2]) == {"sad", "fear"} else None


# --------------------------------------------------------------------------
# per subject
# --------------------------------------------------------------------------
def process_subject(subj):
    qc = dict(subj=subj)
    b = load_behav(subj)
    t, tnote = load_triggers(subj)
    qc["trigger_note"] = tnote

    order_par = parity_order(subj)
    order_trg = trigger_order(t)
    order = order_trg or order_par
    qc["cond_source"] = "triggers" if order_trg else "parity"
    qc["order_parity"] = "-".join(order_par)
    qc["order_triggers"] = "-".join(order_trg) if order_trg else ""
    qc["order_conflict"] = bool(order_trg and order_trg != order_par)
    cond_of_block = {1: "baseline", 2: order[0], 3: order[1]}
    vorder = {order[0]: 1, order[1]: 2}

    wins = rest_windows(t) if t is not None else []
    qc["n_rest_windows"] = len(wins)
    qc["n_blocks_behav"] = int(b["block"].nunique())
    use_trig = t is not None and len(wins) == 3

    # ---- button direction: test both mappings on all screens ----
    tprobes = {}
    if use_trig:
        for k, (i0, i1) in enumerate(wins, start=1):
            tprobes[k] = trigger_probes(t, i0, i1)

    rows = []
    for blk in sorted(b["block"].dropna().unique()):
        blk = int(blk)
        bb = b[b["block"] == blk]
        times = np.sort(bb["elapsed_s"].dropna().unique())
        tp = tprobes.get(blk, [])
        trel = np.array([p["rel_s"] for p in tp]) if tp else np.array([])
        prev = np.nan
        for pi, et in enumerate(times, start=1):
            pr = bb[bb["elapsed_s"] == et]
            match = None
            if len(trel):
                j = int(np.argmin(np.abs(trel - et)))
                if abs(trel[j] - et) <= MATCH_TOL_S:
                    match = tp[j]
            for _, r in pr.iterrows():
                rows.append(dict(
                    subj=subj, root_subj=f"{int(subj) % 100:03d}", block=blk,
                    condition=cond_of_block.get(blk, f"block{blk}"),
                    video_order=vorder.get(cond_of_block.get(blk), 0),
                    cond_source=qc["cond_source"], probe_idx=pi,
                    elapsed_s=et, gap_s=et - prev if pi > 1 else np.nan,
                    fix_interval_s=match["fix_interval_s"] if match else np.nan,
                    trig_matched=match is not None,
                    emotion=r["emotion"], rating=r["rating"], status=r["status"],
                    valid=bool(r["valid"]), probe_order=r["probe_order"],
                    _screens=match["screens"] if match else None))
            prev = et
    long = pd.DataFrame(rows)

    # ---- assign screens to emotions ----
    best = None
    if use_trig and len(long):
        for up in (C_RIGHT, C_LEFT):
            hits = 0
            for _, g in long.groupby(["block", "probe_idx"]):
                sc = g["_screens"].iloc[0]
                if not sc:
                    continue
                ratings = set(g["rating"].dropna())
                hits += sum(simulate(s["presses"], up)[0] in ratings for s in sc)
            if best is None or hits > best[1]:
                best = (up, hits)
        qc["up_button"] = best[0]
    up = best[0] if best else C_RIGHT

    feats = {c: [] for c in ("n_press", "n_reversals", "excess_press", "overshoot", "screen_match")}
    n_match = n_amb = n_fail = 0
    for _, g in long.groupby(["block", "probe_idx"], sort=False):
        sc = g["_screens"].iloc[0] if use_trig else None
        assign = {}
        if sc and len(sc) == 2:
            f = [screen_features(s["presses"], up) for s in sc]
            rS = g.loc[g["emotion"] == "Sadness", "rating"]
            rF = g.loc[g["emotion"] == "Fear", "rating"]
            rS = rS.iloc[0] if len(rS) else np.nan
            rF = rF.iloc[0] if len(rF) else np.nan
            a = f[0]["implied"] == rS and f[1]["implied"] == rF
            bb_ = f[0]["implied"] == rF and f[1]["implied"] == rS
            po = first_emotion(g["probe_order"].iloc[0])
            if a and not bb_:
                assign, tag = {"Sadness": f[0], "Fear": f[1]}, "matched"; n_match += 1
            elif bb_ and not a:
                assign, tag = {"Fear": f[0], "Sadness": f[1]}, "matched"; n_match += 1
            else:
                first = po or "Sadness"
                other = "Fear" if first == "Sadness" else "Sadness"
                assign = {first: f[0], other: f[1]}
                tag = "equal_ratings" if a and bb_ else "mismatch"
                if a and bb_:
                    n_amb += 1
                else:
                    n_fail += 1
        for _, r in g.iterrows():
            f = assign.get(r["emotion"])
            for c in ("n_press", "n_reversals", "excess_press", "overshoot"):
                feats[c].append(f[c] if f else np.nan)
            feats["screen_match"].append(tag if f else "no_screens")
    long = long.drop(columns="_screens")
    for c, v in feats.items():
        long[c] = v
    # screens of non-confirmed ratings are not interpretable as rating behaviour
    long.loc[~long["valid"], ["n_press", "n_reversals", "excess_press", "overshoot"]] = np.nan

    qc.update(n_probes=int(long[["block", "probe_idx"]].drop_duplicates().shape[0]),
              n_valid_ratings=int(long["valid"].sum()),
              trig_matched_frac=round(float(long["trig_matched"].mean()), 3) if len(long) else 0,
              screens_matched=n_match, screens_equal=n_amb, screens_mismatch=n_fail)
    for blk in (1, 2, 3):
        qc[f"n_valid_block{blk}"] = int(long[(long["block"] == blk)]["valid"].sum())

    # ---- post-video + last base rating ----
    pv = []
    post = b[b["post_run"].notna()]
    for run, g in post.groupby("post_run"):
        cond = cond_of_block.get(int(run) + 1)
        for _, r in g.iterrows():
            pv.append(dict(subj=subj, video=cond, video_order=vorder.get(cond, 0),
                           emotion=r["emotion"], rating=r["rating"], valid=bool(r["valid"])))
    base = long[(long["block"] == 1) & long["valid"]]
    if len(base):
        last_p = base["probe_idx"].max()
        for _, r in base[base["probe_idx"] == last_p].iterrows():
            pv.append(dict(subj=subj, video="base_last", video_order=0,
                           emotion=r["emotion"], rating=r["rating"], valid=True))
    qc["n_postvideo_rows"] = len([p for p in pv if p["video"] != "base_last"])
    return long, pd.DataFrame(pv), qc


def main():
    global RAW_DIR, TRIG_DIR, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--trig", default=str(TRIG_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--subjects", nargs="*")
    a = ap.parse_args()
    RAW_DIR, TRIG_DIR, OUT_DIR = Path(a.raw), Path(a.trig), Path(a.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subs = [s.zfill(3) for s in a.subjects] if a.subjects else sorted(
        m.group(1) for p in RAW_DIR.glob("dyn*_results.csv")
        if (m := re.match(r"dyn(\d+)_results\.csv$", p.name)))
    subs = [s for s in subs if int(s) < 200]
    print(f"{len(subs)} subjects")

    longs, pvs, qcs = [], [], []
    for s in subs:
        try:
            l, p, q = process_subject(s)
            longs.append(l); pvs.append(p); qcs.append(q)
            print(f"[{s}] probes={q['n_probes']} valid={q['n_valid_ratings']} "
                  f"cond={q['cond_source']} trig_match={q['trig_matched_frac']} "
                  f"screens ok/eq/bad={q['screens_matched']}/{q['screens_equal']}/{q['screens_mismatch']}"
                  + ("  !! ORDER CONFLICT" if q["order_conflict"] else ""))
        except Exception as e:
            print(f"[{s}] FAILED {type(e).__name__}: {e}")
            qcs.append(dict(subj=s, error=f"{type(e).__name__}: {e}"))

    pd.concat(longs).to_csv(OUT_DIR / "vas_probes_long.csv", index=False)
    pd.concat(pvs).to_csv(OUT_DIR / "vas_postvideo.csv", index=False)
    pd.DataFrame(qcs).to_csv(OUT_DIR / "qc_build.csv", index=False)
    print(f"\nwritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
