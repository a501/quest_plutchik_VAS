# -*- coding: utf-8 -*-
"""
check_cut_dyn_blocks.py
=======================
Check the output of cut_dyn_blocks_python.py: open every cut block in
MEG/2_cut/, read its triggers exactly as the cutting script does (STI101,
min_duration 2 ms) and test what a correct cut must contain.

PER CUT FILE
  base (1b_base)      no video trigger
  sad / fear (2b, 3b) exactly one video trigger, 7 for a *_sad file / 8 for
                      a *_fear file, before the rest_onset
  all                 exactly one complete rest_onset -> rest_offset window
                      of 18-22 min inside the file, number of probe onsets
                      (3). How much is cut before / after the rest is only
                      reported (rest_on_s, end_pad_s, video_at_s), not tested.
  vs ratings          the matching behaviour block
                      (Behav/analysis/behav_per_block/dyn{NNN}_block*_{cond}_behav.csv,
                      written by analyze_behav_vas.py): same number of
                      probes, and probe times (probe onset - rest onset)
                      match elapsed_s within MATCH_TOL_S

PER SUBJECT
  exactly one base, one sad and one fear block; 2b / 3b order agrees with
  the block order of the ratings

Nothing is changed. Plain Python, no SLURM; N_WORKERS files are read at
the same time. Reading the triggers means reading the whole file from
disk, so each file takes as long as copying it once.

Outputs
  MEG/2_cut/cut_check_files.csv      one row per cut file, every measure
  MEG/2_cut/cut_check_subjects.csv   one row per subject, OK / problems

Usage
    python3 check_cut_dyn_blocks.py            # all subjects in 2_cut/
    python3 check_cut_dyn_blocks.py 000 010    # these subjects only
"""

import argparse
import glob
import multiprocessing as mp
import os
import re
import time
import warnings

import mne
import numpy as np
import pandas as pd

# ===================================================================
# CONFIG (same values as cut_dyn_blocks_python.py)
# ===================================================================

CUT_DIR = "/home/ZhangJianfeng_04/emo_study/MEG/2_cut"
PERBLOCK_DIR = "/home/ZhangJianfeng_04/emo_study/Behav/analysis/behav_per_block"

REST_ON, REST_OFF, PROBE_ON = 1, 2, 3
VIDEO = {7: "sad", 8: "fear"}
REST_DUR_MIN_S, REST_DUR_MAX_S = 18 * 60, 22 * 60

N_WORKERS = 4          # files read at the same time
MATCH_TOL_S = 5.0      # probe onset trigger <-> behavioural elapsed_s
MIN_MATCH_FRAC = 0.9   # share of rating probes that must match a probe trigger

split_re = re.compile(r"-\d+\.fif$")
name_re = re.compile(r"^(\d+)_.*_(\d)b_(base\d*|sad|fear)\.fif$")


# ===================================================================
# ONE CUT FILE
# ===================================================================

def behav_block(subj, cond):
    """Unique probe times (elapsed_s) and block number of the rating block."""
    files = glob.glob(os.path.join(PERBLOCK_DIR, f"dyn{subj}_block*_{cond}_behav.csv"))
    if len(files) != 1:
        return None, None
    d = pd.read_csv(files[0])
    el = pd.to_numeric(d["elapsed_s"], errors="coerce").dropna().unique()
    k = int(re.search(r"_block(\d+)_", os.path.basename(files[0])).group(1))
    return np.sort(el), k


def check_file(path):
    name = os.path.basename(path)
    m = name_re.match(name)
    row = dict(file=name, subject=m.group(1) if m else "", block_num=int(m.group(2)) if m else np.nan,
               cond=re.sub(r"\d+$", "", m.group(3)) if m else "", problems=[])
    P = row["problems"]
    if not m:
        P.append("file name does not follow {subj}_..._{N}b_{cond}.fif")
        return row

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
    sf, dur = raw.info["sfreq"], raw.times[-1]
    # initial_event=True: a video block is cut to start exactly at the video
    # trigger, which is then already "on" at the first sample and would be
    # missed by the default find_events
    ev = mne.find_events(raw, stim_channel="STI101", min_duration=0.002,
                         initial_event=True, verbose=False)
    t = (ev[:, 0] - raw.first_samp) / sf
    c = ev[:, 2]
    on, off, probes = t[c == REST_ON], t[c == REST_OFF], t[c == PROBE_ON]
    vids = [(tt, VIDEO[cc]) for tt, cc in zip(t, c) if cc in VIDEO]
    row.update(duration_min=round(dur / 60, 2), n_rest_on=len(on), n_rest_off=len(off),
               n_video=len(vids), videos=",".join(v for _, v in vids),
               n_probe_triggers=len(probes))

    # ---- one complete rest window ----
    if len(on) != 1 or len(off) != 1:
        P.append(f"{len(on)} rest_onset / {len(off)} rest_offset (expected 1/1)")
    if len(on) and len(off):
        r_on, r_off = on[0], off[off > on[0]][0] if np.any(off > on[0]) else np.nan
        rest = r_off - r_on
        row.update(rest_on_s=round(r_on, 2), rest_min=round(rest / 60, 2),
                   end_pad_s=round(dur - r_off, 2))
        if not (REST_DUR_MIN_S <= rest <= REST_DUR_MAX_S):
            P.append(f"rest window {rest / 60:.1f} min (expected 18-22)")
    else:
        r_on = np.nan

    # ---- start of the file ----
    if row["cond"] == "base":
        if vids:
            P.append(f"video trigger(s) in a base block: {row['videos']}")
    else:
        if len(vids) != 1:
            P.append(f"{len(vids)} video triggers (expected 1)")
        if vids:
            v_t, v_lab = vids[0]
            row["video_at_s"] = round(v_t, 2)
            if v_lab != row["cond"]:
                P.append(f"file says {row['cond']} but video trigger is {v_lab}")
            if np.isfinite(r_on) and r_on < v_t:
                P.append("rest_onset before the video")

    # ---- against the ratings ----
    el, k_behav = behav_block(row["subject"], row["cond"])
    row["behav_block_num"] = k_behav
    if el is None:
        row["behav_probes"] = np.nan
        P.append(f"no rating block for '{row['cond']}' in behav_per_block "
                 f"(ratings missing, MEG block itself not checked against ratings)")
    else:
        row["behav_probes"] = len(el)
        if np.isfinite(r_on) and len(probes):
            rel = probes - r_on
            lag = float(np.median([rel[np.argmin(np.abs(rel - e))] - e for e in el]))
            n_m = sum(np.min(np.abs(rel - (e + lag))) <= MATCH_TOL_S for e in el)
            row.update(probe_lag_s=round(lag, 2), probes_matched=int(n_m))
            if n_m < MIN_MATCH_FRAC * len(el):
                P.append(f"only {n_m}/{len(el)} rating probes match a probe trigger "
                         f"- ratings and MEG block do not belong together")
        if abs(len(el) - len(probes)) > 1:
            P.append(f"{len(probes)} probe triggers vs {len(el)} rating probes")
    return row


def safe_check(path):
    """check_file that never raises, so one bad file cannot stop the run."""
    try:
        return check_file(path)
    except Exception as e:
        return dict(file=os.path.basename(path), subject=os.path.basename(path).split("_")[0],
                    problems=[f"could not read: {type(e).__name__}: {e}"])


# ===================================================================
# MAIN
# ===================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("subjects", nargs="*", help="subject IDs, e.g. 000 010 (default: all)")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(CUT_DIR, "*_[0-9]b_*.fif"))
                   if not split_re.search(f) and "preproc" not in os.path.basename(f))
    if a.subjects:
        want = {s.zfill(3) for s in a.subjects}
        files = [f for f in files if os.path.basename(f).split("_")[0] in want]
    print(f"{len(files)} cut file(s) in {CUT_DIR}, {N_WORKERS} at a time\n", flush=True)

    rows, t_start = [], time.time()
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(safe_check, files), start=1):
            rows.append(r)
            print(f"  [{i}/{len(files)}  {time.time() - t_start:5.0f} s]  {r['file']}: "
                  f"{'ok' if not r['problems'] else '; '.join(r['problems'])}", flush=True)
    print()
    F = pd.DataFrame(rows).sort_values("file")
    F["ok"] = F["problems"].apply(len) == 0
    F["problems"] = F["problems"].apply("; ".join)

    # ---- per subject ----
    S = []
    for subj, g in F.groupby("subject", sort=True):
        P = [f"{r.file}: {r.problems}" for r in g.itertuples() if r.problems]
        conds = list(g["cond"])
        for cnd in ("base", "sad", "fear"):
            n = conds.count(cnd)
            if n != 1:
                P.append(f"{n} {cnd} file(s) (expected 1)")
        if "block_num" in g and "behav_block_num" in g:
            for r in g.itertuples():
                if r.cond in ("sad", "fear") and pd.notna(r.behav_block_num) \
                        and r.block_num != r.behav_block_num:
                    P.append(f"{r.cond}: MEG block {r.block_num}b but rating block {int(r.behav_block_num)}")
        S.append(dict(subject=subj, n_files=len(g), files=", ".join(g["file"]),
                      status="OK" if not P else "CHECK", problems=" | ".join(P)))
        print(f"[{subj}] {'OK   ' if not P else 'CHECK'} " + ", ".join(
            f"{r.cond} {r.rest_min if pd.notna(getattr(r, 'rest_min', np.nan)) else '?'} min "
            f"{int(r.probes_matched) if pd.notna(getattr(r, 'probes_matched', np.nan)) else '?'}"
            f"/{int(r.behav_probes) if pd.notna(getattr(r, 'behav_probes', np.nan)) else '?'} probes"
            for r in g.itertuples()))
        for p in P:
            print(f"        - {p}")
    S = pd.DataFrame(S)

    F.to_csv(os.path.join(CUT_DIR, "cut_check_files.csv"), index=False)
    S.to_csv(os.path.join(CUT_DIR, "cut_check_subjects.csv"), index=False)
    print(f"\n{(S.status == 'OK').sum()} OK, {(S.status != 'OK').sum()} to check "
          f"-> {CUT_DIR}/cut_check_subjects.csv (details: cut_check_files.csv)")


if __name__ == "__main__":
    main()
