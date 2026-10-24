# -*- coding: utf-8 -*-
"""
Behavioural VAS rest-period timeseries analysis (emo_study).

For each subject file (dynNNN_results.csv) this script:
  * splits the file into its rest blocks (base, then two emotion blocks),
    using the block definition below,
  * keeps ONLY within_rest rows (drops post_video) and ALL ratings,
    confirmed AND timeouts (project rule: timeouts are always used; the
    'timeout' column marks them),
  * labels the two emotion blocks 'sad' / 'fear' from the video triggers
    of the MEG trigger file (7 = sad, 8 = fear); odd/even subject number
    is only the fallback,
  * analyses the Sadness series and the Fear series SEPARATELY within
    every block (base + sad + fear) -> 6 series per subject,
  * runs a battery of complexity / frequency measures suited to short,
    irregularly-sampled series (N ~ 24),
  * writes one long-format CSV per subject, the per-block audit CSVs and
    one QC table for all subjects.

Plain Python, no SLURM. N_WORKERS subjects run at the same time (default 4).

Block definition (same rules as cut_dyn_blocks_python.py on the MEG side):
  1. A new rest run starts after every run of post_video rows, and also
     where elapsed_s jumps BACK within within_rest rows (elapsed_s is
     block-relative, so a jump back means the rest task was restarted
     without a video in between, e.g. a false start).
  2. A run is a complete block only if its last probe is at or after
     MIN_LAST_PROBE_S (a 20-min rest block has its last probe at ~19 min).
     Shorter runs are aborted / false-start runs and are dropped. This is
     the behavioural counterpart of the 18-22 min rest-window check on the
     triggers.
  3. Position = number of post_video runs before the run:
        0 -> block 1 = base, 1 -> block 2 = first video, 2 -> block 3 =
        second video. If two complete runs share a position, the one with
        more confirmed ratings is kept (later one on a tie) and this is
        flagged.
  4. Which video came first is read from the trigger file
     Behav/MEG_trigger_files/{NNN}_emo_rest_all_triggers.csv. A video
     trigger (7 = sad, 8 = fear) counts only if the next rest_onset (1) ->
     rest_offset (2) window after it lasts 18-22 min, so videos of aborted
     runs are ignored. Fallback when there is no usable trigger file:
     subject parity (odd -> block 2 = sad; root subject 000 and its repeat
     visits 200/300 count as sad-first, like 001). Both orders are stored
     in the QC table and a disagreement is flagged.

Design decisions for the measures (unchanged):
  - The VAS rest series are SHORT (~24 pts) and IRREGULARLY sampled in
    time (elapsed_s). Frequency analysis therefore uses Lomb-Scargle on
    the real timestamps, computed ONCE per series; peak/median freq and
    spectral entropy are all derived from that single periodogram.
  - No sliding-window spectrogram: 24 points cannot support it. We fall
    back to a single peak-frequency estimate over the whole series.
  - Measures with a hard minimum-N requirement (SampEn, MSE, PLE/slope)
    are gated: below their scientific floor they write NaN + a flag
    rather than a fake number.
  - Flat / near-flat series (>90% identical values, common for the
    off-target emotion under floor effects) get the literal string
    'flat' in the value cell.
  - Timeseries-valued outputs (periodogram, ACF) are JSON-serialised
    into a single cell.

Outputs (OUT_DIR = Behav/analysis):
  dyn{NNN}_vas_analysis.csv                    measures, long format
  behav_per_block/dyn{NNN}_block{k}_{role}_behav.csv
        all within_rest rows of the kept block; 'rating' = value as logged
        (timeouts included), 'timeout' = True for timeout rows
  behav_vas_qc.csv                             one row per subject

A subject is SKIPPED if its dyn{NNN}_vas_analysis.csv is newer than its
results file and trigger file. Use --force to redo (needed once after
this block-definition change).

Usage:
    python3 analyze_behav_vas.py                  # all subjects (ID < SUBJ_MAX)
    python3 analyze_behav_vas.py 014 019          # these subjects only
    python3 analyze_behav_vas.py --force          # redo everything
    python3 analyze_behav_vas.py --workers 2      # 2 subjects at a time
"""

import os

# ---- PARALLEL RUN ---------------------------------------------------
# Number of SUBJECTS analysed at the same time.
N_WORKERS = 4

# Split the CPU cores between the workers. Must be set BEFORE numpy is
# imported.
_threads = str(max(1, (os.cpu_count() or N_WORKERS) // N_WORKERS))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _threads)

import argparse
import glob
import json
import multiprocessing as mp
import re
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis

warnings.filterwarnings("ignore")

# ===================================================================
# CONFIG
# ===================================================================

BEHAV_DIR = "/home/ZhangJianfeng_04/emo_study/Behav/raw"
TRIG_DIR = "/home/ZhangJianfeng_04/emo_study/Behav/MEG_trigger_files"
OUT_DIR = "/home/ZhangJianfeng_04/emo_study/Behav/analysis"
PERBLOCK_DIR = os.path.join(OUT_DIR, "behav_per_block")
QC_FILE = os.path.join(OUT_DIR, "behav_vas_qc.csv")

SUBJ_MAX = 200         # analyse subjects with ID < SUBJ_MAX (repeat visits 2xx/3xx excluded)

# ---- sessions saved in several result files -------------------------
# The task was restarted under another ID; the later file is read directly
# after the subject's own file, as one session. The part IDs are never
# analysed as subjects of their own.
SESSION_PARTS = {"034": ["134"], "036": ["136"]}
PART_IDS = {p for parts in SESSION_PARTS.values() for p in parts}

# Condition of every post_video run of the merged session, in order, for
# sessions where the trigger file cannot give it (a video was repeated or a
# rest window has no offset). Checked against the MEG VAS-trigger timing:
#   034: fear #1, [crash - sad #1 not saved], fear #2 (repeat), sad #2
#   036: fear, sad (MEG recording of the sad rest ends after 13 min)
VIDEO_SEQUENCE_OVERRIDE = {"034": ["fear", "fear", "sad"],
                           "036": ["fear", "sad"]}

EMOTIONS = ["Sadness", "Fear"]

# ---- block definition ---------------------------------------------
MIN_LAST_PROBE_S = 15 * 60   # a complete 20-min rest run has its last
                             # probe at ~19 min; runs ending earlier are
                             # aborted / false starts
REST_DUR_MIN_S = 18 * 60     # trigger rest window (rest_onset -> offset)
REST_DUR_MAX_S = 22 * 60     # valid range, as in cut_dyn_blocks_python.py

# ---- training button press at the start of the session ---------------
# The first probe is dropped ONLY if both ratings are 50, both timed out and
# the remaining block-1 ratings are on average >= TRAINING_MIN_DROP lower
# (see training_pair_rows). Same rule in analyze_behav_vas.py and
# 1_build_vas_dataset.py.
TRAINING_VALUE = 50
TRAINING_MIN_DROP = 10

# ---- trigger codes --------------------------------------------------
REST_ONSET_CODE = 1
REST_OFFSET_CODE = 2
VIDEO_SAD_CODE = 7
VIDEO_FEAR_CODE = 8

# ---- minimum-N floors (scientific) --------------------------------
# Below these, the measure is NOT computed (NaN + flag).
MIN_N_GENERIC = 5      # anything at all
MIN_N_SAMPEN = 50      # SampEn is heavily biased for small N (Yentes 2013)
MIN_N_ACW = 10         # autocorrelation window needs a few lags
MIN_N_SLOPE = 12       # 1/f slope needs a usable frequency range
MIN_N_LZC = 8

# ---- flat detection -----------------------------------------------
FLAT_FRACTION = 0.90   # >90% identical -> 'flat'

# ---- SampEn params (kept conservative for short series) -----------
SAMPEN_M = 1           # embedding dim reduced from 3: 24 pts cannot
SAMPEN_R = 0.2         # support m=3. Even m=1 is gated by MIN_N_SAMPEN.


# ===================================================================
# UTILITIES
# ===================================================================

def is_flat(vals):
    """True if series is constant or near-constant (>FLAT_FRACTION identical)."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return True
    # most common value's share
    uniq, counts = np.unique(vals, return_counts=True)
    return (counts.max() / len(vals)) >= FLAT_FRACTION


def json_cell(obj):
    """Serialise a small array/dict into a single CSV cell."""
    def _clean(x):
        if isinstance(x, np.ndarray):
            return [None if (isinstance(v, float) and np.isnan(v)) else float(v)
                    for v in x]
        return x
    if isinstance(obj, dict):
        obj = {k: _clean(v) for k, v in obj.items()}
    else:
        obj = _clean(obj)
    return json.dumps(obj)


# ===================================================================
# TIME-DOMAIN MEASURES
# ===================================================================

def lempel_ziv_complexity(sequence):
    sub_strings = set()
    ind, inc = 0, 1
    while ind + inc <= len(sequence):
        sub = sequence[ind:ind + inc]
        if sub in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub)
            ind += inc
            inc = 1
    return len(sub_strings)


def lzc(vals):
    """Normalised LZC + raw count (raw is more honest at small N)."""
    bin_ts = (vals >= np.median(vals)).astype(int)
    bin_str = "".join(map(str, bin_ts))
    raw = lempel_ziv_complexity(bin_str)
    n = len(vals)
    norm = raw / (n / np.log2(n)) if n > 1 else np.nan
    return norm, raw


def sample_entropy(vals, m=1, r_factor=0.2):
    """Manual SampEn (no EntropyHub dependency). Returns NaN if undefined."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n < m + 2:
        return np.nan
    sd = np.std(vals)
    if sd < 1e-12:
        return np.nan
    x = (vals - np.mean(vals)) / sd
    r = r_factor * np.std(x)

    def _phi(mm):
        pats = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        cnt = 0
        for i in range(len(pats)):
            d = np.max(np.abs(pats - pats[i]), axis=1)
            cnt += np.sum(d <= r) - 1  # exclude self-match
        return cnt

    B = _phi(m)
    A = _phi(m + 1)
    if B <= 0 or A <= 0:
        return np.nan
    return -np.log(A / B)


def autocorr(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals - np.mean(vals)
    n = len(vals)
    ac = np.correlate(vals, vals, mode="full")[n - 1:]
    if ac[0] == 0:
        return np.full(n, np.nan)
    return ac / ac[0]


def acw(vals, threshold):
    """
    Autocorrelation window at a threshold, reported in LAGS (probe steps).

    NOTE: not converted to seconds. The rest probes are irregularly spaced
    in real time (~27-72 s apart), so there is no single sampling interval
    to multiply by; a 'seconds' value would be misleading. The unit here is
    'probe steps'. A lag of k means the autocorrelation first crosses the
    threshold k ratings apart in sequence order.
    """
    ac = autocorr(vals)
    if np.all(np.isnan(ac)):
        return np.nan, ac
    below = np.where(ac <= threshold)[0]
    idx = below[0] if len(below) else len(ac) - 1
    return float(idx), ac


# ===================================================================
# FREQUENCY DOMAIN (Lomb-Scargle, computed ONCE per series)
# ===================================================================

def lomb_scargle_spectrum(t, vals):
    """
    Compute a Lomb-Scargle periodogram on the REAL (irregular) timestamps.
    Frequency grid auto-bounded to the resolvable range:
        f_low  ~ 1 / total_span   (one cycle over the record)
        f_high ~ 1 / (2 * median_dt)  (pseudo-Nyquist for mean spacing)
    Returns freqs (Hz), power, or (None, None) if not computable.
    """
    t = np.asarray(t, dtype=float)
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n < MIN_N_GENERIC:
        return None, None
    if np.std(vals) < 1e-12:
        return None, None

    span = t.max() - t.min()
    if span <= 0:
        return None, None
    dt_med = np.median(np.diff(np.sort(t)))
    if dt_med <= 0:
        return None, None

    f_low = 1.0 / span
    f_high = 1.0 / (2.0 * dt_med)
    if f_high <= f_low:
        return None, None

    freqs = np.linspace(f_low, f_high, 200)
    ang = 2.0 * np.pi * freqs
    v = vals - np.mean(vals)
    try:
        power = lombscargle(t, v, ang, normalize=True)
    except Exception:
        return None, None
    return freqs, power


def freq_measures_from_spectrum(freqs, power):
    """Derive peak freq, median freq, spectral entropy from ONE periodogram."""
    out = {"PeakFreq": np.nan, "MedianFreq": np.nan, "SpectralEntropy": np.nan}
    if freqs is None or power is None or np.all(power == 0):
        return out
    # peak frequency
    out["PeakFreq"] = float(freqs[np.argmax(power)])
    # median frequency (half cumulative power)
    csum = np.cumsum(power)
    total = csum[-1]
    if total > 0:
        out["MedianFreq"] = float(freqs[np.argmax(csum >= total / 2)])
        # spectral entropy (normalised Shannon over the power distribution)
        p = power / total
        eps = 1e-12
        se = -np.sum(p * np.log(p + eps))
        out["SpectralEntropy"] = float(se / np.log(len(power)))
    return out


def spectral_slope(freqs, power):
    """1/f slope in log-log space (positive value). Gated by MIN_N_SLOPE upstream."""
    if freqs is None or power is None or len(freqs) < 3:
        return np.nan
    mask = power > 0
    if mask.sum() < 3:
        return np.nan
    slope, _ = np.polyfit(np.log10(freqs[mask]), np.log10(power[mask]), 1)
    return float(-slope)


# ===================================================================
# PER-SERIES ANALYSIS
# ===================================================================

def analyze_series(t, vals, meta):
    """
    Run the full battery on one (timestamps, values) series.
    Returns a list of long-format row dicts.
    `meta` supplies subject/block/emotion identifiers.
    """
    rows = []
    vals = np.asarray(vals, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(vals)

    def add(measure, value, flag=""):
        rows.append({**meta, "measure": measure, "value": value,
                     "flag": flag, "n_samples": n})

    # ---------- degenerate cases ----------
    if n < MIN_N_GENERIC:
        add("ALL", np.nan, flag=f"too_few_samples(n={n})")
        return rows

    flat = is_flat(vals)
    dt_med = np.median(np.diff(np.sort(t))) if n > 1 else np.nan

    # ---------- basic descriptors (always computable) ----------
    add("mean", float(np.mean(vals)))
    add("std", float(np.std(vals)))
    add("range", float(np.max(vals) - np.min(vals)))
    add("max", float(np.max(vals)))
    add("n_nonzero", int(np.sum(vals != 0)))
    add("frac_nonzero", float(np.mean(vals != 0)))
    if not flat:
        add("skewness", float(scipy_skew(vals)))
        add("kurtosis", float(scipy_kurtosis(vals)))  # excess kurtosis
    else:
        add("skewness", "flat")
        add("kurtosis", "flat")

    # ---------- if flat: short-circuit the shape/complexity measures ----------
    if flat:
        for m in ["LZC", "LZC_raw", "SampEn_m1", "SampEn_m2", "SampEn_m3",
                  "ACW0_lags", "ACW5_lags",
                  "PeakFreq", "MedianFreq", "SpectralEntropy", "SpectralSlope"]:
            add(m, "flat")
        add("periodogram", json_cell({"freqs": [], "power": []}), flag="flat")
        add("acf", json_cell([]), flag="flat")
        return rows

    # ---------- LZC ----------
    if n >= MIN_N_LZC:
        norm, raw = lzc(vals)
        add("LZC", norm)
        add("LZC_raw", int(raw))
    else:
        add("LZC", np.nan, flag=f"too_few_for_lzc(<{MIN_N_LZC})")
        add("LZC_raw", np.nan, flag=f"too_few_for_lzc(<{MIN_N_LZC})")

    # ---------- SampEn at m=1,2,3 (low-confidence at N~24) ----------
    # N is far below the reliable floor for SampEn (Yentes et al. 2013),
    # so every value is flagged low_confidence. Where no template of length
    # m+1 matches (A=0), the estimator is undefined -> NaN with a reason.
    lowconf = f"low_confidence_smallN(n={n})"
    for mm in (1, 2, 3):
        if n < mm + 2:
            add(f"SampEn_m{mm}", np.nan, flag=f"too_few_for_m{mm}(need>={mm+2})")
            continue
        val = sample_entropy(vals, m=mm, r_factor=SAMPEN_R)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            add(f"SampEn_m{mm}", np.nan,
                flag=f"undefined_no_matches;{lowconf}")
        else:
            add(f"SampEn_m{mm}", val, flag=lowconf)

    # ---------- ACW (reported in LAGS / probe steps, not seconds) ----------
    if n >= MIN_N_ACW:
        acw0, ac = acw(vals, threshold=0.0)
        acw5, _ = acw(vals, threshold=0.5)
        add("ACW0_lags", acw0)
        add("ACW5_lags", acw5)
        add("acf", json_cell(ac))
    else:
        add("ACW0_lags", np.nan, flag=f"too_few_for_acw(<{MIN_N_ACW})")
        add("ACW5_lags", np.nan, flag=f"too_few_for_acw(<{MIN_N_ACW})")
        add("acf", json_cell([]), flag=f"too_few_for_acw(<{MIN_N_ACW})")

    # ---------- FREQUENCY: compute the transform ONCE ----------
    freqs, power = lomb_scargle_spectrum(t, vals)
    fm = freq_measures_from_spectrum(freqs, power)
    add("PeakFreq", fm["PeakFreq"])
    add("MedianFreq", fm["MedianFreq"])
    add("SpectralEntropy", fm["SpectralEntropy"])

    # spectral slope only if enough spectral range
    if n >= MIN_N_SLOPE:
        add("SpectralSlope", spectral_slope(freqs, power))
    else:
        add("SpectralSlope", np.nan, flag=f"too_few_for_slope(<{MIN_N_SLOPE})")

    # store the periodogram (single cell)
    if freqs is not None:
        add("periodogram", json_cell({"freqs": freqs, "power": power}))
    else:
        add("periodogram", json_cell({"freqs": [], "power": []}),
            flag="spectrum_uncomputable")

    return rows



# ===================================================================
# BLOCK DEFINITION + CONDITION LABELS
# ===================================================================

def parity_order(subject_id):
    """Fallback video order from the subject number. Root subject 000 (and
    its repeat visits 200/300) counts as sad-first, like 001."""
    root = int(subject_id) % 100
    sad_first = (root % 2 == 1) or root == 0
    return ["sad", "fear"] if sad_first else ["fear", "sad"]


def trigger_video_order(subject_id):
    """Video order from the MEG trigger file. A video trigger counts only
    if the next rest_onset -> rest_offset window after it lasts
    REST_DUR_MIN_S..REST_DUR_MAX_S (videos of aborted runs are ignored).
    Returns (order or None, note)."""
    path = os.path.join(TRIG_DIR, f"{subject_id}_emo_rest_all_triggers.csv")
    if not os.path.exists(path):
        return None, "no trigger file"
    t = pd.read_csv(path)
    if "trigger_code" not in t.columns or "timestamp_s" not in t.columns:
        return None, f"trigger columns missing ({list(t.columns)})"
    codes = t["trigger_code"].to_numpy(dtype=int)
    times = t["timestamp_s"].to_numpy(dtype=float)
    idx = np.arange(len(codes))

    seq, dropped = [], []
    for i in np.flatnonzero(np.isin(codes, [VIDEO_SAD_CODE, VIDEO_FEAR_CODE])):
        label = "sad" if codes[i] == VIDEO_SAD_CODE else "fear"
        on = np.flatnonzero((codes == REST_ONSET_CODE) & (idx > i))
        off = (np.flatnonzero((codes == REST_OFFSET_CODE) & (idx > on[0]))
               if len(on) else [])
        if len(on) == 0 or len(off) == 0:
            dropped.append(f"{label}@{times[i]:.0f}s: no rest window after")
            continue
        dur = times[off[0]] - times[on[0]]
        if not (REST_DUR_MIN_S <= dur <= REST_DUR_MAX_S):
            dropped.append(f"{label}@{times[i]:.0f}s: rest {dur / 60:.1f} min")
            continue
        if not seq or seq[-1] != label:
            seq.append(label)

    note = "; ".join(f"video ignored ({d})" for d in dropped)
    if len(seq) == 2 and set(seq) == {"sad", "fear"}:
        return seq, note
    return None, "; ".join(x for x in (note, f"video sequence {seq} not usable") if x)


def split_rest_runs(df):
    """Assign every within_rest row to a rest run. A new run starts after a
    run of post_video rows, or where elapsed_s jumps back (task restarted
    without a video). Returns a per-row run id (-1 for non-rest rows) and
    the number of post_video runs before each row."""
    ctx = df["context"].astype(str).str.strip().str.lower().to_numpy()
    el = pd.to_numeric(df["elapsed_s"], errors="coerce").to_numpy(dtype=float)
    run_id = np.full(len(df), -1, dtype=int)
    n_video = np.zeros(len(df), dtype=int)
    cur, videos, prev_post, last_t = -1, 0, False, np.nan
    for i in range(len(df)):
        is_post = ctx[i] == "post_video"
        if is_post and not prev_post:
            videos += 1
        n_video[i] = videos
        if ctx[i] == "within_rest":
            if cur < 0 or prev_post or (np.isfinite(el[i]) and np.isfinite(last_t)
                                        and el[i] < last_t):
                cur += 1
                last_t = np.nan
            run_id[i] = cur
            if np.isfinite(el[i]):
                last_t = el[i]
        prev_post = is_post
    return run_id, n_video


def define_blocks(df, subject_id, qc, log):
    """Return {block_num: DataFrame of that block's within_rest rows} with
    block_role filled in, following the block definition in the header."""
    run_id, n_video = split_rest_runs(df)
    df = df.assign(_run=run_id, _pos=n_video)
    rest = df[df["_run"] >= 0]

    runs = []
    for r, g in rest.groupby("_run"):
        last = pd.to_numeric(g["elapsed_s"], errors="coerce").max()
        runs.append(dict(run=r, pos=int(g["_pos"].iloc[0]), n_rows=len(g),
                         n_valid=int(g["valid"].sum()), last_s=last,
                         complete=bool(np.isfinite(last) and last >= MIN_LAST_PROBE_S)))
    qc["n_rest_runs"] = len(runs)
    flags = []
    for r in runs:
        if not r["complete"]:
            flags.append(f"run{r['run'] + 1} dropped (last probe {r['last_s']:.0f}s, "
                         f"{r['n_rows']} rows)")

    # ---- video order: triggers first, parity as fallback ----
    order_par = parity_order(subject_id)
    order_trg, tnote = trigger_video_order(subject_id)
    order = order_trg or order_par
    qc.update(cond_source="triggers" if order_trg else "parity",
              order_triggers="-".join(order_trg) if order_trg else "",
              order_parity="-".join(order_par),
              order_conflict=bool(order_trg and order_trg != order_par),
              trigger_note=tnote)
    if qc["order_conflict"]:
        log.append(f"  !! video order: triggers {qc['order_triggers']} vs "
                   f"parity {qc['order_parity']} -> using triggers")
    elif not order_trg and subject_id not in VIDEO_SEQUENCE_OVERRIDE:
        log.append(f"  ! video order from parity ({tnote})")
    role_of = {0: "base", 1: order[0], 2: order[1]}
    if subject_id in VIDEO_SEQUENCE_OVERRIDE:
        seq = VIDEO_SEQUENCE_OVERRIDE[subject_id]
        role_of = {0: "base", **{k + 1: c for k, c in enumerate(seq)}}
        qc["cond_source"] = "override"
        log.append(f"  video sequence from VIDEO_SEQUENCE_OVERRIDE: {'-'.join(seq)}")

    # first complete block of each condition is kept, later repeats are
    # dropped (same rule as cut_dyn_blocks_python.py)
    blocks, used = {}, set()
    for pos in sorted({r["pos"] for r in runs}):
        cand = [r for r in runs if r["pos"] == pos and r["complete"]]
        if not cand:
            continue
        if pos not in role_of:
            flags.append(f"{len(cand)} complete run(s) after video {pos} dropped "
                         f"(only 2 videos expected)")
            continue
        role = role_of[pos]
        if role in used:
            flags.append(f"block after video {pos} ({role}) dropped: repeat of {role}")
            continue
        num = 1 if role == "base" else 2 + sum(r != "base" for r in used)
        used.add(role)
        keep = max(cand, key=lambda r: (r["n_valid"], r["run"]))
        if len(cand) > 1:
            flags.append(f"{len(cand)} complete runs at block {num}, "
                         f"kept run{keep['run'] + 1} ({keep['n_valid']} ratings)")
        b = df[df["_run"] == keep["run"]].drop(columns=["_run", "_pos"]).copy()
        b.insert(0, "block_num", num)
        b.insert(1, "block_role", role)
        b.insert(2, "cond_source", qc["cond_source"] if pos else "")
        blocks[num] = b.reset_index(drop=True)

    if len(blocks) != 3:
        flags.append(f"{len(blocks)} block(s) found (expected 3)")
    qc["blocks_found"] = ",".join(f"{k}:{blocks[k]['block_role'].iloc[0]}"
                                  for k in sorted(blocks))
    qc["flags"] = "; ".join(flags)
    for f in flags:
        log.append(f"  ! {f}")
    return blocks


def training_pair_rows(df):
    """Index of the 2 rows of the session's FIRST probe if it is the training
    button press, else []. Training press = both ratings are TRAINING_VALUE
    (the start value), both timed out, and the other block-1 ratings are on
    average at least TRAINING_MIN_DROP lower. Only then is it dropped - every
    other timeout is a rating and is used. The probe keeps its index, so the
    neural snippet of that probe has no rating and is skipped as well."""
    ctx = df["context"].astype(str).str.strip().str.lower()
    el = pd.to_numeric(df["elapsed_s"], errors="coerce")
    r = pd.to_numeric(df["rating"], errors="coerce")
    st = (df["status"].astype(str).str.strip().str.lower()
          if "status" in df.columns else pd.Series("", index=df.index))
    post = ctx.eq("post_video").to_numpy()
    end = int(post.argmax()) if post.any() else len(df)       # block 1 = before 1st video
    b1 = df.index[:end][ctx.iloc[:end].eq("within_rest").to_numpy()]
    if not len(b1):
        return []
    pair = b1[el[b1] == el[b1[0]]]
    others = b1.difference(pair)
    if (len(pair) == 2 and (r[pair] == TRAINING_VALUE).all() and st[pair].eq("timeout").all()
            and len(others) and r[others].mean() <= TRAINING_VALUE - TRAINING_MIN_DROP):
        return list(pair)
    return []


def load_results(subject_id):
    """dyn{subject}_results.csv, followed by its SESSION_PARTS files."""
    parts = [subject_id] + SESSION_PARTS.get(subject_id, [])
    df = pd.concat([pd.read_csv(os.path.join(BEHAV_DIR, f"dyn{p}_results.csv"))
                    for p in parts], ignore_index=True)
    df.columns = df.columns.str.strip().str.lower()
    for col in ("context", "elapsed_s", "emotion", "rating"):
        if col not in df.columns:
            raise KeyError(f"column '{col}' missing; has {list(df.columns)}")
    df["emotion"] = df["emotion"].astype(str).str.strip().str.capitalize()
    df["rating_raw"] = pd.to_numeric(df["rating"], errors="coerce")
    status = (df["status"].astype(str).str.strip().str.lower()
              if "status" in df.columns else pd.Series("confirmed", index=df.index))
    df["timeout"] = status.eq("timeout")
    # project rule: timeouts are ALWAYS used - every logged rating is valid
    df["valid"] = df["rating_raw"].notna()
    df["rating"] = df["rating_raw"]
    return df


# ===================================================================
# MAIN PER-SUBJECT DRIVER
# ===================================================================

def write_perblock(subject_id, blocks, log):
    """Write the per-block audit CSVs. Old files of this script for this
    subject are removed first, so a changed block label cannot leave a
    stale file next to the new one. Other files of the subject are only
    reported."""
    os.makedirs(PERBLOCK_DIR, exist_ok=True)
    own = re.compile(rf"^dyn{subject_id}_block\d+_[a-z]+_behav\.csv$")
    for f in glob.glob(os.path.join(PERBLOCK_DIR, f"dyn{subject_id}_*.csv")):
        name = os.path.basename(f)
        if own.match(name):
            os.remove(f)
        else:
            log.append(f"  ! behav_per_block/{name} was not written by this script "
                       f"- remove it by hand, or 5_vas_fluctuation_proof.py "
                       f"excludes that block as a duplicate")
    for k, b in blocks.items():
        b.to_csv(os.path.join(PERBLOCK_DIR,
                              f"dyn{subject_id}_block{k}_{b['block_role'].iloc[0]}_behav.csv"),
                 index=False)


def process_subject(subject_id):
    """One subject, inside one worker. Returns (subject_id, qc row, log lines);
    never raises, so one bad file cannot stop the run."""
    log = [f"[{subject_id}]"]
    qc = dict(subject=subject_id, status="")
    try:
        df = load_results(subject_id)
        tr = training_pair_rows(df)
        qc["training_press_dropped"] = bool(tr)
        if tr:
            log.append("  first probe = training press (50/50, timeout) -> dropped")
            df = df.drop(index=tr).reset_index(drop=True)
        if "subject_nr" in df.columns and int(df["subject_nr"].iloc[0]) != int(subject_id):
            log.append(f"  ! subject_nr in file = {int(df['subject_nr'].iloc[0])} "
                       f"(filename is used)")
        blocks = define_blocks(df, subject_id, qc, log)
        write_perblock(subject_id, blocks, log)

        all_rows = []
        for k, bdf in blocks.items():
            role = bdf["block_role"].iloc[0]
            for emo in EMOTIONS:
                sub = bdf[(bdf["emotion"] == emo) & bdf["valid"]]
                t = pd.to_numeric(sub["elapsed_s"], errors="coerce").to_numpy(dtype=float)
                vals = sub["rating"].to_numpy(dtype=float)
                good = ~np.isnan(t)
                t, vals = t[good], vals[good]
                n_all = int((bdf["emotion"] == emo).sum())
                qc[f"n_{role}_{emo}"] = len(vals)
                qc[f"n_timeout_{role}_{emo}"] = int(((bdf["emotion"] == emo)
                                                     & bdf["timeout"]).sum())
                qc[f"n_missing_{role}_{emo}"] = n_all - len(vals)
                if len(vals) == 0:
                    continue
                meta = {"subject": subject_id,
                        "subject_nr": int(subject_id),
                        "block_role": role,
                        "block_num": k,
                        "emotion": emo}
                all_rows.extend(analyze_series(t, vals, meta))

        out = pd.DataFrame(all_rows)
        out.to_csv(os.path.join(OUT_DIR, f"dyn{subject_id}_vas_analysis.csv"), index=False)
        counts = "  ".join(f"{b['block_role'].iloc[0]} {int(b['valid'].sum())}/{len(b)}"
                           for _, b in sorted(blocks.items()))
        log.append(f"  blocks {qc['blocks_found'] or '-'}  (order: {qc['cond_source']})")
        log.append(f"  ratings used (incl. timeouts)/rows: {counts}")
        qc["status"] = "ok"
    except Exception as e:
        log.append("  X FAILED: " + "".join(traceback.format_exception_only(type(e), e)).strip())
        qc["status"] = f"error: {type(e).__name__}: {str(e)[:150]}"
    return subject_id, qc, log


def is_done(subject_id):
    """Output exists and is newer than every input of this subject."""
    out = os.path.join(OUT_DIR, f"dyn{subject_id}_vas_analysis.csv")
    if not os.path.exists(out):
        return False
    inputs = [os.path.join(BEHAV_DIR, f"dyn{p}_results.csv")
              for p in [subject_id] + SESSION_PARTS.get(subject_id, [])] + [
              os.path.join(TRIG_DIR, f"{subject_id}_emo_rest_all_triggers.csv")]
    newest = max(os.path.getmtime(p) for p in inputs if os.path.exists(p))
    return os.path.getmtime(out) >= newest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("subjects", nargs="*", help="subject IDs, e.g. 014 019 (default: all)")
    ap.add_argument("--force", action="store_true", help="redo subjects already done")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    a = ap.parse_args()

    found = sorted(m.group(1) for p in glob.glob(os.path.join(BEHAV_DIR, "dyn*_results.csv"))
                   if (m := re.match(r"dyn(\d+)_results\.csv$", os.path.basename(p)))
                   and m.group(1) not in PART_IDS)
    if a.subjects:
        want = {s.zfill(3) for s in a.subjects}
        missing = sorted(want - set(found))
        if missing:
            print(f"No results file for: {', '.join(missing)}")
        subs = [s for s in found if s in want]
    else:
        subs = [s for s in found if int(s) < SUBJ_MAX]

    todo = [s for s in subs if a.force or not is_done(s)]
    skipped = [s for s in subs if s not in todo]
    if skipped:
        print(f"SKIP (up to date, use --force to redo): {', '.join(skipped)}")
    if not todo:
        print("Nothing to do.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    n_workers = max(1, min(a.workers, len(todo)))
    print(f"Analysing {len(todo)} subject(s), {n_workers} at a time.\n")
    rows = []
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        for subject_id, qc, log in pool.imap_unordered(process_subject, todo):
            print("\n".join(log), flush=True)
            rows.append(qc)

    # merge into the QC table, so subjects skipped in this run keep their row
    new = pd.DataFrame(rows)
    new["subject"] = new["subject"].astype(str)
    if os.path.exists(QC_FILE):
        old = pd.read_csv(QC_FILE, dtype={"subject": str})
        new = pd.concat([old[~old["subject"].isin(new["subject"])], new], ignore_index=True)
    new.sort_values("subject").to_csv(QC_FILE, index=False)

    this = pd.DataFrame(rows)
    n_ok = int((this["status"] == "ok").sum())
    n_flag = int(this.get("flags", pd.Series(dtype=str)).fillna("").astype(bool).sum())
    n_par = int((this.get("cond_source", pd.Series(dtype=str)) == "parity").sum())
    print(f"\n{'=' * 60}")
    print(f"Done - {n_ok} ok, {len(this) - n_ok} failed, {len(skipped)} skipped | "
          f"{n_flag} with block flags, {n_par} with video order from parity")
    print(f"QC table      : {QC_FILE}")
    print(f"Per-block CSVs: {PERBLOCK_DIR}")


if __name__ == "__main__":
    main()
