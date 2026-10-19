# -*- coding: utf-8 -*-
"""
Behavioural VAS rest-period timeseries analysis (emo_study).

For each subject file (dynNNN_results.csv) this script:
  * keeps ONLY within_rest rows (drops post_video),
  * splits the file into its 3 blocks (base, then two emotion blocks),
  * resolves which emotion block is 'sad' vs 'fear' via odd/even
    counterbalancing of the subject number,
  * analyses the Sadness series and the Fear series SEPARATELY within
    every block (base + sad + fear) -> 6 series per subject,
  * runs a battery of complexity / frequency measures suited to short,
    irregularly-sampled series (N ~ 24),
  * writes one long-format CSV per subject.

Design decisions (see accompanying discussion):
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

Counterbalancing (video order):
    Odd  subjects -> Block 2 = sad,  Block 3 = fear
    Even subjects -> Block 2 = fear, Block 3 = sad
(Block 1 = base, always first, no counterbalancing.)

Usage:
    python3 analyze_behav_vas.py <subject_id>
    python3 analyze_behav_vas.py 019
"""

import os
import sys
import json
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
OUT_DIR = "/home/ZhangJianfeng_04/emo_study/Behav/analysis"

EMOTIONS = ["Sadness", "Fear"]

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
# BLOCK SPLITTING + COUNTERBALANCING
# ===================================================================

def label_and_split_blocks(df, subject_num):
    """
    Assign a block index + role label to EVERY raw row (including post_video)
    BEFORE any rows are dropped, then return:
      - labelled_full : the full raw df with 'block_num' and 'block_role'
                        columns added (post_video rows still present),
      - rest_blocks   : list of per-block DataFrames containing only the
                        within_rest rows, in file order.

    Block boundaries are the post_video markers: each post_video row (or run
    of them) closes the current block; the next within_rest run is a new
    block. Block 0 = base (always first, no video). Blocks 1,2 = the two
    emotion videos, resolved by odd/even counterbalancing.
    """
    d = df.reset_index(drop=True).copy()
    ctx = d["context"].astype(str).values

    # Walk rows; a block starts at the first within_rest after a post_video
    # (or at the very start). post_video rows belong to the block they close.
    block_idx = np.full(len(d), -1, dtype=int)
    cur = 0
    prev_was_post = False
    seen_rest_in_block = False
    for i in range(len(d)):
        is_rest = ctx[i] == "within_rest"
        is_post = ctx[i] == "post_video"
        if is_rest and prev_was_post:
            # transition from a post_video run into new rest -> new block
            cur += 1
            seen_rest_in_block = True
        elif is_rest:
            seen_rest_in_block = True
        block_idx[i] = cur
        prev_was_post = is_post

    d["block_num"] = block_idx + 1  # 1-based

    n_blocks = int(block_idx.max()) + 1 if len(d) else 0
    roles = block_role_map(subject_num, n_blocks)
    d["block_role"] = [roles.get(b, f"block{b+1}") for b in block_idx]

    rest_blocks = []
    for b in range(n_blocks):
        rb = d[(d["block_num"] == b + 1) & (d["context"] == "within_rest")]
        rest_blocks.append(rb.reset_index(drop=True))
    return d, rest_blocks


def block_role_map(subject_num, n_blocks):
    """
    Map block index -> role label.
    Block 0 = base always. Blocks 1,2 depend on odd/even counterbalancing.
        Odd  -> block index1 = sad, index2 = fear
        Even -> block index1 = fear, index2 = sad
    """
    roles = {}
    if n_blocks >= 1:
        roles[0] = "base"
    odd = (subject_num % 2) == 1
    if n_blocks >= 2:
        roles[1] = "sad" if odd else "fear"
    if n_blocks >= 3:
        roles[2] = "fear" if odd else "sad"
    return roles


# ===================================================================
# MAIN PER-SUBJECT DRIVER
# ===================================================================

def process_subject(subject_id):
    fname = f"dyn{subject_id}_results.csv"
    fpath = os.path.join(BEHAV_DIR, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Behav file not found: {fpath}")

    df = pd.read_csv(fpath)
    subject_num = int(df["subject_nr"].iloc[0])

    # Label EVERY row (incl post_video) with block_num + block_role BEFORE
    # dropping anything, then get the within_rest-only per-block frames.
    labelled_full, rest_blocks = label_and_split_blocks(df, subject_num)
    if len(rest_blocks) < 3:
        print(f"WARNING subject {subject_id}: found {len(rest_blocks)} blocks "
              f"(expected 3). Proceeding with what is present.")

    # ---- save the actual behavioural data used, one CSV per block ----
    # This is the audit trail: exactly the within_rest ratings that feed
    # each analysis, per block, with the emotion-block label attached.
    behav_out_dir = os.path.join(OUT_DIR, "behav_per_block")
    os.makedirs(behav_out_dir, exist_ok=True)
    for bidx, rb in enumerate(rest_blocks):
        if len(rb) == 0:
            continue
        role = rb["block_role"].iloc[0]
        bpath = os.path.join(
            behav_out_dir,
            f"dyn{subject_id}_block{bidx+1}_{role}_behav.csv")
        rb.to_csv(bpath, index=False)

    all_rows = []
    for bidx, bdf in enumerate(rest_blocks):
        if len(bdf) == 0:
            continue
        role = bdf["block_role"].iloc[0]
        for emo in EMOTIONS:
            sub = bdf[bdf["emotion"] == emo]
            if len(sub) == 0:
                continue
            t = sub["elapsed_s"].to_numpy(dtype=float)
            vals = sub["rating"].to_numpy(dtype=float)
            good = ~np.isnan(t)
            t, vals = t[good], vals[good]
            meta = {
                "subject": subject_id,
                "subject_nr": subject_num,
                "block_role": role,
                "block_num": bidx + 1,
                "emotion": emo,
            }
            all_rows.extend(analyze_series(t, vals, meta))

    out = pd.DataFrame(all_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"dyn{subject_id}_vas_analysis.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] subject {subject_id}: {len(out)} rows -> {out_path}")
    print(f"     per-block behav CSVs -> {behav_out_dir}")
    return out_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_behav_vas.py <subject_id>  (e.g. 019)")
        sys.exit(1)
    subject_id = sys.argv[1].strip()
    process_subject(subject_id)


if __name__ == "__main__":
    main()
