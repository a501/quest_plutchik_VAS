# -*- coding: utf-8 -*-
"""
extract_plutchik_dynamics.py
=============================
Concatenates the raw Plutchik EMA CSVs, reconstructs the 3x-daily
checkpoint structure, flags/interpolates genuinely missing checkpoints,
and (once wired in) runs the SAME dynamic-measure battery used in
analyze_behav_rt.py / analyze_behav_vas.py on each subject x emotion
timeseries.

The dynamic-measure battery itself (analyze_series / is_flat / autocorr /
lomb_scargle_spectrum) is IMPORTED from analyze_behav_vas.py, unmodified,
so the numbers are identical in method to the rest of the study (VAS +
RT). The scalar reductions of the JSON-blob outputs (ACF / periodogram)
follow the same approach analyze_behav_rt.py added for the same reason:
those curves can't be tested or plotted at group level as-is.

ASSUMPTION - RESOLVED: earlier drafts guessed nominal clock times for
the 3 daily checkpoints. The real export instead has an explicit
check-in-number field (df['slot'], parsed from the "which check-in
today" column - A.1/B.2/C.3), so slot identity comes directly from the
survey, not from rank order or a guess. The Lomb-Scargle t-axis for
interpolated (missing) points uses data-driven nominal hours - the
median REAL check-in time per slot, computed once from the actual
timestamps in load_and_concat() - not a hardcoded guess.

Key design decisions (confirmed with the user):
  - An EMPTY CELL in an existing row means the emotion was rated 0.
    This is applied at load time (fillna(0)), exactly as in
    plot_participant_feedback.py / 3month_4stPlots.py, and is NEVER
    touched by interpolation.
  - A GENUINELY MISSING checkpoint (no row at all for one of the
    3-a-day slots) is what gets interpolated:
      * run of 1 missing slot   -> linear interpolation
      * run of 2-3 missing slots -> cubic spline IF there is enough
        surrounding valid data for a stable fit, else left as NaN
      * run of >3 missing slots (e.g. a whole day+ gone) -> left as
        NaN / excluded, never fabricated
  - Rationale for interpolating (unlike analyze_behav_rt.py, which
    drops missing probes and keeps only real timestamps): the RT
    battery's autocorr() operates on the value array by INDEX, not
    time (see derived_rows() in analyze_behav_rt.py - it calls
    autocorr(v), no t argument). For RT that's fine because gaps
    within a ~20 min block are small and rare. For 3x-daily EMA data,
    leaving slots out entirely would make "lag 1" mean anything from
    a few hours to several days depending on missingness, corrupting
    the ACF-based (inertia) measures. The Lomb-Scargle half of the
    battery IS time-aware (lomb_scargle_spectrum(t, v)) and would be
    unaffected either way.

Usage:
    python3 extract_plutchik_dynamics.py
"""

import os
import re
import sys
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# numpy renamed trapz -> trapezoid in 2.0; support both so the scripts
# run under the cluster's older numpy as well as a current one.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# reuse the battery from the VAS script - single source of truth,
# same import pattern as analyze_behav_rt.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_behav_vas import analyze_series, is_flat, autocorr  # noqa: E402

# nominal clock times for the 3 daily checkpoints, used ONLY to build a
# real time axis (seconds) for the Lomb-Scargle half of the battery -
# see ASSUMPTION note above.
# nominal clock times for the 3 daily checkpoints, used ONLY as a
# fallback if load_and_concat cannot derive them from the real data
# (e.g. a participant with zero valid slots). Actual runs use the
# data-driven medians computed in load_and_concat instead.
NOMINAL_SLOT_HOURS_FALLBACK = {1: 9.0, 2: 16.0, 3: 24.3}

# ===================================================================
# CONFIG
# ===================================================================

RAW_DIR = Path("/home/ZhangJianfeng_04/emo_study/Behav/Pluchik")
OUT_DIR = Path("/home/ZhangJianfeng_04/emo_study/Behav/Pluchik/analysis")

SLEEP_HOUR = 4              # entries before 4am count as the previous waking day
                             # (same rule as plot_participant_feedback.py)
N_SLOTS_PER_DAY = 3

# interpolation gap policy
MAX_GAP_LINEAR = 1          # exactly 1 missing slot -> linear
MAX_GAP_CUBIC = 3           # 2-3 missing slots -> cubic, IF enough context
MIN_VALID_FOR_CUBIC = 8     # need at least this many valid points in the
                             # whole series before attempting a cubic fit
                             # (below this a cubic spline is not meaningfully
                             # better-determined than linear, and risks
                             # overshoot on a short series)

RATING_MIN, RATING_MAX = 0.0, 100.0   # Plutchik VAS bounds - interpolated
                                       # (cubic in particular) values are
                                       # clipped into this range; confirmed
                                       # necessary by testing (see below)

# ---------------------------------------------------------------
# MEASURE SELECTION
# ---------------------------------------------------------------
# Dropped at the user's request, plus the raw/length-dependent ones
# that are not comparable across subjects with different series
# lengths (see LENGTH CONTROL below):
#   skewness, kurtosis  - requested drop
#   n_nonzero           - a raw COUNT; scales directly with series
#                         length. frac_nonzero is the length-invariant
#                         version and is kept.
#   LZC_raw             - the un-normalised complexity count; LZC is
#                         the length-normalised version and is kept.
EXCLUDE_MEASURES = {"skewness", "kurtosis", "n_nonzero", "LZC_raw"}

# ---------------------------------------------------------------
# LENGTH CONTROL
# ---------------------------------------------------------------
# Participants contributed series of different lengths (different
# enrolment durations and different amounts of missingness). Several
# measures in the battery are length-dependent even after
# normalisation:
#   - SampEn is biased at small N (Yentes 2013)
#   - ACW0 / ACW5 are bounded above by the series length
#   - Lomb-Scargle frequency resolution scales with the observed span
#   - LZC converges only slowly with N
# so a between-subject contrast could reflect how long someone stayed
# in the study rather than their emotion dynamics.
#
# Handling, both emitted so they can be compared:
#   "full"     - every valid point the participant has
#   "common_n" - every participant truncated to their FIRST n_common
#                valid points, where n_common = the minimum across
#                participants (per emotion). This is the length-
#                controlled analysis and is the one to report.
# check_length_confound() additionally reports, per measure, the
# correlation between the measure and series length in "full" mode,
# so the size of the problem is visible rather than assumed.
# Default is FULL: each subject's own complete series. common_n truncates
# every subject to the shortest one, which with rolling enrolment (series
# length reflects time enrolled, not compliance) discards most of the data
# from earlier subjects. Set to ("full", "common_n") to emit both and
# compare.
LENGTH_MODES = ("full",)

# The mode used for the figures, the paired t-tests, and (by default) the
# downstream correlation scripts. Everything keys off this one setting.
PRIMARY_LENGTH_MODE = LENGTH_MODES[0]

# Plutchik's 8 primary emotions. Pairwise paired t-tests are run on
# these only: all 52 emotions would be 1326 pairs per measure, which is
# unreadable and (with n=7) uninterpretable. The violin panels still
# show all 52 emotions.
PRIMARY_EMOTIONS = ["joy", "trust", "fear", "sadness",
                    "disgust", "anger", "anticipation", "surprise"]


# ===================================================================
# SUBJECT EDITS  (data exclusions, applied to the Plutchik EMA only)
# ===================================================================
# Recorded here rather than done by hand, so the exclusions travel with
# the code and appear in the run log. Each edit is reported at runtime
# with how many check-ins it removed and the resulting date range.
#
# These affect the PLUTCHIK EMA ONLY. MEG, VAS and questionnaire data
# for the same subjects are untouched.
SUBJECT_EDITS = {
    # subject 002: only 6 days enrolled, 33% observed - too sparse for
    # any dynamic measure. Excluded from the EMA entirely.
    2:  {"exclude": True,
         "reason": "too little EMA data (6 days, 33% observed)"},

    # subject 021: an isolated first rating, then a ~20-day break, then a
    # continuous run. Dropping the isolated start removes the 59-slot gap
    # altogether; the remaining series may still contain 1-2 missing
    # check-ins, which interpolate normally.
    21: {"start_on": "2026-07-17",
         "reason": "isolated first rating before a long break; keep the "
                   "continuous run from 17 July"},

    # subject 024: early fragment discarded.
    24: {"start_on": "2026-07-06",
         "reason": "discard the fragment before 6 July"},
}


def apply_subject_edits(df):
    """Apply SUBJECT_EDITS, reporting exactly what each one removed."""
    if not SUBJECT_EDITS:
        return df
    print("\nApplying subject edits (Plutchik EMA only):")
    for pid, ed in sorted(SUBJECT_EDITS.items()):
        m = df["participant"] == pid
        if not m.any():
            print(f"  P{pid:03d}: not present in the data - edit skipped")
            continue
        n0 = int(m.sum())
        if ed.get("exclude"):
            df = df[~m]
            print(f"  P{pid:03d}: EXCLUDED, {n0} submission(s) removed "
                  f"({ed['reason']})")
            continue
        if ed.get("start_on"):
            cut = pd.to_datetime(ed["start_on"]).date()
            drop = m & (df["sleep_day"] < cut)
            df = df[~drop]
            rem = df[df["participant"] == pid]
            print(f"  P{pid:03d}: dropped {int(drop.sum())} submission(s) "
                  f"before {cut}; {len(rem)} remain, now "
                  f"{min(rem['sleep_day'])} to {max(rem['sleep_day'])} "
                  f"({ed['reason']})")
    return df.reset_index(drop=True)





# ===================================================================
# COLUMN / DAY STRUCTURE  (reused from 3month_4stPlots.py load_data())
# ===================================================================

EMOTION_ZH_EN = {
    "joy": "\u611f\u60a6", "trust": "\u4fe1\u4efb", "fear": "\u6050\u60e7",
    "sadness": "\u60b2\u4f24", "disgust": "\u538c\u6076", "anger": "\u751f\u6c14",
    "anticipation": "\u671f\u5f85", "love": "\u7231", "submission": "\u5c48\u670d",
    "surprise": "\u60ca\u8bb6", "alarm": "\u656c\u754f",
    "disappiontment": "\u5931\u671b", "remorse": "\u61ca\u6094",
    "contempt": "\u9119\u5955", "aggression": "\u653b\u51fb",
    "optimism": "\u4e50\u89c2", "guilt": "\u5185\u759a", "curiosity": "\u597d\u5947",
    "despair": "\u7edd\u671b", "unbelief": "\u4e0d\u4fe1", "envy": "\u5978\u5ffb",
    "cynicism": "\u6124\u4e16\u5ac9\u4fd7", "pride": "\u81ea\u8c6a", "hope": "\u5e0c\u671b",
    "delight": "\u9ad8\u5174", "sentimentality": "\u591a\u6101\u5584\u611f",
    "shame": "\u7f9e\u6127", "outrage": "\u6124\u6012", "pessimism": "\u6096\u89c2",
    "morbidness": "\u75c5\u6001", "dominance": "\u4f18\u52bf", "anxiety": "\u7126\u8651",
    "bittersweetness": "\u6096\u559c\u4ea4\u52a0", "ambivalence": "\u77db\u76fe",
    "frozenness": "\u60e7\u6012\u4ea4\u52a0", "confusion": "\u4e0d\u77e5\u6240\u63aa",
    "serenity": "\u5e73\u9759", "ecstacy": "\u72c2\u559c", "interest": "\u5173\u5fc3",
    "vigilance": "\u8b66\u60d5", "rage": "\u66b4\u6012", "annoyance": "\u4e0d\u8010\u70e6",
    "loathing": "\u618e\u6068", "boredom": "\u538c\u5026", "pensiveness": "\u4f24\u611f",
    "grief": "\u6096\u75db", "distraction": "\u4e0d\u89e3", "amazement": "\u60ca\u8bc7",
    "terror": "\u60ca\u609a", "apprehension": "\u62c5\u5fc3", "admiration": "\u5d07\u656c",
    "acceptance": "\u63a5\u53d7",
}


def _extract_english(col):
    m = re.search(r'[(\uff08]([a-zA-Z]+[\w\s]*)[\uff09)]', col)
    return m.group(1).strip().lower() if m else None


def _sleep_day(ts):
    """Return the waking-day date a timestamp belongs to."""
    if ts.hour < SLEEP_HOUR:
        return (ts - pd.Timedelta(days=1)).date()
    return ts.date()


def _detect_and_parse_timestamps(raw_col, source_name, now=None,
                                  max_bad_frac=0.02):
    """
    Robustly parse the raw timestamp column WITHOUT assuming a fixed
    format, and WITHOUT silently guessing on an ambiguous month/day
    order - the ambiguity matters here (a wrong guess would silently
    scramble every date with day > 12, not throw an error).

    Strategy: try each known candidate format; keep only formats that
    parse EVERY value; among those, prefer ones producing zero
    future-dated timestamps (an already-submitted EMA response cannot
    be dated after "now" - real Plutchik participants have not
    responded from the future). If more than one candidate survives
    both filters, it is genuinely ambiguous and this prints a loud
    warning rather than picking silently.

    If NO candidate matches every value but one matches almost all of
    them (>= 1 - max_bad_frac), the specific unmatched rows are printed
    in full (index + raw value) and returned as NaT - load_one_csv drops
    those rows with a loud count, rather than the whole file failing
    for a couple of bad rows. If the unmatched fraction is bigger than
    that, it still hard-fails, since that suggests something more
    fundamental than a couple of stray rows.
    """
    now = now or pd.Timestamp.now()
    candidates = [
        ("%d-%b-%Y %H:%M:%S", "day-Mon-year, with seconds (original format)"),
        ("%m/%d/%Y %H:%M:%S", "month/day/year, with seconds"),
        ("%m/%d/%Y %H:%M",    "month/day/year, no seconds"),
        ("%d/%m/%Y %H:%M:%S", "day/month/year, with seconds"),
        ("%d/%m/%Y %H:%M",    "day/month/year, no seconds"),
    ]
    n = len(raw_col)
    results = {}
    for fmt, desc in candidates:
        parsed = pd.to_datetime(raw_col, format=fmt, errors="coerce")
        results[fmt] = dict(desc=desc, parsed=parsed,
                             n_ok=int(parsed.notna().sum()),
                             n_future=int((parsed > now).sum()))

    full = {fmt: r for fmt, r in results.items() if r["n_ok"] == n}

    if not full:
        best_fmt, best_r = max(results.items(), key=lambda kv: kv[1]["n_ok"])
        n_bad = n - best_r["n_ok"]
        bad_idx = raw_col.index[best_r["parsed"].isna()]
        bad_vals = raw_col.loc[bad_idx]
        if n_bad / n > max_bad_frac:
            raise ValueError(
                f"{source_name}: no candidate timestamp format matched "
                f"every value (best: '{best_fmt}' matched {best_r['n_ok']}/{n}, "
                f"{n_bad} unmatched - too many to auto-drop). "
                f"First unmatched rows:\n" +
                "\n".join(f"    row {i}: {v!r}" for i, v in bad_vals.head(10).items()))
        print(f"  {source_name}: timestamp format detected -> {best_r['desc']} "
              f"({best_fmt}), but {n_bad} row(s) didn't match and will be "
              f"DROPPED:")
        for i, v in bad_vals.items():
            print(f"      row {i}: {v!r}")
        return best_r["parsed"]

    no_future = {fmt: r for fmt, r in full.items() if r["n_future"] == 0}
    pool = no_future if no_future else full
    if len(pool) == 1:
        fmt, r = next(iter(pool.items()))
        print(f"  {source_name}: timestamp format detected -> {r['desc']} "
              f"({fmt})")
        return r["parsed"]

    fmt, r = next(iter(pool.items()))
    print(f"  {source_name}: WARNING - timestamp format is AMBIGUOUS from "
          f"the data alone, multiple formats fit every value:")
    for f2, r2 in pool.items():
        print(f"      - {r2['desc']}  ({f2})  future_dates_if_used={r2['n_future']}")
    print(f"    Defaulting to '{fmt}' ({r['desc']}) - PLEASE CONFIRM this "
          f"against a known participant/date before trusting day-of-month "
          f"ordering in the results.")
    return r["parsed"]


def load_one_csv(csv_path):
    """Load a single raw Plutchik export, same renaming/typing as
    3month_4stPlots.py's load_data() but WITHOUT the fillna/exclusions
    step yet (done after concatenation so it only happens once)."""
    df = pd.read_csv(csv_path)
    rename_map = {}
    for col in df.columns:
        if "\u60c5\u7eea\u8bc4\u5206" in col:
            en = _extract_english(col)
            if en:
                rename_map[col] = en
    df = df.rename(columns=rename_map)

    ts_col = "\u5f00\u59cb\u7b54\u9898\u65f6\u95f4"
    if ts_col not in df.columns:
        raise KeyError(
            f"{csv_path}: expected timestamp column '{ts_col}' not found. "
            f"Columns present: {list(df.columns)[:10]}... - export format "
            f"may have changed further than just the date string; please "
            f"paste the header row.")
    df["timestamp"] = _detect_and_parse_timestamps(df[ts_col], Path(csv_path).name)
    n_nat = int(df["timestamp"].isna().sum())
    if n_nat:
        df = df[df["timestamp"].notna()].copy()
        print(f"  {Path(csv_path).name}: dropped {n_nat} row(s) with an "
              f"unparseable timestamp (shown above).")

    pid_col = "1.\u4f60\u7684\u5b9e\u9a8c\u7f16\u53f7"
    if pid_col not in df.columns:
        raise KeyError(
            f"{csv_path}: expected participant-ID column '{pid_col}' not "
            f"found. Columns present: {list(df.columns)[:10]}... - please "
            f"paste the header row so this can be updated.")
    # Known data-entry name substitutions (applied before int cast).
    # Source: COMPLIANCE_Plutchik.py
    _name_subs = {
        "\u6881\u539f": "019",         # 梁原  -> 019
        "\u9648\u82e5\u840c": "045",   # 陈若萌 -> 045
        ":053": "053"}
    for _bad, _good in _name_subs.items():
        if _bad in df[pid_col].values:
            df[pid_col] = df[pid_col].replace(_bad, _good)
            print(f"  {Path(csv_path).name}: replaced '{_bad}' -> {_good}")
    df["participant"] = df[pid_col].astype(int)

    # real slot label from the survey itself ("A.1"/"B.2"/"C.3" = which
    # check-in of the day) - authoritative, used instead of inferring
    # slot from chronological rank within the sleep_day.
    slot_cols = [c for c in df.columns if "\u7b2c\u51e0\u6b21\u6253\u5361" in c]
    if not slot_cols:
        raise KeyError(
            f"{csv_path}: no check-in-number column found (expected one "
            f"containing '\u7b2c\u51e0\u6b21\u6253\u5361'). Columns present: "
            f"{list(df.columns)[:10]}... - please paste the header row.")
    slot_col = slot_cols[0]
    slot_num = df[slot_col].astype(str).str.extract(r"(\d+)")[0]
    bad_slot = slot_num.isna() | ~slot_num.astype("Int64").isin(
        range(1, N_SLOTS_PER_DAY + 1))
    if bad_slot.any():
        print(f"  {Path(csv_path).name}: {int(bad_slot.sum())} row(s) have "
              f"an unrecognised check-in value and will be DROPPED: "
              f"{df.loc[bad_slot, slot_col].unique().tolist()}")
        df = df[~bad_slot].copy()
        slot_num = slot_num[~bad_slot]
    df["slot"] = slot_num.astype(int)

    df["_source_file"] = Path(csv_path).name
    return df


def load_and_concat(raw_dir):
    """Concatenate every CSV in raw_dir, dedup on (participant, timestamp)
    in case the 3 exports have overlapping windows, then apply the
    sleep-day / hour columns and the empty-cell-means-0 rule exactly once,
    on the full concatenated set."""
    csvs = sorted(Path(raw_dir).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {raw_dir}")
    print(f"Found {len(csvs)} file(s): {[c.name for c in csvs]}")

    dfs = [load_one_csv(c) for c in csvs]
    df = pd.concat(dfs, ignore_index=True)

    n_before = len(df)
    df = df.drop_duplicates(subset=["participant", "timestamp"], keep="first")
    n_dupes = n_before - len(df)
    if n_dupes:
        print(f"  Dropped {n_dupes} duplicate row(s) (same participant+"
              f"timestamp - within a file or repeated across files).")

    df = df.sort_values(["participant", "timestamp"]).reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    df["sleep_day"] = df["timestamp"].apply(_sleep_day)
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    # hour-of-day remapped onto the SAME wrap-around convention as
    # _sleep_day (clock hours < SLEEP_HOUR belong to the tail end of the
    # previous sleep-day), so slot timing stays monotonic within a
    # sleep_day even when slot 3 lands just after midnight
    df["sleep_day_hour"] = np.where(df["hour"] < SLEEP_HOUR,
                                     df["hour"] + 24.0, df["hour"])

    df = apply_subject_edits(df)

    dup_slot = df.duplicated(subset=["participant", "sleep_day", "slot"], keep=False)
    if dup_slot.any():
        print(f"  WARNING: {int(dup_slot.sum())} row(s) share the same "
              f"participant+sleep_day+slot (duplicate/extra submission) - "
              f"kept as-is, not dropped:")
        print(df.loc[dup_slot, ["participant", "sleep_day", "slot", "timestamp"]]
              .to_string(index=False))

    present = [c for c in EMOTION_ZH_EN if c in df.columns]
    if not present:
        raise ValueError("No emotion columns recognised after renaming - "
                          "check the raw CSV header format.")
    df = apply_subject_edits(df)
    df[present] = df[present].fillna(0)   # empty cell = rated 0, NOT missing
    print(f"  {len(df)} rows / {len(present)} emotions / "
          f"{df['participant'].nunique()} participant(s)")

    nominal_hours = (df.groupby("slot")["sleep_day_hour"].median()
                     .reindex(range(1, N_SLOTS_PER_DAY + 1)))
    print(f"  Data-driven nominal check-in times (median, hour-of-sleep-day): "
          + ", ".join(f"slot {s}: {h:.1f}h" for s, h in nominal_hours.items()))

    return df, present, nominal_hours.to_dict()


# ===================================================================
# SLOT GRID + MISSINGNESS
# ===================================================================

def build_slot_grid(df, pid):
    """
    For one participant: use the REAL slot label from the survey itself
    (df['slot'], parsed from the "which check-in today" field - A.1/B.2/
    C.3), not an inferred chronological rank, then build the FULL
    expected grid of (sleep_day x slot 1..N_SLOTS_PER_DAY) spanning that
    participant's first to last active sleep_day. Real entries are
    left-joined onto the grid; anything not matched is a genuinely
    missing checkpoint (NaN in every emotion column).

    Returns a DataFrame with one row per expected slot, columns:
    sleep_day, slot, seq (global chronological index used as the
    interpolation x-axis), + one column per emotion (NaN if missing),
    + is_missing (bool).
    """
    pdata = df[df["participant"] == pid].sort_values("timestamp").copy()

    # a (sleep_day, slot) should be unique; if the survey logged two
    # submissions under the same slot that day, keep the LAST (matches
    # drop_duplicates(keep="first") already applied upstream on exact
    # participant+timestamp dupes - this is a different case: same slot,
    # different timestamp) and say so loudly rather than silently merging.
    dup = pdata.duplicated(subset=["sleep_day", "slot"], keep="last")
    if dup.any():
        print(f"  P{pid}: {int(dup.sum())} duplicate (sleep_day, slot) "
              f"submission(s) - keeping the LATEST, dropping the earlier one(s).")
        pdata = pdata[~dup]

    all_days = pd.date_range(min(pdata["sleep_day"]), max(pdata["sleep_day"]),
                              freq="D").date
    grid = pd.MultiIndex.from_product(
        [all_days, range(1, N_SLOTS_PER_DAY + 1)], names=["sleep_day", "slot"]
    ).to_frame(index=False)
    grid["seq"] = np.arange(len(grid))  # global chronological index

    emo_cols = [c for c in EMOTION_ZH_EN if c in pdata.columns]
    merged = grid.merge(pdata[["sleep_day", "slot", *emo_cols]],
                         on=["sleep_day", "slot"], how="left")
    merged["is_missing"] = merged[emo_cols].isna().all(axis=1)

    n_days = len(all_days)
    n_missing_slots = int(merged["is_missing"].sum())
    print(f"  P{pid}: {n_days} day(s) span, "
          f"{n_days * N_SLOTS_PER_DAY - n_missing_slots}/"
          f"{n_days * N_SLOTS_PER_DAY} checkpoints present "
          f"({n_missing_slots} missing).")
    return merged


# np.trapezoid was introduced in numpy 2.0 as the rename of np.trapz;
# resolve whichever exists so the script runs on either.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _gap_runs(is_missing):
    """Return list of (start_idx, end_idx_inclusive) for each run of
    consecutive True values in a boolean array."""
    runs = []
    start = None
    for i, m in enumerate(is_missing):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(is_missing) - 1))
    return runs


def interpolate_series(seq, values):
    """
    Apply the gap-length-dependent interpolation policy to one
    subject x emotion series.

    seq    : array of global chronological slot indices (int, may have
             gaps if e.g. the participant only has data starting mid-grid
             is not expected here - seq is dense/contiguous by construction)
    values : array of ratings, NaN where the checkpoint is genuinely missing

    Returns (values_interpolated, meta) where meta records what was done
    per gap for auditing (never silent).
    """
    v = np.asarray(values, dtype=float).copy()
    n = len(v)
    is_missing = np.isnan(v)
    n_valid = int((~is_missing).sum())
    meta = []

    if not is_missing.any():
        return v, meta

    for (s, e) in _gap_runs(is_missing):
        gap_len = e - s + 1
        # need real points on both sides to interpolate at all
        has_left = s > 0 and not np.isnan(v[s - 1])
        has_right = e < n - 1 and not np.isnan(v[e + 1])
        if not (has_left and has_right):
            meta.append(dict(start=s, end=e, gap_len=gap_len,
                              method="none", reason="at series edge"))
            continue

        if gap_len <= MAX_GAP_LINEAR:
            method = "linear"
        elif gap_len <= MAX_GAP_CUBIC and n_valid >= MIN_VALID_FOR_CUBIC:
            method = "cubic"
        else:
            meta.append(dict(start=s, end=e, gap_len=gap_len, method="none",
                              reason=f"gap too long / too little context "
                                     f"(n_valid={n_valid})"))
            continue

        valid_idx = np.where(~is_missing)[0]
        if method == "linear":
            x0, x1 = s - 1, e + 1
            y0, y1 = v[x0], v[x1]
            for i in range(s, e + 1):
                v[i] = y0 + (y1 - y0) * (i - x0) / (x1 - x0)
        else:  # cubic, fit on ALL valid points in the series (not just
               # the local neighbours) - a proper global spline, more
               # stable than a 4-point local fit for short local runs
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(valid_idx, v[valid_idx])
            for i in range(s, e + 1):
                v[i] = float(cs(i))

        clipped = False
        for i in range(s, e + 1):
            if v[i] < RATING_MIN or v[i] > RATING_MAX:
                clipped = True
            v[i] = float(np.clip(v[i], RATING_MIN, RATING_MAX))
        meta.append(dict(start=s, end=e, gap_len=gap_len, method=method,
                          reason="clipped to scale bounds" if clipped else ""))

    return v, meta


def process_participant(df, emo_cols, pid, nominal_hours, n_common=None):
    """n_common: dict emotion -> int, truncate each series to its first
    n_common valid points ('common_n' mode). None = 'full' mode."""
    grid = build_slot_grid(df, pid)
    day0 = min(grid["sleep_day"])
    slot_rows, battery_rows = [], []
    mode = "full" if n_common is None else "common_n"

    for emo in emo_cols:
        v_interp, imeta = interpolate_series(grid["seq"].to_numpy(),
                                              grid[emo].to_numpy())
        out = grid[["sleep_day", "slot", "seq"]].copy()
        out["participant"] = pid
        out["emotion"] = emo
        out["rating_raw"] = grid[emo].to_numpy()
        out["rating"] = v_interp
        out["was_interpolated"] = grid[emo].isna() & ~np.isnan(v_interp)
        out["still_missing"] = np.isnan(v_interp)

        if mode == "full":
            slot_rows.append(out)
            for m in imeta:
                if m["method"] != "none":
                    print(f"  P{pid} {emo}: seq {m['start']}-{m['end']} "
                          f"(gap={m['gap_len']}) -> {m['method']}"
                          + (f"  [{m['reason']}]" if m["reason"] else ""))
        else:
            # keep only the first n_common VALID points
            k = n_common.get(emo)
            if k is not None:
                valid_pos = np.where(~out["still_missing"].to_numpy())[0]
                if len(valid_pos) > k:
                    cutoff = valid_pos[k - 1]
                    out = out.iloc[:cutoff + 1].copy()

        n_good = int((~out["still_missing"]).sum())
        meta = {"participant": pid, "emotion": emo, "length_mode": mode}
        if n_good < 5:  # analyze_series' own MIN_N_GENERIC floor
            battery_rows.append({**meta, "measure": "ALL", "value": np.nan,
                                  "flag": f"too_few_samples(n={n_good})",
                                  "n_samples": n_good})
            continue
        battery_rows.extend(run_battery(out, meta, day0, nominal_hours))

    slots = pd.concat(slot_rows, ignore_index=True) if slot_rows else None
    return slots, pd.DataFrame(battery_rows)


# ===================================================================
# DYNAMIC-MEASURE BATTERY  (imported from analyze_behav_vas.py)
# ===================================================================

def slot_to_seconds(sleep_day, slot, day0, nominal_hours):
    """Real-time position (seconds since day0) for one grid slot, for the
    Lomb-Scargle t-axis only. nominal_hours is data-driven (median actual
    check-in time per slot, computed in load_and_concat), not guessed."""
    day_offset = (sleep_day - day0).days
    hour = nominal_hours[slot]
    return day_offset * 86400.0 + hour * 3600.0


def derived_rows(t, v, meta):
    """
    Scalar reductions of the JSON-blob curves analyze_series() emits
    (periodogram, acf), so they're usable at group level - same
    rationale and same measures as analyze_behav_rt.py's derived_rows().
    Also adds CV (raw-scale coefficient of variation), which
    analyze_series() does not compute.
    """
    rows = []

    def add(measure, value, flag=""):
        rows.append({**meta, "measure": measure, "value": value,
                     "flag": flag, "n_samples": len(v)})

    n = len(v)
    if n < 2:
        return rows

    mu, sd = float(np.mean(v)), float(np.std(v))
    if abs(mu) < 1e-9:
        add("CV", np.nan, flag="mean_near_zero")
    else:
        add("CV", sd / mu)

    if sd < 1e-12:
        return rows  # flat - analyze_series already wrote 'flat' for ACF/spectrum

    ac = autocorr(v)
    if not np.all(np.isnan(ac)):
        neg = np.where(ac <= 0)[0]
        first_zero = int(neg[0]) if len(neg) else len(ac)
        if first_zero >= 2:
            add("ACF_area_to_zero", float(_trapz(ac[:first_zero],
                                                        np.arange(first_zero))))
        else:
            add("ACF_area_to_zero", np.nan, flag="no_positive_lags")
        add("ACF_lag1", float(ac[1]) if len(ac) > 1 else np.nan)
        add("ACF_min", float(np.nanmin(ac)))
        add("ACF_min_lag", float(np.nanargmin(ac)))

    return rows


def run_battery(grid_long_one_series, meta, day0, nominal_hours):
    """
    grid_long_one_series: rows for ONE participant x ONE emotion, sorted
    by seq, with columns seq/sleep_day/slot/rating/still_missing.

    Runs analyze_series() on the interpolated values (still-missing
    points excluded - those are real, policy-driven gaps, not data), plus
    the derived scalar reductions. Returns a list of long-format row dicts.
    """
    g = grid_long_one_series.sort_values("seq")
    keep = ~g["still_missing"].to_numpy()
    v = g.loc[keep, "rating"].to_numpy(dtype=float)
    t = np.array([slot_to_seconds(sd, sl, day0, nominal_hours)
                  for sd, sl in zip(g.loc[keep, "sleep_day"],
                                     g.loc[keep, "slot"])])

    rows = analyze_series(t, v, meta)
    rows.extend(derived_rows(t, v, meta))
    return rows


# ===================================================================
# PLOTTING - descriptive violin + jitter, one dot per subject, PPT-sized
# grid of subfigures (16:9, ~13.33x7.5in per figure, per stored preference)
# ===================================================================

def check_length_confound(battery, out_dir):
    """How much does each measure track series length? Reported on the
    'full' mode, where lengths differ. A large |r| means that measure's
    between-subject differences are partly duration, not dynamics -
    read the 'common_n' results for those."""
    d = battery[battery["length_mode"] == "full"].copy()
    d["value_num"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value_num", "n_samples"])
    rows = []
    for meas, g in d.groupby("measure"):
        # one value per participant per emotion; correlate across all
        if g["participant"].nunique() < 3:
            continue
        r = np.corrcoef(g["n_samples"], g["value_num"])[0, 1] \
            if g["n_samples"].std() > 0 else np.nan
        rows.append(dict(measure=meas, r_with_series_length=r,
                         n=len(g)))
    res = pd.DataFrame(rows).sort_values(
        "r_with_series_length", key=lambda s: s.abs(), ascending=False)
    p = Path(out_dir) / "length_confound_check.csv"
    res.to_csv(p, index=False)
    print(f"\n  Length-confound check -> {p}")
    strong = res[res["r_with_series_length"].abs() > 0.3]
    if len(strong):
        print(f"  Measures correlating |r|>0.3 with series length "
              f"(prefer the common_n results for these):")
        print(strong.to_string(index=False))
    return res



VIOLIN_LEGEND = ("red line = mean   \u00b7   dashed = median   \u00b7   "
                 "dotted = 25th/75th percentile   \u00b7   "
                 "dots = individual subjects")


def _violin_caption(fig, n=None):
    """One-line key so the lines on a violin are self-explanatory."""
    txt = VIOLIN_LEGEND + (f"   \u00b7   n = {n} subjects" if n else "")
    fig.text(0.5, 0.005, txt, ha="center", va="bottom", fontsize=9,
             color="#333333")


def _overlay_mean(ax, values_by_pos, orient="v"):
    """Draw the MEAN as a wide flat marker on top of a violin.

    seaborn's inner="quartile" gives median and quartiles but never the
    mean, and for skewed distributions those differ - worth seeing both.
    values_by_pos: {x_position: array_of_values}.
    """
    for pos, vals in values_by_pos.items():
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        ax.plot([pos - 0.28, pos + 0.28], [v.mean(), v.mean()],
                color="#b00020", lw=2.0, solid_capstyle="butt",
                zorder=6, clip_on=False)


def _shared_emotion_order(d):
    """One emotion order used by every measure's figure, so the panels can
    be read side by side. Derived from the 'mean' measure (i.e. the mean
    Plutchik rating per emotion); falls back to the first measure
    alphabetically if 'mean' is absent."""
    ref = "mean" if "mean" in set(d["measure"]) else sorted(d["measure"].unique())[0]
    g = d[d["measure"] == ref].groupby("emotion")["value_num"].mean()
    return list(g.sort_values(ascending=False).index), ref


def emotion_pairwise_ttests(d, length_mode):
    """Paired (repeated-measures) t-tests between EVERY pair of emotions,
    for every measure. All 52 emotions -> 1326 pairs per measure. Results
    go to CSV only; no significance marks are drawn on the figures.
    UNCORRECTED, per request."""
    from scipy import stats
    rows = []
    for meas, gm in d.groupby("measure"):
        wide = gm.pivot_table(index="participant", columns="emotion",
                              values="value_num")
        emos = sorted(wide.columns)
        for i in range(len(emos)):
            for j in range(i + 1, len(emos)):
                a, b = emos[i], emos[j]
                pair = wide[[a, b]].dropna()
                base = dict(measure=meas, emo_a=a, emo_b=b,
                            n_pairs=len(pair), length_mode=length_mode)
                if len(pair) < 3:
                    rows.append({**base, "t": np.nan, "p": np.nan,
                                 "mean_diff": np.nan, "cohens_dz": np.nan,
                                 "flag": "n<3"})
                    continue
                diff = pair[a].to_numpy() - pair[b].to_numpy()
                sd = np.std(diff, ddof=1)
                if np.allclose(diff, 0):
                    rows.append({**base, "t": np.nan, "p": np.nan,
                                 "mean_diff": 0.0, "cohens_dz": np.nan,
                                 "flag": "identical"})
                    continue
                t, p = stats.ttest_rel(pair[a], pair[b])
                rows.append({**base, "t": float(t), "p": float(p),
                             "mean_diff": float(np.mean(diff)),
                             "cohens_dz": float(np.mean(diff) / sd) if sd > 0
                             else np.nan, "flag": ""})
    return pd.DataFrame(rows)


def plot_emotion_comparison(battery, out_dir, length_mode=None,
                            sort_mode="self", n_rows=3):
    """
    One PPT-sized figure per measure: violin+jitter for ALL emotions,
    sorted by mean and wrapped across n_rows continuous rows so the
    labels stay legible. Violin fill follows a colour gradient keyed to
    each emotion's mean, so the sort is visible at a glance.

    sort_mode:
      "self"   - each measure sorted by its own mean
      "shared" - one order (from the 'mean' measure) reused everywhere,
                 so figures are comparable side by side

    No significance annotation is drawn - see emotion_pairwise_ttests().
    """
    length_mode = length_mode or PRIMARY_LENGTH_MODE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors
    import seaborn as sns

    d = battery[battery["length_mode"] == length_mode].copy()
    d["value_num"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value_num"])
    if d.empty:
        print("  Nothing numeric to plot for emotion comparison.")
        return

    shared_order, ref_meas = _shared_emotion_order(d)
    if sort_mode == "shared":
        print(f"  Shared emotion order taken from the '{ref_meas}' measure.")

    for meas, gm in d.groupby("measure"):
        means = gm.groupby("emotion")["value_num"].mean()
        if sort_mode == "self":
            order = list(means.sort_values(ascending=False).index)
        else:
            order = [e for e in shared_order if e in means.index]
        if not order:
            continue

        per_row = int(np.ceil(len(order) / n_rows))
        norm = mcolors.Normalize(vmin=means.min(), vmax=means.max())
        cmap = cm.get_cmap("viridis")

        fig, axes = plt.subplots(n_rows, 1, figsize=(13.33, 7.5),
                                 squeeze=False)
        order_desc = ("emotions ordered by THIS measure's own mean"
                      if sort_mode == "self" else
                      f"emotions in the SHARED order (by mean {ref_meas}, "
                      f"identical across all measures - NOT this measure's "
                      f"own mean)")
        fig.suptitle(f"{meas} across all {len(order)} emotions   "
                     f"({length_mode}, n={gm['participant'].nunique()} "
                     f"subjects)\n{order_desc}",
                     fontsize=15, fontweight="bold")

        for r in range(n_rows):
            ax = axes[r][0]
            chunk = order[r * per_row:(r + 1) * per_row]
            if not chunk:
                ax.set_visible(False)
                continue
            sub = gm[gm["emotion"].isin(chunk)]
            pal = {e: cmap(norm(means[e])) for e in chunk}
            sns.violinplot(data=sub, x="emotion", y="value_num", order=chunk,
                           ax=ax, inner="quartile", cut=0, linewidth=0.6,
                           palette=pal, saturation=0.9)
            for ln in ax.lines:            # quartile/median lines
                ln.set_color("#1a1a2e"); ln.set_linewidth(1.0)
            for v in ax.collections:
                v.set_alpha(0.75)
            sns.stripplot(data=sub, x="emotion", y="value_num", order=chunk,
                          ax=ax, color="#1a1a2e", size=3, jitter=0.22,
                          alpha=0.8, linewidth=0)
            _overlay_mean(ax, {i: sub.loc[sub["emotion"] == e, "value_num"]
                               for i, e in enumerate(chunk)})
            ax.set_xlabel(""); ax.set_ylabel(meas if r == 1 else "",
                                             fontsize=9)
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.set_axisbelow(True)
            sns.despine(ax=ax, left=True)

        _violin_caption(fig, gm["participant"].nunique())
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        p_out = Path(out_dir) / f"emocmp_{meas}_{length_mode}_{sort_mode}.png"
        fig.savefig(p_out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {p_out}")


def plot_violin_grid(battery_df, out_dir, title_prefix="Plutchik dynamics",
                     length_mode=None):
    """
    One PPT-sized figure per emotion, with one subplot per measure
    (violin + jittered per-subject dots). Numeric measures only
    ('flat'/NaN-string cells and JSON blobs are excluded).
    """
    length_mode = length_mode or PRIMARY_LENGTH_MODE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = battery_df[battery_df["length_mode"] == length_mode].copy()
    d["value_num"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value_num"])
    if d.empty:
        print("  Nothing numeric to plot.")
        return

    NCOLS = 5
    for emo, gemo in d.groupby("emotion"):
        measures = sorted(gemo["measure"].unique())
        nrows = int(np.ceil(len(measures) / NCOLS))
        fig, axes = plt.subplots(nrows, NCOLS, figsize=(13.33, 7.5),
                                  squeeze=False)
        fig.suptitle(f"{title_prefix} - {emo} ({length_mode})",
                     fontsize=13, fontweight="bold")

        for i, meas in enumerate(measures):
            ax = axes[i // NCOLS][i % NCOLS]
            sub = gemo[gemo["measure"] == meas]
            sns.violinplot(y=sub["value_num"], ax=ax, inner="quartile",
                            color="#cfe2f3", cut=0, linewidth=0.9)
            for ln in ax.lines:
                ln.set_color("#1a1a2e"); ln.set_linewidth(1.0)
            sns.stripplot(y=sub["value_num"], ax=ax, color="#2c3e50",
                          size=4, jitter=0.15, alpha=0.7)
            _overlay_mean(ax, {0: sub["value_num"]})
            ax.set_title(meas, fontsize=9)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(labelsize=7)
            sns.despine(ax=ax)

        for i in range(len(measures), nrows * NCOLS):
            axes[i // NCOLS][i % NCOLS].set_visible(False)

        _violin_caption(fig, gemo["participant"].nunique())
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = Path(out_dir) / f"plutchik_dynamics_{emo}_{length_mode}_violin.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    df, emo_cols, nominal_hours = load_and_concat(RAW_DIR)
    nominal_hours = {k: (v if pd.notna(v) else NOMINAL_SLOT_HOURS_FALLBACK[k])
                     for k, v in nominal_hours.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pids = sorted(df["participant"].unique())

    # ---- pass 1: full-length ----
    all_slots, all_battery = [], []
    for pid in pids:
        print(f"\n-- Participant {pid} (full) --")
        slot_df, battery_df = process_participant(df, emo_cols, pid,
                                                  nominal_hours)
        all_slots.append(slot_df)
        all_battery.append(battery_df)
    slots = pd.concat(all_slots, ignore_index=True)

    # ---- how many valid points does each participant have per emotion? ----
    valid_counts = (slots[~slots["still_missing"]]
                    .groupby(["emotion", "participant"]).size()
                    .rename("n_valid").reset_index())
    n_common = valid_counts.groupby("emotion")["n_valid"].min().to_dict()
    span = valid_counts.groupby("emotion")["n_valid"].agg(["min", "max"])
    print(f"\nSeries length across participants: "
          f"min={int(span['min'].min())}, max={int(span['max'].max())} "
          f"valid points.")
    if "common_n" in LENGTH_MODES:
        print("  Also running a length-controlled pass: every participant "
              "truncated to their first n_common points per emotion.")
    else:
        print("  Using FULL series (no truncation). Add 'common_n' to "
              "LENGTH_MODES for the length-controlled comparison.")

    # ---- pass 2: length-controlled (only if requested) ----
    for pid in (pids if "common_n" in LENGTH_MODES else []):
        print(f"\n-- Participant {pid} (common_n) --")
        _, battery_df = process_participant(df, emo_cols, pid, nominal_hours,
                                            n_common=n_common)
        all_battery.append(battery_df)

    battery = pd.concat(all_battery, ignore_index=True)
    n_before = len(battery)
    battery = battery[~battery["measure"].isin(EXCLUDE_MEASURES)]
    print(f"\nDropped {n_before - len(battery)} row(s) for excluded measures "
          f"{sorted(EXCLUDE_MEASURES)}")

    slots_path = OUT_DIR / "plutchik_slots_interpolated.csv"
    battery_path = OUT_DIR / "plutchik_dynamic_measures.csv"
    slots.to_csv(slots_path, index=False)
    battery.to_csv(battery_path, index=False)
    print(f"[OK] {len(slots)} slot rows -> {slots_path}")
    print(f"[OK] {len(battery)} measure rows -> {battery_path}")

    check_length_confound(battery, OUT_DIR)

    plot_dir = OUT_DIR / "figures"
    plot_dir.mkdir(exist_ok=True)
    for mode in LENGTH_MODES:
        plot_violin_grid(battery, plot_dir, length_mode=mode)
    emocmp_dir = OUT_DIR / "figures_emotion_comparison"
    emocmp_dir.mkdir(exist_ok=True)
    for sort_mode in ("self", "shared"):
        plot_emotion_comparison(battery, emocmp_dir,
                                length_mode=PRIMARY_LENGTH_MODE,
                                sort_mode=sort_mode)

    # paired t-tests over ALL emotion pairs -> CSV only, no marks on figures
    d = battery[battery["length_mode"] == PRIMARY_LENGTH_MODE].copy()
    d["value_num"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value_num"])
    tt = emotion_pairwise_ttests(d, PRIMARY_LENGTH_MODE)
    tt_path = OUT_DIR / f"emotion_pairwise_ttests_{PRIMARY_LENGTH_MODE}.csv"
    tt.to_csv(tt_path, index=False)
    n_ok = int(tt["p"].notna().sum())
    print(f"\n[OK] {len(tt)} paired t-tests ({n_ok} computable) -> {tt_path}")
    print("     UNCORRECTED. With this many tests, ~5% of them will reach")
    print("     p<.05 by chance alone - read the table as a whole, not cell "
          "by cell.")


if __name__ == "__main__":
    main()
