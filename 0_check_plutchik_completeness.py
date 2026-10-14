# -*- coding: utf-8 -*-
"""
0_check_plutchik_completeness.py
================================
Standalone completeness check for the Plutchik EMA timeseries. Run this
BEFORE 1_extract_plutchik_dynamics.py to see how much data each subject
actually has, where the gaps fall, and what the interpolation policy
would do with them.

Changes nothing and writes no analysis output - diagnostics only.

WHAT IT SHOWS
-------------
A checkpoint is either submitted or not: if a submission is missing,
every emotion is missing for that slot at once. (An EMPTY CELL inside a
submitted row is NOT missing - it means the emotion was rated 0, per the
study convention.) So completeness is a property of the 3-per-day
checkpoint grid, not of individual emotions, and can be shown compactly.
The script verifies this assumption holds and reports any violation.

Per subject:
  * a compliance raster - every expected checkpoint from their first to
    last active day, coloured observed / interpolated / still missing
  * the actual timeseries for a sample of emotions, with interpolated
    points marked, so the effect of the gaps on the data is visible

Group level:
  * per-subject completeness bars
  * distribution of gap lengths across the whole sample

Outputs (to OUT_DIR):
  completeness_per_subject.csv - one row per subject
  gaps_detail.csv              - one row per gap, with the method the
                                 interpolation policy would apply
  figures/                     - the plots above

Usage:
    python3 0_check_plutchik_completeness.py
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ===================================================================
# CONFIG  (kept identical to 1_extract_plutchik_dynamics.py)
# ===================================================================

RAW_DIR = Path("/home/ZhangJianfeng_04/emo_study/Behav/Pluchik")
OUT_DIR = Path("/home/ZhangJianfeng_04/emo_study/Behav/Pluchik/completeness_check")

SLEEP_HOUR = 4           # entries before 4am belong to the previous waking day
N_SLOTS_PER_DAY = 3

MAX_GAP_LINEAR = 1       # 1 missing slot        -> linear
MAX_GAP_CUBIC = 3        # 2-3 missing slots     -> cubic (if enough context)
MIN_VALID_FOR_CUBIC = 8  # below this, cubic is not better determined

# emotions drawn in the timeseries panel (all 52 would be illegible)
SAMPLE_EMOTIONS = ["joy", "trust", "fear", "sadness",
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
# Specific bad days to delete (spurious/erroneous submissions).
# Source: COMPLIANCE_Plutchik.py.  Key = participant, value = list of
# calendar dates to drop (the "date" column, before sleep-day remap).
BAD_DAYS = {
    28: ["2026-08-01"],
     3: ["2026-06-22"],
     1: ["2026-05-31", "2026-06-04"],
    21: ["2026-06-27"],
    40: ["2026-09-14", "2026-09-15"],
}

SUBJECT_EDITS = {
    # subject 002: only 6 days enrolled, 33 % observed - too sparse for
    # any dynamic measure. Excluded from the EMA entirely.
    2:  {"exclude": True,
         "reason": "too little EMA data (6 days, 33 % observed)"},

    # subject 021: an isolated first rating, then a ~20-day break, then a
    # continuous run.  Dropping the isolated start removes the 59-slot gap
    # altogether; remaining series may still contain 1-2 missing check-ins,
    # which interpolate normally.
    21: {"start_on": "2026-07-17",
         "reason": "isolated first rating before a long break; keep the "
                   "continuous run from 17 July"},

    # subject 024: early fragment discarded.
    24: {"start_on": "2026-07-06",
         "reason": "discard the fragment before 6 July"},

    # subject 040: long early fragment of poor quality; keep from 30 Sep.
    40: {"start_on": "2026-09-30",
         "reason": "discard everything before 30 September 2026"},
}

# NOTE: the compliance script contained a fully commented-out list that
# excluded ALL subjects from the analysis:
#   for pid in [1,2,3,8,9,6,10,5,11,13,14,22,26,7,23,24,25,27,28,29,
#               19,15,30,32,40]:
#       df = df[~(df["participant"]==pid)]
# That block was NOT active (commented out) and is NOT applied here.


def apply_subject_edits(df):
    """Apply BAD_DAYS point exclusions then SUBJECT_EDITS trims/excludes,
    reporting exactly what each change removed."""
    print("\nApplying subject edits (Plutchik EMA only):")

    # --- bad-day deletions (single calendar dates) ---
    # BAD_DAYS filters on CALENDAR date (timestamp.dt.date), NOT sleep_day,
    # matching the compliance script exactly. A 1am submission on Aug 1st
    # has date=Aug 1 but sleep_day=Jul 31; filtering on sleep_day would
    # miss it.
    if BAD_DAYS:
        cal_date = df["timestamp"].dt.date
        for pid, dates in sorted(BAD_DAYS.items()):
            for d_str in dates:
                d = pd.Timestamp(d_str).date()
                mask = (df["participant"] == pid) & (cal_date == d)
                n = int(mask.sum())
                if n:
                    df = df[~mask]
                    cal_date = df["timestamp"].dt.date   # refresh after drop
                    print(f"  P{pid:03d}: deleted {n} submission(s) on {d} "
                          f"(bad-day exclusion)")

    if not SUBJECT_EDITS:
        return df
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
            if rem.empty:
                print(f"  P{pid:03d}: dropped {int(drop.sum())} submission(s) "
                      f"before {cut}; 0 remain — subject has no data from "
                      f"{cut} onward ({ed['reason']})")
            else:
                print(f"  P{pid:03d}: dropped {int(drop.sum())} submission(s) "
                      f"before {cut}; {len(rem)} remain, now "
                      f"{min(rem['sleep_day'])} to {max(rem['sleep_day'])} "
                      f"({ed['reason']})")
    return df.reset_index(drop=True)






# ===================================================================
# COHERENT-BLOCK EXTRACTION + SUBJECT FILTERING
# ===================================================================
# Steps after SUBJECT_EDITS (hardcoded date trims):
#
#   1. COHERENT BLOCK: per subject, keep only the longest continuous run
#      of sleep-days where no gap exceeds MAX_ZERO_DAY_RUN consecutive
#      days without any submission.
#   2. MINIMUM DURATION: subjects whose remaining block is shorter than
#      MIN_EFFECTIVE_DAYS are excluded.
#   3. EXPLICIT HARD EXCLUDES: HARD_EXCLUDE list regardless of block length.
#
# All exclusions are printed with their reason.

MAX_ZERO_DAY_RUN = 2     # ≤2 consecutive zero-rating days: tolerated
                          # 3+ consecutive zero-rating days: breaks the block
MIN_EFFECTIVE_DAYS = 60   # subjects with a coherent block < this are excluded
HARD_EXCLUDE = {2, 13}    # excluded regardless of block length


def _find_coherent_block(pdata):
    """
    Find the longest run of consecutive sleep-days where no gap of
    (MAX_ZERO_DAY_RUN + 1)+ consecutive zero-rating days occurs.
    Returns (block_start, block_end, block_n_days).
    """
    if pdata.empty:
        return None, None, 0

    first = pdata["sleep_day"].min()
    last  = pdata["sleep_day"].max()
    all_days = pd.date_range(first, last, freq="D").date
    rated = set(pdata["sleep_day"].unique())

    best_start = best_end = None
    best_len = 0

    cur_start = all_days[0]
    zero_run = 0 if all_days[0] in rated else 1

    i = 1
    while i < len(all_days):
        d = all_days[i]
        if d in rated:
            zero_run = 0
            i += 1
        else:
            zero_run += 1
            if zero_run > MAX_ZERO_DAY_RUN:
                # block ends MAX_ZERO_DAY_RUN days before today
                end_idx = i - MAX_ZERO_DAY_RUN - 1
                if end_idx >= 0:
                    block_end = all_days[end_idx]
                    block_len = (pd.Timestamp(block_end) -
                                 pd.Timestamp(cur_start)).days + 1
                    if block_len > best_len:
                        best_len = block_len
                        best_start, best_end = cur_start, block_end
                # advance to the next rated day after this gap
                i += 1
                while i < len(all_days) and all_days[i] not in rated:
                    i += 1
                if i < len(all_days):
                    cur_start = all_days[i]
                    zero_run = 0
                else:
                    cur_start = None
                    break
            else:
                i += 1

    if cur_start is not None:
        block_end = all_days[-1]
        block_len = (pd.Timestamp(block_end) -
                     pd.Timestamp(cur_start)).days + 1
        if block_len > best_len:
            best_len = block_len
            best_start, best_end = cur_start, block_end

    return best_start, best_end, best_len


def apply_coherent_block_filter(df):
    """
    Trim each participant to their longest coherent block, then remove
    subjects below MIN_EFFECTIVE_DAYS or in HARD_EXCLUDE.
    Returns (filtered_df, exclusion_log).
    """
    print("\nCoherent-block extraction and subject filtering:")
    print(f"  Max tolerated zero-rating gap : {MAX_ZERO_DAY_RUN} days")
    print(f"  Minimum block length          : {MIN_EFFECTIVE_DAYS} days")
    print(f"  Hard-excluded subjects        : {sorted(HARD_EXCLUDE)}")

    keep_rows = []
    exclusion_log = []
    included_log  = []

    for pid in sorted(df["participant"].unique()):
        pdata = df[df["participant"] == pid]

        if pid in HARD_EXCLUDE:
            exclusion_log.append((pid, "hard-excluded (HARD_EXCLUDE list)"))
            continue

        block_start, block_end, block_days = _find_coherent_block(pdata)

        if block_start is None or block_days == 0:
            exclusion_log.append((pid, "no usable data found"))
            continue

        if block_days < MIN_EFFECTIVE_DAYS:
            exclusion_log.append((
                pid,
                f"coherent block only {block_days} days "
                f"({block_start} to {block_end}), "
                f"below {MIN_EFFECTIVE_DAYS}-day minimum"))
            continue

        in_block = ((pdata["sleep_day"] >= block_start) &
                    (pdata["sleep_day"] <= block_end))
        n_trimmed = int((~in_block).sum())
        keep_rows.append(pdata[in_block])
        included_log.append((pid, block_start, block_end, block_days, n_trimmed))

    print(f"\n  Included ({len(included_log)} subject(s)):")
    for pid, bs, be, bd, nt in included_log:
        trim_note = f"  [{nt} submission(s) outside block trimmed]" if nt else ""
        print(f"    P{pid:03d}: {bs} to {be}  ({bd} days){trim_note}")

    print(f"\n  Excluded ({len(exclusion_log)} subject(s)):")
    for pid, reason in exclusion_log:
        print(f"    P{pid:03d}: {reason}")

    if not keep_rows:
        raise RuntimeError("All subjects were excluded — check the thresholds.")

    result = pd.concat(keep_rows, ignore_index=True)
    print(f"\n  {result['participant'].nunique()} subject(s) retained, "
          f"{len(result)} submission(s) total.")
    return result, exclusion_log


MIN_GAP_MINUTES = 60   # entries within this window are near-duplicates


def remove_close_duplicates(df):
    """
    Per participant, walk entries in time order and keep an entry only if
    it is at least MIN_GAP_MINUTES after the last KEPT entry.  A run of
    several close-together submissions collapses to its first one rather
    than keeping every other one.  Mirrors COMPLIANCE_Plutchik.py exactly.
    Returns (clean_df, n_dropped).
    """
    keep_idx, drop_idx = [], []
    for pid, grp in df.sort_values("timestamp").groupby("participant"):
        last_kept_ts = None
        for idx, ts in zip(grp.index, grp["timestamp"]):
            gap_ok = (last_kept_ts is None or
                      (ts - last_kept_ts).total_seconds() / 60 >= MIN_GAP_MINUTES)
            if gap_ok:
                keep_idx.append(idx)
                last_kept_ts = ts
            else:
                drop_idx.append(idx)
    clean = df.loc[keep_idx].sort_values("timestamp").reset_index(drop=True)
    if drop_idx:
        print(f"  remove_close_duplicates: dropped {len(drop_idx)} "
              f"submission(s) within {MIN_GAP_MINUTES} min of the previous "
              f"kept entry (per participant)")
    return clean, len(drop_idx)


# ===================================================================
# LOADING  (same rules as the extract script)
# ===================================================================

def _extract_english(col):
    m = re.search(r'[(\uff08]([a-zA-Z]+[\w\s]*)[\uff09)]', col)
    return m.group(1).strip().lower() if m else None


def _sleep_day(ts):
    if ts.hour < SLEEP_HOUR:
        return (ts - pd.Timedelta(days=1)).date()
    return ts.date()


def _detect_timestamps(raw_col, source_name, now=None):
    """Try several formats; prefer one that parses everything and yields
    no future-dated responses. Tolerates a few unparseable rows."""
    now = now or pd.Timestamp.now()
    cands = [("%d-%b-%Y %H:%M:%S", "day-Mon-year w/ seconds"),
             ("%m/%d/%Y %H:%M:%S", "month/day/year w/ seconds"),
             ("%m/%d/%Y %H:%M",    "month/day/year, no seconds"),
             ("%d/%m/%Y %H:%M:%S", "day/month/year w/ seconds"),
             ("%d/%m/%Y %H:%M",    "day/month/year, no seconds")]
    n = len(raw_col)
    res = {}
    for fmt, desc in cands:
        p = pd.to_datetime(raw_col, format=fmt, errors="coerce")
        res[fmt] = dict(desc=desc, parsed=p, n_ok=int(p.notna().sum()),
                        n_future=int((p > now).sum()))
    full = {f: r for f, r in res.items() if r["n_ok"] == n}
    if not full:
        best_f, best_r = max(res.items(), key=lambda kv: kv[1]["n_ok"])
        n_bad = n - best_r["n_ok"]
        print(f"  {source_name}: {best_r['desc']}; {n_bad} unparseable row(s) dropped")
        return best_r["parsed"]
    no_future = {f: r for f, r in full.items() if r["n_future"] == 0} or full
    f, r = next(iter(no_future.items()))
    print(f"  {source_name}: timestamp format -> {r['desc']}")
    return r["parsed"]


def load_and_concat(raw_dir):
    csvs = sorted(Path(raw_dir).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {raw_dir}")
    print(f"Found {len(csvs)} file(s): {[c.name for c in csvs]}")

    frames = []
    for c in csvs:
        df = pd.read_csv(c)
        ren = {}
        for col in df.columns:
            if "\u60c5\u7eea\u8bc4\u5206" in col:
                en = _extract_english(col)
                if en:
                    ren[col] = en
        df = df.rename(columns=ren)

        # If the survey wording changed mid-study, multiple CSV columns can
        # map to the same English name after renaming. Merge duplicates by
        # back-filling across them so no data is lost (matches compliance
        # script logic exactly).
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            print(f"  {c.name}: merging {len(dupes)} duplicate column(s) "
                  f"(survey text changed mid-study): {dupes}")
            merged = {}
            for name in dict.fromkeys(df.columns):
                block = df.loc[:, df.columns == name]
                merged[name] = (block.iloc[:, 0] if block.shape[1] == 1
                                else block.bfill(axis=1).iloc[:, 0])
            df = pd.DataFrame(merged, index=df.index)

        ts_col = "\u5f00\u59cb\u7b54\u9898\u65f6\u95f4"
        df["timestamp"] = _detect_timestamps(df[ts_col], c.name)
        df = df[df["timestamp"].notna()].copy()
        pid_col_0 = "1.\u4f60\u7684\u5b9e\u9a8c\u7f16\u53f7"
        # Known data-entry name substitutions (applied before int cast)
        _name_subs = {
            "\u6881\u539f": "019",         # 梁原  -> 019
            "\u9648\u82e5\u840c": "045",   # 陈若萌 -> 045
            ":053": "053"}
        for _bad, _good in _name_subs.items():
            if _bad in df[pid_col_0].values:
                df[pid_col_0] = df[pid_col_0].replace(_bad, _good)
                print(f"  {c.name}: replaced '{_bad}' -> {_good}")
        df["participant"] = df[pid_col_0].astype(int)
        slot_cols = [x for x in df.columns if "\u7b2c\u51e0\u6b21\u6253\u5361" in x]
        if not slot_cols:
            raise KeyError(f"{c.name}: no check-in-number column found")
        num = df[slot_cols[0]].astype(str).str.extract(r"(\d+)")[0]
        ok = num.notna() & num.astype("Int64").isin(range(1, N_SLOTS_PER_DAY + 1))
        if (~ok).any():
            print(f"  {c.name}: dropped {int((~ok).sum())} row(s) with an "
                  f"unrecognised check-in value")
        df = df[ok].copy()
        df["slot"] = num[ok].astype(int)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["participant", "timestamp"], keep="first")
    if before - len(df):
        print(f"  Dropped {before-len(df)} duplicate row(s) "
              f"(same participant+timestamp)")

    df, _ = remove_close_duplicates(df)

    df = df.sort_values(["participant", "timestamp"]).reset_index(drop=True)
    df["sleep_day"] = df["timestamp"].apply(_sleep_day)
    df = apply_subject_edits(df)
    df, _exclusion_log = apply_coherent_block_filter(df)
    emo = [c for c in df.columns
           if isinstance(c, str) and c.islower() and c.isalpha()
           and c not in ("timestamp", "participant", "slot", "sleep_day")]
    df[emo] = df[emo].fillna(0)      # empty cell = rated 0, NOT missing
    print(f"  {len(df)} submissions / {len(emo)} emotions / "
          f"{df['participant'].nunique()} participants\n")
    return df, emo


# ===================================================================
# GRID + GAPS
# ===================================================================

def _gap_runs(mask):
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def classify_gap(gap_len, n_valid, at_edge):
    """Which interpolation the policy in the extract script would apply."""
    if at_edge:
        return "none (series edge)"
    if gap_len <= MAX_GAP_LINEAR:
        return "linear"
    if gap_len <= MAX_GAP_CUBIC and n_valid >= MIN_VALID_FOR_CUBIC:
        return "cubic"
    return "none (too long / too little context)"


def build_grid(df, emo, pid):
    p = df[df["participant"] == pid].sort_values("timestamp").copy()
    dup = p.duplicated(subset=["sleep_day", "slot"], keep="last")
    n_dup = int(dup.sum())
    p = p[~dup]

    days = pd.date_range(min(p["sleep_day"]), max(p["sleep_day"]), freq="D").date
    grid = pd.MultiIndex.from_product(
        [days, range(1, N_SLOTS_PER_DAY + 1)],
        names=["sleep_day", "slot"]).to_frame(index=False)
    grid["seq"] = np.arange(len(grid))
    g = grid.merge(p[["sleep_day", "slot", *emo]],
                   on=["sleep_day", "slot"], how="left")
    g["observed"] = g[emo].notna().any(axis=1)

    # sanity: missingness should be all-or-nothing across emotions
    partial = g[emo].notna().any(axis=1) & g[emo].isna().any(axis=1)
    n_partial = int(partial.sum())

    mask = (~g["observed"]).to_numpy()
    n_valid = int(g["observed"].sum())
    g["status"] = np.where(g["observed"], "observed", "missing")
    gaps = []
    for s, e in _gap_runs(mask):
        glen = e - s + 1
        at_edge = (s == 0) or (e == len(mask) - 1)
        method = classify_gap(glen, n_valid, at_edge)
        if method in ("linear", "cubic"):
            g.loc[s:e, "status"] = "interpolated"
        gaps.append(dict(participant=pid, seq_start=s, seq_end=e,
                         gap_slots=glen,
                         start_day=g.loc[s, "sleep_day"],
                         start_slot=int(g.loc[s, "slot"]),
                         method=method))
    return g, pd.DataFrame(gaps), n_dup, n_partial


# ===================================================================
# PLOTS
# ===================================================================

STATUS_COLOR = {"observed": "#2e7d32",
                "interpolated": "#f9a825",
                "missing": "#c62828"}


def plot_subject(g, pid, emo, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import seaborn as sns

    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5], hspace=0.35)

    # --- compliance raster: slot x day ---
    ax0 = fig.add_subplot(gs[0])
    days = sorted(g["sleep_day"].unique())
    dayidx = {d: i for i, d in enumerate(days)}
    for _, r in g.iterrows():
        ax0.add_patch(plt.Rectangle(
            (dayidx[r["sleep_day"]] - .5, r["slot"] - .5), 1, 1,
            facecolor=STATUS_COLOR[r["status"]], edgecolor="white", lw=.4))
    ax0.set_xlim(-.5, len(days) - .5); ax0.set_ylim(.5, N_SLOTS_PER_DAY + .5)
    ax0.set_yticks(range(1, N_SLOTS_PER_DAY + 1))
    ax0.set_ylabel("check-in", fontsize=9)
    step = max(1, len(days) // 25)
    ax0.set_xticks(range(0, len(days), step))
    ax0.set_xticklabels([str(days[i]) for i in range(0, len(days), step)],
                        rotation=90, fontsize=6)
    ax0.invert_yaxis()
    n_obs = int((g["status"] == "observed").sum())
    n_int = int((g["status"] == "interpolated").sum())
    n_mis = int((g["status"] == "missing").sum())
    tot = len(g)
    ax0.set_title(f"Subject {pid:03d} - {tot} expected check-ins over "
                  f"{len(days)} days:  {n_obs} observed ({100*n_obs/tot:.0f}%),  "
                  f"{n_int} interpolated,  {n_mis} still missing",
                  fontsize=12, fontweight="bold")
    ax0.legend(handles=[Patch(facecolor=v, label=k)
                        for k, v in STATUS_COLOR.items()],
               loc="upper right", fontsize=7, ncol=3, framealpha=.9)

    # --- timeseries for a sample of emotions ---
    ax1 = fig.add_subplot(gs[1])
    shown = [e for e in SAMPLE_EMOTIONS if e in emo][:8]
    pal = sns.color_palette("tab10", len(shown))
    for c, e in zip(pal, shown):
        y = g[e].astype(float).to_numpy()
        ax1.plot(g["seq"], y, lw=.9, color=c, alpha=.85, label=e)
        obs = g["status"].to_numpy() == "observed"
        ax1.plot(g["seq"][obs], y[obs], "o", ms=2.2, color=c, alpha=.9)
    for _, r in g[g["status"] != "observed"].iterrows():
        ax1.axvspan(r["seq"] - .5, r["seq"] + .5,
                    color=STATUS_COLOR[r["status"]], alpha=.16, lw=0)
    ax1.set_xlabel("check-in index (3 per day)", fontsize=9)
    ax1.set_ylabel("rating", fontsize=9)
    ax1.set_title("Sample emotion timeseries (raw values; shaded = "
                  "not submitted)", fontsize=10)
    ax1.legend(fontsize=6, ncol=8, loc="upper center",
               bbox_to_anchor=(.5, -.18), frameon=False)
    ax1.tick_params(labelsize=7)
    ax1.grid(alpha=.25, lw=.5); ax1.set_axisbelow(True)
    sns.despine(ax=ax1)

    p = Path(out_dir) / f"completeness_subject_{pid:03d}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_group(summary, gaps, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5),
                             gridspec_kw={"width_ratios": [2, 1]})
    s = summary.sort_values("n_expected", ascending=False)
    x = np.arange(len(s))
    ax = axes[0]
    ax.bar(x, s["n_observed"], color=STATUS_COLOR["observed"], label="observed")
    ax.bar(x, s["n_interpolated"], bottom=s["n_observed"],
           color=STATUS_COLOR["interpolated"], label="interpolated")
    ax.bar(x, s["n_still_missing"],
           bottom=s["n_observed"] + s["n_interpolated"],
           color=STATUS_COLOR["missing"], label="still missing")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i:03d}" for i in s["participant"]],
                       rotation=90, fontsize=7)
    ax.set_xlabel("subject", fontsize=9)
    ax.set_ylabel("check-ins", fontsize=9)
    ax.set_title("Series length and completeness per subject\n"
                 "(length reflects time enrolled - collection is ongoing)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); sns.despine(ax=ax)

    ax = axes[1]
    if len(gaps):
        vc = gaps["gap_slots"].value_counts().sort_index()
        ax.bar(vc.index, vc.values, color="#546e7a")
        ax.set_xlabel("gap length (consecutive missed check-ins)", fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.set_title(f"Gap-length distribution\n"
                     f"(<={MAX_GAP_LINEAR} linear, <={MAX_GAP_CUBIC} cubic, "
                     f"longer left as NaN)", fontsize=11, fontweight="bold")
        for xx, yy in zip(vc.index, vc.values):
            ax.text(xx, yy, str(yy), ha="center", va="bottom", fontsize=7)
    else:
        ax.text(.5, .5, "no gaps", ha="center"); ax.axis("off")
    sns.despine(ax=ax)

    fig.tight_layout()
    p = Path(out_dir) / "completeness_group_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {p}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    df, emo = load_and_concat(RAW_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    rows, all_gaps = [], []
    for pid in sorted(df["participant"].unique()):
        g, gaps, n_dup, n_partial = build_grid(df, emo, pid)
        if n_dup:
            print(f"  P{pid:03d}: {n_dup} duplicate (day, slot) submission(s) "
                  f"- kept the latest")
        if n_partial:
            print(f"  P{pid:03d}: *** {n_partial} checkpoint(s) have SOME "
                  f"emotions present and some absent - the all-or-nothing "
                  f"assumption does not hold here, check these rows ***")
        vc = g["status"].value_counts()
        n_obs = int(vc.get("observed", 0))
        n_int = int(vc.get("interpolated", 0))
        n_mis = int(vc.get("missing", 0))
        rows.append(dict(
            participant=pid,
            first_day=min(g["sleep_day"]), last_day=max(g["sleep_day"]),
            n_days=g["sleep_day"].nunique(), n_expected=len(g),
            n_observed=n_obs, n_interpolated=n_int, n_still_missing=n_mis,
            pct_observed=round(100 * n_obs / len(g), 1),
            pct_usable=round(100 * (n_obs + n_int) / len(g), 1),
            longest_gap=int(gaps["gap_slots"].max()) if len(gaps) else 0,
            n_gaps=len(gaps), n_partial_checkpoints=n_partial))
        all_gaps.append(gaps)
        plot_subject(g, pid, emo, fig_dir)

    summary = pd.DataFrame(rows)
    gaps = pd.concat(all_gaps, ignore_index=True) if all_gaps else pd.DataFrame()
    summary.to_csv(OUT_DIR / "completeness_per_subject.csv", index=False)
    gaps.to_csv(OUT_DIR / "gaps_detail.csv", index=False)
    plot_group(summary, gaps, fig_dir)

    print("\n" + "=" * 72)
    print(summary[["participant", "n_days", "n_expected", "n_observed",
                   "n_still_missing", "pct_observed", "longest_gap"]]
          .to_string(index=False))
    print("=" * 72)
    print(f"\nSeries length spans {summary['n_expected'].min()} to "
          f"{summary['n_expected'].max()} check-ins "
          f"({summary['n_expected'].min()/N_SLOTS_PER_DAY:.0f} to "
          f"{summary['n_expected'].max()/N_SLOTS_PER_DAY:.0f} days).")
    if len(gaps):
        print("Gaps by interpolation method the policy would apply:")
        print(gaps["method"].value_counts().to_string())
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
