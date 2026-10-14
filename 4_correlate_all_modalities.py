# -*- coding: utf-8 -*-
"""
correlate_all_modalities.py
===========================
Exploratory correlations between three "person-level" variable families
and three "task-level" families from the MEG-session VAS ratings.

PREDICTOR side (person-level, one value per subject):
  A  quest    - questionnaire subscale/total scores
                <- quest/analysis/questionnaire_scores_wide.csv
  B  plutmean - mean Plutchik EMA rating per emotion (the "emotion score")
                <- Pluchik/analysis/plutchik_slots_interpolated.csv
  C  plutdyn  - Plutchik EMA dynamic measures, per emotion x measure
                <- Pluchik/analysis/plutchik_dynamic_measures.csv

OUTCOME side (task-level, from the MEG-session VAS):
  D  vasdyn   - dynamic measures of the WITHIN-REST VAS series,
                per block_role (base/sad/fear) x emotion (Sadness/Fear)
                <- Behav/analysis/dyn{NNN}_vas_analysis.csv
  E  vasmean  - mean within-rest VAS rating per block_role x emotion
                <- Behav/analysis/behav_per_block/*.csv
  F  vaspost  - the single post-video VAS rating per video x emotion
                <- Behav/raw/dyn{NNN}_results.csv, context == post_video

--------------------------------------------------------------------
WHY THERE ARE NO POST-VIDEO *DYNAMICS*
--------------------------------------------------------------------
analyze_behav_vas.py deliberately drops post_video rows and analyses
only within_rest series: each video is followed by a single rating per
emotion, so there is no timeseries to run a battery on. Those single
values are still informative as scalars, so they are carried here as
family F (4 variables: 2 videos x 2 emotions) - a value, not a battery.
Set INCLUDE_POST_VIDEO = False to drop them.

The "post-video VAS dynamics" in the original request therefore map
onto family D: the dynamics of the rest-period ratings, which is what
the 20-minute rest blocks actually provide.

--------------------------------------------------------------------
SUBJECT IDs
--------------------------------------------------------------------
The three sources number subjects differently:
  - Plutchik : the in-survey experiment-number field (e.g. 7, 13, 22)
  - quest    : the filename prefix (e.g. 001_Pysch_quest -> 1)
  - VAS      : the subject number in dyn{NNN}_results.csv (e.g. 019 -> 19)
All are coerced to int and joined on that. The script PRINTS the
overlap and every ID present in one source but missing from another,
rather than silently inner-joining - check that report before reading
any result. If the numbering does NOT correspond across sources, supply
ID_MAP below.

--------------------------------------------------------------------
STATISTICS
--------------------------------------------------------------------
No multiple-comparison correction, per request. The number of tests is
large (all 52 emotions are retained), so the script reports the
observed vs chance-expected count of p<.05 for every family pairing.
A yield at or near 1.0x chance means the block is null and individual
cells in it should not be interpreted. Spearman by default: with small
n a single subject dominates a Pearson r, and several measures (ACW
lags, frequencies) are not plausibly linear-normal.

Usage:
    python3 correlate_all_modalities.py
"""

import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ===================================================================
# CONFIG
# ===================================================================

BASE = Path("/home/ZhangJianfeng_04/emo_study/Behav")

QUEST_WIDE   = BASE / "quest/analysis/questionnaire_scores_wide.csv"
PLUT_SLOTS   = BASE / "Pluchik/analysis/plutchik_slots_interpolated.csv"
PLUT_DYN     = BASE / "Pluchik/analysis/plutchik_dynamic_measures.csv"
VAS_ANALYSIS = BASE / "analysis"                 # dyn{NNN}_vas_analysis.csv
VAS_PERBLOCK = BASE / "analysis/behav_per_block"
VAS_RAW      = BASE / "raw"                      # dyn{NNN}_results.csv

OUT_DIR = BASE / "analysis/corr_all_modalities"

# Both correlation methods are run in a single pass, each into its own
# subfolder of OUT_DIR. They answer slightly different questions:
#   spearman - rank-based; robust to outliers and to non-linear but
#              monotonic relationships. The safer default at small n,
#              where one subject can otherwise dominate an r.
#   pearson  - linear; more powerful IF the relationship really is
#              linear and roughly normal, but a single extreme subject
#              can carry it entirely.
# Where the two disagree sharply for a given cell, that disagreement is
# itself the finding: it usually means an outlier or a non-linear shape.
METHODS = ("spearman", "pearson")
METHOD = METHODS[0]           # rebound per method by main(); do not edit
PLUT_LENGTH_MODE = "full"     # each subject's FULL series. common_n would
                              # truncate everyone to the shortest subject,
                              # which with ongoing enrolment discards months
                              # of data from early subjects. Set to
                              # "common_n" for the length-controlled check.
MIN_N = 5                     # minimum complete pairs to COMPUTE a correlation
# Spearman saturates at exactly |r|=1 whenever a handful of pairs happen to
# rank monotonically: at n=5 that is ~1.7% of tests, i.e. ~766 spurious
# "perfect" correlations in 46,000. Ranking by |r| therefore fills the
# top-hits figure with the least trustworthy pairs. MIN_N_RANK is the
# separate, higher floor used for RANKING and for the top-scatters figure.
MIN_N_RANK = 10

# Dynamic measures excluded from the correlation figures (still present in
# the CSV). Distribution-shape and raw-count measures, per request.
EXCLUDE_VAS_MEASURES = {"kurtosis", "skewness", "frac_nonzero", "n_nonzero",
                        "range", "max", "min", "LZC_raw"}

# Condition x rating column order: conditions blocked, with the two ratings
# adjacent inside each block, so the figures read left-to-right by condition.
COND_ORDER = ["base", "sad", "fear"]
RATING_ORDER = ["Sadness", "Fear"]

# Plain-language names for the outcome families, used in figure titles -
# "vasdyn"/"vasraw" mean nothing to a reader.
# Measure-name abbreviations used in the variable names, so the columns
# read as base_m_Sadness / base_sd_Sadness / base_LZC_Sadness rather than
# the opaque "base|Sadness|mean".
MEASURE_ABBREV = {"mean": "m", "std": "sd"}

# Short measure names used in the VAS variable labels: the block mean and
# SD get compact abbreviations, everything else keeps its battery name.
# Result reads e.g. base_m_Sadness, base_sd_Sadness, sad_LZC_Fear.
MEASURE_ABBREV = {"mean": "m", "std": "sd"}


def vas_varname(cond, rating, measure=None):
    """{condition}_{measure}_{rating}, e.g. base_m_Sadness."""
    if measure is None:
        return f"{cond}_{rating}"
    return f"{cond}_{MEASURE_ABBREV.get(measure, measure)}_{rating}"


# Short tags used in FILENAMES, so a figure can be identified from its
# name alone without opening it. The internal family keys (vasdyn,
# vasraw, plutdyn...) say nothing to a reader.
FAMILY_TAG = {
    "quest":    "quest",
    "plutmean": "EMAmean",     # mean Plutchik EMA rating per emotion
    "plutdyn":  "EMAdyn",      # dynamic measures of the EMA timeseries
    "vasdyn":   "VASdyn",      # dynamics of the within-block VAS series
    "vasmean":  "VASblockmean",  # mean within-block VAS rating
    "vasraw":   "VASsingle",   # single pre-/post-video ratings
}


FAMILY_LABEL = {
    "vasdyn":  "within-block VAS",
    "vasmean": "within-block VAS mean",
    "vasraw":  "single VAS ratings",
}
INCLUDE_POST_VIDEO = True

# Optional explicit cross-source ID mapping, e.g.
#   ID_MAP = {"plutchik": {7: 107}, "quest": {}, "vas": {}}
ID_MAP = None

PRIMARY_EMOTIONS = ["joy", "trust", "fear", "sadness",
                    "disgust", "anger", "anticipation", "surprise"]

# plotting: how many predictor rows per figure page
# --- heatmap geometry -------------------------------------------------
# Cells are drawn SQUARE (ax.set_aspect("equal")). The number of columns
# per page is derived from CELL_IN rather than fixed, so cell size - and
# therefore label legibility - stays constant no matter how many rows a
# figure has. Bigger CELL_IN = bigger, more readable cells but more pages.
CELL_IN = 0.46          # target square cell size, inches
FIG_W = 13.33           # PPT widescreen width
FIG_H_MAX = 7.5         # PPT widescreen height
_Y_LABEL_IN = 2.8       # reserved for y tick labels
_CBAR_IN = 1.1          # reserved for the colourbar
_X_LABEL_IN = 2.3       # reserved for rotated x tick labels
_TITLE_IN = 0.8

# Which predictor families appear in the PER-MEASURE figures. All
# families always go into the CSV and into the family-level figures;
# this only controls the by-measure set, whose page count is the
# product of (predictors) x (measures).
#
# With every family included that is 75 + 52 + 697 = 824 predictors,
# i.e. ~42 pages per measure and several hundred images - too many to
# actually look at. quest + plutmean keeps the questionnaire scores and
# all 52 emotion means (the comparison usually wanted) at a readable
# size. Add "plutdyn" here if you want the 697 emotion x measure
# variables in these figures too.
PER_MEASURE_FAMILIES = ("quest", "plutmean")

# In the per-measure figures the Plutchik DYNAMICS are included too, but
# only the MATCHING measure: the ACW0 figure pairs ACW0-of-emotion with
# ACW0-of-VAS, the LZC figure pairs LZC with LZC, and so on. Correlating
# ACW0-of-VAS against, say, SampEn-of-emotion is not a comparison anyone
# reads off these plots, and including every combination multiplies the
# page count for no gain. Set False to leave the dynamics out entirely.
MATCHED_MEASURE_DYNAMICS = True

# font sizes
FS_TITLE = 16
FS_XTICK = 11
FS_YTICK = 13
FS_ANNOT = 10
FS_CBAR = 11


# ===================================================================
# LOADERS -> each returns a wide frame indexed by subject (int)
# ===================================================================

def _apply_id_map(idx, source):
    if not ID_MAP or source not in ID_MAP:
        return idx
    m = ID_MAP[source]
    return pd.Index([m.get(i, i) for i in idx], name="subject")


def load_quest():
    if not QUEST_WIDE.exists():
        print(f"  [skip] quest: {QUEST_WIDE} not found")
        return pd.DataFrame()
    q = pd.read_csv(QUEST_WIDE)
    q = q.rename(columns={q.columns[0]: "subject"}).set_index("subject")
    q.index = _apply_id_map(q.index.astype(int), "quest")
    q.columns = [f"quest|{c}" for c in q.columns]
    print(f"  quest    : {q.shape[1]:4d} vars x {q.shape[0]} subjects")
    return q


def load_plutchik_means():
    """Mean EMA rating per emotion per subject. Uses the interpolated
    series with genuinely-missing slots excluded, so a subject with more
    missingness is not pulled toward interpolated values."""
    if not PLUT_SLOTS.exists():
        print(f"  [skip] plutmean: {PLUT_SLOTS} not found")
        return pd.DataFrame()
    s = pd.read_csv(PLUT_SLOTS)
    s = s[~s["still_missing"]]
    m = (s.groupby(["participant", "emotion"])["rating"].mean()
          .unstack("emotion"))
    m.index = _apply_id_map(m.index.astype(int), "plutchik")
    m.index.name = "subject"
    m.columns = [f"plutmean|{c}" for c in m.columns]
    print(f"  plutmean : {m.shape[1]:4d} vars x {m.shape[0]} subjects")
    return m


def load_plutchik_dynamics():
    if not PLUT_DYN.exists():
        print(f"  [skip] plutdyn: {PLUT_DYN} not found")
        return pd.DataFrame()
    d = pd.read_csv(PLUT_DYN)
    if "length_mode" in d.columns:
        d = d[d["length_mode"] == PLUT_LENGTH_MODE]
    d["value_num"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value_num"])
    w = d.pivot_table(index="participant", columns=["emotion", "measure"],
                      values="value_num")
    w.columns = [f"plutdyn|{e}|{m}" for e, m in w.columns]
    w.index = _apply_id_map(w.index.astype(int), "plutchik")
    w.index.name = "subject"
    print(f"  plutdyn  : {w.shape[1]:4d} vars x {w.shape[0]} subjects "
          f"({PLUT_LENGTH_MODE})")
    return w


def load_vas_dynamics():
    """Long-format per-subject dyn{NNN}_vas_analysis.csv ->
    one column per block_role x emotion x measure."""
    files = sorted(VAS_ANALYSIS.glob("dyn*_vas_analysis.csv"))
    if not files:
        print(f"  [skip] vasdyn: no dyn*_vas_analysis.csv in {VAS_ANALYSIS}")
        return pd.DataFrame()
    frames = []
    for f in files:
        m = re.search(r"dyn(\d+)_vas_analysis", f.name)
        if not m:
            continue
        d = pd.read_csv(f)
        d["subject"] = int(m.group(1))
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["value_num"] = pd.to_numeric(d["value"], errors="coerce")
    n_flat = int((d["value"].astype(str) == "flat").sum())
    d = d.dropna(subset=["value_num"])
    w = d.pivot_table(index="subject",
                      columns=["block_role", "emotion", "measure"],
                      values="value_num")
    w.columns = [f"vasdyn|{vas_varname(b, e, m)}"
                 for b, e, m in w.columns]
    w.index = _apply_id_map(w.index.astype(int), "vas")
    print(f"  vasdyn   : {w.shape[1]:4d} vars x {w.shape[0]} subjects "
          f"({len(files)} files, {n_flat} 'flat' cells dropped)")
    return w


def load_vas_block_means(vasdyn=None):
    """Mean WITHIN-REST rating per block_role x emotion, from the
    behav_per_block audit files.

    This is the SAME quantity the battery already reports as its 'mean'
    measure (both are the mean of the same within_rest series), so it is
    used only to CROSS-CHECK that the two sources agree, and is then
    dropped rather than correlated twice. If they disagree the mismatch is
    printed - that would mean behav_per_block and dyn*_vas_analysis.csv
    were produced from different data, which matters well beyond here.
    """
    files = sorted(VAS_PERBLOCK.glob("*.csv"))
    if not files:
        print(f"  [skip] vasmean cross-check: no CSVs in {VAS_PERBLOCK}")
        return pd.DataFrame()
    frames = []
    for f in files:
        m = re.search(r"(\d+)", f.name)
        if not m:
            continue
        d = pd.read_csv(f)
        d["subject"] = int(m.group(1))
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    rating_col = _find_col(d, ["rating", "response", "vas", "value"])
    if rating_col is None:
        print(f"  [skip] vasmean cross-check: no rating column found")
        return pd.DataFrame()
    d[rating_col] = pd.to_numeric(d[rating_col], errors="coerce")
    w = (d.dropna(subset=[rating_col])
          .pivot_table(index="subject", columns=["block_role", "emotion"],
                       values=rating_col, aggfunc="mean"))
    w.columns = [f"{b}_m_{e}" for b, e in w.columns]
    w.index = _apply_id_map(w.index.astype(int), "vas")

    if vasdyn is not None and len(vasdyn):
        shared = [c for c in w.columns if f"vasdyn|{c}" in vasdyn.columns]
        if shared:
            diffs = []
            for c in shared:
                a = w[c]
                b = vasdyn[f"vasdyn|{c}"]
                j = pd.concat([a, b], axis=1).dropna()
                if len(j):
                    diffs.append((c, float((j.iloc[:, 0] - j.iloc[:, 1])
                                           .abs().max())))
            worst = max((d for _, d in diffs), default=0.0)
            if worst < 1e-6:
                print(f"  vasmean cross-check: block means match the "
                      f"battery's 'mean' exactly ({len(shared)} variables) "
                      f"- dropped as duplicate.")
            else:
                print(f"  *** vasmean cross-check: block means DIFFER from "
                      f"the battery's 'mean' (max abs diff {worst:.4g}). ***")
                print(f"  behav_per_block and dyn*_vas_analysis.csv appear to "
                      f"come from different data - investigate before "
                      f"trusting either.")
                for c, dd in sorted(diffs, key=lambda x: -x[1])[:6]:
                    print(f"      {c}: max abs diff {dd:.4g}")
    return pd.DataFrame()   # never correlated - duplicate of vasdyn mean


# -------------------------------------------------------------------
# VIDEO ORDER  (which induction video came first, per subject)
# -------------------------------------------------------------------
# Two independent sources:
#   (a) the MEG trigger stream - 7 = video_sad_onset, 8 = video_fear_onset;
#       whichever fires first is video 1. Authoritative.
#   (b) the odd/even counterbalancing rule on subject number.
# (a) is preferred where a trigger file exists, because parity breaks on
# aborted or re-run sessions. Any disagreement is printed loudly.
#
# PARITY EXCEPTION: subject 000 is treated as ODD (sad video first), like
# 001. In Python 0 % 2 == 0, so a plain parity test would send 000 down
# the even branch and silently swap its sad/fear labels.

TRIGGER_DIR = BASE / "MEG_trigger_files"
TRIG_VIDEO_SAD, TRIG_VIDEO_FEAR = 7, 8


def _parity_order(subject):
    """Counterbalancing fallback. Returns {'sad': idx, 'fear': idx}."""
    is_odd = (subject % 2 == 1) or (subject == 0)   # 000 treated as odd
    return ({"sad": 1, "fear": 2} if is_odd else {"sad": 2, "fear": 1})


def _trigger_order(subject):
    """Read the actual video order out of the trigger stream, or None."""
    cands = list(TRIGGER_DIR.glob(f"{subject:03d}_*triggers*.csv"))
    if not cands:
        return None
    d = pd.read_csv(cands[0])
    tcol = _find_col(d, ["trigger", "code", "value"])
    timecol = _find_col(d, ["time", "onset", "sample"])
    if tcol is None or timecol is None:
        return None
    d[tcol] = pd.to_numeric(d[tcol], errors="coerce")
    d[timecol] = pd.to_numeric(d[timecol], errors="coerce")
    sad = d.loc[d[tcol] == TRIG_VIDEO_SAD, timecol].min()
    fear = d.loc[d[tcol] == TRIG_VIDEO_FEAR, timecol].min()
    if pd.isna(sad) or pd.isna(fear):
        return None
    return ({"sad": 1, "fear": 2} if sad < fear else {"sad": 2, "fear": 1})


def build_video_order(subjects):
    """Per-subject video order, trigger-derived where possible."""
    rows, mismatches = [], []
    for s in subjects:
        par = _parity_order(s)
        trig = _trigger_order(s)
        used, src_name = (trig, "trigger") if trig else (par, "parity")
        if trig and trig != par:
            mismatches.append((s, par, trig))
        rows.append(dict(subject=s, sad_video_order=used["sad"],
                         fear_video_order=used["fear"], order_source=src_name))
    vo = pd.DataFrame(rows).set_index("subject")
    n_trig = int((vo["order_source"] == "trigger").sum())
    print(f"  video order: {n_trig}/{len(vo)} from triggers, "
          f"{len(vo)-n_trig} from parity (000 treated as odd)")
    if mismatches:
        print(f"  *** WARNING: trigger vs parity DISAGREE for "
              f"{len(mismatches)} subject(s). Trigger order used. This "
              f"implies existing block_role labels may be wrong: ***")
        for s, p, t in mismatches:
            print(f"      subject {s:03d}: parity says {p}, triggers say {t}")
    return vo


def load_vas_single_ratings(video_order):
    """
    Six raw single ratings per subject:
      pre_video_{Sadness,Fear}   - LAST within_rest rating of the base
                                   block, i.e. the rating immediately
                                   before the first video starts
      post_sad_{Sadness,Fear}    - the rating right after the sad video
      post_fear_{Sadness,Fear}   - the rating right after the fear video

    Post-video ratings are keyed by WHICH VIDEO PRECEDED THEM, resolved
    through video_order - not by chronological index. Ordering them
    chronologically would mix post-sad and post-fear ratings across
    subjects, because the video order is counterbalanced.

    No dynamic measures are computed on these: each is a single value per
    emotion, so there is no series to run a battery on.
    """
    files = sorted(VAS_RAW.glob("dyn*_results.csv"))
    if not files:
        print(f"  [skip] vas single ratings: no dyn*_results.csv in {VAS_RAW}")
        return pd.DataFrame()

    rows = {}
    for f in files:
        m = re.search(r"dyn(\d+)_results", f.name)
        if not m:
            continue
        subj = int(m.group(1))
        d = pd.read_csv(f)
        ctx = _find_col(d, ["context"])
        emo_col = _find_col(d, ["emotion"])
        rating_col = _find_col(d, ["rating", "response", "vas", "value"])
        if ctx is None or emo_col is None or rating_col is None:
            print(f"  [skip] {f.name}: missing context/emotion/rating column")
            continue
        d[rating_col] = pd.to_numeric(d[rating_col], errors="coerce")
        d["_row"] = np.arange(len(d))
        ctx_s = d[ctx].astype(str).str.strip()
        rec = {}

        # ---- pre-video: last within_rest rating BEFORE the first video ----
        post_rows = d.loc[ctx_s == "post_video", "_row"]
        if len(post_rows):
            first_video_row = post_rows.min()
            pre = d[(ctx_s == "within_rest") & (d["_row"] < first_video_row)]
            for e, g in pre.groupby(emo_col):
                g = g.dropna(subset=[rating_col])
                if len(g):
                    rec[f"pre_video_{e}"] = float(
                        g.sort_values("_row")[rating_col].iloc[-1])

        # ---- post-video: keyed by which video preceded ----
        pv = d[ctx_s == "post_video"].copy()
        if len(pv) and subj in video_order.index:
            pv = pv.sort_values("_row")
            pv["event_idx"] = pv.groupby(emo_col).cumcount() + 1
            idx2cond = {video_order.loc[subj, "sad_video_order"]: "sad",
                        video_order.loc[subj, "fear_video_order"]: "fear"}
            for _, r in pv.iterrows():
                cond = idx2cond.get(int(r["event_idx"]))
                if cond and pd.notna(r[rating_col]):
                    rec[f"post_{cond}_{r[emo_col]}"] = float(r[rating_col])
        rows[subj] = rec

    if not rows:
        return pd.DataFrame()
    w = pd.DataFrame.from_dict(rows, orient="index")
    w.index = _apply_id_map(w.index.astype(int), "vas")
    w.index.name = "subject"
    w = w.reindex(sorted(w.columns), axis=1)
    w.columns = [f"vasraw|{c}" for c in w.columns]
    print(f"  vasraw   : {w.shape[1]:4d} vars x {w.shape[0]} subjects "
          f"(single ratings; no dynamics possible)")
    return w


def _find_col(df, candidates):
    low = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for cand in candidates:
        for lc, orig in low.items():
            if cand in lc:
                return orig
    return None


# ===================================================================
# ID RECONCILIATION
# ===================================================================

def report_id_overlap(families):
    """Never silently inner-join: show what each source has."""
    print("\n--- subject ID reconciliation ---")
    sets = {name: set(df.index) for name, df in families.items() if len(df)}
    for name, s in sets.items():
        print(f"  {name:9s}: n={len(s):3d}  {sorted(s)}")
    if not sets:
        raise RuntimeError("No input families could be loaded.")
    common = set.intersection(*sets.values())
    print(f"\n  IN ALL SOURCES: n={len(common)} -> {sorted(common)}")
    for name, s in sets.items():
        missing = s - common
        if missing:
            print(f"  in {name} but not in all: {sorted(missing)}")
    if len(common) < MIN_N:
        print(f"\n  WARNING: only {len(common)} subject(s) present in every "
              f"source. Correlations at this n are descriptive only - a "
              f"single subject moves any r substantially. If this number "
              f"looks too low, the ID numbering probably does not match "
              f"across sources; set ID_MAP in the config.")
    return sorted(common)


# ===================================================================
# CORRELATION
# ===================================================================

CONDITIONS = ("base", "sad", "fear")


def condition_of(out_family, outcome):
    """Which experimental condition an outcome variable belongs to.

    The three outcome families encode it differently:
      vasdyn  : "base|Sadness|LZC"      -> first token is the block_role
      vasmean : "base|Sadness"          -> first token is the block_role
      vasraw  : "pre_video_Sadness"     -> the pre-video rating is taken in
                                           the base block, so it maps to
                                           'base'
                "post_sad_Sadness"      -> 'sad'   (rating after sad video)
                "post_fear_Sadness"     -> 'fear'  (rating after fear video)

    Anything unrecognised returns 'other' rather than being forced into a
    condition, so it stays visible instead of silently landing in the
    wrong figure.
    """
    if out_family in ("vasdyn", "vasmean"):
        head = outcome.split("|")[0].strip().lower()
        return head if head in CONDITIONS else "other"
    if out_family == "vasraw":
        low = outcome.lower()
        if low.startswith("pre_video"):
            return "base"
        if low.startswith("post_sad"):
            return "sad"
        if low.startswith("post_fear"):
            return "fear"
    return "other"


def predictor_label(pred_family, predictor):
    """Axis label that makes the variable's kind unambiguous.

    A column labelled just 'grief' (the mean EMA rating) and one labelled
    'grief|ACW0_lags' (a dynamic measure of the same emotion) are very
    different quantities, and nothing on the axis said which was which.

        plutmean -> mean:grief
        plutdyn  -> ACW0_lags:grief
        quest    -> unchanged (already unambiguous)
    """
    if pred_family == "plutmean":
        return f"mean:{predictor}"
    if pred_family == "plutdyn" and "|" in predictor:
        emo, meas = predictor.split("|", 1)
        return f"{meas}:{emo}"
    return predictor


def drop_excluded_predictor_measures(P):
    """EXCLUDE_VAS_MEASURES was only filtering the VAS outcome side. The
    same measures also appear as PREDICTORS (plutdyn|{emotion}|{measure}),
    so kurtosis/range/max/... were still showing up as rows in the
    correlation figures. Filter them there too, for consistency."""
    drop = []
    for c in P.columns:
        if not c.startswith("plutdyn|"):
            continue
        parts = c.split("|")
        if len(parts) >= 3 and parts[2] in EXCLUDE_VAS_MEASURES:
            drop.append(c)
    if drop:
        print(f"  dropped {len(drop)} plutdyn predictor(s) whose measure is "
              f"in EXCLUDE_VAS_MEASURES (kurtosis/range/max/...)")
    return P.drop(columns=drop)


def correlate_families(pred, outc, subjects):
    from scipy import stats
    fn = stats.spearmanr if METHOD == "spearman" else stats.pearsonr

    P = pred.reindex(subjects)
    O = outc.reindex(subjects)
    rows = []
    for pc in P.columns:
        pv = P[pc]
        if pv.notna().sum() < MIN_N or pv.std(skipna=True) in (0, np.nan):
            continue
        for oc in O.columns:
            pair = pd.DataFrame({"p": pv, "o": O[oc]}).dropna()
            if len(pair) < MIN_N:
                continue
            if pair["p"].std() == 0 or pair["o"].std() == 0:
                continue
            r, p = fn(pair["p"], pair["o"])
            if not np.isfinite(r):
                continue
            pfam, pname = pc.split("|", 1)
            ofam, oname = oc.split("|", 1)
            rows.append(dict(pred_family=pfam, predictor=pname,
                             out_family=ofam, outcome=oname,
                             condition=condition_of(ofam, oname),
                             r=float(r), p=float(p), n=len(pair)))
    return pd.DataFrame(rows)


def compliance_covariates():
    """Two per-subject compliance covariates, correlated against every
    questionnaire score as a diagnostic:

      series_length_slots  - how many check-ins the subject's grid spans.
                             In an ONGOING study this mostly reflects how
                             long they have been enrolled, so it should be
                             unrelated to trait scores. If it is not, any
                             length-sensitive dynamic measure can correlate
                             with a scale through enrolment rather than
                             through emotion dynamics.
      frac_interpolated    - proportion of the series that was filled in.
                             Interpolated stretches are smooth by
                             construction, which DEFLATES LZC and SampEn
                             and INFLATES autocorrelation. This fraction
                             varies several-fold across subjects and tracks
                             compliance, so it biases those measures
                             per-subject rather than uniformly.

    Both are diagnostics, not corrections: a strong relationship here means
    the corresponding measures need care, not that the script fixes them.
    """
    if not PLUT_SLOTS.exists():
        return pd.DataFrame()
    s = pd.read_csv(PLUT_SLOTS)
    one = s[s["emotion"] == s["emotion"].iloc[0]]   # missingness is per-slot
    g = one.groupby("participant").agg(
        grid_span_slots=("seq", "size"),          # first-to-last day, incl. gaps
        n_interpolated=("was_interpolated", "sum"),
        n_still_missing=("still_missing", "sum"))
    # two different senses of "length", both worth checking: how long the
    # subject was enrolled, and how many points actually feed the battery
    g["n_valid_points"] = g["grid_span_slots"] - g["n_still_missing"]
    g["frac_interpolated"] = g["n_interpolated"] / g["grid_span_slots"]
    g["frac_observed"] = 1 - (g["n_interpolated"] + g["n_still_missing"]) \
        / g["grid_span_slots"]
    g.index = _apply_id_map(g.index.astype(int), "plutchik")
    g.index.name = "subject"
    return g[["grid_span_slots", "n_valid_points",
              "frac_interpolated", "frac_observed"]]


def report_compliance_confound(quest, subjects, out_dir):
    """Correlate the compliance covariates against every questionnaire
    score, so the confound is measured rather than assumed."""
    from scipy import stats
    cov = compliance_covariates()
    if cov.empty or quest.empty:
        return pd.DataFrame()
    fn = stats.spearmanr if METHOD == "spearman" else stats.pearsonr
    rows = []
    for cc in cov.columns:
        for qc in quest.columns:
            pair = pd.DataFrame({"c": cov[cc], "q": quest[qc]}).reindex(
                subjects).dropna()
            if len(pair) < MIN_N or pair["c"].std() == 0 or pair["q"].std() == 0:
                continue
            r, p = fn(pair["c"], pair["q"])
            rows.append(dict(covariate=cc, quest_score=qc.replace("quest|", ""),
                             r=float(r), p=float(p), n=len(pair)))
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res.to_csv(Path(out_dir) / "compliance_vs_questionnaires.csv", index=False)
    print("\n--- compliance covariates vs questionnaire scores ---")
    for cc, g in res.groupby("covariate"):
        n_sig = int((g["p"] < .05).sum())
        strongest = g.reindex(g["r"].abs().sort_values(ascending=False).index).head(3)
        print(f"  {cc}: {n_sig}/{len(g)} at p<.05 "
              f"(chance ~{0.05*len(g):.0f}); strongest:")
        for _, r in strongest.iterrows():
            print(f"      {r.quest_score:<34s} r={r.r:+.2f} p={r.p:.3f} n={int(r.n)}")
    print("  If these are at chance, enrolment length and interpolation are")
    print("  unrelated to trait scores and the confound is not operating.")
    return res


def report_n_coverage(res):
    """How many complete pairs actually went into each correlation. The
    header n on a heatmap is the range across its cells, not a single
    value: variables added to the battery partway through (EAQ, TSQ) have
    far fewer complete cases than the rest, and those small-n pairs are
    exactly the ones that produce spurious |r|=1."""
    print("\n--- complete-pair count (n) across correlations ---")
    q = res["n"].describe(percentiles=[.05, .25, .5, .75, .95])
    print(f"  min={int(q['min'])}  p05={int(q['5%'])}  median={int(q['50%'])}  "
          f"p95={int(q['95%'])}  max={int(q['max'])}")
    low = res[res["n"] < MIN_N_RANK]
    if len(low):
        print(f"  {len(low)} of {len(res)} correlations ({100*len(low)/len(res):.1f}%) "
              f"have n < {MIN_N_RANK} and are EXCLUDED from ranking.")
        worst = (low.groupby("predictor")["n"].agg(["size", "max"])
                 .sort_values("size", ascending=False).head(10))
        print(f"  predictors contributing most low-n correlations:")
        print(worst.to_string())
    sat = res[res["r"].abs() >= 0.999]
    if len(sat):
        print(f"  {len(sat)} correlation(s) at |r|>=0.999 - of these, "
              f"{int((sat['n'] < MIN_N_RANK).sum())} have n < {MIN_N_RANK} "
              f"(rank saturation, not signal).")
    return res["n"].describe()


def report_yield(res):
    print("\n--- p<.05 yield vs chance, per family pairing ---")
    print("  (a ratio near 1.0 means that block is indistinguishable from "
          "noise;\n   do not pick individual cells out of such a block)")
    out = []
    for (pf, of, cond), g in res.groupby(["pred_family", "out_family",
                                          "condition"]):
        n = len(g)
        sig = int((g["p"] < .05).sum())
        exp = 0.05 * n
        ratio = sig / exp if exp else np.nan
        out.append(dict(pred_family=pf, out_family=of, condition=cond,
                        n_tests=n,
                        n_sig=sig, n_expected=round(exp, 1),
                        ratio_vs_chance=round(ratio, 2)))
    t = pd.DataFrame(out).sort_values("ratio_vs_chance", ascending=False)
    print(t.to_string(index=False))
    return t


# ===================================================================
# PLOTTING
# ===================================================================

def _stars(pmat):
    """Significance markers as a plain ndarray.

    Avoids DataFrame.map, which only exists in pandas >= 2.1 (it was
    .applymap before that) - the cluster env raised AttributeError here.
    np.where works on every version.
    """
    return np.where(pmat.notna().to_numpy() & (pmat.to_numpy() < .05), "*", "")


VAS_NAME_RE = re.compile(r"^(base|sad|fear)_(.+)_(Sadness|Fear)$")


def parse_vas_name(name):
    """'base_m_Sadness' -> ('base', 'm', 'Sadness'). The measure part may
    itself contain underscores (SampEn_m2, ACF_lag1, ACW0_lags), so the
    condition is taken from the front and the rating from the end."""
    m = VAS_NAME_RE.match(name)
    return (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)


def _blocked_columns(g, out_family):
    """Column order: conditions blocked (base, sad, fear), the two ratings
    adjacent inside each block. Returns (ordered_cols, block_boundaries)."""
    outcomes = list(g["outcome"].unique())
    cols, bounds = [], []
    if out_family == "vasraw":
        groups = [("pre_video", "pre-video"),
                  ("post_sad", "post-sad"),
                  ("post_fear", "post-fear")]
        for prefix, _ in groups:
            block = []
            for rating in RATING_ORDER:
                for oc in outcomes:
                    if oc.startswith(prefix) and oc.endswith(rating):
                        block.append(oc)
            if block:
                cols.extend(block)
                bounds.append(len(cols))
    else:
        for cond in COND_ORDER:
            block = []
            for rating in RATING_ORDER:
                for oc in outcomes:
                    c, _, r = parse_vas_name(oc)
                    if c == cond and r == rating:
                        block.append(oc)
            if block:
                cols.extend(block)
                bounds.append(len(cols))
    return cols, bounds[:-1], []


def _draw_heatmap_pages(mat, pmat, title, stem, out_dir,
                        y_block_bounds=None, cols_per_page=None,
                        nmat=None):
    """
    Render one correlation matrix across as many PPT-sized pages as it
    takes to stay legible.

    ORIENTATION RULE: the factor with MORE variables goes on the X axis,
    the factor with FEWER goes on Y. `mat` arrives with predictors as
    rows; it is transposed here whenever that leaves more variables on
    the short axis. Paging always runs along X, so each page keeps the
    same small set of Y labels for comparison.

    GEOMETRY: cells are forced square and sized from CELL_IN, and the
    figure height follows the number of rows instead of being pinned to
    the full slide height. Previously the height was fixed, so a figure
    with few rows stretched each cell into a tall thin sliver and wasted
    the space that the tick labels needed.

    y_block_bounds: indices on the OUTCOME axis to draw a separating rule
    after (the condition blocks).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if mat.shape[0] > mat.shape[1]:
        mat, pmat = mat.T, pmat.T
        if nmat is not None:
            nmat = nmat.T          # must follow mat, not be guessed from shape
        transposed = True
    else:
        transposed = False
    # `y_block_bounds` indexes the OUTCOME axis. Outcomes start as columns,
    # so the rules are vertical; after a transpose outcomes become rows and
    # the rules must be horizontal.
    ybounds = y_block_bounds if transposed else None
    xbounds = y_block_bounds if not transposed else None

    n_rows_total = mat.shape[0]
    plot_w = FIG_W - _Y_LABEL_IN - _CBAR_IN
    plot_h = FIG_H_MAX - _X_LABEL_IN - _TITLE_IN
    per_page = cols_per_page or max(6, int(plot_w / CELL_IN))
    rows_per_page = max(4, int(plot_h / CELL_IN))

    # per-cell N alongside the figure (written once, before page loop)
    if nmat is not None:
        try:
            nmat.reindex(index=mat.index, columns=mat.columns).to_csv(
                Path(out_dir) / f"{stem}_n.csv")
        except Exception as e:
            print(f"  [warn] could not write {stem}_n.csv: {e}")

    col_pages = max(1, int(np.ceil(mat.shape[1] / per_page)))
    row_pages = max(1, int(np.ceil(n_rows_total / rows_per_page)))
    total_pages = col_pages * row_pages

    for rp in range(row_pages):
        rsl = slice(rp * rows_per_page, (rp + 1) * rows_per_page)
        rmat, rpmat = mat.iloc[rsl], pmat.iloc[rsl]
        if rmat.shape[0] == 0:
            continue
        n_rows = rmat.shape[0]
        fig_h = min(FIG_H_MAX, n_rows * CELL_IN + _X_LABEL_IN + _TITLE_IN)
        page_ybounds = (
            [b - rp * rows_per_page for b in ybounds
             if rp * rows_per_page < b < (rp + 1) * rows_per_page]
            if ybounds else None)

        for pg in range(col_pages):
            sl = slice(pg * per_page, (pg + 1) * per_page)
            sub, psub = rmat.iloc[:, sl], rpmat.iloc[:, sl]
            if sub.shape[1] == 0:
                continue
            annot = _stars(psub)
            fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
            sns.heatmap(sub, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                        annot=annot, fmt="", annot_kws={"size": FS_ANNOT},
                        cbar_kws={"label": f"{METHOD} r", "shrink": .85},
                        linewidths=0.4, linecolor="white", square=True)
            if page_ybounds:
                for bnd in page_ybounds:
                    ax.axhline(bnd, color="#111111", linewidth=2.6)
            if xbounds:
                for bnd in xbounds:
                    ax.axvline(bnd, color="#111111", linewidth=2.6)
            tag = ""
            if total_pages > 1:
                tag = (f"   [rows {rp+1}/{row_pages}, cols {pg+1}/{col_pages}]"
                       if row_pages > 1 else f"   [page {pg+1}/{col_pages}]")
            import textwrap
            full_title = textwrap.fill(title + tag, width=64)
            ax.set_title(full_title, fontsize=FS_TITLE, fontweight="bold",
                         pad=14, loc="left")
            ax.tick_params(axis="x", labelrotation=90, labelsize=FS_XTICK)
            ax.tick_params(axis="y", labelrotation=0, labelsize=FS_YTICK)
            ax.set_xlabel(""); ax.set_ylabel("")
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=FS_CBAR)
            cbar.set_label(f"{METHOD} r", fontsize=FS_CBAR + 1)
            fig.tight_layout()
            suffix = (f"_r{rp+1}p{pg+1}" if row_pages > 1 else f"_p{pg+1}")
            p = Path(out_dir) / f"{stem}{suffix}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  Saved: {p}")


def plot_per_measure_heatmaps(res, out_dir):
    """
    One figure PER dynamic measure. Rows are the 6 condition x rating VAS
    outcomes (conditions blocked, ratings adjacent). Columns are the
    questionnaire scores, the mean EMA rating per emotion, and - when
    MATCHED_MEASURE_DYNAMICS is on - the SAME dynamic measure computed on
    each emotion's EMA timeseries.

    So the ACW0 figure holds ACW0-of-VAS against ACW0-of-emotion; the LZC
    figure holds LZC against LZC. Mismatched pairings are deliberately
    excluded.
    """
    d = res[res["out_family"] == "vasdyn"].copy()
    if d.empty:
        return
    parsed = d["outcome"].apply(parse_vas_name)
    d["cond"] = [p[0] for p in parsed]
    d["meas"] = [p[1] for p in parsed]
    d["rating"] = [p[2] for p in parsed]
    n_bad = int(d["cond"].isna().sum())
    if n_bad:
        print(f"  [warn] {n_bad} vasdyn variable(s) did not match the "
              f"expected {{cond}}_{{measure}}_{{rating}} naming; skipped")
        d = d[d["cond"].notna()]
    if d.empty:
        return

    _unabbrev = {v: k for k, v in MEASURE_ABBREV.items()}
    d["meas_full"] = d["meas"].map(lambda m: _unabbrev.get(m, m))
    d = d[~d["meas_full"].isin(EXCLUDE_VAS_MEASURES)]
    if d.empty:
        print("  [skip] per-measure figures: all measures excluded")
        return

    for (meas, meas_full), gm_all in d.groupby(["meas", "meas_full"]):
        keep = gm_all["pred_family"].isin(PER_MEASURE_FAMILIES)
        n_matched = 0
        if MATCHED_MEASURE_DYNAMICS:
            same = ((gm_all["pred_family"] == "plutdyn")
                    & gm_all["predictor"].str.endswith(f"|{meas_full}"))
            n_matched = int(same.sum())
            keep = keep | same
        gm = gm_all[keep].copy()
        if gm.empty:
            continue
        gm["plabel"] = [predictor_label(f, p) for f, p
                        in zip(gm["pred_family"], gm["predictor"])]

        cols, bounds, _ = _blocked_columns(gm, "vasdyn")
        if not cols:
            continue
        mat = gm.pivot_table(index="plabel", columns="outcome", values="r")
        pmat = gm.pivot_table(index="plabel", columns="outcome", values="p")
        nmat = gm.pivot_table(index="plabel", columns="outcome", values="n")
        cols = [c for c in cols if c in mat.columns]
        mat = mat[cols]
        pmat = pmat.reindex(index=mat.index, columns=cols)
        nmat = nmat.reindex(index=mat.index, columns=cols)
        nice = [f"{parse_vas_name(c)[0]} {parse_vas_name(c)[2]}" for c in cols]
        mat.columns = pmat.columns = nmat.columns = nice

        # minimal title: names the measure and says which side is dynamic
        rhs = "EMA dyn + mean" if n_matched else "EMA mean"
        title = (f"{meas_full}   VAS dyn x {rhs}   "
                 f"({METHOD}, n={int(gm['n'].min())}-{int(gm['n'].max())}, "
                 f"* p<.05 unc.)")
        rhs_tag = ("EMAdyn+EMAmean+quest" if n_matched
                   else "EMAmean+quest")
        stem = f"bymeasure_{meas_full}__VASdyn_x_{rhs_tag}__{METHOD}"
        _draw_heatmap_pages(mat, pmat, title, stem,
                            out_dir, y_block_bounds=bounds, nmat=nmat)


def plot_scalar_family_heatmaps(res, out_dir):
    """Same layout for the single-value families: within-block mean
    ratings, and the pre/post-video single ratings."""
    for fam in ("vasmean", "vasraw"):
        g = res[(res["out_family"] == fam)
                & (res["pred_family"].isin(PER_MEASURE_FAMILIES))]
        if g.empty:
            continue
        cols, bounds, _ = _blocked_columns(g, fam)
        mat = g.pivot_table(index="predictor", columns="outcome", values="r")
        pmat = g.pivot_table(index="predictor", columns="outcome", values="p")
        cols = [c for c in cols if c in mat.columns] or list(mat.columns)
        mat, pmat = mat[cols], pmat.reindex(index=mat.index, columns=cols)
        nmat = g.pivot_table(index="predictor", columns="outcome",
                             values="n").reindex(index=mat.index, columns=cols)
        relabel = {p: predictor_label(f, p) for f, p
                   in zip(g["pred_family"], g["predictor"])}
        mat.index = [relabel.get(i, i) for i in mat.index]
        pmat.index = [relabel.get(i, i) for i in pmat.index]
        nmat.index = [relabel.get(i, i) for i in nmat.index]
        nice = [c.replace("_", " ") for c in cols]
        mat.columns = pmat.columns = nmat.columns = nice
        nicefam = ("within-block mean rating" if fam == "vasmean"
                   else "pre-video & post-video single ratings")
        title = (f"{nicefam}   ({METHOD}, n={int(g['n'].min())}-"
                 f"{int(g['n'].max())}, * p<.05 uncorrected)")
        stem = (f"blockmean__VASblockmean_x_EMAmean+quest__{METHOD}"
                if fam == "vasmean" else
                f"singlerating__VASprepostvideo_x_EMAmean+quest__{METHOD}")
        _draw_heatmap_pages(mat, pmat, title, stem, out_dir,
                            y_block_bounds=bounds, nmat=nmat)


def plot_family_heatmaps(res, out_dir):
    """Whole predictor-family x outcome-family matrices, paged."""
    for (pf, of), g in res.groupby(["pred_family", "out_family"]):
        mat = g.pivot_table(index="predictor", columns="outcome", values="r")
        pmat = g.pivot_table(index="predictor", columns="outcome",
                             values="p").reindex(index=mat.index,
                                                 columns=mat.columns)
        title = (f"{pf}  x  {FAMILY_LABEL.get(of, of)}   "
                 f"({METHOD}, n={int(g['n'].min())}-{int(g['n'].max())}, "
                 f"* p<.05 uncorrected)")
        nmat = g.pivot_table(index="predictor", columns="outcome",
                             values="n").reindex(index=mat.index,
                                                 columns=mat.columns)
        stem = (f"family__{FAMILY_TAG.get(pf, pf)}"
                f"_x_{FAMILY_TAG.get(of, of)}__{METHOD}")
        _draw_heatmap_pages(mat, pmat, title, stem, out_dir, nmat=nmat)


def plot_top_scatters(res, allvars, out_dir, top_n=20):
    """Scatters for the strongest |r|. With small n most 'hits' visibly
    hinge on one or two points - that is the thing to look for."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    elig = res[res["n"] >= MIN_N_RANK]
    if elig.empty:
        print(f"  [skip] top scatters: no correlation has n >= {MIN_N_RANK}")
        return
    dropped = len(res) - len(elig)
    print(f"  top scatters: ranking over {len(elig)} correlations with "
          f"n >= {MIN_N_RANK} ({dropped} excluded as too few pairs)")
    top = elig.reindex(elig["r"].abs().sort_values(ascending=False).index).head(top_n)
    ncols, nrows = 5, int(np.ceil(len(top) / 5))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.33, 7.5), squeeze=False)
    fig.suptitle(f"Strongest {len(top)} correlations with n>={MIN_N_RANK} "
                 f"({METHOD}, UNCORRECTED - check for single-point leverage)",
                 fontsize=12, fontweight="bold")
    for i, (_, r) in enumerate(top.iterrows()):
        ax = axes[i // ncols][i % ncols]
        xc = f"{r.pred_family}|{r.predictor}"
        yc = f"{r.out_family}|{r.outcome}"
        if xc not in allvars.columns or yc not in allvars.columns:
            ax.axis("off"); continue
        pair = allvars[[xc, yc]].dropna()
        ax.scatter(pair[xc], pair[yc], s=22, color="#2c3e50")
        ax.set_title(f"{r.predictor}\nvs {r.outcome}\n"
                     f"r={r.r:.2f} p={r.p:.3f} n={int(r.n)}", fontsize=5)
        ax.tick_params(labelsize=5)
        sns.despine(ax=ax)
    for i in range(len(top), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = Path(out_dir) / (f"topcorr_{len(top)}strongest__allfamilies__"
                         f"{METHOD}_minN{MIN_N_RANK}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ===================================================================
# MAIN
# ===================================================================

def run_one_method(method, predictors, outcomes, subjects, P, O, allvars):
    """Everything downstream of loading, for a single correlation method.
    METHOD is a module-level global read by the plotting and reporting
    helpers, so it is rebound here rather than threaded through every
    signature."""
    global METHOD
    METHOD = method

    out_dir = OUT_DIR / method
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  {method.upper()}\n{'='*70}")

    res = correlate_families(P, O, subjects)
    if res.empty:
        print("No correlations could be computed.")
        return None
    out = out_dir / f"corr_all_modalities_{method}.csv"
    res.to_csv(out, index=False)
    print(f"[OK] {len(res)} correlations -> {out}")

    if "quest" in predictors:
        report_compliance_confound(predictors["quest"], subjects, out_dir)
    report_n_coverage(res)
    ytab = report_yield(res)
    ytab.to_csv(out_dir / "yield_vs_chance.csv", index=False)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    bym = out_dir / "figures_by_measure"
    bym.mkdir(exist_ok=True)
    plot_per_measure_heatmaps(res, bym)
    plot_scalar_family_heatmaps(res, bym)
    plot_family_heatmaps(res, fig_dir)
    plot_top_scatters(res, allvars, fig_dir)
    return res


def compare_methods(results):
    """Where do the two methods disagree? A large spearman-vs-pearson gap
    on the same variable pair points at an outlier or a non-linear
    relationship, so it is worth listing rather than leaving buried."""
    if len(results) < 2:
        return
    keys = ["pred_family", "predictor", "out_family", "outcome"]
    (m1, r1), (m2, r2) = list(results.items())[:2]
    merged = r1.merge(r2, on=keys, suffixes=(f"_{m1}", f"_{m2}"))
    if merged.empty:
        return
    merged["abs_diff"] = (merged[f"r_{m1}"] - merged[f"r_{m2}"]).abs()
    merged = merged.sort_values("abs_diff", ascending=False)
    p = OUT_DIR / f"method_comparison_{m1}_vs_{m2}.csv"
    merged.to_csv(p, index=False)
    big = merged[merged["abs_diff"] > 0.3]
    print(f"\n--- {m1} vs {m2} ---")
    print(f"  overall agreement: r = "
          f"{merged[f'r_{m1}'].corr(merged[f'r_{m2}']):.3f} across "
          f"{len(merged)} shared pairs")
    print(f"  {len(big)} pair(s) differ by |dr| > 0.3 - inspect these for "
          f"outlier leverage or non-linearity")
    print(f"  full comparison -> {p}")


def main():
    print("Loading variable families...")
    predictors = {
        "quest":    load_quest(),
        "plutmean": load_plutchik_means(),
        "plutdyn":  load_plutchik_dynamics(),
    }
    _vasdyn = load_vas_dynamics()
    load_vas_block_means(_vasdyn)          # cross-check only, returns empty
    outcomes = {"vasdyn": _vasdyn}
    if INCLUDE_POST_VIDEO:
        vas_subjects = sorted({int(re.search(r"dyn(\d+)_results", f.name).group(1))
                               for f in VAS_RAW.glob("dyn*_results.csv")
                               if re.search(r"dyn(\d+)_results", f.name)})
        video_order = build_video_order(vas_subjects)
        outcomes["vasraw"] = load_vas_single_ratings(video_order)

    predictors = {k: v for k, v in predictors.items() if len(v)}
    outcomes = {k: v for k, v in outcomes.items() if len(v)}
    if not predictors or not outcomes:
        raise RuntimeError("Need at least one predictor and one outcome "
                           "family; check the paths in CONFIG.")

    subjects = report_id_overlap({**predictors, **outcomes})
    if len(subjects) < MIN_N:
        print(f"\nStopping: {len(subjects)} shared subject(s) is below "
              f"MIN_N={MIN_N}. Fix the ID mapping or wait for more data.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    P = pd.concat(predictors.values(), axis=1)
    P = drop_excluded_predictor_measures(P)
    O = pd.concat(outcomes.values(), axis=1)
    allvars = pd.concat([P, O], axis=1).reindex(subjects)

    # data is loaded once; only the correlation step repeats per method
    results = {}
    for method in METHODS:
        r = run_one_method(method, predictors, outcomes, subjects,
                           P, O, allvars)
        if r is not None:
            results[method] = r
    compare_methods(results)


if __name__ == "__main__":
    # Which correlation method(s) to run:
    #   python3 4_correlate_all_modalities.py            -> both (default)
    #   python3 4_correlate_all_modalities.py spearman   -> spearman only
    #   python3 4_correlate_all_modalities.py pearson    -> pearson only
    #   python3 4_correlate_all_modalities.py both       -> both, explicitly
    import sys as _sys
    _arg = _sys.argv[1].lower() if len(_sys.argv) > 1 else None
    if _arg in ("spearman", "pearson"):
        METHODS = (_arg,)
    elif _arg in (None, "both", "all"):
        pass                      # keep the configured default
    else:
        raise SystemExit(f"Unknown argument {_arg!r}. Use: spearman | "
                         f"pearson | both (or no argument for both).")
    print(f"Running correlation method(s): {', '.join(METHODS)}")
    main()
