# -*- coding: utf-8 -*-
"""
score_questionnaires.py
=======================
Scores the per-subject psychometric battery in
/home/ZhangJianfeng_04/emo_study/Behav/quest/quest and produces
descriptive violin+jitter plots (one dot per subject), PPT-sized.

Each subject has:
  NNN_Pysch_quest_*.csv  - 11 scales, 265 items, PsychoPy slider output
  BDI_NNN_*.csv          - BDI-II, already contains BDI_total

--------------------------------------------------------------------
SCALING - continuous slider -> published Likert points
--------------------------------------------------------------------
All scales were administered on a CONTINUOUS slider (slider.response
holds values like 1.025, 9.9583), NOT on each instrument's published
Likert anchors (TAS-20 is 1-5, ASI-3 0-4, MAIA-2 0-5, etc.).

Each response is converted to the instrument's published Likert scale
(LIKERT_ANCHORS below) BEFORE any reverse scoring or aggregation:

    likert = round( a + (x - slider_min) / (slider_max - slider_min) * (b - a) )

i.e. the slider is stretched linearly onto [a, b] and snapped to the
nearest integer point (halves round up), clipped to [a, b]. This
assumes the slider's two ends correspond to the instrument's two end
anchors and that its points are equally spaced along the slider.

After conversion:
  * Reverse scoring uses the PUBLISHED anchors: rev(x) = (a + b) - x.
  * Every score is emitted as MEAN-PER-ITEM on the Likert scale
    (`value`, used in the wide file and downstream), and also as the
    published-style SUM of items (`sum_value`, only when every item of
    that score was answered; in questionnaire_scores_wide_sums.csv).
  * Sums are on the published metric, but the responses were still
    given on a slider, not on labelled response options, so treat
    published cutoffs / norms with caution.

Scales with no entry in LIKERT_ANCHORS (currently TSQ) are left on the
continuous slider scale and reported as such.

Set SCORE_ON_LIKERT = False to reproduce the old continuous scoring.

SLIDER_MIN / SLIDER_MAX are auto-detected from the pooled data across
all subjects and printed; override them below if the PsychoPy slider
was configured with known fixed bounds (recommended - the conversion
depends directly on them).

--------------------------------------------------------------------
SCORING KEYS - provenance
--------------------------------------------------------------------
The 'No' column in the raw file is a GLOBAL item counter (1-265 across
all 11 scales), NOT per-scale numbering. Once each scale's offset is
subtracted, the file's own ReverseScored flags were checked against the
published keys and AGREE for TAS, DERS, MAIA-2 and CTQ. The file's
flags are therefore trusted where present, with two documented
exceptions handled below:

  * FFMQ  - file has NO reverse flags but the instrument genuinely has
            19. Patched from the published key (Baer et al. 2006).
  * MW    - the 'ReverseScored' column holds D/S, which are the
            Deliberate/Spontaneous SUBSCALE labels, not reverse flags.
            Used as subscales; no MW item is reverse scored.

ASI-3, GASP, ASQ, PERS and SCS legitimately have no reverse items, so
their empty flag column is correct, not missing metadata.

Subscale membership is NOT in the file for most scales and comes from
the published keys encoded in SUBSCALES below. Each scale carries a
`verified` field:
    "published"  - subscale key taken from the published instrument
    "item_text"  - subscale key derived from this file's item text
                   (PERS and ASQ; both resolve unambiguously - PERS is
                   a perfect 6-cycle of valence x activation/duration/
                   intensity, ASQ splits 8/7/5 by content)
    "in_file"    - subscale labels supplied by the file itself
Anything marked "item_text" is worth a sanity check against your own
protocol before publication.

Usage:
    python3 2_score_questionnaires.py
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

QUEST_DIR = Path("/home/ZhangJianfeng_04/emo_study/Behav/quest/quest")
OUT_DIR = Path("/home/ZhangJianfeng_04/emo_study/Behav/quest/analysis")

# Set to a (min, max) tuple to override auto-detection.
SLIDER_BOUNDS_OVERRIDE = None

# Convert slider responses to the published Likert points (see docstring).
# False = old behaviour: score on the continuous slider scale.
SCORE_ON_LIKERT = True

# Published response anchors (min, max) per scale, as named in the raw
# file's 'Scale' column. Check these against the versions you administered.
LIKERT_ANCHORS = {
    "TAS":    (1, 5),   # TAS-20 (Bagby et al. 1994)
    "DERS":   (1, 5),   # DERS-36 (Gratz & Roemer 2004)
    "FFMQ":   (1, 5),   # FFMQ-39 (Baer et al. 2006)
    "MAIA-2": (0, 5),   # MAIA-2 (Mehling et al. 2018)
    "ASI3":   (0, 4),   # ASI-3 (Taylor et al. 2007)
    "CTQ":    (1, 5),   # CTQ-28 (Bernstein et al. 2003)
    "GASP":   (1, 7),   # GASP (Cohen et al. 2011)
    "PERS":   (1, 5),   # PERS (Becerra et al. 2019)
    "ASQ":    (1, 5),   # Affective Style Questionnaire (Hofmann & Kashdan 2010)
    "SCS":    (1, 5),   # Self-Compassion Scale (Neff 2003) - confirm which SCS
    "MW":     (1, 7),   # MW-D / MW-S (Carriere et al. 2013)
    "EAQ":    (1, 3),   # EAQ-30 (Rieffe et al. 2008)
    # "TSQ": no published anchors known here -> left on the slider scale
}

# minimum fraction of a subscale's items a subject must have answered
# for that subscale score to be emitted (else NaN)
MIN_ITEM_COVERAGE = 0.75

# Scales scored ITEM BY ITEM rather than as subscales/totals. Each item
# becomes its own variable (e.g. TSQ_item01 ... TSQ_item15) and flows
# downstream into the correlations like any other score.
#
# TSQ carries no Dimension and no ReverseScored metadata in the raw file,
# so no subscales exist to compute and no total is emitted: a total would
# require guessing which items are reverse-keyed, and the item wording
# mixes directions (most items read negatively, item 8 reads positively).
ITEM_LEVEL_SCALES = {"TSQ"}

# Scales added to the battery partway through data collection - subjects
# run before the change simply have no rows for them, and will come out
# as NaN rather than as an error.
LATE_ADDITION_SCALES = {"TSQ", "EAQ"}


# ===================================================================
# SCORING KEYS
# ===================================================================
# Item numbers are WITHIN-SCALE positions (1-based), after subtracting
# each scale's global offset.

REVERSE_PATCH = {
    # FFMQ: file has no flags; published key (Baer et al. 2006), 19 items
    "FFMQ": [3, 5, 8, 10, 12, 13, 14, 16, 17, 18, 22, 23, 25, 28, 30,
             34, 35, 38, 39],
}

# Scales whose empty ReverseScored column is CORRECT (no reverse items)
NO_REVERSE_EXPECTED = {"ASI3", "GASP", "ASQ", "PERS", "SCS", "MW", "TSQ"}

SUBSCALES = {
    "TAS": dict(verified="published", subs={
        "DIF_DifficultyIdentifyingFeelings": [1, 3, 6, 7, 9, 13, 14],
        "DDF_DifficultyDescribingFeelings":  [2, 4, 11, 12, 17],
        "EOT_ExternallyOrientedThinking":    [5, 8, 10, 15, 16, 18, 19, 20],
    }),
    "DERS": dict(verified="published", subs={
        "Nonacceptance": [11, 12, 21, 23, 25, 29],
        "Goals":         [13, 18, 20, 26, 33],
        "Impulse":       [3, 14, 19, 24, 27, 32],
        "Awareness":     [2, 6, 8, 10, 17, 34],
        "Strategies":    [15, 16, 22, 28, 30, 31, 35, 36],
        "Clarity":       [1, 4, 5, 7, 9],
    }),
    "FFMQ": dict(verified="published", subs={
        "Observe":       [1, 6, 11, 15, 20, 26, 31, 36],
        "Describe":      [2, 7, 12, 16, 22, 27, 32, 37],
        "ActAware":      [5, 8, 13, 18, 23, 28, 34, 38],
        "Nonjudge":      [3, 10, 14, 17, 25, 30, 35, 39],
        "Nonreact":      [4, 9, 19, 21, 24, 29, 33],
    }),
    "MAIA-2": dict(verified="published", subs={
        "Noticing":            [1, 2, 3, 4],
        "NotDistracting":      [5, 6, 7, 8, 9, 10],
        "NotWorrying":         [11, 12, 13, 14, 15],
        "AttentionRegulation": [16, 17, 18, 19, 20, 21, 22],
        "EmotionalAwareness":  [23, 24, 25, 26, 27],
        "SelfRegulation":      [28, 29, 30, 31],
        "BodyListening":       [32, 33, 34],
        "Trusting":            [35, 36, 37],
    }),
    "ASI3": dict(verified="published", subs={
        "Physical":  [3, 4, 7, 8, 12, 15],
        "Cognitive": [2, 5, 10, 14, 16, 18],
        "Social":    [1, 6, 9, 11, 13, 17],
    }),
    "CTQ": dict(verified="published", subs={
        "EmotionalAbuse":    [3, 8, 14, 18, 25],
        "PhysicalAbuse":     [9, 11, 12, 15, 17],
        "SexualAbuse":       [20, 21, 23, 24, 27],
        "EmotionalNeglect":  [5, 7, 13, 19, 28],
        "PhysicalNeglect":   [1, 2, 4, 6, 26],
        "MinimizationDenial": [10, 16, 22],
    }),
    "GASP": dict(verified="published", subs={
        "GuiltNegativeBehaviorEvaluation": [1, 9, 14, 16],
        "GuiltRepair":                     [2, 5, 11, 15],
        "ShameNegativeSelfEvaluation":     [3, 6, 10, 13],
        "ShameWithdraw":                   [4, 7, 8, 12],
    }),
    "PERS": dict(verified="item_text", subs={
        "PositiveActivation": [1, 7, 13, 19, 25],
        "PositiveDuration":   [3, 9, 15, 21, 27],
        "PositiveIntensity":  [5, 11, 17, 23, 29],
        "NegativeActivation": [2, 8, 14, 20, 26],
        "NegativeDuration":   [4, 10, 16, 22, 28],
        "NegativeIntensity":  [6, 12, 18, 24, 30],
    }),
    "ASQ": dict(verified="item_text", subs={
        "Concealing": [1, 5, 9, 10, 13, 15, 18, 20],
        "Adjusting":  [2, 4, 7, 8, 12, 16, 19],
        "Tolerating": [3, 6, 11, 14, 17],
    }),
    # MW and SCS subscales come from the file's own columns - see
    # build_subscales_from_file()
}

# Scales where a "total" score is NOT meaningful and should be skipped
# (multidimensional with non-additive / opposing subscales)
NO_TOTAL = {"PERS", "ASQ", "SCS", "MW", "GASP", "MAIA-2", "TSQ", "EAQ"}


# ===================================================================
# LOADING
# ===================================================================

def find_subject_files(quest_dir):
    r"""Map subject_id -> {'quest': path, 'bdi': path, 'suffix': str}.

    Filenames may carry a letter suffix (e.g. 001b_Pysch_quest...), which
    normally marks a repeat session. A regex of ^(\d+)_ would not match
    those and would drop the subject SILENTLY, so the suffix is parsed
    explicitly and reported. Where both a plain and a suffixed file exist
    for one subject the LAST by filename order is used and the collision
    is printed - it is a data-management decision, not one to make here.
    """
    quest_dir = Path(quest_dir)
    subs, suffixed, collisions = {}, [], []
    for p in sorted(quest_dir.glob("*.csv")):
        name = p.name
        m_q = re.match(r"^(\d+)([A-Za-z]*)_Pysch_quest", name, flags=re.I)
        m_b = re.match(r"^BDI_(\d+)([A-Za-z]*)_", name, flags=re.I)
        if m_q:
            sid, suf = int(m_q.group(1)), m_q.group(2)
            if suf:
                suffixed.append((name, sid, suf))
            if "quest" in subs.get(sid, {}):
                collisions.append((sid, subs[sid]["quest"].name, name))
            subs.setdefault(sid, {})["quest"] = p
            subs[sid]["suffix"] = suf
        elif m_b:
            sid, suf = int(m_b.group(1)), m_b.group(2)
            if suf:
                suffixed.append((name, sid, suf))
            subs.setdefault(sid, {})["bdi"] = p

    if suffixed:
        print(f"  {len(suffixed)} file(s) carry a letter suffix (usually a "
              f"repeat session) - parsed, not skipped:")
        for name, sid, suf in suffixed:
            print(f"      {name}  -> subject {sid} (suffix '{suf}')")
    if collisions:
        print(f"  *** {len(collisions)} subject(s) have MORE THAN ONE quest "
              f"file. The later filename is used - confirm that is right: ***")
        for sid, a, b in collisions:
            print(f"      subject {sid}: {a}  vs  {b}   -> using {b}")
    return subs


def load_quest_items(csv_path):
    """Return long-form item table for one subject: scale, pos, response,
    reverse_flag_raw. 'pos' is the WITHIN-SCALE 1-based item number."""
    df = pd.read_csv(csv_path)
    need = {"No", "Scale", "slider.response", "ReverseScored"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{csv_path.name}: missing column(s) {sorted(missing)}")

    d = df[df["Scale"].notna()].copy()
    d["Scale"] = d["Scale"].astype(str).str.strip()
    # global No -> within-scale position. The offset is taken BEFORE
    # dropping unanswered items, so a skipped first item does not shift
    # every other item of that scale down by one.
    offs = d.groupby("Scale")["No"].transform("min") - 1
    d["pos"] = (d["No"] - offs).astype(int)
    n_items_scale = d.groupby("Scale")["pos"].nunique().to_dict()
    d = d[d["slider.response"].notna()].copy()
    d["response"] = d["slider.response"].astype(float)
    keep = ["Scale", "pos", "response", "ReverseScored", "Dimension"]
    if "English_Translation" in df.columns:
        d["English_Translation"] = d.get("English_Translation")
        keep.append("English_Translation")
    d = d[keep]
    d.attrs["n_items_scale"] = n_items_scale
    # the in-file participant field, for cross-checking against the filename
    # kept as the RAW value: it may be non-numeric (e.g. '001b' for a
    # repeat session), so it is never blindly int()-cast - that crashed.
    d.attrs["participant_field"] = (
        str(df["participant"].dropna().iloc[0]).strip()
        if "participant" in df.columns and df["participant"].notna().any()
        else None)
    return d


def detect_slider_bounds(all_items):
    """Infer the slider range from the pooled responses across subjects."""
    if SLIDER_BOUNDS_OVERRIDE is not None:
        lo, hi = SLIDER_BOUNDS_OVERRIDE
        print(f"  Slider bounds: OVERRIDDEN to [{lo}, {hi}]")
        return float(lo), float(hi)
    obs_lo = float(all_items["response"].min())
    obs_hi = float(all_items["response"].max())
    lo, hi = float(np.floor(obs_lo)), float(np.ceil(obs_hi))
    print(f"  Slider bounds: observed [{obs_lo:.3f}, {obs_hi:.3f}] "
          f"-> using [{lo:.0f}, {hi:.0f}] for reverse scoring.")
    if hi - lo < 2:
        raise ValueError("Implausible slider range - set SLIDER_BOUNDS_OVERRIDE.")
    print(f"    NOTE: these bounds define the slider -> Likert conversion. "
          f"Set SLIDER_BOUNDS_OVERRIDE if the slider's true range is known.")
    per_scale = all_items.groupby("Scale")["response"].agg(["min", "max"])
    print("    observed range per scale:")
    for sc, r in per_scale.iterrows():
        print(f"      {sc:<8s} [{r['min']:.3f}, {r['max']:.3f}]")
    return lo, hi


def scale_bounds(scale, slider_lo, slider_hi):
    """(lo, hi) of the scale the responses are scored on, and whether it
    is the published Likert scale."""
    if SCORE_ON_LIKERT and scale in LIKERT_ANCHORS:
        a, b = LIKERT_ANCHORS[scale]
        return float(a), float(b), True
    return slider_lo, slider_hi, False


def slider_to_likert(x, slider_lo, slider_hi, a, b):
    """Map continuous slider values linearly onto [a, b] and snap to the
    nearest integer Likert point (halves round up)."""
    x = np.asarray(x, dtype=float)
    scaled = a + (x - slider_lo) / (slider_hi - slider_lo) * (b - a)
    return np.clip(np.floor(scaled + 0.5), a, b)


# ===================================================================
# SCORING
# ===================================================================

def resolve_reverse_items(scale, g):
    """Which within-scale item numbers are reverse-scored, combining the
    file's own flags with the documented FFMQ patch."""
    if scale in REVERSE_PATCH:
        return set(REVERSE_PATCH[scale]), "published_patch"
    if scale == "MW":
        return set(), "none (D/S column is subscale labels, not reverse flags)"
    if scale in NO_REVERSE_EXPECTED and g["ReverseScored"].isna().all():
        return set(), "none (scale has no reverse items)"
    flagged = g.loc[g["ReverseScored"].notna(), "pos"]
    if scale == "DERS":
        # DERS encodes reverse as the literal 1.0, non-reverse as 5.0
        flagged = g.loc[g["ReverseScored"].astype(str).str.strip() == "1.0", "pos"]
    rev = set(int(x) for x in flagged)
    if not rev and scale not in NO_REVERSE_EXPECTED:
        print(f"    WARNING: {scale} has no reverse items flagged and no "
              f"patch - verify this is correct.")
    return rev, "file_flags"


def build_subscales_from_file(scale, g):
    """Subscales taken from the raw file's own columns.

    MW is a special case: its subscale labels live in the ReverseScored
    column as D/S (Deliberate/Spontaneous), not in Dimension.

    Every other scale with a populated Dimension column (SCS, EAQ) has
    its subscales read straight from it.
    """
    if scale == "MW":
        col = g["ReverseScored"].astype(str).str.strip()
        mapping = {"D": "Deliberate", "S": "Spontaneous"}
        subs = {}
        for code, name in mapping.items():
            items = g.loc[col == code, "pos"].astype(int).tolist()
            if items:
                subs[name] = sorted(items)
        return subs
    if g["Dimension"].notna().any():
        subs = {}
        for dim, gg in g.groupby("Dimension"):
            if pd.isna(dim):
                continue
            key = re.sub(r"[^A-Za-z0-9]+", "_", str(dim).strip()).strip("_")
            subs[key] = sorted(gg["pos"].astype(int).tolist())
        return subs
    return {}


def score_subject(sid, items, slider_lo, slider_hi):
    """Return long-form rows: subject, questionnaire, score_name, value."""
    rows = []
    for scale, g in items.groupby("Scale"):
        g = g.sort_values("pos")
        rev_items, rev_src = resolve_reverse_items(scale, g)

        v = g.set_index("pos")["response"].astype(float)
        lo, hi, on_likert = scale_bounds(scale, slider_lo, slider_hi)
        if on_likert:
            v = pd.Series(slider_to_likert(v.values, slider_lo, slider_hi,
                                           lo, hi), index=v.index)
        metric = f"likert_{lo:g}-{hi:g}" if on_likert else "slider"
        v_scored = v.copy()
        for p in rev_items:
            if p in v_scored.index:
                v_scored.loc[p] = (lo + hi) - v_scored.loc[p]

        if scale in ITEM_LEVEL_SCALES:
            for p in sorted(v_scored.index):
                rows.append(dict(subject=sid, questionnaire=scale,
                                 score=f"{scale}_item{int(p):02d}",
                                 value=float(v_scored.loc[p]),
                                 sum_value=float(v_scored.loc[p]),
                                 n_items=1, coverage=1.0, metric=metric,
                                 key_source="item_level",
                                 reverse_source=rev_src))
            continue

        if scale in SUBSCALES:
            subs = SUBSCALES[scale]["subs"]
            verified = SUBSCALES[scale]["verified"]
        else:
            subs = build_subscales_from_file(scale, g)
            verified = "in_file"
            if not subs:
                # A scale with no published key here, no Dimension column
                # and not marked item-level. Rather than emit nothing and
                # let it disappear unnoticed, emit a total and say so.
                print(f"    NOTE: '{scale}' has no subscale key and no "
                      f"Dimension column - emitting a TOTAL only. Add it to "
                      f"SUBSCALES or ITEM_LEVEL_SCALES if that is not right.")
                verified = "unkeyed_total_only"

        for sub_name, item_nums in subs.items():
            present = [p for p in item_nums if p in v_scored.index]
            cov = len(present) / len(item_nums) if item_nums else 0
            val = float(v_scored.loc[present].mean()) if cov >= MIN_ITEM_COVERAGE \
                else np.nan
            # a published-style sum is only defined when nothing is missing
            total = float(v_scored.loc[present].sum()) if cov == 1.0 else np.nan
            rows.append(dict(subject=sid, questionnaire=scale,
                             score=f"{scale}_{sub_name}", value=val,
                             sum_value=total,
                             n_items=len(item_nums), coverage=cov,
                             metric=metric,
                             key_source=verified, reverse_source=rev_src))

        if scale not in NO_TOTAL:
            n_expected = items.attrs.get("n_items_scale", {}).get(
                scale, len(v_scored))
            cov_t = len(v_scored) / n_expected if n_expected else 0
            rows.append(dict(subject=sid, questionnaire=scale,
                             score=f"{scale}_TOTAL",
                             value=float(v_scored.mean()),
                             sum_value=(float(v_scored.sum())
                                        if cov_t == 1.0 else np.nan),
                             n_items=n_expected, coverage=cov_t,
                             metric=metric,
                             key_source=verified, reverse_source=rev_src))
    return rows


def load_bdi(path, sid):
    """BDI file already carries BDI_total; fall back to summing item_ cols.
    BDI uses its own 0-3 item scale (NOT the slider), so it is reported as
    the conventional SUM, unlike the slider-based scales above."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    if "BDI_total" in df.columns and df["BDI_total"].notna().any():
        total = float(df["BDI_total"].dropna().iloc[0])
        src = "BDI_total column"
    else:
        item_cols = [c for c in df.columns if re.match(r"^item_\d+$", c)]
        if not item_cols:
            raise KeyError(f"{path.name}: no BDI_total and no item_ columns")
        total = float(df[item_cols].iloc[0].sum())
        src = f"summed {len(item_cols)} item_ columns"
    return dict(subject=sid, questionnaire="BDI", score="BDI_TOTAL",
                value=total, sum_value=total, n_items=21, coverage=1.0,
                metric="bdi_0-3_sum", key_source=src, reverse_source="n/a")


def audit_subject_ids(per_subject_items, subs):
    r"""Record the subject number as it appears in each of the two places
    it is stored: the FILENAME prefix and the in-file `participant`
    column.

    The FILENAME IS AUTHORITATIVE, by decision - the in-file participant
    field is not used for anything. It is recorded here only so the
    discrepancies stay visible in subject_id_audit.csv if they ever
    matter later. Known harmless cases: subject 000 was renamed from 001,
    and the second participant was temporarily saved as 001b, so several
    in-file values legitimately lag behind the filenames.

    The field is parsed on leading digits, so '001b' reads as 1 rather
    than crashing an int() cast.
    """
    rows = []
    for sid, items in sorted(per_subject_items.items()):
        raw = items.attrs.get("participant_field")
        digits = re.match(r"^\s*(\d+)", str(raw)) if raw is not None else None
        parsed = int(digits.group(1)) if digits else None
        rows.append(dict(subject_used=sid,
                         source="filename (authoritative)",
                         participant_field_raw=raw,
                         participant_field_parsed=parsed,
                         filename_suffix=subs.get(sid, {}).get("suffix", ""),
                         fields_agree=(parsed == sid)))
    audit = pd.DataFrame(rows)
    n_diff = int((~audit["fields_agree"]).sum())
    if n_diff:
        print(f"  Subject IDs: taken from filenames. ({n_diff} file(s) have a "
              f"different number in their internal 'participant' column; "
              f"ignored by design - see subject_id_audit.csv.)")
    else:
        print("  Subject IDs: taken from filenames.")
    return audit


def report_scale_coverage(per_subject_items):
    """Which scales each subject actually has - the battery gained TSQ and
    EAQ partway through, so earlier subjects legitimately lack them."""
    rows = []
    for sid, items in sorted(per_subject_items.items()):
        present = set(items["Scale"].unique())
        rows.append(dict(subject=sid, n_scales=len(present),
                         **{s: (s in present) for s in sorted(
                             set().union(*[set(i["Scale"].unique())
                                           for i in per_subject_items.values()]))}))
    cov = pd.DataFrame(rows)
    for s in LATE_ADDITION_SCALES:
        if s in cov.columns:
            missing = cov.loc[~cov[s], "subject"].tolist()
            if missing:
                print(f"  {s}: absent for {len(missing)} subject(s) "
                      f"(added to the battery later) -> {missing}")
    return cov


def write_item_legend(per_subject_items, out_dir, scale):
    """Item number -> English wording, as CSV, so the numbered violins can
    be read without going back to the raw file. Pooled across subjects
    (the wording is identical in every file); any subject-to-subject
    disagreement in wording is reported rather than silently collapsed."""
    text_map, conflicts = {}, []
    for sid, items in per_subject_items.items():
        g = items[items["Scale"] == scale]
        if "English_Translation" not in g.columns:
            continue
        for _, r in g.iterrows():
            t = r.get("English_Translation")
            if pd.isna(t):
                continue
            pos, t = int(r["pos"]), str(t).strip()
            if pos in text_map and text_map[pos] != t:
                conflicts.append((pos, sid))
            text_map.setdefault(pos, t)
    if not text_map:
        print(f"  [skip] {scale} item legend: no English_Translation column")
        return None
    if conflicts:
        print(f"  WARNING: {scale} item wording differs between subjects for "
              f"item(s) {sorted({p for p, _ in conflicts})} - first seen kept.")
    leg = pd.DataFrame(
        {"item_number": sorted(text_map),
         "score_name": [f"{scale}_item{p:02d}" for p in sorted(text_map)],
         "english_translation": [text_map[p] for p in sorted(text_map)]})
    p = Path(out_dir) / f"{scale}_item_legend.csv"
    leg.to_csv(p, index=False)
    print(f"  Saved: {p}")
    return leg


# ===================================================================
# PLOTTING
# ===================================================================


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


def plot_quest_violins(scores, out_dir, ncols=4):
    """One PPT-sized figure per questionnaire, one subplot per score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = scores.dropna(subset=["value"])
    for q, gq in d.groupby("questionnaire"):
        score_names = sorted(gq["score"].unique())
        nrows = int(np.ceil(len(score_names) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13.33, 7.5),
                                 squeeze=False)
        kind = "items" if q in ITEM_LEVEL_SCALES else "subscale scores"
        fig.suptitle(f"{q} - {kind} (n={gq['subject'].nunique()} subjects)"
                     + ("   [see item legend figure for wording]"
                        if q in ITEM_LEVEL_SCALES else ""),
                     fontsize=13, fontweight="bold")
        for i, sn in enumerate(score_names):
            ax = axes[i // ncols][i % ncols]
            sub = gq[gq["score"] == sn]
            sns.violinplot(y=sub["value"], ax=ax, inner="quartile",
                           color="#d8e8d0", cut=0, linewidth=0.9)
            for ln in ax.lines:
                ln.set_color("#1a1a2e"); ln.set_linewidth(1.0)
            sns.stripplot(y=sub["value"], ax=ax, color="#2c3e50",
                          size=5, jitter=0.15, alpha=0.75)
            _overlay_mean(ax, {0: sub["value"]})
            ax.set_title(sn.replace(f"{q}_", ""), fontsize=9)
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.tick_params(labelsize=7)
            sns.despine(ax=ax)
        for i in range(len(score_names), nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)
        _violin_caption(fig, gq["subject"].nunique())
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        p = Path(out_dir) / f"quest_{q.replace('-','')}_violin.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")


def plot_all_scores_overview(scores, out_dir, per_fig=20):
    """Every score across all questionnaires, paged into PPT-sized grids."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = scores.dropna(subset=["value"])
    names = sorted(d["score"].unique())
    ncols = 5
    for page, start in enumerate(range(0, len(names), per_fig), 1):
        chunk = names[start:start + per_fig]
        nrows = int(np.ceil(len(chunk) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13.33, 7.5),
                                 squeeze=False)
        fig.suptitle(f"All questionnaire scores ({page})", fontsize=13,
                     fontweight="bold")
        for i, sn in enumerate(chunk):
            ax = axes[i // ncols][i % ncols]
            sub = d[d["score"] == sn]
            sns.violinplot(y=sub["value"], ax=ax, inner="quartile",
                           color="#d8e8d0", cut=0, linewidth=0.9)
            for ln in ax.lines:
                ln.set_color("#1a1a2e"); ln.set_linewidth(1.0)
            sns.stripplot(y=sub["value"], ax=ax, color="#2c3e50",
                          size=4, jitter=0.15, alpha=0.75)
            _overlay_mean(ax, {0: sub["value"]})
            ax.set_title(sn, fontsize=7)
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.tick_params(labelsize=6)
            sns.despine(ax=ax)
        for i in range(len(chunk), nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)
        _violin_caption(fig, d["subject"].nunique())
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        p = Path(out_dir) / f"quest_ALL_overview_p{page}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    subs = find_subject_files(QUEST_DIR)
    if not subs:
        raise FileNotFoundError(f"No questionnaire CSVs found in {QUEST_DIR}")
    print(f"Found {len(subs)} subject(s) with questionnaire files.")
    n_bdi = sum(1 for v in subs.values() if v.get("bdi"))
    print(f"  {n_bdi}/{len(subs)} have a BDI file "
          f"({len(subs)-n_bdi} missing - expected, will be added later).")

    per_subject_items = {}
    for sid, paths in sorted(subs.items()):
        if not paths.get("quest"):
            print(f"  P{sid}: no Pysch_quest file, skipping scale scoring.")
            continue
        per_subject_items[sid] = load_quest_items(paths["quest"])

    if not per_subject_items:
        raise RuntimeError("No Pysch_quest files could be loaded.")

    audit = audit_subject_ids(per_subject_items, subs)
    cov = report_scale_coverage(per_subject_items)

    all_items = pd.concat(per_subject_items.values(), ignore_index=True)
    slider_lo, slider_hi = detect_slider_bounds(all_items)

    rows = []
    for sid, items in sorted(per_subject_items.items()):
        rows.extend(score_subject(sid, items, slider_lo, slider_hi))
    for sid, paths in sorted(subs.items()):
        if paths.get("bdi"):
            try:
                rows.append(load_bdi(paths["bdi"], sid))
            except Exception as e:
                print(f"  P{sid}: BDI load failed - {e}")

    scores = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_long = OUT_DIR / "questionnaire_scores_long.csv"
    scores.to_csv(out_long, index=False)

    wide = scores.pivot_table(index="subject", columns="score", values="value")
    out_wide = OUT_DIR / "questionnaire_scores_wide.csv"
    wide.to_csv(out_wide)

    # published-style sums (complete subscales only); kept in a separate
    # file so downstream scripts reading the wide file see one value per score
    wide_sums = scores.pivot_table(index="subject", columns="score",
                                   values="sum_value")
    out_sums = OUT_DIR / "questionnaire_scores_wide_sums.csv"
    wide_sums.to_csv(out_sums)

    print(f"\n[OK] {len(scores)} score rows / {scores['score'].nunique()} "
          f"distinct scores -> {out_long}")
    print(f"[OK] subject x score matrix (mean per item) -> {out_wide}")
    print(f"[OK] subject x score matrix (sum of items)  -> {out_sums}")
    print("  Scoring metric per questionnaire:")
    for q, m in scores.groupby("questionnaire")["metric"].first().items():
        print(f"      {q:<8s} {m}")

    low = scores[(scores["coverage"] < 1.0) & scores["coverage"].notna()]
    if len(low):
        print(f"\n  {len(low)} score(s) computed from partially answered "
              f"subscales (coverage < 100%):")
        print(low[["subject", "score", "coverage"]].to_string(index=False))

    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    audit.to_csv(OUT_DIR / "subject_id_audit.csv", index=False)
    cov.to_csv(OUT_DIR / "scale_coverage_per_subject.csv", index=False)
    for sc in sorted(ITEM_LEVEL_SCALES):
        write_item_legend(per_subject_items, OUT_DIR, sc)
    plot_quest_violins(scores, fig_dir)
    plot_all_scores_overview(scores, fig_dir)


if __name__ == "__main__":
    main()
