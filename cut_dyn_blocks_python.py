# -*- coding: utf-8 -*-
"""
cut_dyn_blocks_python.py
========================
Cut each subject's MEG rest-task recording(s) into 3 block segments:
  {subj}_*_1b_base.fif   — baseline rest block
  {subj}_*_2b_sad.fif    — sad video + rest  (or fear)
  {subj}_*_3b_fear.fif   — fear video + rest (or sad)

Resume logic (per subject, not per file):
  1. Check 2_cut/ for existing 1b_base, 2b_*, 3b_* files for this subject.
  2. If all 3 exist → skip subject entirely (no file is opened).
  3. If any missing → open the raw file(s) and cut what's missing.

Block numbering is per SUBJECT, not per raw file: when a session was split
over several raw files (e.g. 036_emo_rest1 + 036_emo_rest_2_3), the files are
processed in recording order (raw.info['meas_date'] = the date/time the
recording was started, stored in the FIF header) and the numbering continues
from one file to the next (1b, 2b, 3b).

Repeated video triggers: if a video trigger is followed by another video
trigger before its rest_onset (video started, stopped, restarted), only the
LAST trigger before that rest is used — the one followed by the full rest
block with no other video trigger in between.

Recording stopped during a video (e.g. 042): if a raw file ends after a video
trigger whose rest never started or never finished there, and the NEXT file
starts with a rest that has no video trigger before it, that rest is the
video's block (labelled with the video's condition, cut from rest_onset - 2 s;
the video itself is not in that file). The log says so.

Recording stopped during a rest (e.g. 036 sad): a video rest without
rest_offset that runs to the end of the file is kept as a PARTIAL block if
it lasted at least PARTIAL_REST_MIN_S.

Repeated conditions (e.g. 034, where fear and sad were shown again after a
crash): only the FIRST valid block of each condition is cut; later repeats
are skipped and logged.

Log file: everything printed is also written to
  {OUTPUT_DIR}/logs/cut_dyn_blocks_YYYYmmdd_HHMMSS.log

Block detection is trigger-based only — filename content is never assumed:
  Base blocks  : rest_offset count − video_onset count
  Video blocks : video_onset count (code 7=sad, 8=fear)
  Onset/offset:
    Block 1 (base) : onset = rest_onset (code 1) − 2 s
    Block 2/3      : onset = video_onset (code 7/8), no pre-pad
    All blocks     : offset = rest_offset (code 2) + 2 s

Input  : /home/ZhangJianfeng_04/emo_study/MEG/1_raw/{subject}/**/*emo*rest*tsss*.fif
         ONLY tSSS-processed files are used. Searched at any depth below
         the subject folder (with or without a session subfolder). Where
         both a _tsss_mc and a plain _tsss version of the same recording
         exist, the _tsss_mc one is used.
Output : /home/ZhangJianfeng_04/emo_study/MEG/2_cut/{subject}_..._{1b_base,2b_*,3b_*}.fif
         Output names always start with the subject ID, even when the raw
         file name does not.

Plain Python, no SLURM. Usage:
    python3 cut_dyn_blocks_python.py            # process ALL subjects found
    python3 cut_dyn_blocks_python.py 001        # process subject 001 only
"""

import os
import sys
import re
import glob
import mne
import numpy as np
import warnings
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_ROOT  = '/home/ZhangJianfeng_04/emo_study/MEG/1_raw'
OUTPUT_DIR  = '/home/ZhangJianfeng_04/emo_study/MEG/2_cut'
FILE_PATTERN = '*emo*rest*tsss*.fif'     # tSSS files only

# ── Trigger codes ──────────────────────────────────────────────────────────────
BLOCK_ONSET_CODE = 10
REST_ONSET_CODE  =  1
REST_OFFSET_CODE =  2
VIDEO_SAD_CODE   =  7
VIDEO_FEAR_CODE  =  8

# ── Rest duration validation ───────────────────────────────────────────────────
REST_DUR_MIN_S = 18 * 60   # below this = aborted/practice run
REST_DUR_MAX_S = 22 * 60   # above this = something wrong
PARTIAL_REST_MIN_S = 10 * 60   # a video rest cut off by the END OF THE FILE
                               # (no rest_offset, recording stopped) is kept
                               # as a partial block if at least this long

# ── Padding ────────────────────────────────────────────────────────────────────
PAD_BEFORE_BLOCK1_S = 2.0   # seconds before rest_onset, block 1 only
PAD_AFTER_S         = 2.0   # seconds after rest_offset, all blocks

# ── Optional subject filter ────────────────────────────────────────────────────
subject_filter = sys.argv[1] if len(sys.argv) > 1 else None

split_re = re.compile(r'-\d+\.fif$')


class Tee:
    """Write everything printed to the terminal AND to a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(
    LOG_DIR, f"cut_dyn_blocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_file = open(LOG_PATH, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)


def find_tsss_files(subj_id):
    """All tSSS rest recordings of one subject, at any depth below
    INPUT_ROOT/{subj_id}/. MNE split parts (-1.fif, -2.fif) are excluded
    because MNE reads them together with their first part. If a recording
    exists both as _tsss_mc and as plain _tsss, only _tsss_mc is kept."""
    files = sorted(
        f for f in glob.glob(os.path.join(INPUT_ROOT, subj_id, '**', FILE_PATTERN),
                             recursive=True)
        if not split_re.search(f) and 'tsss' in os.path.basename(f).lower())
    have_mc = {re.sub(r'_tsss_mc\.fif$', '', f, flags=re.I)
               for f in files if re.search(r'_tsss_mc\.fif$', f, flags=re.I)}
    kept = []
    for f in files:
        base = re.sub(r'_tsss\.fif$', '', f, flags=re.I)
        if base != f and base in have_mc:
            print(f"    (ignoring {os.path.basename(f)} - _tsss_mc version exists)")
            continue
        kept.append(f)
    return kept


def subject_outputs_complete(subj_id):
    """
    Check if all 3 output blocks already exist for this subject in OUTPUT_DIR.
    Looks for any file matching:
      {subj_id}_*_1b_base*.fif
      {subj_id}_*_2b_*.fif      (sad or fear)
      {subj_id}_*_3b_*.fif      (sad or fear)
    Also warns (does not fix) when the outputs look wrong: a block number
    other than 1b-3b (e.g. 4b), a block present more than once, or the two
    video blocks not being one sad + one fear. Such outputs come from the
    old per-file numbering - fix them with fix_double_2b.py (two 2b) or
    fix_cut_names.py before cutting again.
    Returns (complete: bool, missing: list of block labels,
             wrong: bool - True if any WARNING was printed)
    """
    missing = []
    wrong = False
    conditions = []
    for pattern, label in [
        (f"{subj_id}_*_1b_base*.fif", "1b_base"),
        (f"{subj_id}_*_2b_*.fif",     "2b"),
        (f"{subj_id}_*_3b_*.fif",     "3b"),
    ]:
        matches = [f for f in glob.glob(os.path.join(OUTPUT_DIR, pattern))
                   if not split_re.search(f)]
        if not matches:
            missing.append(label)
        elif len(matches) > 1:
            wrong = True
            print(f"  ! WARNING: {label} exists {len(matches)}x: "
                  f"{[os.path.basename(f) for f in matches]}")
        if label != "1b_base":
            conditions += [re.search(r'_\db_([a-z]+)\.fif$', f).group(1)
                           for f in matches]
    extra = [f for f in glob.glob(os.path.join(OUTPUT_DIR, f"{subj_id}_*_[4-9]b_*.fif"))
             if not split_re.search(f)]
    if extra:
        wrong = True
        print(f"  ! WARNING: unexpected block number(s): "
              f"{[os.path.basename(f) for f in extra]}")
    if len(conditions) >= 2 and sorted(conditions) != ['fear', 'sad']:
        wrong = True
        print(f"  ! WARNING: video blocks are {conditions}, expected one sad + one fear")
    return len(missing) == 0, missing, wrong


def find_key_events(raw, pending=None):
    """
    For each file, find valid blocks by:

    VIDEO BLOCKS:
      For each video trigger (code 7/8), find the next rest_onset and
      rest_offset after it. If rest_offset - rest_onset is 19-22 min → valid.
      If not → skip and report.

    BASE BLOCKS:
      Find rest_onset → rest_offset pairs with 19-22 min duration that have
      no video trigger between the preceding rest_offset and this rest_onset.

    CONTINUED VIDEO BLOCK:
      pending = condition of a video at the end of the PREVIOUS file whose
      rest was not recorded there. The first rest of this file is that
      video's block if no video trigger comes before it in this file.

    All times are relative to raw.first_samp (i.e. raw.times[0]=0 reference).

    Returns (blocks, dangling):
      blocks   : list of dicts with keys
                   'type'      : 'base' or 'video'
                   'tmin_s'    : cut start time (seconds, relative to file start)
                   'tmax_s'    : cut end time
                   'condition' : 'base', 'sad', 'fear'
      dangling : condition of the LAST video trigger of this file if its rest
                 did not start or did not end in this file, else None
    """
    events = mne.find_events(
        raw, stim_channel='STI101', min_duration=0.002, verbose=False
    )
    sfreq      = raw.info['sfreq']
    first_samp = raw.first_samp
    total_dur  = raw.times[-1]

    def to_s(sample):
        """Convert absolute sample to time relative to file start."""
        return (sample - first_samp) / sfreq

    rest_on_all  = events[events[:, 2] == REST_ONSET_CODE,  0]
    rest_off_all = events[events[:, 2] == REST_OFFSET_CODE, 0]
    video_all    = events[np.isin(events[:, 2], [VIDEO_SAD_CODE, VIDEO_FEAR_CODE])]

    blocks = []
    dangling = None

    # ── VIDEO BLOCKS ───────────────────────────────────────────────────────
    for i, vid in enumerate(video_all):
        is_last = i == len(video_all) - 1
        t_vid  = to_s(vid[0])
        code   = int(vid[2])
        label  = 'sad' if code == VIDEO_SAD_CODE else 'fear'

        next_on  = rest_on_all[rest_on_all > vid[0]]
        if len(next_on) == 0:
            print(f"  ! video {label} t={t_vid:.1f}s → no rest_onset after → SKIP")
            if is_last:
                dangling = label
            continue
        ron = next_on[0]

        # another video trigger before this rest_onset → this one was
        # aborted/restarted; the later trigger owns the rest block
        later_vids = video_all[(video_all[:, 0] > vid[0]) & (video_all[:, 0] < ron)]
        if len(later_vids) > 0:
            nxt = later_vids[0]
            nxt_label = 'sad' if int(nxt[2]) == VIDEO_SAD_CODE else 'fear'
            print(f"  ✗ video {label} t={t_vid:.1f}s → superseded by video {nxt_label} "
                  f"t={to_s(nxt[0]):.1f}s before rest_onset → SKIP (repeated trigger)")
            continue

        next_off = rest_off_all[rest_off_all > ron]
        if len(next_off) == 0:
            rec = total_dur - to_s(ron)
            if is_last and rec >= PARTIAL_REST_MIN_S:
                print(f"  ✓ video {label} t={t_vid:.1f}s → rest {to_s(ron):.1f}s–END of file "
                      f"({rec/60:.1f} min, no rest_offset) → PARTIAL block")
                blocks.append({
                    'type'       : 'video',
                    'condition'  : label,
                    'tmin_s'     : t_vid,
                    'tmax_s'     : total_dur,
                    'roff_sample': first_samp + len(raw.times) - 1,
                })
                continue
            print(f"  ! video {label} t={t_vid:.1f}s → no rest_offset after rest_onset → SKIP")
            if is_last:
                dangling = label
            continue
        roff = next_off[0]

        dur = (roff - ron) / sfreq
        if not (REST_DUR_MIN_S <= dur <= REST_DUR_MAX_S):
            print(f"  ✗ video {label} t={t_vid:.1f}s → rest {to_s(ron):.1f}–{to_s(roff):.1f}s "
                  f"({dur/60:.1f} min) → SKIP (wrong duration)")
            continue

        print(f"  ✓ video {label} t={t_vid:.1f}s → rest {to_s(ron):.1f}–{to_s(roff):.1f}s "
              f"({dur/60:.1f} min) → VALID")
        blocks.append({
            'type'       : 'video',
            'condition'  : label,
            'tmin_s'     : t_vid,
            'tmax_s'     : min(to_s(roff) + PAD_AFTER_S, total_dur),
            'roff_sample': roff,
        })

    # ── BASE BLOCKS ────────────────────────────────────────────────────────
    used_roff = {b['roff_sample'] for b in blocks}

    for roff in rest_off_all:
        if roff in used_roff:
            continue

        preceding_on = rest_on_all[rest_on_all < roff]
        if len(preceding_on) == 0:
            continue
        ron = preceding_on[-1]
        dur = (roff - ron) / sfreq

        if not (REST_DUR_MIN_S <= dur <= REST_DUR_MAX_S):
            print(f"  ✗ base candidate {to_s(ron):.1f}–{to_s(roff):.1f}s "
                  f"({dur/60:.1f} min) → SKIP (wrong duration)")
            continue

        if pending and not len(video_all[video_all[:, 0] < ron]):
            # first rest of this file, no video before it, and the previous
            # file stopped during a video -> it is that video's rest block
            print(f"  ✓ rest {to_s(ron):.1f}–{to_s(roff):.1f}s ({dur/60:.1f} min) has no "
                  f"video trigger before it, previous file ended during the {pending} "
                  f"video → {pending.upper()} block (rest only, video not in this file)")
            blocks.append({
                'type'       : 'video',
                'condition'  : pending,
                'tmin_s'     : max(0.0, to_s(ron) - PAD_BEFORE_BLOCK1_S),
                'tmax_s'     : min(to_s(roff) + PAD_AFTER_S, total_dur),
                'roff_sample': roff,
            })
            pending = None
            continue

        prev_roff = max([b['roff_sample'] for b in blocks] + [0])
        vids_in_window = video_all[
            (video_all[:, 0] > prev_roff) & (video_all[:, 0] < ron)
        ]
        if len(vids_in_window) > 0:
            print(f"  ✗ base candidate {to_s(ron):.1f}–{to_s(roff):.1f}s "
                  f"→ SKIP (video trigger present before rest_onset)")
            continue

        print(f"  ✓ base block: rest {to_s(ron):.1f}–{to_s(roff):.1f}s "
              f"({dur/60:.1f} min) → VALID")
        blocks.append({
            'type'       : 'base',
            'condition'  : 'base',
            'tmin_s'     : max(0.0, to_s(ron) - PAD_BEFORE_BLOCK1_S),
            'tmax_s'     : min(to_s(roff) + PAD_AFTER_S, total_dur),
            'roff_sample': roff,
        })

    blocks.sort(key=lambda b: b['tmin_s'])

    if pending:
        print(f"  (previous file ended during the {pending} video, but this file "
              f"does not start with a rest without video - not used)")

    if not blocks:
        raise RuntimeError("No valid blocks found.")

    print(f"  → {sum(b['type']=='base' for b in blocks)} base, "
          f"{sum(b['type']=='video' for b in blocks)} video block(s) found.")
    if dangling:
        print(f"  ! file ends during the {dangling} video - its rest may continue "
              f"in the next file")

    return blocks, dangling


def get_condition_label(video_code):
    if video_code == VIDEO_SAD_CODE:
        return 'sad'
    if video_code == VIDEO_FEAR_CODE:
        return 'fear'
    return None   # unknown — will be skipped, not saved


def cut_and_save(raw, tmin_s, tmax_s, out_path):
    raw_cut = raw.copy().crop(tmin=tmin_s, tmax=tmax_s)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*naming conventions.*',
                                category=RuntimeWarning)
        raw_cut.save(out_path, overwrite=False, split_naming='neuromag', verbose=False)
    print(f"    Saved ({raw_cut.times[-1]/60:.2f} min)  ->  {out_path}")


def process_file(fif_path, subj_id, raw, base_count, video_count, pending=None,
                 done_conditions=None):
    """Cut one raw file. base_count / video_count = number of base / video
    blocks already found in this subject's EARLIER raw files, so numbering
    continues across split recordings. pending = condition of a video the
    previous file ended in (see find_key_events). done_conditions = video
    conditions already cut from earlier blocks of this subject (updated here;
    repeats are skipped).
    Returns (base_count, video_count, dangling)."""
    if done_conditions is None:
        done_conditions = set()
    fname = os.path.basename(fif_path)
    stem  = os.path.splitext(fname)[0]
    # outputs must start with the subject ID: 2_cut/ is flat, so a raw file
    # named without it (e.g. emo_rest_tsss_mc.fif) would otherwise produce
    # the same output name for every subject and not be found by the
    # completeness check or by Preproc_MEG_dyn_cut.py
    if not stem.startswith(f"{subj_id}_"):
        stem = f"{subj_id}_{stem}"
    print(f"\n  File : {stem}")
    print(f"  meas_date={raw.info['meas_date']}")

    sfreq     = raw.info['sfreq']
    total_dur = raw.times[-1]
    print(f"  sfreq={sfreq} Hz | duration={total_dur/60:.2f} min")

    try:
        blocks, dangling = find_key_events(raw, pending)
    except Exception as e:
        print(f"  X Could not find events: {e}")
        return base_count, video_count, None

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for block in blocks:
        condition = block['condition']
        tmin_s    = block['tmin_s']
        tmax_s    = block['tmax_s']

        if block['type'] == 'base':
            suffix   = '_1b_base' if base_count == 0 else f'_1b_base{base_count+1}'
            out_path = os.path.join(OUTPUT_DIR, f"{stem}{suffix}.fif")
            base_count += 1
        else:
            if condition in done_conditions:
                print(f"  ✗ {condition} block {tmin_s:.1f}s → repeat of an earlier "
                      f"{condition} block → SKIP (first one is kept)")
                continue
            done_conditions.add(condition)
            block_num  = 2 + video_count
            out_path   = os.path.join(OUTPUT_DIR, f"{stem}_{block_num}b_{condition}.fif")
            video_count += 1

        if os.path.isfile(out_path):
            print(f"  {os.path.basename(out_path)} already exists — skipping.")
            continue

        print(f"  Cutting {os.path.basename(out_path)}: "
              f"{tmin_s:.1f}s → {tmax_s:.1f}s ({(tmax_s-tmin_s)/60:.1f} min)")
        cut_and_save(raw, tmin_s, tmax_s, out_path)

    return base_count, video_count, dangling


def read_raw(fif_path):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*naming conventions.*',
                                category=RuntimeWarning)
        return mne.io.read_raw_fif(fif_path, preload=False, verbose=False)


# ── Collect subjects ───────────────────────────────────────────────────────────
# Subject IDs = numeric folder names directly under INPUT_ROOT
subj_ids = sorted(set(
    os.path.basename(d)
    for d in glob.glob(os.path.join(INPUT_ROOT, '*'))
    if os.path.isdir(d) and re.match(r'^\d+$', os.path.basename(d))
))

if subject_filter:
    subj_ids = [s for s in subj_ids if s == subject_filter]

print(f"Found {len(subj_ids)} subject(s): {subj_ids}")

# ── Main loop — per subject ────────────────────────────────────────────────────
n_ok, n_err, n_skip = 0, 0, 0
wrong_subjects = []

for subj_id in subj_ids:
    print(f"\n{'='*60}")
    print(f"Subject : {subj_id}")

    # Step 1: check output folder
    complete, missing, wrong = subject_outputs_complete(subj_id)
    if wrong:
        # cutting now would add correctly named files NEXT TO the old wrong
        # ones (same block twice under two names) - fix the names first
        print(f"  X Old wrongly numbered files in 2_cut/ — subject SKIPPED. "
              f"Fix names first (fix_double_2b.py / fix_cut_names.py).")
        wrong_subjects.append(subj_id)
        continue
    if complete:
        print(f"  -> All 3 blocks exist in 2_cut/ — skipping subject.")
        n_skip += 1
        continue

    print(f"  -> Missing: {missing} — will process raw file(s).")

    # Step 2: find all raw files for this subject
    raw_files = find_tsss_files(subj_id)

    if not raw_files:
        print(f"  X No tSSS files ({FILE_PATTERN}) found under "
              f"{os.path.join(INPUT_ROOT, subj_id)}.")
        n_err += 1
        continue

    print(f"  Raw files found ({len(raw_files)}):")
    for f in raw_files:
        print(f"    {os.path.relpath(f, INPUT_ROOT)}")

    # Step 3: open all raw files and put them in recording order
    # (meas_date = recording start time; file name order as fallback)
    opened = []
    for i, fif_path in enumerate(raw_files):
        try:
            opened.append((fif_path, read_raw(fif_path), i))
        except Exception as e:
            print(f"  X Error opening {os.path.basename(fif_path)}: {e}")
            n_err += 1
    if all(r.info['meas_date'] is not None for _, r, _ in opened):
        opened.sort(key=lambda x: x[1].info['meas_date'])
    else:
        print("  ! meas_date missing - using file name order")

    # Step 4: process each raw file; block numbering continues across files
    base_count, video_count, pending, done_conditions = 0, 0, None, set()
    for fif_path, raw, _ in opened:
        try:
            base_count, video_count, pending = process_file(
                fif_path, subj_id, raw, base_count, video_count, pending,
                done_conditions)
            n_ok += 1
        except Exception as e:
            print(f"  X Error on {os.path.basename(fif_path)}: {e}")
            n_err += 1

    # Step 5: report final status for this subject
    complete_after, missing_after, wrong_after = subject_outputs_complete(subj_id)
    if wrong_after:
        print(f"  ! Subject {subj_id} outputs look wrong - see WARNING above")
        wrong_subjects.append(subj_id)
    elif complete_after:
        print(f"  ✓ Subject {subj_id} complete.")
    else:
        print(f"  ! Subject {subj_id} still missing: {missing_after}")

print(f"\n{'='*60}")
print(f"Done — {n_ok} file(s) processed | {n_skip} subject(s) skipped | {n_err} error(s)")
if wrong_subjects:
    print(f"Wrongly numbered, NOT processed: {wrong_subjects}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Log file  : {LOG_PATH}")