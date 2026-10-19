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

# ── Padding ────────────────────────────────────────────────────────────────────
PAD_BEFORE_BLOCK1_S = 2.0   # seconds before rest_onset, block 1 only
PAD_AFTER_S         = 2.0   # seconds after rest_offset, all blocks

# ── Optional subject filter ────────────────────────────────────────────────────
subject_filter = sys.argv[1] if len(sys.argv) > 1 else None

split_re = re.compile(r'-\d+\.fif$')


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
    Returns (complete: bool, missing: list of block labels)
    """
    missing = []
    for pattern, label in [
        (f"{subj_id}_*_1b_base*.fif", "1b_base"),
        (f"{subj_id}_*_2b_*.fif",     "2b"),
        (f"{subj_id}_*_3b_*.fif",     "3b"),
    ]:
        matches = [f for f in glob.glob(os.path.join(OUTPUT_DIR, pattern))
                   if not split_re.search(f)]
        if not matches:
            missing.append(label)
    return len(missing) == 0, missing


def find_key_events(raw):
    """
    For each file, find valid blocks by:

    VIDEO BLOCKS:
      For each video trigger (code 7/8), find the next rest_onset and
      rest_offset after it. If rest_offset - rest_onset is 19-22 min → valid.
      If not → skip and report.

    BASE BLOCKS:
      Find rest_onset → rest_offset pairs with 19-22 min duration that have
      no video trigger between the preceding rest_offset and this rest_onset.

    All times are relative to raw.first_samp (i.e. raw.times[0]=0 reference).

    Returns: list of dicts with keys:
      'type'      : 'base' or 'video'
      'tmin_s'    : cut start time (seconds, relative to file start)
      'tmax_s'    : cut end time
      'condition' : 'base', 'sad', 'fear'
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

    # ── VIDEO BLOCKS ───────────────────────────────────────────────────────
    for vid in video_all:
        t_vid  = to_s(vid[0])
        code   = int(vid[2])
        label  = 'sad' if code == VIDEO_SAD_CODE else 'fear'

        next_on  = rest_on_all[rest_on_all > vid[0]]
        if len(next_on) == 0:
            print(f"  ! video {label} t={t_vid:.1f}s → no rest_onset after → SKIP")
            continue
        ron = next_on[0]

        next_off = rest_off_all[rest_off_all > ron]
        if len(next_off) == 0:
            print(f"  ! video {label} t={t_vid:.1f}s → no rest_offset after rest_onset → SKIP")
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

    if not blocks:
        raise RuntimeError("No valid blocks found.")

    print(f"  → {sum(b['type']=='base' for b in blocks)} base, "
          f"{sum(b['type']=='video' for b in blocks)} video block(s) found.")

    return blocks


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


def process_file(fif_path, subj_id):
    fname = os.path.basename(fif_path)
    stem  = os.path.splitext(fname)[0]
    # outputs must start with the subject ID: 2_cut/ is flat, so a raw file
    # named without it (e.g. emo_rest_tsss_mc.fif) would otherwise produce
    # the same output name for every subject and not be found by the
    # completeness check or by Preproc_MEG_dyn_cut.py
    if not stem.startswith(f"{subj_id}_"):
        stem = f"{subj_id}_{stem}"
    print(f"\n  File : {stem}")

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*naming conventions.*',
                                category=RuntimeWarning)
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
    sfreq     = raw.info['sfreq']
    total_dur = raw.times[-1]
    print(f"  sfreq={sfreq} Hz | duration={total_dur/60:.2f} min")

    try:
        blocks = find_key_events(raw)
    except Exception as e:
        print(f"  X Could not find events: {e}")
        return []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cuts_made   = []
    base_count  = 0
    video_count = 0

    for block in blocks:
        condition = block['condition']
        tmin_s    = block['tmin_s']
        tmax_s    = block['tmax_s']

        if block['type'] == 'base':
            suffix   = '_1b_base' if base_count == 0 else f'_1b_base{base_count+1}'
            out_path = os.path.join(OUTPUT_DIR, f"{stem}{suffix}.fif")
            base_count += 1
        else:
            block_num  = 2 + video_count
            out_path   = os.path.join(OUTPUT_DIR, f"{stem}_{block_num}b_{condition}.fif")
            video_count += 1

        if os.path.isfile(out_path):
            print(f"  {os.path.basename(out_path)} already exists — skipping.")
            cuts_made.append(out_path)
            continue

        print(f"  Cutting {os.path.basename(out_path)}: "
              f"{tmin_s:.1f}s → {tmax_s:.1f}s ({(tmax_s-tmin_s)/60:.1f} min)")
        cut_and_save(raw, tmin_s, tmax_s, out_path)
        cuts_made.append(out_path)

    return cuts_made


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

for subj_id in subj_ids:
    print(f"\n{'='*60}")
    print(f"Subject : {subj_id}")

    # Step 1: check output folder
    complete, missing = subject_outputs_complete(subj_id)
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

    # Step 3: process each raw file
    for fif_path in raw_files:
        try:
            process_file(fif_path, subj_id)
            n_ok += 1
        except Exception as e:
            print(f"  X Error on {os.path.basename(fif_path)}: {e}")
            n_err += 1

    # Step 4: report final status for this subject
    complete_after, missing_after = subject_outputs_complete(subj_id)
    if complete_after:
        print(f"  ✓ Subject {subj_id} complete.")
    else:
        print(f"  ! Subject {subj_id} still missing: {missing_after}")

print(f"\n{'='*60}")
print(f"Done — {n_ok} file(s) processed | {n_skip} subject(s) skipped | {n_err} error(s)")
print(f"Output dir: {OUTPUT_DIR}")