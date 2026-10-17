# -*- coding: utf-8 -*-
"""
cut_rest.py
===========
Cut each subject's resting-state MEG recording into ONE continuous segment:

    {stem}_cut.fif

Cut definition
--------------
  START : the FIRST trigger 1 in the file (rest period begins).
  END   : START + 5 min (REST_DURATION_S).

  If a task was restarted, the LAST trigger-1 marks the valid run — see
  USE_LAST_ONSET below. By default the first trigger 1 is used.

Trigger codes
-------------
   1  rest onset
   2  rest offset (not used for the cut)

Input  : /home/ZhangJianfeng_04/emo_study/MEG/1_raw/{subject}/**/*rest*tsss*.fif
         ONLY tSSS files are used, found at any depth below the subject
         folder (with or without a session subfolder). If both _tsss_mc and
         plain _tsss exist for the same recording, _tsss_mc is used.
         Files named *emo_rest* are EXCLUDED (different paradigm).
         If a subject has no matching file, every .fif in its folder is
         listed, so you can see why.
Output : /home/ZhangJianfeng_04/emo_study/Rest_task/2_cut/{subject}_..._cut.fif
         (always starts with the subject ID, even if the raw name does not)

Usage:
    python3 cut_rest.py              # all subjects
    python3 cut_rest.py 031          # subject 031 only
    python3 cut_rest.py --dry-run    # report only, write nothing
    python3 cut_rest.py 031 --dry-run
"""

import os
import sys
import re
import glob
import warnings

import mne
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_ROOT   = '/home/ZhangJianfeng_04/emo_study/MEG/1_raw'
OUTPUT_DIR   = '/home/ZhangJianfeng_04/emo_study/Rest_task/2_cut'
FILE_PATTERN = '*rest*tsss*.fif'   # tSSS files only
EXCLUDE_SUBSTR = 'emo_rest'        # skip the emo_rest paradigm

# ── Trigger codes ──────────────────────────────────────────────────────────────
REST_ONSET_CODE  = 1
REST_OFFSET_CODE = 2

# ── Cut definition ─────────────────────────────────────────────────────────────
REST_DURATION_S = 5 * 60.0         # 5 minutes after trigger 1
USE_LAST_ONSET  = False            # True -> use the last trigger 1 (task restart)

# ── Plausibility check ─────────────────────────────────────────────────────────
# The file must be long enough to contain a full 5-min segment after onset.
MIN_TAIL_S = REST_DURATION_S       # required seconds available after the onset

split_re = re.compile(r'-\d+\.fif$')

# ── Arguments ──────────────────────────────────────────────────────────────────
args           = [a for a in sys.argv[1:]]
DRY_RUN        = '--dry-run' in args
subject_filter = next((a for a in args if not a.startswith('-')), None)


def find_cut_window(raw):
    """
    Determine the cut window for one rest file.

    Returns dict:
        tmin_s, tmax_s   : crop bounds relative to file start
        n_onset          : number of trigger-1 events found
        onset_s          : time (s) of the onset used
    Raises RuntimeError if the file cannot be cut.
    """
    events = mne.find_events(
        raw, stim_channel='STI101', min_duration=0, shortest_event=1, verbose=False
    )
    if len(events) == 0:
        raise RuntimeError("no events on STI101")

    sfreq      = raw.info['sfreq']
    first_samp = raw.first_samp
    total_dur  = raw.times[-1]

    def to_s(sample):
        return (sample - first_samp) / sfreq

    samples = events[:, 0].astype(float)
    codes   = events[:, 2].astype(int)

    onset_samples = samples[codes == REST_ONSET_CODE]
    if len(onset_samples) == 0:
        raise RuntimeError(f"no rest-onset trigger (code {REST_ONSET_CODE}) in file")

    onset = onset_samples[-1] if USE_LAST_ONSET else onset_samples[0]
    tmin_s = to_s(onset)
    tmax_s = tmin_s + REST_DURATION_S

    avail_after = total_dur - tmin_s
    if avail_after < MIN_TAIL_S - 1.0:      # allow ~1 s slack
        raise RuntimeError(
            f"only {avail_after:.1f}s available after onset, "
            f"need {REST_DURATION_S:.0f}s — recording too short")

    # clamp to file end if marginally short
    tmax_s = min(tmax_s, total_dur)

    if tmin_s >= tmax_s:
        raise RuntimeError("computed start is after the end — check triggers")

    return {
        'tmin_s'  : tmin_s,
        'tmax_s'  : tmax_s,
        'n_onset' : len(onset_samples),
        'onset_s' : tmin_s,
    }


def report_window(info):
    tmin, tmax = info['tmin_s'], info['tmax_s']
    dur_min    = (tmax - tmin) / 60.0
    print(f"  Rest-onset triggers  : {info['n_onset']}")
    if info['n_onset'] > 1:
        which = 'last' if USE_LAST_ONSET else 'first'
        print(f"  ! {info['n_onset']} onsets found — using the {which} one "
              f"(t={info['onset_s']:.1f}s).")
    print(f"  Cut window           : {tmin:.1f}s -> {tmax:.1f}s  ({dur_min:.2f} min)")


def cut_and_save(raw, tmin_s, tmax_s, out_path):
    raw_cut = raw.copy().crop(tmin=tmin_s, tmax=tmax_s)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*naming conventions.*',
                                category=RuntimeWarning)
        raw_cut.save(out_path, overwrite=False, split_naming='neuromag', verbose=False)
    print(f"    Saved ({raw_cut.times[-1]/60:.2f} min)  ->  {out_path}")


def find_rest_tsss_files(subj_id):
    """tSSS rest recordings of one subject, at any depth below
    INPUT_ROOT/{subj_id}/. Excludes emo_rest and MNE split parts
    (-1.fif, -2.fif; MNE reads those with their first part). If a
    recording exists as both _tsss_mc and plain _tsss, keeps _tsss_mc."""
    files = sorted(
        f for f in glob.glob(os.path.join(INPUT_ROOT, subj_id, '**', FILE_PATTERN),
                             recursive=True)
        if not split_re.search(f)
        and EXCLUDE_SUBSTR not in os.path.basename(f).lower())
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


def process_file(fif_path, subj_id):
    """Returns True if a cut file exists (written now or already present)."""
    fname = os.path.basename(fif_path)
    stem  = os.path.splitext(fname)[0]
    # 2_cut/ is flat: the output must carry the subject ID, or the
    # "already cut?" check never finds it and subjects can collide
    if not stem.startswith(f"{subj_id}_"):
        stem = f"{subj_id}_{stem}"
    print(f"\n  File : {stem}")

    out_path = os.path.join(OUTPUT_DIR, f"{stem}_cut.fif")
    if os.path.isfile(out_path):
        print(f"  {os.path.basename(out_path)} already exists — skipping.")
        return True

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*naming conventions.*',
                                category=RuntimeWarning)
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)

    print(f"  sfreq={raw.info['sfreq']} Hz | duration={raw.times[-1]/60:.2f} min")

    try:
        info = find_cut_window(raw)
    except RuntimeError as e:
        print(f"  X NOT CUT: {e}")
        return False

    report_window(info)

    if DRY_RUN:
        print("  (dry run — nothing written)")
        return False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cut_and_save(raw, info['tmin_s'], info['tmax_s'], out_path)
    return True


# ── Collect subjects ───────────────────────────────────────────────────────────
subj_ids = sorted(set(
    os.path.basename(d)
    for d in glob.glob(os.path.join(INPUT_ROOT, '*'))
    if os.path.isdir(d) and re.match(r'^\d+$', os.path.basename(d))
))

if subject_filter:
    subj_ids = [s for s in subj_ids if s == subject_filter]

print(f"Found {len(subj_ids)} subject(s): {subj_ids}")
if DRY_RUN:
    print("DRY RUN — no files will be written.")

# ── Main loop — per subject ────────────────────────────────────────────────────
n_ok, n_err, n_skip = 0, 0, 0

for subj_id in subj_ids:
    print(f"\n{'='*70}")
    print(f"Subject : {subj_id}")

    # Step 1 — already cut?
    existing = [f for f in glob.glob(os.path.join(OUTPUT_DIR, f"{subj_id}_*_cut.fif"))
                if not split_re.search(f)]
    if existing:
        print(f"  -> Cut file already exists in 2_cut/ — skipping subject.")
        n_skip += 1
        continue

    # Step 2 — find raw file(s), excluding emo_rest
    raw_files = find_rest_tsss_files(subj_id)

    if not raw_files:
        print(f"  X No rest tSSS file ({FILE_PATTERN}, not emo_rest) for "
              f"subject {subj_id}.")
        present = sorted(glob.glob(os.path.join(INPUT_ROOT, subj_id, '**', '*.fif'),
                                   recursive=True))
        if present:
            print(f"    .fif files that ARE in its folder:")
            for f in present:
                print(f"      {os.path.relpath(f, INPUT_ROOT)}")
        else:
            print(f"    (no .fif files at all under {os.path.join(INPUT_ROOT, subj_id)})")
        n_err += 1
        continue

    print(f"  Raw files found ({len(raw_files)}):")
    for f in raw_files:
        print(f"    {os.path.relpath(f, INPUT_ROOT)}")

    # Step 3 — process
    subj_ok = False
    for fif_path in raw_files:
        try:
            if process_file(fif_path, subj_id):
                subj_ok = True
        except Exception as e:
            print(f"  X Error on {os.path.basename(fif_path)}: {e}")

    if subj_ok:
        print(f"\n  Subject {subj_id} complete.")
        n_ok += 1
    else:
        print(f"\n  ! Subject {subj_id} produced no cut file — check the log above.")
        n_err += 1

print(f"\n{'='*70}")
print(f"Done — {n_ok} subject(s) cut | {n_skip} skipped | {n_err} failed")
print(f"Output dir: {OUTPUT_DIR}")
