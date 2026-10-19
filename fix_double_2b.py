# -*- coding: utf-8 -*-
"""
fix_double_2b.py
================
Repair subjects that have TWO 2b blocks in 2_cut/ (split recordings cut by
the old cut_dyn_blocks_python.py, which numbered blocks per raw file), e.g.
  015_emo_rest_1_2_tsss_mc_2b_sad.fif    (recorded first)  -> stays 2b
  015_emo_rest_3_tsss_mc_2b_fear.fif     (recorded later)  -> becomes 3b

Which one is "later" is read from the cut files themselves: recording start
(meas_date) + offset of the cut in that recording (first_samp). File names
are not used for this.

For the later 2b, below MEG/ (except 1_raw/):
  - every file whose name contains its cut stem is renamed 2b -> 3b
    (2_cut, 3_preproc, ... - steps that ADD text to the name are found too)
  - SourceSpace/stc/{subj}_2b_{cond}_*.stc is renamed to {subj}_3b_{cond}_*
  - if the 3b name already exists (the new cut script already cut it, as for
    015), the old 2b file is a duplicate of the same block and is MOVED to
    MEG/_trash_cut_fix/ instead
  - SourceSpace/label_tc/{subj}_*_label_tc.npz stores the block names inside,
    so it is moved to the trash as well; SS_emo_full.py then recomputes it

Nothing is deleted: everything removed goes to MEG/_trash_cut_fix/.

Usage:
    python3 fix_double_2b.py           # dry run: prints what it would do
    python3 fix_double_2b.py --apply   # really rename / move
"""

import os
import re
import sys
import glob
import shutil
import datetime
import warnings
from collections import defaultdict

import mne

MEG       = '/home/ZhangJianfeng_04/emo_study/MEG'
CUT_DIR   = os.path.join(MEG, '2_cut')
TC_DIR    = os.path.join(MEG, 'SourceSpace', 'label_tc')
TRASH_DIR = os.path.join(MEG, '_trash_cut_fix')
SKIP_DIRS = {'1_raw', '_trash_cut_fix'}       # top level of MEG, never touched

cut_re = re.compile(r'^(\d+)_.*_(\d)b_([a-z]+)\.fif$')   # split parts -1.fif don't match


def start_time(path):
    """Absolute start of a cut: recording start + offset of the cut in it."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = mne.io.read_raw_fif(path, preload=False, verbose=False)
    if r.info['meas_date'] is None:
        return None
    return r.info['meas_date'] + datetime.timedelta(
        seconds=r.first_samp / r.info['sfreq'])


def find_cases():
    """[(subj, cond, old_stem, new_stem)] for every subject with two 2b."""
    cuts = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(CUT_DIR, '*.fif'))):
        m = cut_re.match(os.path.basename(f))
        if m:
            cuts[m.group(1)].append((int(m.group(2)), m.group(3), f))

    cases = []
    for subj, blocks in sorted(cuts.items()):
        two = [b for b in blocks if b[0] == 2]
        if len(two) < 2:
            continue
        print(f"\n{subj}: 2b used {len(two)}x")
        if len(two) > 2 or two[0][1] == two[1][1]:
            print("  ! more than two 2b, or both the same emotion - SKIPPED, check by hand")
            continue
        times = [(start_time(p), c, p) for _, c, p in two]
        if any(t is None for t, _, _ in times):
            print("  ! no meas_date - cannot tell which was recorded later - SKIPPED")
            continue
        times.sort(key=lambda x: x[0])
        for t, _, p in times:
            print(f"  {t:%Y-%m-%d %H:%M:%S}  {os.path.basename(p)}")
        _, cond, later = times[1]
        old = os.path.basename(later)[:-4]
        new = re.sub(r'_2b_([a-z]+)$', r'_3b_\1', old)
        three = [b for b in blocks if b[0] == 3]
        if three and [os.path.basename(p)[:-4] for _, _, p in three] != [new]:
            print(f"  ! a different 3b already exists "
                  f"({[os.path.basename(p) for _, _, p in three]}) - SKIPPED, check by hand")
            continue
        print(f"  -> later one is 3b: {old} -> {new}"
              f"{'   (3b already cut - old 2b goes to trash)' if three else ''}")
        cases.append((subj, cond, old, new))
    return cases


def plan(cases):
    """[(op, src, dst)] with op 'RENAME' or 'TRASH'."""
    ops = []

    def trash(src):
        return os.path.join(TRASH_DIR, os.path.relpath(src, MEG))

    for dirpath, dirnames, filenames in os.walk(MEG):
        if dirpath == MEG:
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            src = os.path.join(dirpath, fn)
            for subj, cond, old, new in cases:
                rx = re.compile(re.escape(old) + r'(?![A-Za-z0-9])')
                stc_old = f'{subj}_2b_{cond}_'
                if rx.search(fn):
                    dst_name = rx.sub(new, fn)
                elif fn.startswith(stc_old) and fn.endswith('.stc'):
                    dst_name = f'{subj}_3b_{cond}_' + fn[len(stc_old):]
                elif dirpath == TC_DIR and fn.startswith(f'{subj}_') \
                        and fn.endswith('_label_tc.npz'):
                    ops.append(('TRASH', src, trash(src)))
                    continue
                else:
                    continue
                dst = os.path.join(dirpath, dst_name)
                if os.path.exists(dst):            # already redone under the new name
                    ops.append(('TRASH', src, trash(src)))
                else:
                    ops.append(('RENAME', src, dst))
    return ops


if __name__ == '__main__':
    apply = '--apply' in sys.argv
    print('APPLY - renaming / moving files' if apply
          else 'DRY RUN - nothing is changed (use --apply)')

    cases = find_cases()
    if not cases:
        print('\nNothing to fix.')
        sys.exit(0)

    ops = plan(cases)
    print(f"\n{len(ops)} operation(s):")
    for op, src, dst in ops:
        print(f"  {op:<6} {os.path.relpath(src, MEG)}\n"
              f"         -> {os.path.relpath(dst, MEG)}")

    if apply:
        for op, src, dst in ops:
            if os.path.exists(dst):
                print(f"  ! {os.path.relpath(dst, MEG)} exists - {os.path.basename(src)} left alone")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        print('\nDone. Rerun SS_emo_full.py for: '
              + ' '.join(sorted({c[0] for c in cases})))
