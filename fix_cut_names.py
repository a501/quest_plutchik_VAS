# -*- coding: utf-8 -*-
"""
fix_cut_names.py
================
Repair block names produced by the OLD cut_dyn_blocks_python.py, in 2_cut/
AND in every later processing step.

The old script numbered blocks per raw file (not per subject) and did not
handle repeated video triggers, so some subjects got wrong names:
  046 (split recording) : rest_3 file got _2b_sad   -> should be _3b_sad
  054, 034 (fear trigger twice): _2b_fear (aborted start) + _3b_fear + _4b_sad
                          -> trash _2b_fear, _3b_fear -> _2b_fear, _4b_sad -> _3b_sad

Every file or folder anywhere below SEARCH_ROOT whose name CONTAINS an old
block stem (followed by a non-letter/digit or the end of the name) is
renamed by replacing that stem, so later steps that ADD text to the name
(e.g. ..._2b_sad_ica.fif, ..._2b_sad-epo.fif) are found too. Files to
delete are MOVED to {SEARCH_ROOT}/_trash_cut_fix/ (not deleted), keeping
their folder structure, so nothing is lost if a decision was wrong.

Text INSIDE files (CSV tables, logs that list file names) is NOT changed.

Usage:
    python3 fix_cut_names.py           # dry run: only prints what it would do
    python3 fix_cut_names.py --apply   # really rename / move
    python3 fix_cut_names.py --scan    # list subjects in 2_cut/ that look wrong

Before --apply for a duplicate-trigger subject, check in the log of the new
cut_dyn_blocks_python.py (or with the trigger printout) that the FIRST video
trigger is the one marked "superseded".
"""

import os
import re
import sys
import glob
import shutil
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
SEARCH_ROOT = '/home/ZhangJianfeng_04/emo_study/MEG'
CUT_DIR     = os.path.join(SEARCH_ROOT, '2_cut')
TRASH_DIR   = os.path.join(SEARCH_ROOT, '_excluded')   # None-entries are MOVED here
SKIP_DIRS   = {'1_raw', '_trash_cut_fix', '_excluded'}   # never touched (top level of SEARCH_ROOT)

# ── Fixes, applied IN THIS ORDER: (old stem, new stem or None = move to trash
#    [, folder below SEARCH_ROOT the entry is limited to])
# The fixes for 046, 054 and 034 were already applied / redone (054 and 034
# were recut with the new cut script). Do NOT put them back: running
# ('054_emo_rest_tsss_mc_2b_fear', None) again would trash the CORRECT new file.
FIXES = [
    # 034: fear shown twice, MEG complete for both -> fear #1 is used, fear #2
    # and everything made from it is MOVED to MEG/_excluded/. Works whether
    # the files still have the old names (2b_fear, 3b_fear, 4b_sad) or were
    # already renamed (2b_1_fear, 2b_2_fear, 3b_sad). Order matters.
    ('034_emo_rest_tsss_mc_3b_fear',   None),                            # fear #2, old name
    ('034_emo_rest_tsss_mc_2b_2_fear', None),                            # fear #2, new name
    ('034_emo_rest_tsss_mc_2b_1_fear', '034_emo_rest_tsss_mc_2b_fear'),  # fear #1
    ('034_emo_rest_tsss_mc_4b_sad',    '034_emo_rest_tsss_mc_3b_sad'),   # sad #2
    # source space: STCs ({subj}_{block}_{cond}_{ver}_eloreta-?h.stc)
    ('034_3b_fear',   None),
    ('034_2b_2_fear', None),
    ('034_2b_1_fear', '034_2b_fear'),
    ('034_4b_sad',    '034_3b_sad'),
    # source-space dynamics ({subj}_ecg_static.csv, ...) contain fear #2 ->
    # moved, then recomputed. ONLY in SourceSpace/dynamics: the label_tc npz
    # was cut in place (fear #2 removed, sad 4b -> 3b) and stays.
    ('034_ecg', None, 'SourceSpace/dynamics'),
    ('034_eog', None, 'SourceSpace/dynamics'),
]

block_re = re.compile(r'^(\d+)_.*_(\d)b_(?:\d+_)?([a-z]+\d*)\.fif$')
split_re = re.compile(r'-\d+\.fif$')


def stem_re(stem):
    """old stem, not followed by a letter or digit (so _1b_base != _1b_base2)"""
    return re.compile(re.escape(stem) + r'(?![A-Za-z0-9])')


def collect_paths():
    """All files and folders below SEARCH_ROOT, except SKIP_DIRS."""
    paths = set()
    for dirpath, dirnames, filenames in os.walk(SEARCH_ROOT):
        if dirpath == SEARCH_ROOT:
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in dirnames + filenames:
            paths.add(os.path.join(dirpath, name))
    return paths


def plan(paths):
    """Simulate all FIXES in order on the set of paths. Returns the list of
    operations (op, src, dst) and a list of problems (conflicts)."""
    ops, problems = [], []
    for old, new, *only in FIXES:
        rx = stem_re(old)
        limit = os.path.join(SEARCH_ROOT, only[0]) + os.sep if only else SEARCH_ROOT
        # deepest first, so files inside a renamed folder are handled before it
        hits = sorted((p for p in paths if rx.search(os.path.basename(p))
                       and p.startswith(limit)),
                      key=lambda p: -p.count(os.sep))
        if not hits:
            print(f"  (no match for {old})")
        for src in hits:
            if new is None:
                dst = os.path.join(TRASH_DIR, os.path.relpath(src, SEARCH_ROOT))
                op = 'TRASH '
            else:
                dst = os.path.join(os.path.dirname(src),
                                   rx.sub(new, os.path.basename(src)))
                op = 'RENAME'
            if dst in paths or (op == 'TRASH ' and os.path.exists(dst)):
                problems.append(f"{op} {src}\n        -> target exists: {dst}")
                continue
            ops.append((op, src, dst))
            # update the simulated file tree (also everything inside a folder)
            moved = {p for p in paths if p == src or p.startswith(src + os.sep)}
            paths -= moved
            if op == 'RENAME':
                paths |= {dst + p[len(src):] for p in moved}
    return ops, problems


def scan():
    """Flag subjects whose 2_cut/ outputs do not look like 1b_base, 2b, 3b
    with one sad + one fear."""
    by_subj = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(CUT_DIR, '*.fif'))):
        if split_re.search(f):
            continue
        m = block_re.match(os.path.basename(f))
        if m:
            by_subj[m.group(1)].append((int(m.group(2)), m.group(3), os.path.basename(f)))
    n_bad = 0
    for subj, blocks in sorted(by_subj.items()):
        nums = sorted(b[0] for b in blocks)
        conds = sorted(b[1] for b in blocks if b[0] > 1)
        if nums == [1, 2, 3] and conds == ['fear', 'sad']:
            continue
        n_bad += 1
        print(f"\n{subj}: blocks {nums}, video conditions {conds}")
        for b in blocks:
            print(f"    {b[2]}")
    print(f"\n{n_bad} of {len(by_subj)} subject(s) look wrong.")


if __name__ == '__main__':
    if '--scan' in sys.argv:
        scan()
        sys.exit(0)

    apply = '--apply' in sys.argv
    print(f"{'APPLY' if apply else 'DRY RUN (nothing is changed, use --apply)'}")
    print(f"Searching below {SEARCH_ROOT} (skipping {sorted(SKIP_DIRS)})\n")

    ops, problems = plan(collect_paths())

    for op, src, dst in ops:
        print(f"{op} {os.path.relpath(src, SEARCH_ROOT)}\n"
              f"    -> {os.path.relpath(dst, SEARCH_ROOT)}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S) - nothing applied, fix these first:")
        for p in problems:
            print(f"  ! {p}")
        sys.exit(1)

    print(f"\n{len(ops)} operation(s).")
    if not apply:
        sys.exit(0)

    for op, src, dst in ops:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    print("Done.")
