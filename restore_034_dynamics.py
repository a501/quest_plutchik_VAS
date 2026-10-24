# -*- coding: utf-8 -*-
"""
restore_034_dynamics.py
=======================
After fix_cut_names.py --apply moved ALL 034 dynamics files to _excluded/,
this script puts them back with fear #2 removed and sad renumbered, so 034
has the same 1b/2b/3b blocks as every other subject.

034 dynamics were computed from the OLD cut, so the block column holds:
    1b = base   2b = fear #1 (kept)   3b = fear #2 (DROPPED)   4b = sad #2
-> rows of 3b are dropped, 4b is renamed to 3b (same as was done to the
   034 label_tc npz, see fix_cut_names.py history).

What it does
------------
  CSV files  ({subj}_ecg_static.csv, _sliding.csv, _eog_static.csv, etc.):
    - reads from EXCLUDED_DYN
    - drops rows whose block is DROP_BLOCK, renames RENAME_BLOCKS
    - writes to DYN_DIR (skips if already present unless --overwrite)

  Series folders (series/{subj}_*_win*/: ACW0.npy, ..., index.npz):
    - the block of each row/window must be stored in index.npz; the arrays
      are cut along the axis of that length and the labels renamed
    - if index.npz holds no block labels the folder is NOT restored (it
      would bring fear #2 back) and is reported as a warning

Usage
-----
    python3 restore_034_dynamics.py           # dry run
    python3 restore_034_dynamics.py --apply
    python3 restore_034_dynamics.py --apply --overwrite
"""

import os
import re
import sys
import glob
import shutil

import numpy as np
import pandas as pd

MEG          = '/home/ZhangJianfeng_04/emo_study/MEG'
DYN_DIR      = os.path.join(MEG, 'SourceSpace', 'dynamics')
EXCLUDED_DYN = os.path.join(MEG, '_excluded', 'SourceSpace', 'dynamics')
SUBJ         = '034'

DROP_BLOCK    = '3b'            # fear #2 (old cut name 3b_fear)
RENAME_BLOCKS = {'4b': '3b'}    # sad #2 (old cut name 4b_sad -> 3b_sad)
OLD_BLOCKS    = {'1b', '2b', '3b', '4b'}

# ── helpers ─────────────────────────────────────────────────────────────────

def block_of(value) -> str:
    """'3b', '3b_fear', b'3b' -> '3b'; anything else -> ''"""
    if isinstance(value, bytes):
        value = value.decode()
    m = re.match(r'(\d+b)(?:_|$)', str(value).strip())
    return m.group(1) if m else ''


def rename_label(value):
    """Rename the block part of a label, keep the rest ('4b_sad' -> '3b_sad')."""
    b = block_of(value)
    if b in RENAME_BLOCKS:
        return RENAME_BLOCKS[b] + str(value).strip()[len(b):]
    return value


def find_block_col(df: pd.DataFrame):
    """Return the first column whose values look like block labels."""
    candidates = [c for c in df.columns
                  if re.search(r'block|cond|label', c, re.I)]
    if candidates:
        return candidates[0]
    # fallback: any column with string values like '2b' / '2b_fear'
    for c in df.columns:
        if df[c].dtype == object and df[c].astype(str).str.match(r'\d+b(_|$)').any():
            return c
    return None


def find_block_key(index):
    """Key in index.npz whose values are block labels (1b..4b)."""
    for k in index.files:
        a = np.asarray(index[k]).ravel()
        if a.size < 2 or a.dtype.kind not in 'OUS':
            continue
        if {block_of(v) for v in a} <= OLD_BLOCKS and DROP_BLOCK in {block_of(v) for v in a}:
            return k
    return None


# ── main ────────────────────────────────────────────────────────────────────

apply     = '--apply'     in sys.argv
overwrite = '--overwrite' in sys.argv
print('APPLY' if apply else 'DRY RUN (use --apply to execute)')
print(f'Drop block {DROP_BLOCK}, rename {RENAME_BLOCKS}')
print()

if not os.path.isdir(EXCLUDED_DYN):
    sys.exit(f'Source not found: {EXCLUDED_DYN}')

errors = []

# ── 1. CSV files ─────────────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(EXCLUDED_DYN, f'{SUBJ}_*.csv')))
if not csv_files:
    print(f'  (no CSVs found for {SUBJ} in {EXCLUDED_DYN})')
else:
    for src in csv_files:
        fn  = os.path.basename(src)
        dst = os.path.join(DYN_DIR, fn)

        if os.path.exists(dst) and not overwrite:
            print(f'  SKIP   {fn}  (already in dynamics/, use --overwrite)')
            continue

        df = pd.read_csv(src)
        col = find_block_col(df)
        if col is None:
            print(f'  WARN   {fn}  — no block column found; columns: {list(df.columns)}')
            errors.append(fn)
            continue

        blocks = df[col].map(block_of)
        found  = set(blocks.unique())
        if not found <= OLD_BLOCKS or DROP_BLOCK not in found:
            print(f'  WARN   {fn}  — unexpected blocks {sorted(found)}, expected {sorted(OLD_BLOCKS)}')
            errors.append(fn)
            continue

        mask = blocks == DROP_BLOCK
        out  = df[~mask].copy()
        out[col] = out[col].map(rename_label)
        print(f'  CSV    {fn}')
        print(f'         block col = "{col}"  |  keep {len(out)} rows, drop {mask.sum()} ({DROP_BLOCK})')
        print(f'         blocks after: {out[col].map(block_of).value_counts().sort_index().to_dict()}')

        if apply:
            os.makedirs(DYN_DIR, exist_ok=True)
            out.to_csv(dst, index=False)
            print(f'         -> written to {os.path.relpath(dst, MEG)}')

# ── 2. Series folders ────────────────────────────────────────────────────────
series_src = os.path.join(EXCLUDED_DYN, 'series')
if os.path.isdir(series_src):
    for win_dir in sorted(os.listdir(series_src)):
        if not win_dir.startswith(SUBJ + '_'):
            continue
        src_dir = os.path.join(series_src, win_dir)
        dst_dir = os.path.join(DYN_DIR, 'series', win_dir)
        print(f'\n  SERIES {win_dir}/')

        if os.path.exists(dst_dir) and not overwrite:
            print(f'         SKIP (already in dynamics/series/, use --overwrite)')
            continue

        idx_path = os.path.join(src_dir, 'index.npz')
        if not os.path.isfile(idx_path):
            print(f'         WARN no index.npz -> cannot tell blocks apart, NOT restored')
            errors.append(win_dir)
            continue
        index = np.load(idx_path, allow_pickle=True)
        key = find_block_key(index)
        if key is None:
            print(f'         WARN no block labels in index.npz -> NOT restored')
            for k in index.files:
                a = np.asarray(index[k])
                print(f'              {k}: shape {a.shape}, dtype {a.dtype}, e.g. {a.ravel()[:4].tolist()}')
            errors.append(win_dir)
            continue

        labels = np.asarray(index[key])
        keep   = np.array([block_of(v) != DROP_BLOCK for v in labels.ravel()])
        n      = keep.size
        print(f'         block labels in index.npz["{key}"]: keep {keep.sum()} of {n}')

        new_index = {k: index[k] for k in index.files}
        new_index[key] = np.array([str(rename_label(v.decode() if isinstance(v, bytes) else v))
                                   for v in labels.ravel()[keep]])
        # other per-block arrays in index.npz (same length) are cut too
        for k in index.files:
            a = np.asarray(index[k])
            if k != key and a.ndim >= 1 and a.shape[0] == n:
                new_index[k] = a[keep]

        new_arrays, bad = {}, False
        for f in sorted(os.listdir(src_dir)):
            if not f.endswith('.npy'):
                continue
            arr  = np.load(os.path.join(src_dir, f), allow_pickle=True)
            axes = [ax for ax, s in enumerate(arr.shape) if s == n]
            if len(axes) != 1:
                print(f'         WARN {f}: shape {arr.shape}, no single axis of length {n}')
                bad = True
                continue
            new_arrays[f] = np.compress(keep, arr, axis=axes[0])
            print(f'         {f}: {arr.shape} -> {new_arrays[f].shape}')
        if bad:
            print(f'         -> NOT restored')
            errors.append(win_dir)
            continue

        if apply:
            os.makedirs(dst_dir, exist_ok=True)
            for f, a in new_arrays.items():
                np.save(os.path.join(dst_dir, f), a)
            np.savez(os.path.join(dst_dir, 'index.npz'), **new_index)
            for f in os.listdir(src_dir):        # anything else: copy as-is
                if not f.endswith('.npy') and f != 'index.npz':
                    shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            print(f'         -> written to {os.path.relpath(dst_dir, MEG)}')
else:
    print('\n  (no series/ folder found)')

print()
if errors:
    print(f'{len(errors)} WARNING(S) — not restored, check manually: {errors}')
else:
    print('No problems.')
if not apply:
    print('\nRe-run with --apply to execute.')
