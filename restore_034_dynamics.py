# -*- coding: utf-8 -*-
"""
restore_034_dynamics.py
=======================
After fix_cut_names.py --apply moved ALL 034 dynamics files to _excluded/,
this script puts them back with fear #2 rows removed.

What it does
------------
  CSV files  ({subj}_ecg_static.csv, _sliding.csv, _eog_static.csv, etc.):
    - reads from EXCLUDED_DYN
    - drops rows where the block column matches any FEAR2_PATTERNS
    - writes to DYN_DIR (skips if already present unless --overwrite)

  Series folders (series/{subj}_ecg_win*/):
    - files whose *basename* matches a FEAR2_PATTERNS entry are left in _excluded
    - all other files are copied back to DYN_DIR/series/...

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

import pandas as pd

MEG          = '/home/ZhangJianfeng_04/emo_study/MEG'
DYN_DIR      = os.path.join(MEG, 'SourceSpace', 'dynamics')
EXCLUDED_DYN = os.path.join(MEG, '_excluded', 'SourceSpace', 'dynamics')
SUBJ         = '034'

# Block labels that belong to fear #2 — all naming variants that may appear
# in the CSV's block column.  Matched as exact string or as substring.
# Blocks to KEEP — dynamics CSV uses bare block numbers: 1b, 2b, 3b, 4b
# For 034: 1b=base, 2b=fear#1, 3b=fear#2 (EXCLUDE), 4b=sad (keep but will be renumbered)
# We keep 1b (base), 2b (fear#1), 4b (sad); drop 3b (fear#2).
KEEP_BLOCKS = {'1b', '2b', '4b'}

# ── helpers ─────────────────────────────────────────────────────────────────

def is_fear2(value) -> bool:
    return str(value).strip() not in KEEP_BLOCKS


def find_block_col(df: pd.DataFrame):
    """Return the first column whose values look like block labels."""
    candidates = [c for c in df.columns
                  if re.search(r'block|cond|label', c, re.I)]
    if candidates:
        return candidates[0]
    # fallback: any column with string values containing 'b_'
    for c in df.columns:
        if df[c].dtype == object and df[c].str.contains(r'\db_', na=False).any():
            return c
    return None


# ── main ────────────────────────────────────────────────────────────────────

apply    = '--apply'     in sys.argv
overwrite = '--overwrite' in sys.argv
print('APPLY' if apply else 'DRY RUN (use --apply to execute)')
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

        mask   = df[col].apply(is_fear2)
        n_drop = mask.sum()
        n_keep = (~mask).sum()
        print(f'  CSV    {fn}')
        print(f'         block col = "{col}"  |  keep {n_keep} rows, drop {n_drop} (fear #2)')
        if n_drop:
            dropped = df.loc[mask, col].unique().tolist()
            print(f'         dropped block labels: {dropped}')

        if apply:
            os.makedirs(DYN_DIR, exist_ok=True)
            df[~mask].to_csv(dst, index=False)
            print(f'         -> written to {os.path.relpath(dst, MEG)}')

# ── 2. Series folders ────────────────────────────────────────────────────────
series_src = os.path.join(EXCLUDED_DYN, 'series')
if os.path.isdir(series_src):
    for win_dir in sorted(os.listdir(series_src)):
        if not win_dir.startswith(SUBJ + '_'):
            continue
        src_dir = os.path.join(series_src, win_dir)
        dst_dir = os.path.join(DYN_DIR, 'series', win_dir)
        files   = sorted(os.listdir(src_dir))
        # Series files are per-metric (not per-block), copy all back
        print(f'\n  SERIES {win_dir}/')
        print(f'         copy all {len(files)} files (per-metric, not per-block)')
        if apply:
            os.makedirs(dst_dir, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(src_dir, f),
                             os.path.join(dst_dir, f))
            print(f'         -> copied to {os.path.relpath(dst_dir, MEG)}')
else:
    print('\n  (no series/ folder found)')

print()
if errors:
    print(f'{len(errors)} WARNING(S) — check manually: {errors}')
else:
    print('No problems.')
if not apply:
    print('\nRe-run with --apply to execute.')
