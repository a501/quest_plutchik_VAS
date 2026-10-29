#!/usr/bin/env python3
"""
check_dynamics_coverage.py
==========================
For each subject, print how many rows and how many NaNs exist per measure
in the static and sliding CSVs.  Use this to spot aborted runs (a measure
with fewer rows than the others) or bad data (excessive NaNs).

Usage
-----
    python3 check_dynamics_coverage.py
    python3 check_dynamics_coverage.py --subjects 001 054
    python3 check_dynamics_coverage.py --csv-dir /path/to/dynamics
"""

import argparse
import glob
import os
import os.path as op
import sys

import numpy as np
import pandas as pd

DEFAULT_DIR = '/home/ZhangJianfeng_04/emo_study/MEG/SourceSpace/dynamics'


def check_static(path, subject):
    """Check one subject's static CSV."""
    df = pd.read_csv(path)
    # count rows and NaNs per measure
    measures = sorted(df['measure'].unique())
    rows = []
    for m in measures:
        sub = df[df['measure'] == m]
        n = len(sub)
        # NaN in the value column (could be actual NaN or the string 'sat')
        n_nan = sub['value'].apply(
            lambda v: v != v  # NaN check
            or (isinstance(v, str) and v.lower() == 'nan')
        ).sum()
        n_sat = (sub['value'] == 'sat').sum()
        rows.append({'measure': m, 'rows': n, 'NaN': int(n_nan),
                     'sat': int(n_sat)})
    return pd.DataFrame(rows)


def check_sliding(path, subject):
    """Check one subject's sliding CSV."""
    df = pd.read_csv(path)
    # group by (measure, win_sec, stat)
    grouped = df.groupby(['measure', 'win_sec'])
    rows = []
    for (m, ws), sub in grouped:
        n = len(sub)
        n_nan = sub['value'].apply(
            lambda v: v != v
            or (isinstance(v, str) and v.lower() == 'nan')
        ).sum()
        n_sat = (sub['value'] == 'sat').sum()
        n_stats = sub['stat'].nunique()
        rows.append({'measure': m, 'win_sec': ws, 'rows': n,
                     'stats': n_stats, 'NaN': int(n_nan), 'sat': int(n_sat)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument('--csv-dir', default=DEFAULT_DIR,
                    help=f'directory with *_static.csv / *_sliding.csv '
                         f'(default: {DEFAULT_DIR})')
    ap.add_argument('--subjects', nargs='+', metavar='ID',
                    help='only check these subjects (e.g. --subjects 001 054)')
    a = ap.parse_args()

    statics = sorted(glob.glob(op.join(a.csv_dir, '*_static.csv')))
    if not statics:
        sys.exit(f'No *_static.csv found in {a.csv_dir}')

    for sf in statics:
        stem = op.basename(sf).replace('_static.csv', '')
        # extract subject code (first token before _)
        subj = stem.split('_')[0]
        if a.subjects and subj not in a.subjects:
            continue

        slif = sf.replace('_static.csv', '_sliding.csv')

        print('=' * 75)
        print(f'SUBJECT: {subj}  ({stem})')
        print('=' * 75)

        # --- static ---
        print(f'\n  STATIC  ({op.basename(sf)})')
        try:
            st = check_static(sf, subj)
            # get expected row count (mode = most common)
            expected = int(st['rows'].mode().iloc[0]) if len(st) else 0
            print(f'  Expected rows/measure: {expected}  '
                  f'(= units × snippets)')
            print(f'  {"measure":<35} {"rows":>6} {"NaN":>6} {"sat":>5}  note')
            print(f'  {"-"*35} {"-"*6} {"-"*6} {"-"*5}  ----')
            for _, r in st.iterrows():
                note = ''
                if r['rows'] < expected:
                    note = f'INCOMPLETE ({r["rows"]}/{expected})'
                elif r['rows'] > expected:
                    note = f'EXTRA ({r["rows"]}/{expected})'
                if r['NaN'] > 0:
                    pct = 100 * r['NaN'] / r['rows'] if r['rows'] else 0
                    note += f'  {pct:.0f}% NaN'
                if r['sat'] > 0:
                    pct = 100 * r['sat'] / r['rows'] if r['rows'] else 0
                    note += f'  {pct:.0f}% sat'
                print(f'  {r["measure"]:<35} {r["rows"]:>6} '
                      f'{r["NaN"]:>6} {r["sat"]:>5}  {note}')
        except Exception as e:
            print(f'  ERROR reading: {e}')

        # --- sliding ---
        if op.isfile(slif):
            print(f'\n  SLIDING  ({op.basename(slif)})')
            try:
                sl = check_sliding(slif, subj)
                if len(sl):
                    expected_sl = int(sl['rows'].mode().iloc[0])
                    print(f'  Expected rows/(measure,win): {expected_sl}')
                    print(f'  {"measure":<20} {"win_sec":>7} {"rows":>6} '
                          f'{"stats":>5} {"NaN":>6} {"sat":>5}  note')
                    print(f'  {"-"*20} {"-"*7} {"-"*6} '
                          f'{"-"*5} {"-"*6} {"-"*5}  ----')
                    for _, r in sl.iterrows():
                        note = ''
                        if r['rows'] < expected_sl:
                            note = f'INCOMPLETE ({r["rows"]}/{expected_sl})'
                        if r['NaN'] > 0:
                            pct = 100 * r['NaN'] / r['rows'] if r['rows'] else 0
                            note += f'  {pct:.0f}% NaN'
                        if r['sat'] > 0:
                            pct = 100 * r['sat'] / r['rows'] if r['rows'] else 0
                            note += f'  {pct:.0f}% sat'
                        print(f'  {r["measure"]:<20} {r["win_sec"]:>7.1f} '
                              f'{r["rows"]:>6} {r["stats"]:>5} '
                              f'{r["NaN"]:>6} {r["sat"]:>5}  {note}')
            except Exception as e:
                print(f'  ERROR reading: {e}')
        else:
            print(f'\n  SLIDING  — not found')

        print()


if __name__ == '__main__':
    main()
