#!/usr/bin/env python3
"""
run_SS_dyn_dynamics.py
======================
Runs SS_dynamics.py on every *_label_tc.npz in the dyn-task label_tc
directory, 4 subjects at a time. No SLURM needed — run directly on
fox-c18 or inside an srun session.

Each subject is a separate subprocess; a crash in one never affects others.
SS_dynamics.py itself parallelises over ROI units via joblib, so each
subprocess uses multiple cores internally.

Usage
-----
    nohup python3 run_SS_dyn_dynamics.py > dyn_dyn_batch.out 2>&1 &

    # skip already finished subjects
    nohup python3 run_SS_dyn_dynamics.py --skip-done > dyn_dyn_batch.out 2>&1 &

    # specific subjects
    python3 run_SS_dyn_dynamics.py --subjects 001 003 206

    # faster sliding window (fewer steps)
    python3 run_SS_dyn_dynamics.py --slide-step 5

    # fewer parallel jobs if memory is tight
    python3 run_SS_dyn_dynamics.py --jobs 2

Monitor
-------
    tail -f dyn_dyn_batch.out
    tail -f /home/ZhangJianfeng_04/emo_study/MEG/SourceSpace/logs/dyn_dyn_<subj>.log
    ps aux | grep SS_dynamics.py
Stop:
    pkill -f SS_dynamics.py
"""

import argparse
import glob
import os
import os.path as op
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# ============================================================
# PATHS — keep in sync with SS_dynamics.py and run_SS_dynamics.sh
# ============================================================
TC_DIR   = '/home/ZhangJianfeng_04/emo_study/MEG/SourceSpace/label_tc'
OUT_DIR  = '/home/ZhangJianfeng_04/emo_study/MEG/SourceSpace/dynamics'
LOG_DIR  = '/home/ZhangJianfeng_04/emo_study/MEG/SourceSpace/logs'
SCRIPT   = '/home/ZhangJianfeng_04/emo_study/MEG/SS_dynamics.py'
PYTHON   = '/home/ZhangJianfeng_04/.conda/envs/MEG_env/bin/python3'

# joblib parallelises over units inside each job — keep BLAS single-threaded
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# Pattern: NNN_ecg_label_tc.npz  (dyn task, ecg version)
NPZ_RE = re.compile(r'^(\d+)_ecg_label_tc\.npz$')


# ============================================================
# HELPERS
# ============================================================
def discover_npz():
    """Return sorted list of (subject_code, npz_path)."""
    pairs = []
    for path in sorted(glob.glob(op.join(TC_DIR, '*_label_tc.npz'))):
        m = NPZ_RE.match(op.basename(path))
        if m:
            pairs.append((m.group(1), path))
    return pairs


def already_done(subject):
    """True if the static CSV already exists and is non-empty."""
    f = op.join(OUT_DIR, f'{subject}_ecg_static.csv')
    return op.isfile(f) and op.getsize(f) > 0


def run_one(subject, npz_path, log_dir, extra_args):
    """Run SS_dynamics.py for one subject; return (rc, elapsed_s, log_path)."""
    log_path = op.join(log_dir, f'dyn_dyn_{subject}.log')
    cmd = [PYTHON, SCRIPT, npz_path] + extra_args
    t0 = time.time()
    with open(log_path, 'w') as fh:
        fh.write(f'# cmd  : {" ".join(cmd)}\n')
        fh.write(f'# start: {datetime.now():%Y-%m-%d %H:%M:%S}\n')
        fh.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
        rc = proc.wait()
        elapsed = time.time() - t0
        fh.write(f'# end  : {datetime.now():%Y-%m-%d %H:%M:%S} '
                 f'exit={rc} elapsed={elapsed/60:.1f} min\n')
    return rc, elapsed, log_path


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument('--subjects', nargs='+', metavar='ID',
                    help='explicit subject codes, e.g. --subjects 001 003 206')
    ap.add_argument('--jobs', type=int, default=4,
                    help='subjects running simultaneously (default 4)')
    ap.add_argument('--skip-done', action='store_true',
                    help='skip subjects whose static CSV already exists')
    ap.add_argument('--log-dir', default=LOG_DIR)
    ap.add_argument('--dry-run', action='store_true',
                    help='print what would run, do nothing')
    # passed through to SS_dynamics.py
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no-series', action='store_true')
    ap.add_argument('--slide-step', type=int, default=5,
                    help='sliding window step in samples (default 5, '
                         'use 1 for full resolution — much slower)')
    a = ap.parse_args()

    if not op.isfile(SCRIPT):
        sys.exit(f'Script not found: {SCRIPT}')
    os.makedirs(a.log_dir, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- discover ----
    all_pairs = discover_npz()
    if not all_pairs:
        sys.exit(f'No *_label_tc.npz files found in {TC_DIR}')

    if a.subjects:
        want = set(a.subjects)
        pairs = [(s, p) for s, p in all_pairs if s in want]
        unknown = want - {s for s, _ in pairs}
        if unknown:
            print(f'WARNING: no npz found for: {" ".join(sorted(unknown))}')
    else:
        pairs = all_pairs

    # ---- skip-done filter ----
    skipped = []
    if a.skip_done and not a.overwrite:
        todo = []
        for s, p in pairs:
            (skipped if already_done(s) else todo).append((s, p))
        pairs = todo

    # ---- extra args ----
    extra = [f'--slide-step={a.slide_step}']
    if a.overwrite:
        extra.append('--overwrite')
    if a.no_series:
        extra.append('--no-series')

    # ---- summary ----
    print('=' * 65)
    print(f'SS_dynamics (dyn task) — batch launcher')
    print(f'Node       : {os.uname().nodename}')
    print(f'Start      : {datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'Script     : {SCRIPT}')
    print(f'Concurrency: {a.jobs} subjects at a time')
    print(f'Slide step : {a.slide_step} samples')
    if skipped:
        print(f'Already done (skip): {" ".join(s for s, _ in skipped)}')
    print(f'To run     : {len(pairs)}')
    print('=' * 65)
    for s, p in pairs:
        done_tag = '  [will overwrite]' if (a.overwrite and already_done(s)) else ''
        print(f'  {s}  {op.basename(p)}{done_tag}')

    if not pairs:
        print('Nothing to run.')
        return 0

    if a.dry_run:
        print('\n[dry run] nothing executed.')
        return 0

    # ---- run ----
    results = []
    t_all = time.time()

    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, s, p, a.log_dir, extra): s
                for s, p in pairs}
        n_done = 0
        for fut in as_completed(futs):
            s = futs[fut]
            n_done += 1
            try:
                rc, elapsed, lp = fut.result()
            except Exception as e:
                rc, elapsed, lp = 1, 0.0, '?'
                print(f'[{n_done}/{len(pairs)}] {s}  EXCEPTION: {e}', flush=True)
            results.append((s, rc, elapsed, lp))
            status = 'OK' if rc == 0 else f'FAILED (exit {rc})'
            print(f'[{n_done}/{len(pairs)}]  {datetime.now():%H:%M:%S}  '
                  f'{s:6}  {status}  {elapsed/60:.1f} min', flush=True)

    # ---- summary ----
    total_elapsed = time.time() - t_all
    failed = [(s, rc) for s, rc, _, _ in results if rc != 0]
    ok = len(results) - len(failed)

    lines = [
        '',
        '=' * 65,
        f'BATCH SUMMARY  {datetime.now():%Y%m%d_%H%M%S}  '
        f'total {total_elapsed/60:.1f} min',
        '=' * 65,
        f'{"subject":<10} {"exit":>6} {"min":>8}  log',
    ]
    for s, rc, elapsed, lp in sorted(results, key=lambda x: x[0]):
        flag = '' if rc == 0 else '  <-- FAILED'
        lines.append(f'{s:<10} {rc:>6} {elapsed/60:>8.1f}  '
                     f'{op.basename(lp)}{flag}')
    lines += ['', f'{ok} succeeded, {len(failed)} failed']
    if failed:
        lines.append('Failed: ' + ', '.join(s for s, _ in failed))
        kill = {rc for _, rc in failed if rc in (137, -9)}
        py   = {rc for _, rc in failed if rc not in (137, -9)}
        if kill:
            lines.append('exit 137/-9 = OOM kill — reduce --jobs')
        if py:
            lines.append('exit 1 = exception in SS_dynamics.py — check log')

    text = '\n'.join(lines)
    print(text)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summ = op.join(a.log_dir, f'dyn_dyn_batch_summary_{stamp}.txt')
    with open(summ, 'w') as f:
        f.write(text + '\n')
    print(f'\nSummary -> {summ}')
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
