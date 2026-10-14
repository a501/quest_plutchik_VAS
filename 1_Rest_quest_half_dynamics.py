#!/usr/bin/env python3
"""
1_Rest_quest_half_dynamics.py
=============================
ANALYSIS TYPE: SLIDING WINDOW (win1s) -- 2-min halves + full 5 min.

Builds the half/full dynamics table from the existing SS_rest_dynamics.py
output. Nothing is recomputed from raw timeseries.

ROW LABELLING -- the important bit
----------------------------------
The per-window arrays in dynamics/series/<NNN>_ecg_win1s/*.npy have one row
per (unit x method), in exactly the order the label_tc npz stores them. So
unit_name / unit_level / method from

    label_tc/<NNN>_ecg_rest_label_tc.npz

are the authoritative row labels. The static CSV is long-format (one row per
unit x method x measure) and CANNOT be used for row alignment -- using it is
what produced the roi_0000 placeholder names.

Only level == 'roi' and method == 'mean_flip' rows are kept.

FILE STRUCTURE (confirmed on disk)
    dynamics/<NNN>_ecg_sliding.csv          summary stats, not per-window
    dynamics/<NNN>_ecg_static.csv           long format, unit_name/measure/value
    dynamics/series/<NNN>_ecg_win1s/        per-window arrays, (n_rows, n_win)
        ACW0.npy ACW50.npy LZC.npy SpectralEntropy.npy MedianFreq.npy
    label_tc/<NNN>_ecg_rest_label_tc.npz    unit_name, unit_level, method

Measures: ACW0, ACW50, LZC, SpecEn (from SpectralEntropy), MedianFreq.
SampEn and SpecSlope are not produced by SS_rest_dynamics.py.

Halves:
    first  : window centre <  120 s
    second : window centre >  (T_total - 120 s)
    full   : every window (full 5-min win1s estimator)

OUTPUT
    analysis/rest_half_dynamics_win1s.csv
    analysis/rest_half_dynamics_win1s_diag.csv

    export PATH="/home/ZhangJianfeng_04/.conda/envs/MEG_env/bin:$PATH"
    python3 1_Rest_quest_half_dynamics.py
"""

import os
import os.path as op
import glob
import re
import sys

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
BASE    = '/home/ZhangJianfeng_04/emo_study/Rest_task/SourceSpace'
DYN_DIR = op.join(BASE, 'dynamics')
SER_DIR = op.join(DYN_DIR, 'series')
TC_DIR  = op.join(BASE, 'label_tc')      # authoritative row order
OUT_DIR = op.join(BASE, 'analysis')

VERSION    = 'ecg'
METHOD     = 'mean_flip'                 # primary orientation method
LEVEL      = 'roi'                       # drop individual-label rows
WINDOW_S   = 1.0
HALF_LEN_S = 120.0

# SS_rest_dynamics.py settings, used only if index.npz / label_tc are missing.
# Sliding windows are 1 s and 2 s, advanced SLIDE_STEP samples at a time, at
# native 500 Hz with no downsampling, so the derived sliding series has an
# effective rate of 500/5 = 100 Hz.
SLIDE_STEP_SAMPLES = 5
SFREQ_FALLBACK     = 500.0

# .npy filename stem -> internal measure name
NPY_MEASURE_MAP = {
    'ACW0':            'ACW0',
    'ACW50':           'ACW50',
    'LZC':             'LZC',
    'SpectralEntropy': 'SpecEn',
    'MedianFreq':      'MedianFreq',
}
MEASURES = list(NPY_MEASURE_MAP.values())

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# SUBJECTS
# ============================================================
def discover_subjects():
    subs = set()
    if op.isdir(SER_DIR):
        for d in os.listdir(SER_DIR):
            m = re.fullmatch(rf'(\d{{3}})_{VERSION}_win1s', d)
            if m:
                subs.add(m.group(1))
    for f in glob.glob(op.join(DYN_DIR, f'*_{VERSION}_static.csv')):
        m = re.match(r'(\d{3})_', op.basename(f))
        if m:
            subs.add(m.group(1))
    return sorted(subs)


# ============================================================
# ROW LABELS FROM label_tc npz
# ============================================================
def load_row_labels(sub):
    """
    Returns (unit_name, unit_level, method, sfreq, n_times) from the
    label_tc npz. sfreq and n_times are needed to convert window INDEX to
    window CENTRE TIME, because SS_rest_dynamics.py steps the window by a
    few SAMPLES, not by one second -- assuming a 1 s step stretches a 5-min
    recording into hours and destroys the half assignment.
    """
    f = op.join(TC_DIR, f'{sub}_{VERSION}_rest_label_tc.npz')
    if not op.isfile(f):
        print(f'  [{sub}] label_tc npz missing: {f}')
        return None, None, None, None, None
    try:
        z = np.load(f, allow_pickle=True)
        sfreq   = float(np.asarray(z['sfreq']).ravel()[0])
        n_times = int(np.asarray(z['n_times']).ravel()[0])
        return ([str(x) for x in z['unit_name']],
                [str(x) for x in z['unit_level']],
                [str(x) for x in z['method']],
                sfreq, n_times)
    except Exception as e:
        print(f'  [{sub}] could not read label_tc npz: {e}')
        return None, None, None, None, None


# ============================================================
# SERIES METADATA (index.npz)
# ============================================================
def load_series_index(folder):
    """
    SS_rest_dynamics.py writes a self-describing index.npz beside the per-
    measure .npy files. Prefer its stored step / sampling rate over deriving
    them, since the sliding step is 5 SAMPLES (10 ms at 500 Hz), not 1 s.

    Returns dict with any of: step (samples), sfreq, win_s, n_win, rows.
    """
    f = op.join(folder, 'index.npz')
    if not op.isfile(f):
        return {}
    try:
        z = np.load(f, allow_pickle=True)
    except Exception:
        return {}

    def grab(*names):
        for n in names:
            if n in z.files:
                a = np.asarray(z[n]).ravel()
                if a.size:
                    try:
                        return float(a[0])
                    except (TypeError, ValueError):
                        return a[0]
        return None

    return {
        'step':  grab('slide_step', 'step', 'step_samples'),
        'sfreq': grab('sfreq', 'fs', 'sampling_rate'),
        'win_s': grab('win_s', 'window_s', 'win_sec', 'window'),
        'n_win': grab('n_win', 'n_windows'),
    }


# ============================================================
# LOAD PER-WINDOW ARRAYS
# ============================================================
def load_from_npy(sub):
    folder = op.join(SER_DIR, f'{sub}_{VERSION}_win1s')
    if not op.isdir(folder):
        return None

    names, levels, methods, sfreq, n_times = load_row_labels(sub)
    idx = load_series_index(folder)
    if idx.get('sfreq'):
        sfreq = idx['sfreq']

    rows = []
    for npy_stem, measure_name in NPY_MEASURE_MAP.items():
        path = op.join(folder, f'{npy_stem}.npy')
        if not op.isfile(path):
            continue

        arr = np.load(path, allow_pickle=True, mmap_mode='r')
        arr = np.asarray(arr)

        # saturation markers stored as strings in object arrays
        if arr.dtype == object:
            sat = np.zeros(arr.shape, dtype=bool)
            num = np.zeros(arr.shape, dtype=float)
            for idx in np.ndindex(arr.shape):
                v = arr[idx]
                if isinstance(v, str) and v.lower().strip() == 'sat':
                    sat[idx] = True
                    num[idx] = np.nan
                else:
                    try:
                        num[idx] = float(v)
                    except (TypeError, ValueError):
                        num[idx] = np.nan
            arr, sat_arr = num, sat
        else:
            arr = arr.astype(np.float32)
            sat_arr = np.zeros(arr.shape, dtype=bool)

        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
            sat_arr = sat_arr[np.newaxis, :]
        if arr.ndim != 2:
            print(f'  [{sub}] {npy_stem}: unexpected shape {arr.shape} -- skipped')
            continue

        n_rows, n_win = arr.shape

        # --- window centre times -------------------------------------
        # SS_rest_dynamics.py advances the window by SLIDE_STEP SAMPLES
        # (5 samples = 10 ms at 500 Hz), NOT by one second. Window i has
        # centre (i*step + win/2) / sfreq.
        # Preference order:
        #   1. step + sfreq stored in the series index.npz
        #   2. derived from the recording length in the label_tc npz
        #   3. the hardcoded fallback below
        step_s = None
        if idx.get('step') and sfreq:
            step_s = float(idx['step']) / float(sfreq)
        elif sfreq and n_times and n_win > 1:
            step_s = ((n_times / sfreq) - WINDOW_S) / (n_win - 1)
        if not step_s or not np.isfinite(step_s) or step_s <= 0:
            step_s = SLIDE_STEP_SAMPLES / SFREQ_FALLBACK
            print(f'  [{sub}] step not recoverable -- assuming '
                  f'{SLIDE_STEP_SAMPLES} samples @ {SFREQ_FALLBACK:.0f} Hz')
        t_center = np.arange(n_win) * step_s + WINDOW_S / 2.0

        if names and len(names) == n_rows:
            r_names, r_levels, r_methods = names, levels, methods
        else:
            if names:
                print(f'  [{sub}] {npy_stem}: npy rows {n_rows} != '
                      f'label_tc rows {len(names)} -- placeholders used')
            r_names   = [f'roi_{i:04d}' for i in range(n_rows)]
            r_levels  = ['unknown'] * n_rows
            r_methods = ['unknown'] * n_rows

        # keep only the rows we want, before expanding to long format
        keep = [i for i in range(n_rows)
                if r_levels[i] == LEVEL and r_methods[i] == METHOD]
        if not keep:                       # fall back to everything
            keep = list(range(n_rows))

        for i in keep:
            rows.append(pd.DataFrame({
                'subject':  sub,
                'ROI':      r_names[i],
                'level':    r_levels[i],
                'method':   r_methods[i],
                'measure':  measure_name,
                'win_s':    WINDOW_S,
                't_center': t_center,
                'value':    np.asarray(arr[i], dtype=float),
                'sat':      np.asarray(sat_arr[i], dtype=bool),
            }))

    return pd.concat(rows, ignore_index=True) if rows else None


# ============================================================
# MAIN
# ============================================================
def main():
    subjects = discover_subjects()
    if not subjects:
        sys.exit(f'No win1s series folders found under {SER_DIR}')

    print(f'Subjects found: {len(subjects)}')
    print(f'Measures: {MEASURES}')
    print(f'Keeping level={LEVEL!r}, method={METHOD!r}')

    all_rows, diag_rows = [], []

    for sub in subjects:
        df = load_from_npy(sub)
        if df is None or df.empty:
            print(f'  [{sub}] no win1s data -- skipped')
            continue

        t_max = float(df['t_center'].max())
        n_win_sub = int(df.groupby(['ROI', 'measure']).size().max())

        df['half'] = np.where(
            df['t_center'] < HALF_LEN_S, 'first',
            np.where(df['t_center'] > (t_max - HALF_LEN_S), 'second', 'middle'))

        df_full = df.copy()
        df_full['half'] = 'full'
        df = pd.concat([df[df['half'] != 'middle'], df_full], ignore_index=True)

        g = df.groupby(['subject', 'ROI', 'level', 'method', 'measure', 'half'],
                       dropna=False)
        agg = g.agg(
            value     = ('value', 'mean'),
            sd        = ('value', 'std'),
            n_windows = ('value', 'size'),
            n_valid   = ('value', lambda s: int(s.notna().sum())),
            n_sat     = ('sat',   'sum'),
        ).reset_index()
        agg['win_s']     = WINDOW_S
        agg['T_total_s'] = t_max

        all_rows.append(agg)

        d2 = agg.groupby(['ROI', 'measure']).agg(
            n_sat     = ('n_sat',     'sum'),
            n_windows = ('n_windows', 'sum'),
            n_valid   = ('n_valid',   'sum'),
        ).reset_index()
        d2['subject']  = sub
        d2['sat_rate'] = d2['n_sat'] / d2['n_windows'].replace(0, np.nan)
        d2['nan_rate'] = 1 - d2['n_valid'] / d2['n_windows'].replace(0, np.nan)
        diag_rows.append(d2)

        step_est = (t_max - WINDOW_S / 2.0) / max(n_win_sub - 1, 1) \
            if n_win_sub else np.nan
        print(f'  [{sub}] T={t_max:.1f}s  step={step_est*1000:.1f}ms  '
              f'ROIs={agg.ROI.nunique()}  rows={len(agg)}')

    if not all_rows:
        sys.exit('Nothing assembled.')

    out  = pd.concat(all_rows,  ignore_index=True)
    diag = pd.concat(diag_rows, ignore_index=True)

    f_out  = op.join(OUT_DIR, 'rest_half_dynamics_win1s.csv')
    f_diag = op.join(OUT_DIR, 'rest_half_dynamics_win1s_diag.csv')
    out.to_csv(f_out,  index=False)
    diag.to_csv(f_diag, index=False)

    print(f'\nSLIDING WINDOW (win1s), 2-min halves + full 5 min')
    print(f'  {f_out}  ({len(out)} rows, {out.subject.nunique()} subjects, '
          f'{out.ROI.nunique()} ROIs)')
    print(f'  {f_diag}')

    sample = sorted(out.ROI.unique())[:8]
    print(f'  example ROI names: {sample}')

    # NOTE: the .npy arrays are float32, so a saturated ACW window is stored
    # as a number or NaN, not as the string 'sat'. sat_rate below is therefore
    # structurally 0 and is NOT evidence that ACW0 did not saturate. Judge
    # saturation from nan_rate, and from the ACW0 values sitting at the window
    # ceiling (= WINDOW_S).
    print('\nNaN rate per measure (win1s)  [use this, not sat_rate]:')
    nr = diag.groupby('measure')['nan_rate'].mean().sort_values(ascending=False)
    for m, v in nr.items():
        if not np.isnan(v):
            flag = '  <-- HIGH' if v > 0.2 else ''
            print(f'  {m:<16s} {v:6.1%}{flag}')

    # how often does ACW0 sit at the 1 s window ceiling?
    acw = out[(out.measure == 'ACW0') & out.value.notna()]
    if len(acw):
        at_ceiling = float((acw['value'] >= WINDOW_S * 0.99).mean())
        print(f'\n  ACW0 at the {WINDOW_S:.0f}s window ceiling: '
              f'{at_ceiling:6.1%} of ROI x half cells')


if __name__ == '__main__':
    main()
