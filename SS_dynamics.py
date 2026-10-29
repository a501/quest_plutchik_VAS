#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SS_dynamics.py
==============
Dynamic measures on the source-space ROI / label timeseries produced by
SS_label_tc.py. One process per {subject}_{version}_label_tc.npz.

Measures
--------
STATIC (one value per snippet)
    LZC, ACW0, ACW50, MedianFreq, SpectralEntropy, SpectralSlope, PeakFreq
    IRASA            -> IRASA_slope, IRASA_oscpower
    SampEn_m2                  m = 2, r = 0.2 x SD (field-standard)
    IRASA_slope / _intercept / _r2 / _oscpower   via neurodsp.compute_irasa
    MSE_costa_m2_*             canonical Costa (EntropyHub RadNew=0):
                               tolerance fixed at r * SD(original)
    MSE_norm_m2_*              tolerance rescaled per scale (RadNew=1);
                               reproduces the existing sensor-level m_mse
    MSE_*_{CI,AUC,slope,max_entropy,scale_at_max,entropy_range}
    MSE_*_curve                the full 20-scale curve, JSON-encoded

SLIDING (1 s window, step configurable; default step = 1 sample)
    LZC, ACW0, ACW50, SpectralEntropy, MedianFreq
    summarised as mean / sd / cv / median / p05 / p95 / min / max / slope,
    plus sat_frac for each ACW measure.

    SpectralSlope is not slid — see the note by SLIDING_MEASURES.

    Where an ACW series contains ANY saturated window (the autocorrelation
    never crossed the threshold inside the 1 s window, so the value is the
    window edge rather than a measurement), every summary statistic for
    that measure is written as the string 'sat' instead of a number. The
    same applies to a saturated static ACW value.

Speed
-----
SampEn and MSE come from EntropyHub, the published reference
implementation, rather than a hand-rolled one. That costs roughly 9x
more time per snippet than a bespoke vectorised version but removes any
question about whether the numbers match the literature. SampEn is the
bottleneck at ~4.6 s per snippet; the two MSEn calls add ~7 s.

Usage
-----
    python3 SS_dynamics.py /path/to/001_eog_label_tc.npz
    python3 SS_dynamics.py <npz> --jobs 32 --save-series
    python3 SS_dynamics.py <npz> --slide-step 5      # 5x faster sliding
"""

import argparse
import json
import os
import os.path as op
import sys
import traceback
from itertools import product

import contextlib
import io

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import periodogram, welch
from joblib import Parallel, delayed

import EntropyHub as eh
import antropy as ant
from neurodsp.aperiodic import compute_irasa

# ------------------------------------------------------------------
OUT_DIR = '/home/ZhangJianfeng_04/emo_study/MEG/SourceSpace/dynamics'

# Analyse at whatever rate the npz holds — currently the 500 Hz acquisition
# rate, no downsampling. Note for the methods section: relative to 250 Hz
# this lowers LZC to ~0.37x and SampEn to ~0.84x, because oversampling makes
# consecutive samples more similar. The factor is constant across ROIs and
# conditions, so contrasts are unaffected; absolute values are not
# comparable to anything computed at another rate.
ANALYSIS_SFREQ = None

# Two window lengths. Source-space timeseries are smoother than sensor-level
# ones (leakage and rank reduction), so a 1 s window may resolve less of
# their structure; 2 s is run alongside rather than instead. Everything
# downstream carries a win_sec column so the two never get mixed.
SLIDE_WINDOWS_S = (1.0, 2.0)

# Step 5 samples (10 ms at 500 Hz). At step 1 consecutive windows shared
# 499 of 500 samples, so the series was oversampled ~500-fold relative to
# its own ~1 Hz bandwidth; step 5 keeps it heavily oversampled but cuts the
# window count and the stored series fivefold.
SLIDE_STEP_SAMPLES = 5

FMIN, FMAX = 0.5, 45.0
SLOPE_FMIN, SLOPE_FMAX = 1.0, 45.0

# IRASA needs valid spectrum above max(hset) * band_hi. Verified clean (no
# Nyquist warning) at both 250 and 500 Hz for a 1-45 Hz band.
IRASA_BAND = (1.0, 45.0)

# Spec_lowhigh_ratio boundaries. In the behavioural RT battery this measure
# is a robust stand-in for SpectralSlope on ~25-point series with ~11 usable
# frequency bins, where a log-log fit is barely estimable. That argument
# does not carry over here — 9500 samples give thousands of bins and both
# SpectralSlope and IRASA_slope are well constrained — so the boundaries are
# chosen physiologically rather than inherited: 8 Hz splits the aperiodic-
# dominated low band from alpha and above, 13 Hz splits alpha-and-below from
# beta and above.
LOWHIGH_SPLITS_HZ = (8.0, 13.0)

# The behavioural RT battery splits at np.median(freqs) — the midpoint of
# the frequency AXIS, not the power-weighted median frequency. Emitted here
# too, as Spec_lowhigh_median, so the source-space and behavioural values
# are directly comparable despite the very different frequency ranges.
LOWHIGH_AXIS_MEDIAN = True
IRASA_WIN_SEC = 4.0
IRASA_HSET = [round(1.1 + 0.05 * i, 2) for i in range(17)]   # 1.10 ... 1.90

SAMPEN_M = 2
SAMPEN_R = 0.2

MSE_MAX_SCALE = 20
MSE_M = 2
MSE_R = 0.15

# SpectralSlope is deliberately NOT slid: a 1 s window gives 1 Hz resolution,
# so a 1-45 Hz log-log fit rests on ~45 bins whose lowest is a single cycle.
# The static version (19 s, Welch-averaged) is the trustworthy one.
# The ACF battery (ACF_area_to_zero / _lag1 / _min / _min_lag) is STATIC
# ONLY. Inside a 1-2 s window the autocorrelation is too short-range for
# those descriptors to mean much, and ACW0/ACW50 already carry the
# window-level decay.
SLIDING_MEASURES = ['LZC', 'ACW0', 'ACW50', 'SpectralEntropy', 'MedianFreq']

# Any saturation at all disqualifies an ACW series: a value pinned to the
# window edge is not a measurement. Raise this (e.g. 0.01) to tolerate a
# few bad windows.
ACW_SAT_TOL = 0.0
SAT_TOKEN = 'sat'

# --- Measure groups for incremental skip logic ---
# Each group maps to the set of measure names it produces in the static CSV.
# A group is skipped only when ALL its measures already exist.
_MSE_MEASURES = set()
for _v in ('costa', 'norm'):
    for _k in ('CI', 'AUC', 'slope', 'max_entropy', 'scale_at_max',
               'entropy_range'):
        _MSE_MEASURES.add(f'MSE_{_v}_m{MSE_M}_{_k}')
    _MSE_MEASURES.add(f'MSE_{_v}_m{MSE_M}_curve')
_MSE_MEASURES.add('MSE_n_points_min')

STATIC_MEASURE_GROUPS = {
    'LZC':        {'LZC'},
    'ACW':        {'ACW0', 'ACW50'},
    'spectral':   {'MedianFreq', 'SpectralEntropy', 'SpectralSlope', 'PeakFreq'},
    'IRASA':      {'IRASA_slope', 'IRASA_intercept', 'IRASA_r2', 'IRASA_oscpower'},
    'SampEn':     {f'SampEn_m{SAMPEN_M}'},
    'MSE':        _MSE_MEASURES,
    'ACF':        {'ACF_area_to_zero', 'ACF_lag1', 'ACF_min', 'ACF_min_lag'},
    'spec_extra': {'Spec_centroid', 'Spec_AUC'}
                  | {f'Spec_lowhigh_{hz:g}Hz' for hz in LOWHIGH_SPLITS_HZ}
                  | ({'Spec_lowhigh_median'} if LOWHIGH_AXIS_MEDIAN else set()),
}
# ------------------------------------------------------------------

try:
    from numpy import trapezoid as _trapz
except ImportError:
    from numpy import trapz as _trapz


# ==================================================================
# LZC
# ==================================================================
def _lempel_ziv(s):
    i, k, l, k_max, c, n = 0, 1, 1, 1, 1, len(s)
    while True:
        if l + k > n:
            c += 1
            break
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
        else:
            k_max = max(k_max, k)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def m_lzc(x, fs=None):
    """
    Median-binarised Lempel-Ziv complexity, normalised by n/log2(n).

    Uses antropy's parser rather than the in-house one. Verified
    bit-identical on random-walk, white-noise and sine+noise signals at
    both window lengths (np.allclose true in every case) — same algorithm,
    13-25x faster. That matters because LZC runs once per sliding window,
    ~17,500 times per snippet, and was 69% of the total runtime.
    """
    n = len(x)
    if n < 2:
        return np.nan
    b = (x >= np.median(x)).astype(np.uint8)
    return float(ant.lziv_complexity(b, normalize=True))


# ==================================================================
# spectral (static)
# ==================================================================
def _psd(x, fs, method='periodogram'):
    if method == 'welch':
        nperseg = min(len(x), max(64, int(fs)))
        return welch(x, fs=fs, nperseg=nperseg)
    return periodogram(x, fs=fs)


def m_median_freq(x, fs):
    f, p = _psd(x, fs)
    m = (f >= FMIN) & (f <= FMAX)
    f, p = f[m], p[m]
    if len(p) == 0 or p.sum() == 0:
        return np.nan
    c = np.cumsum(p)
    return float(f[np.argmax(c >= c[-1] / 2)])


def m_spectral_entropy(x, fs):
    f, p = _psd(x, fs)
    p = p[(f >= FMIN) & (f <= FMAX)]
    if len(p) == 0 or p.sum() == 0:
        return np.nan
    pn = p / p.sum()
    return float(-np.sum(pn * np.log(pn + 1e-12)) / np.log(len(p)))


def m_spectral_slope(x, fs):
    f, p = _psd(x, fs, method='welch')
    m = (f >= SLOPE_FMIN) & (f <= SLOPE_FMAX)
    f, p = f[m], p[m]
    if len(p) < 3:
        return np.nan
    slope, _ = np.polyfit(np.log10(f), np.log10(np.maximum(p, 1e-20)), 1)
    return float(-slope)


def m_peak_freq(x, fs):
    f, p = _psd(x, fs)
    m = (f >= FMIN) & (f <= FMAX)
    f, p = f[m], p[m]
    return float(f[np.argmax(p)]) if len(p) else np.nan


def m_irasa(x, fs):
    """
    IRASA via neurodsp — the published Wen & Liu (2016) method: resample by
    a set of non-integer factors h and their reciprocals, take the geometric
    mean of each h / 1-h pair, then the MEDIAN across the set to estimate
    the fractal spectrum. The oscillatory spectrum is the residual.

    neurodsp rather than yasa: identical algorithm, but it needs only
    numpy / scipy / matplotlib, whereas yasa pins mne>=1.3 and would risk
    upgrading the MNE that the source-reconstruction stage runs on. It also
    uses resample_poly with exact integer up/down ratios, which is cleaner
    than resampling to a rounded sample count.

    Returns slope (positive exponent), intercept, R^2 of the log-log fit,
    and the integrated positive oscillatory power. R^2 is computed here
    because neurodsp's fit_irasa returns only intercept and slope; a poor
    fit means the exponent should not be trusted for that snippet.
    """
    try:
        freqs, psd_ap, psd_osc = compute_irasa(
            x, fs=fs, f_range=IRASA_BAND, hset=np.array(IRASA_HSET),
            nperseg=int(IRASA_WIN_SEC * fs))
        m = np.isfinite(freqs) & np.isfinite(psd_ap) & (freqs > 0) & (psd_ap > 0)
        if m.sum() < 3:
            return np.nan, np.nan, np.nan, np.nan
        lf, lp = np.log10(freqs[m]), np.log10(psd_ap[m])
        slope, intercept = np.polyfit(lf, lp, 1)
        resid = lp - (slope * lf + intercept)
        ss_res, ss_tot = float(np.sum(resid ** 2)), float(np.sum((lp - lp.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        oscpow = float(_trapz(np.maximum(psd_osc, 0), freqs))
        return float(-slope), float(intercept), float(r2), oscpow
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ==================================================================
# ACW (static)
# ==================================================================
def _acf_fft(x):
    n = len(x)
    xc = x - x.mean()
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    F = np.fft.rfft(xc, n=nfft)
    ac = np.fft.irfft(np.abs(F) ** 2, n=nfft)[:n]
    return ac / ac[0] if ac[0] > 0 else None


def m_acw(x, fs, threshold):
    if np.std(x) < 1e-12:
        return np.nan, False
    a = _acf_fft(x)
    if a is None:
        return np.nan, False
    below = np.where(a <= threshold)[0]
    if len(below):
        return float(below[0]) / fs, False
    return float(len(a) - 1) / fs, True          # saturated


def m_acf_battery(x, fs):
    """
    Extra autocorrelation descriptors, ported from the behavioural battery.

    ACF_area_to_zero  integral of the ACF up to its first zero crossing, in
                      seconds. A graded version of ACW0: two signals can
                      cross zero at the same lag while decaying at very
                      different rates, and the area separates them.
    ACF_lag1          autocorrelation at one sample. At 500 Hz this is
                      essentially a smoothness index and will sit near 1.
    ACF_min           most negative autocorrelation — oscillatory rebound.
    ACF_min_lag       lag of that minimum, in seconds.
    """
    out = {k: np.nan for k in ('ACF_area_to_zero', 'ACF_lag1',
                               'ACF_min', 'ACF_min_lag')}
    if np.std(x) < 1e-12:
        return out
    a = _acf_fft(x)
    if a is None or len(a) < 3:
        return out
    out['ACF_lag1'] = float(a[1])
    z = np.where(a <= 0.0)[0]
    stop = int(z[0]) if len(z) else len(a)
    if stop >= 2:
        out['ACF_area_to_zero'] = float(_trapz(a[:stop],
                                               np.arange(stop) / fs))
    i = int(np.argmin(a))
    out['ACF_min'] = float(a[i])
    out['ACF_min_lag'] = float(i) / fs
    return out


def m_spec_extra(x, fs):
    """
    Spectral descriptors beyond the median: centroid, in-band area, and
    low/high power ratios at fixed physiological boundaries.

    Spec_AUC is in-band power AFTER z-scoring, so it reads as the share of
    the signal's variance that falls inside FMIN-FMAX rather than an
    absolute power.
    """
    out = {'Spec_centroid': np.nan, 'Spec_AUC': np.nan}
    for hz in LOWHIGH_SPLITS_HZ:
        out[f'Spec_lowhigh_{hz:g}Hz'] = np.nan
    if LOWHIGH_AXIS_MEDIAN:
        out['Spec_lowhigh_median'] = np.nan
    f, p = _psd(x, fs)
    band = (f >= FMIN) & (f <= FMAX)
    fb, pb = f[band], p[band]
    if len(pb) == 0 or pb.sum() <= 0:
        return out
    out['Spec_centroid'] = float(np.sum(fb * pb) / np.sum(pb))
    out['Spec_AUC'] = float(_trapz(pb, fb))
    splits = [(f'Spec_lowhigh_{hz:g}Hz', hz) for hz in LOWHIGH_SPLITS_HZ]
    if LOWHIGH_AXIS_MEDIAN:
        splits.append(('Spec_lowhigh_median', float(np.median(fb))))
    for key, hz in splits:
        lo, hi = pb[fb <= hz].sum(), pb[fb > hz].sum()
        out[key] = float(lo / hi) if hi > 0 else np.nan
    return out


# ==================================================================
# SampEn / MSE  — EntropyHub (the published reference implementation)
# ==================================================================
@contextlib.contextmanager
def _quiet():
    """EntropyHub prints progress dots; keep the logs readable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def static_sampen(x, m=SAMPEN_M, r=SAMPEN_R):
    """
    SampEn on the z-scored series, tolerance r * SD.

    eh.SampEn returns entropies for embedding dimensions 0..m; we take
    the last element, which is the value for m.
    """
    sd = np.std(x)
    if sd < 1e-12:
        return np.nan
    xs = (x - np.mean(x)) / sd
    try:
        with _quiet():
            out = eh.SampEn(xs, m=m, tau=1, r=r * np.std(xs))
        v = float(out[0][-1])
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


MSE_SUMMARY_KEYS = ('CI', 'AUC', 'slope', 'max_entropy',
                    'scale_at_max', 'entropy_range')


def summarise_mse_curve(MSx, CI, max_scale):
    """
    Shape descriptors for an MSE curve. The area alone cannot distinguish a
    falling curve (uncorrelated noise) from a rising one (long-range
    correlated), which is the contrast MSE exists to show — hence the slope
    and the location of the maximum.
    """
    out = {k: np.nan for k in MSE_SUMMARY_KEYS}
    v = np.asarray(MSx, float)
    ok = np.isfinite(v)
    sc = np.arange(1, len(v) + 1)[ok]
    v = v[ok]
    if v.size == 0:
        return out
    out['CI'] = float(CI) if np.isfinite(CI) else float(v.sum())
    out['AUC'] = float(_trapz(v, sc) / max_scale)
    out['max_entropy'] = float(v.max())
    out['scale_at_max'] = float(sc[int(np.argmax(v))])
    out['entropy_range'] = float(v.max() - v.min())
    if v.size >= 2:
        out['slope'] = float(np.polyfit(sc, v, 1)[0])
    return out


def mse_both(x, m=MSE_M, r=MSE_R, scales=MSE_MAX_SCALE):
    """
    Multiscale entropy in both tolerance conventions, via eh.MSEn.

      costa : RadNew=0 — the tolerance is fixed at r * SD(original series)
              and held across every scale, so the variance reduction from
              coarse-graining is part of the measure. This is the
              canonical Costa et al. definition.

      norm  : RadNew=1 — the tolerance is rescaled to r * SD of each
              coarse-grained sub-sequence. This reproduces the behaviour
              of the existing sensor-level m_mse, which re-z-scores at
              every scale inside m_sampen.

    Returns a dict with, for each variant: the full curve, the complexity
    index (sum over scales, EntropyHub's CI), and the trapezoid area
    normalised by max_scale (the summary the sensor pipeline used).
    """
    out = {}
    sd0 = np.std(x)
    for variant, radnew, rad in (('costa', 0, r * sd0), ('norm', 1, r)):
        curve, summ = None, {k: np.nan for k in MSE_SUMMARY_KEYS}
        if sd0 >= 1e-12:
            try:
                with _quiet():
                    Mobj = eh.MSobject('SampEn', m=m, tau=1, r=rad)
                    MSx, CI = eh.MSEn(x, Mobj, Scales=scales,
                                      Methodx='coarse', RadNew=radnew)
                curve = np.asarray(MSx, float)
                summ = summarise_mse_curve(curve, CI, scales)
            except Exception:
                pass
        out[variant] = dict(curve=curve, **summ)
    return out


# ==================================================================
# sliding batches (shared transform per window)
# ==================================================================
# Windows are processed in chunks so peak memory does not scale with the
# number of windows. At win=1000, step=1 the full frame matrix is ~68 MB and
# its FFT autocorrelation ~280 MB; multiplied by the worker count that is
# enough to get workers OOM-killed. Chunking bounds it to a few MB each.
FRAME_CHUNK = 1024


def _n_windows(n, win, step):
    return (n - win) // step + 1


def _frame_chunks(x, win, step, chunk=FRAME_CHUNK):
    """Yield (offset, frames) so callers never hold every window at once."""
    n_win = _n_windows(len(x), win, step)
    idx = np.arange(win)
    for a in range(0, n_win, chunk):
        b = min(a + chunk, n_win)
        starts = np.arange(a, b) * step
        yield a, x[starts[:, None] + idx[None, :]].astype(float)


def _frames(x, win, step):
    starts = np.arange(0, len(x) - win + 1, step)
    return x[starts[:, None] + np.arange(win)[None, :]].astype(float)


def sliding_spectral(x, fs, win, step):
    n_win = _n_windows(len(x), win, step)
    freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    band = (freqs >= FMIN) & (freqs <= FMAX)
    sb = (freqs >= SLOPE_FMIN) & (freqs <= SLOPE_FMAX)
    fb = freqs[band]
    se = np.full(n_win, np.nan)
    mf = np.full(n_win, np.nan)
    sl = np.full(n_win, np.nan)
    logf = np.log10(freqs[sb]) if sb.sum() >= 3 else None
    A = (np.vstack([logf, np.ones_like(logf)]).T if logf is not None else None)

    for off, fr in _frame_chunks(x, win, step):
        fr = fr - fr.mean(axis=1, keepdims=True)
        psd = (np.abs(np.fft.rfft(fr, axis=1)) ** 2) / (fs * win)
        pb = psd[:, band]
        tot = pb.sum(axis=1)
        good = tot > 0
        if band.sum() and good.any():
            pn = pb[good] / tot[good][:, None]
            e = -np.sum(np.where(pn > 0, pn * np.log(pn + 1e-12), 0), axis=1)
            se[off:off + len(fr)][good] = e / np.log(band.sum())
            csum = np.cumsum(pb[good], axis=1)
            half = csum[:, -1] / 2
            mf[off:off + len(fr)][good] = fb[np.argmax(csum >= half[:, None],
                                                       axis=1)]
        if A is not None:
            logp = np.log10(np.maximum(psd[:, sb], 1e-20))
            coef, *_ = np.linalg.lstsq(A, logp.T, rcond=None)
            sl[off:off + len(fr)] = -coef[0]
    return {'SpectralEntropy': se, 'MedianFreq': mf, 'SpectralSlope': sl}


def _sliding_acf_all(x, fs, win, step):
    """
    ACW0 and ACW50 from one chunked autocorrelation pass.

    The wider ACF battery is deliberately NOT computed here — it is static
    only. See the note by SLIDING_MEASURES.
    """
    n_win = _n_windows(len(x), win, step)
    nfft = 1
    while nfft < 2 * win:
        nfft *= 2

    out = {k: np.full(n_win, np.nan) for k in ('ACW0', 'ACW50')}
    sat0 = np.zeros(n_win, bool)
    sat5 = np.zeros(n_win, bool)

    for off, fr in _frame_chunks(x, win, step):
        fr = fr - fr.mean(axis=1, keepdims=True)
        ac = np.fft.irfft(np.abs(np.fft.rfft(fr, n=nfft, axis=1)) ** 2,
                          n=nfft, axis=1)[:, :win]
        lag0 = ac[:, 0].copy()
        nz = lag0 > 0
        ac[nz] /= lag0[nz][:, None]
        for j in np.where(nz)[0]:
            w = off + j
            a = ac[j]
            b0 = np.where(a <= 0.0)[0]
            b5 = np.where(a <= 0.5)[0]
            if len(b0):
                out['ACW0'][w] = b0[0] / fs
            else:
                out['ACW0'][w] = (win - 1) / fs
                sat0[w] = True
            if len(b5):
                out['ACW50'][w] = b5[0] / fs
            else:
                out['ACW50'][w] = (win - 1) / fs
                sat5[w] = True
        del ac
    frac = (lambda v: float(v.mean()) if n_win else np.nan)
    return out, {'ACW0': frac(sat0), 'ACW50': frac(sat5)}


def sliding_lzc(x, win, step):
    n_win = _n_windows(len(x), win, step)
    out = np.empty(n_win)
    for off, fr in _frame_chunks(x, win, step):
        bits = (fr >= np.median(fr, axis=1, keepdims=True)).astype(np.uint8)
        for j, row in enumerate(bits):
            out[off + j] = ant.lziv_complexity(row, normalize=True)
    return out


# ==================================================================
# per-unit worker
# ==================================================================
SUMMARY_KEYS = ('mean', 'sd', 'cv', 'median', 'p05', 'p95',
                'min', 'max', 'slope')


def summarise(v):
    """
    Distribution of a sliding measure across windows.

    sd and cv are the point of the exercise: they say how much the measure
    itself fluctuates within the snippet. slope is the linear trend across
    window index, which catches monotone drift that the spread statistics
    would not distinguish from noise.
    """
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    idx = np.arange(v.size)[ok]
    v = v[ok]
    if v.size == 0:
        return {k: np.nan for k in SUMMARY_KEYS}
    mu, sd = float(v.mean()), float(v.std())
    out = {'mean': mu, 'sd': sd,
           'cv': (sd / abs(mu)) if abs(mu) > 1e-12 else np.nan,
           'median': float(np.median(v)),
           'p05': float(np.percentile(v, 5)),
           'p95': float(np.percentile(v, 95)),
           'min': float(v.min()), 'max': float(v.max()),
           'slope': np.nan}
    if v.size >= 2:
        out['slope'] = float(np.polyfit(idx, v, 1)[0])
    return out


def process_unit(row_data, meta, fs, windows_s, step, series_slots, row_idx,
                 skip_static_groups=frozenset(),
                 skip_sliding_keys=frozenset()):
    """
    row_data: (n_snippets, n_samples), already z-scored.

    Static measures depend only on the snippet, so they are computed once.
    The sliding battery is run separately for every window length in
    windows_s; every sliding row and every stored series carries win_sec,
    so the two never get conflated downstream.

    skip_static_groups : frozenset of group names (keys of STATIC_MEASURE_GROUPS)
        whose static measures already exist and should not be recomputed.
    skip_sliding_keys  : frozenset of (measure_name, win_sec) tuples
        whose sliding results already exist and should not be recomputed.
    """
    static_rows, slide_rows = [], []
    # series_slots: {win_sec: {measure: path_to_npy}} or None.
    # Workers write their own row slice into a preallocated memmap instead
    # of returning arrays. Returning them would push ~9 GB through joblib's
    # pickle channel back to the parent, which is what killed the workers.
    save_series = series_slots is not None
    do_any_static = len(skip_static_groups) < len(STATIC_MEASURE_GROUPS)

    for si in range(row_data.shape[0]):
        x = np.asarray(row_data[si], float)
        m = {**meta, 'snippet_idx': si}
        if not np.isfinite(x).all() or np.std(x) < 1e-12:
            if do_any_static:
                static_rows.append({**m, 'measure': 'ALL', 'value': np.nan,
                                    'flag': 'flat_or_nan'})
            continue

        vals = {}
        curves = {}

        if 'LZC' not in skip_static_groups:
            vals['LZC'] = m_lzc(x)

        if 'ACW' not in skip_static_groups:
            a0, sat0 = m_acw(x, fs, 0.0)
            a5, sat5 = m_acw(x, fs, 0.5)
            vals['ACW0'] = SAT_TOKEN if sat0 else a0
            vals['ACW50'] = SAT_TOKEN if sat5 else a5

        if 'spectral' not in skip_static_groups:
            vals['MedianFreq'] = m_median_freq(x, fs)
            vals['SpectralEntropy'] = m_spectral_entropy(x, fs)
            vals['SpectralSlope'] = m_spectral_slope(x, fs)
            vals['PeakFreq'] = m_peak_freq(x, fs)

        if 'IRASA' not in skip_static_groups:
            ir_s, ir_i, ir_r2, ir_p = m_irasa(x, fs)
            vals['IRASA_slope'] = ir_s
            vals['IRASA_intercept'] = ir_i
            vals['IRASA_r2'] = ir_r2
            vals['IRASA_oscpower'] = ir_p

        if 'SampEn' not in skip_static_groups:
            vals[f'SampEn_m{SAMPEN_M}'] = static_sampen(x)

        if 'MSE' not in skip_static_groups:
            mse = mse_both(x)
            for variant, d in mse.items():
                for k in MSE_SUMMARY_KEYS:
                    vals[f'MSE_{variant}_m{MSE_M}_{k}'] = d[k]
                if d['curve'] is not None:
                    curves[f'MSE_{variant}_m{MSE_M}_curve'] = d['curve']
            vals['MSE_n_points_min'] = float(len(x) // MSE_MAX_SCALE)

        if 'ACF' not in skip_static_groups:
            vals.update(m_acf_battery(x, fs))

        if 'spec_extra' not in skip_static_groups:
            vals.update(m_spec_extra(x, fs))

        for k, v in vals.items():
            static_rows.append({**m, 'measure': k, 'value': v, 'flag': ''})
        # full MSE curves kept as JSON so the shape stays inspectable
        for k, c in curves.items():
            static_rows.append({**m, 'measure': k,
                                'value': json.dumps([None if not np.isfinite(q)
                                                     else round(float(q), 6)
                                                     for q in c]),
                                'flag': 'curve'})

        # --- sliding, once per window length ---
        for win_s in windows_s:
            win = int(round(win_s * fs))
            if win >= len(x):
                continue
            # which sliding measures still need computing for this window?
            todo_slide = [n for n in SLIDING_MEASURES
                          if (n, win_s) not in skip_sliding_keys]
            if not todo_slide:
                continue

            # compute only the batches we need
            need_spectral = any(n in ('SpectralEntropy', 'MedianFreq')
                                for n in todo_slide)
            need_acf = any(n in ('ACW0', 'ACW50') for n in todo_slide)
            need_lzc = 'LZC' in todo_slide

            batched = {}
            sat_frac = {}
            if need_spectral:
                batched.update(sliding_spectral(x, fs, win, step))
            if need_acf:
                acf_out, sf = _sliding_acf_all(x, fs, win, step)
                batched.update(acf_out)
                sat_frac.update(sf)
            if need_lzc:
                batched['LZC'] = sliding_lzc(x, win, step)

            for name in todo_slide:
                ser = batched[name]
                sf = sat_frac.get(name)
                bad = sf is not None and np.isfinite(sf) and sf > ACW_SAT_TOL
                stats = ({k: SAT_TOKEN for k in SUMMARY_KEYS} if bad
                         else summarise(ser))
                for stat, val in stats.items():
                    slide_rows.append({**m, 'win_sec': win_s, 'measure': name,
                                       'stat': stat, 'value': val,
                                       'n_win': len(ser)})
                if sf is not None:
                    slide_rows.append({**m, 'win_sec': win_s, 'measure': name,
                                       'stat': 'sat_frac', 'value': sf,
                                       'n_win': len(ser)})
                if save_series and win_s in series_slots \
                        and name in series_slots[win_s]:
                    mm = np.load(series_slots[win_s][name], mmap_mode='r+')
                    n = min(len(ser), mm.shape[2])
                    mm[row_idx, si, :n] = np.asarray(ser[:n], np.float32)
                    del mm
    return static_rows, slide_rows


# ==================================================================
def _detect_done(f_static, f_slide):
    """
    Load existing CSVs and figure out which measure groups / sliding keys
    are already computed.  Returns (existing_static_df, existing_sliding_df,
    skip_static_groups, skip_sliding_keys).
    """
    existing_static = None
    existing_sliding = None
    done_static_measures = set()
    done_sliding_keys = set()      # (measure_name, win_sec)

    if op.isfile(f_static) and op.getsize(f_static) > 0:
        try:
            existing_static = pd.read_csv(f_static)
            done_static_measures = set(existing_static['measure'].unique())
        except Exception:
            existing_static = None

    if op.isfile(f_slide) and op.getsize(f_slide) > 0:
        try:
            existing_sliding = pd.read_csv(f_slide)
            for mname in existing_sliding['measure'].unique():
                for ws in existing_sliding.loc[
                        existing_sliding['measure'] == mname, 'win_sec'].unique():
                    done_sliding_keys.add((str(mname), float(ws)))
        except Exception:
            existing_sliding = None

    skip_static_groups = set()
    for grp, members in STATIC_MEASURE_GROUPS.items():
        if members <= done_static_measures:
            skip_static_groups.add(grp)

    return (existing_static, existing_sliding,
            frozenset(skip_static_groups), frozenset(done_sliding_keys))


def run(npz_path, jobs, slide_step, save_series, overwrite):
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = op.basename(npz_path).replace('_label_tc.npz', '')
    f_static = op.join(OUT_DIR, f'{stem}_static.csv')
    f_slide = op.join(OUT_DIR, f'{stem}_sliding.csv')

    # --- incremental skip detection ---
    existing_static = None
    existing_sliding = None
    skip_static_groups = frozenset()
    skip_sliding_keys = frozenset()

    if not overwrite:
        (existing_static, existing_sliding,
         skip_static_groups, skip_sliding_keys) = _detect_done(f_static, f_slide)

        all_static_done = (skip_static_groups
                           == frozenset(STATIC_MEASURE_GROUPS.keys()))
        all_sliding_done = all(
            (m, w) in skip_sliding_keys
            for m in SLIDING_MEASURES for w in SLIDE_WINDOWS_S)

        if all_static_done and all_sliding_done:
            series_ok = True
            if save_series:
                series_ok = all(
                    op.isfile(op.join(series_dir(stem, w), 'index.npz'))
                    for w in SLIDE_WINDOWS_S)
            if series_ok:
                print(f'{stem}: all measures already computed, skipping')
                return 0

    d = np.load(npz_path, allow_pickle=True)
    data = d['data']                      # (n_rows, n_snip, n_samp)
    fs_stored = float(d['sfreq'])
    fs = ANALYSIS_SFREQ or fs_stored
    zmu, zsd = np.asarray(d['zref_mean']), np.asarray(d['zref_sd'])

    if abs(fs - fs_stored) > 1e-6:
        import mne
        print(f'  resampling {fs_stored:.0f} -> {fs:.0f} Hz', flush=True)
        n_out = int(round(data.shape[2] * fs / fs_stored))
        res = np.empty(data.shape[:2] + (n_out,), np.float32)
        for r in range(data.shape[0]):
            for si in range(data.shape[1]):
                y = data[r, si].astype(np.float64)
                res[r, si] = (mne.filter.resample(y, down=fs_stored / fs,
                                                  npad='auto')[:n_out]
                              if np.isfinite(y).all() else np.nan)
        data = res
        zmu = np.array([np.nanmean(data[r]) for r in range(data.shape[0])])
        zsd = np.array([np.nanstd(data[r]) for r in range(data.shape[0])])

    n_rows, n_snip, n_samp = data.shape
    print(f'{stem}: {n_rows} unit-method rows, {n_snip} snippets, '
          f'{n_samp} samples @ {fs:.0f} Hz (stored {fs_stored:.0f} Hz)')
    for w in SLIDE_WINDOWS_S:
        win = int(round(w * fs))
        print(f'  sliding win={w:g}s ({win} samples), step={slide_step} -> '
              f'{(n_samp - win) // slide_step + 1} windows/snippet')

    # --- report incremental status ---
    todo_static = sorted(set(STATIC_MEASURE_GROUPS) - skip_static_groups)
    done_static = sorted(skip_static_groups)
    todo_sliding = sorted({m for m in SLIDING_MEASURES
                           for w in SLIDE_WINDOWS_S
                           if (m, w) not in skip_sliding_keys})
    done_sliding = sorted({m for m in SLIDING_MEASURES
                           if all((m, w) in skip_sliding_keys
                                  for w in SLIDE_WINDOWS_S)})
    if done_static or done_sliding:
        print(f'  [incremental] skipping static groups: '
              f'{", ".join(done_static) or "(none)"}')
        print(f'  [incremental] skipping sliding measures: '
              f'{", ".join(done_sliding) or "(none)"}')
        print(f'  [incremental] computing static groups: '
              f'{", ".join(todo_static) or "(none)"}')
        print(f'  [incremental] computing sliding measures: '
              f'{", ".join(todo_sliding) or "(none)"}')

    keep_rows, tasks = [], []
    for r in range(n_rows):
        if not np.isfinite(zsd[r]) or zsd[r] < 1e-30:
            continue
        keep_rows.append(r)
        tasks.append(((data[r] - zmu[r]) / zsd[r],
                      dict(subject=str(d['subject']), version=str(d['version']),
                           level=str(d['unit_level'][r]),
                           unit_name=str(d['unit_name'][r]),
                           method=str(d['method'][r]),
                           n_vertices=int(d['n_vertices'][r]),
                           pca_var=float(d['pca_var'][r]),
                           ori_coherence=float(d['ori_coherence'][r]))))

    slots = (make_series_slots(stem, len(tasks), n_snip, n_samp, fs,
                               slide_step) if save_series else None)

    print(f'  {len(tasks)} of {n_rows} rows to process on {jobs} workers')
    res = Parallel(n_jobs=jobs, backend='loky', verbose=5)(
        delayed(process_unit)(z, meta, fs, SLIDE_WINDOWS_S, slide_step,
                              slots, i,
                              skip_static_groups, skip_sliding_keys)
        for i, (z, meta) in enumerate(tasks))

    snip_meta = pd.DataFrame({
        'snippet_idx': np.arange(n_snip),
        'block': d['snip_block'], 'condition': d['snip_condition'],
        'vas_time': d['snip_vas_time']})

    st = pd.DataFrame([r for a, _ in res for r in a])
    sl = pd.DataFrame([r for _, b in res for r in b])
    if len(st):
        st = st.merge(snip_meta, on='snippet_idx', how='left')
    if len(sl):
        sl = sl.merge(snip_meta, on='snippet_idx', how='left')

    # --- merge with existing CSV rows (keep old, add new) ---
    if existing_static is not None and len(st) > 0:
        new_measures = set(st['measure'].unique())
        keep = existing_static[~existing_static['measure'].isin(new_measures)]
        st = pd.concat([keep, st], ignore_index=True)
    elif existing_static is not None and len(st) == 0:
        st = existing_static

    if existing_sliding is not None and len(sl) > 0:
        new_keys = set(zip(sl['measure'], sl['win_sec']))
        keep_mask = [
            (m, w) not in new_keys
            for m, w in zip(existing_sliding['measure'],
                            existing_sliding['win_sec'])]
        sl = pd.concat([existing_sliding[keep_mask], sl], ignore_index=True)
    elif existing_sliding is not None and len(sl) == 0:
        sl = existing_sliding

    if len(st):
        st.to_csv(f_static, index=False)
    if len(sl):
        sl.to_csv(f_slide, index=False)
    print(f'  static  : {len(st):,} rows -> {op.basename(f_static)}')
    print(f'  sliding : {len(sl):,} rows -> {op.basename(f_slide)}')

    if save_series:
        write_series_index(stem, keep_rows, d, fs, slide_step, n_samp)
    return 0


def series_dir(stem, win_s):
    return op.join(OUT_DIR, 'series', f'{stem}_win{win_s:g}s')


def make_series_slots(stem, n_rows, n_snip, n_samp, fs, step):
    """
    Preallocate one .npy per measure per window length, NaN-filled.

    Stored as plain .npy rather than a single npz so workers can write
    disjoint row slices concurrently through mmap_mode='r+', and so a later
    pass can memory-map one measure without loading ~4.6 GB.
    """
    slots = {}
    for win_s in SLIDE_WINDOWS_S:
        win = int(round(win_s * fs))
        n_win = _n_windows(n_samp, win, step)
        d = series_dir(stem, win_s)
        os.makedirs(d, exist_ok=True)
        slots[win_s] = {}
        for name in SLIDING_MEASURES:
            path = op.join(d, f'{name}.npy')
            arr = np.lib.format.open_memmap(
                path, mode='w+', dtype=np.float32,
                shape=(n_rows, n_snip, n_win))
            arr[:] = np.nan
            arr.flush()
            del arr
            slots[win_s][name] = path
        print(f'  allocated {len(SLIDING_MEASURES)} x '
              f'({n_rows}, {n_snip}, {n_win}) float32 in {op.basename(d)}')
    return slots


def write_series_index(stem, keep_rows, d, fs, step, n_samp):
    """
    Metadata beside the memmapped measure arrays, so a later pass can open
    a series directory on its own and still know which ROI, extraction
    method, block and condition every element belongs to.

    effective_bandwidth_hz is recorded deliberately: with step=1 the series
    is a moving average of length win_sec sampled at fs, so it is
    oversampled by win_sec*fs and carries essentially nothing above
    1/win_sec Hz. Spectral measures applied to these series need a band
    around that, not the 0.5-45 Hz used on the raw signal.
    """
    kr = np.asarray(keep_rows, int)
    for win_s in SLIDE_WINDOWS_S:
        win = int(round(win_s * fs))
        n_win = _n_windows(n_samp, win, step)
        dd = series_dir(stem, win_s)
        np.savez(op.join(dd, 'index.npz'),
                 measure_names=np.array(SLIDING_MEASURES),
                 unit_name=np.asarray(d['unit_name'])[kr],
                 unit_level=np.asarray(d['unit_level'])[kr],
                 method=np.asarray(d['method'])[kr],
                 n_vertices=np.asarray(d['n_vertices'])[kr],
                 pca_var=np.asarray(d['pca_var'])[kr],
                 ori_coherence=np.asarray(d['ori_coherence'])[kr],
                 snip_block=d['snip_block'],
                 snip_condition=d['snip_condition'],
                 snip_vas_time=d['snip_vas_time'],
                 win_start_samples=np.arange(n_win, dtype=np.int64) * step,
                 subject=str(d['subject']), version=str(d['version']),
                 sfreq_source=fs, sfreq_series=fs / step,
                 win_sec=win_s, win_samples=win, step_samples=step,
                 snippet_dur_s=float(d['snippet_dur_s']),
                 effective_bandwidth_hz=1.0 / win_s)
        mb = len(SLIDING_MEASURES) * len(kr) * len(d['snip_block']) * n_win * 4 / 1e6
        print(f'  series  : win={win_s:g}s  {op.basename(dd)}/  '
              f'{len(SLIDING_MEASURES)} measures, {mb:.0f} MB')


def n_jobs_from_env():
    for v in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        q = os.environ.get(v)
        if q and q.isdigit() and int(q) > 0:
            return int(q)
    return os.cpu_count() or 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--slide-step', type=int, default=SLIDE_STEP_SAMPLES)
    ap.add_argument('--no-series', action='store_true',
                    help='skip writing the full sliding series')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    jobs = a.jobs or n_jobs_from_env()
    try:
        return run(a.npz, jobs, a.slide_step, not a.no_series, a.overwrite)
    except Exception:
        print('FAILED:\n' + traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
