#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SS_emo_full.py
==============
Complete source-space pipeline for one subject x version, from the
preprocessed block files through to the label/ROI timeseries.

Replaces SS_emo_linux.py + SS_label_tc.py. Fully self-contained: the ROI
definitions are inlined below, so no other project file is needed.

Why one job per SUBJECT x VERSION rather than per block
-------------------------------------------------------
The pca_flip spatial weights are fitted jointly across base / sad / fear so
that a condition difference cannot be a spatial-filter difference. That
needs every block of a subject in memory at once, which a per-block array
job cannot do. Handling all blocks in one job also means the coregistration,
BEM, forward model and noise covariance are computed exactly once, instead
of three concurrent tasks racing to write the same file.

Stages (each skipped if its output already exists AND validates)
----------------------------------------------------------------
  0  label_tc/{subj}_{ver}_label_tc.npz   -> whole job skipped
  1  discover blocks + empty room
  2  coreg/{subj}-trans.fif
  3  coreg/{subj}_bem-sol.fif
  4  coreg/{subj}_fwd.fif
  5  coreg/{subj}_noise-cov.fif
  6  labels + ROI union (in memory)
  7  stc/{stem}_eloreta-{lh,rh}.stc        per block
  8  joint mean_flip + pca_flip extraction across blocks
  9  write label_tc npz

"Validates" means the file is actually readable, and for .stc that its size
matches the size implied by its own header. A file truncated by a killed job
or a full disk is treated as absent and recomputed.

Usage
-----
    python3 SS_emo_full.py 001 eog
    python3 SS_emo_full.py 001 ecg --overwrite
    python3 SS_emo_full.py 001 eog --force-stage fwd,stc
"""

import argparse
import glob
import os
import os.path as op
import re
import struct
import sys
import tempfile
import traceback
import warnings

import numpy as np
import mne
from mne.coreg import Coregistration
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from mne import read_labels_from_annot

mne.set_log_level('ERROR')

# ==================================================================
# CONFIG
# ==================================================================
BASE      = '/home/ZhangJianfeng_04/emo_study/MEG'
PREPROC   = op.join(BASE, '3_preproc')
RAW_ROOT  = op.join(BASE, '1_raw')
FS_DIR    = '/home/ZhangJianfeng_04/emo_study/freesurfer'
SS_DIR    = op.join(BASE, 'SourceSpace')
COREG_DIR = op.join(SS_DIR, 'coreg')
STC_DIR   = op.join(SS_DIR, 'stc')
TC_DIR    = op.join(SS_DIR, 'label_tc')
LOG_DIR   = op.join(SS_DIR, 'logs')

SNIPPET_DUR_S = 19.0
TARGET_SFREQ  = 500.0        # == acquisition rate -> no resampling
PARC          = 'HCPMMP1_resampled'

SPACING       = 'oct6'
BEM_ICO       = 5
BEM_COND      = (0.3,)       # single-shell, MEG only
MINDIST       = 5.0
SNR           = 3.0
INV_METHOD    = 'eLORETA'
PICK_ORI      = 'normal'
ICP_ITER      = 20
ICP_NASION_W  = 10.0
HSP_OUTLIER_M = 5.0 / 1000
COREG_WARN_MM = 5.0

METHODS = ('mean_flip', 'pca_flip')

STAGES = ('trans', 'bem', 'fwd', 'cov', 'stc', 'extract')
# ==================================================================


def log(msg=''):
    print(msg, flush=True)


def hdr(t):
    log(f"\n{'=' * 62}\n{t}\n{'=' * 62}")


def log_error(subject, version, stage, msg):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(op.join(LOG_DIR, f'errors_{subject}_{version}.txt'), 'a') as f:
        f.write(f'{stage}\t{msg}\n')


# ==================================================================
# MEG subject -> T1/FreeSurfer subject
# ==================================================================
# Repeat-scan subjects are filed under a new MEG code whose leading digit
# encodes the visit number: a participant first scanned as 009 returns as
# 209 (2nd visit) or 309 (3rd visit). They share the same T1/FreeSurfer
# recon as their first-visit code, so 209 and 309 both resolve to 009.
#
# RULE: any 3-digit code whose leading digit is 2 or 3 is a repeat scan
# and maps to the equivalent 0xx root (2xx -> 0xx, 3xx -> 0xx).
# Codes 0xx and 1xx are treated as root subjects with their own T1.
#
# EXCEPTIONS — subjects that break the rule in either direction:
#   • A 2xx/3xx code that IS an independent root (has its own T1):
#     add to REPEAT_SCAN_EXCEPTIONS to suppress the automatic redirect.
#   • A 0xx/1xx code that shares another subject's T1 (rare):
#     add to T1_SUBJECT_OVERRIDES for an explicit redirect.
#
# Empty room and trans/fwd/cov always stay keyed to the visit-specific
# MEG code (head position and noise differ per session).
# BEM stays keyed to the resolved T1 root (anatomy is shared).

REPEAT_SCAN_EXCEPTIONS = set()   # e.g. {'234'} if 234 is an independent root

T1_SUBJECT_OVERRIDES = {
    '206': '006',   # 006 has no MEG; its recording is filed as 206
}


def t1_root_subject(subject):
    """
    Map a MEG subject code to its root T1/FreeSurfer code.

    2xx -> 0xx  (2nd visit)
    3xx -> 0xx  (3rd visit)
    unless the code is listed in REPEAT_SCAN_EXCEPTIONS.
    Explicit entries in T1_SUBJECT_OVERRIDES take priority over the rule.
    """
    s = str(subject)
    if s in T1_SUBJECT_OVERRIDES:
        return T1_SUBJECT_OVERRIDES[s]
    if s not in REPEAT_SCAN_EXCEPTIONS and re.fullmatch(r'[23]\d{2}', s):
        return '0' + s[1:]
    return s


def resolve_fs_subject(subject):
    """
    Find the FreeSurfer subject folder for `subject`, falling back to the
    root T1 subject for repeat scans. Tries, in order: sub{subject},
    sub{int(subject)}, then the same two forms for the root code (only if
    different from `subject`). Returns the matching folder name, or None
    if nothing resolves.
    """
    candidates = [subject, str(int(subject))]
    root = t1_root_subject(subject)
    if root != subject:
        candidates += [root, str(int(root))]
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        name = f'sub{c}'
        if op.isdir(op.join(FS_DIR, name)):
            return name
    return None


# ==================================================================
# atomic write + validity
# ==================================================================
def atomic(path, write_fn):
    """
    Write via a temp file in the same directory, then os.replace().

    os.replace is atomic on POSIX, so a reader either sees the old file or
    the complete new one, never a half-written one. Two jobs racing on the
    same per-subject artifact both produce a valid file; one simply wins.
    """
    d = op.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_', suffix=op.basename(path))
    os.close(fd)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if op.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


MIN_BYTES = {'trans': 200, 'bem': 10_000, 'fwd': 100_000,
             'cov': 10_000, 'npz': 1_000}


def _too_small(path, kind):
    try:
        n = os.path.getsize(path)
    except OSError:
        return True
    if n < MIN_BYTES.get(kind, 0):
        log(f'  ! {op.basename(path)} is only {n} bytes — recomputing')
        return True
    return False


def valid_artifact(path, kind):
    """
    Is this artifact present, parseable AND sane?

    Three layers, cheapest first:
      1. size floor  — catches a zero-length or stub file instantly
      2. full parse  — MNE walks the FIF tag structure, so a file truncated
                       by a killed job or a full disk raises here
      3. content     — a file can parse and still be useless (a forward
                       solution with no sources, a covariance of dimension
                       zero), so check the thing we actually need from it

    Anything that fails is treated as absent and recomputed.
    """
    if not op.isfile(path) or _too_small(path, kind):
        return False
    try:
        if kind == 'trans':
            t = mne.read_trans(path)
            arr = np.asarray(t['trans'], float)
            if arr.shape != (4, 4) or not np.isfinite(arr).all():
                raise ValueError('trans matrix not a finite 4x4')
        elif kind == 'bem':
            b = mne.read_bem_solution(path, verbose=False)
            if not len(b.get('surfs', [])):
                raise ValueError('no surfaces in BEM solution')
        elif kind == 'fwd':
            f = mne.read_forward_solution(path, verbose=False)
            if f['nsource'] < 1 or f['nchan'] < 1:
                raise ValueError(f"nsource={f['nsource']} nchan={f['nchan']}")
            if not np.isfinite(f['sol']['data']).all():
                raise ValueError('non-finite values in the gain matrix')
        elif kind == 'cov':
            c = mne.read_cov(path, verbose=False)
            if c['dim'] < 1 or not np.isfinite(c['data']).all():
                raise ValueError('empty or non-finite covariance')
        else:
            raise ValueError(f'unknown artifact kind {kind}')
        return True
    except Exception as e:
        log(f'  ! {op.basename(path)} present but invalid '
            f'({str(e).splitlines()[0][:70]}) — recomputing')
        return False


def npz_valid(path, n_units, n_methods):
    """
    A savez_compressed killed part-way leaves a file that exists but is
    unreadable or short. Check the archive opens, carries every key the
    downstream script indexes on, has consistent axis lengths, and is not
    entirely NaN.
    """
    if not op.isfile(path) or _too_small(path, 'npz'):
        return False
    need = ('data', 'sfreq', 'snippet_dur_s', 'subject', 'version',
            'snip_block', 'snip_condition', 'snip_vas_time',
            'unit_name', 'unit_level', 'method', 'n_vertices',
            'pca_var', 'ori_coherence', 'zref_mean', 'zref_sd')
    try:
        with np.load(path, allow_pickle=True) as d:
            missing = [k for k in need if k not in d.files]
            if missing:
                raise ValueError(f'missing keys {missing}')
            data = d['data']
            if data.ndim != 3:
                raise ValueError(f'data.ndim={data.ndim}')
            n_rows, n_snip, n_samp = data.shape
            if n_rows != n_units * n_methods:
                raise ValueError(f'{n_rows} rows, expected '
                                 f'{n_units * n_methods}')
            for k in ('unit_name', 'unit_level', 'method', 'n_vertices',
                      'pca_var', 'ori_coherence', 'zref_mean', 'zref_sd'):
                if len(d[k]) != n_rows:
                    raise ValueError(f'{k} has {len(d[k])} entries, '
                                     f'data has {n_rows} rows')
            for k in ('snip_block', 'snip_condition', 'snip_vas_time'):
                if len(d[k]) != n_snip:
                    raise ValueError(f'{k} has {len(d[k])} entries, '
                                     f'data has {n_snip} snippets')
            if n_snip < 1 or n_samp < 100:
                raise ValueError(f'shape {data.shape} too small')
            if not np.isfinite(data).any():
                raise ValueError('data is entirely NaN')
        return True
    except Exception as e:
        log(f'  ! {op.basename(path)} present but invalid '
            f'({str(e).splitlines()[0][:70]}) — recomputing')
        return False


def stc_covers_units(stc, src, units):
    """
    Does this saved STC contain every vertex the CURRENT ROI set needs?

    The inverse is applied with label=roi_union, so an STC physically holds
    only the vertices of the ROI set it was computed for. Adding ROIs whose
    labels were not in that union leaves their vertices absent, and the
    extraction would silently produce empty rows rather than failing. So the
    needed vertices are compared against what the file actually has.

    Label vertices are intersected with the source space first: oct6
    decimation means most surface vertices are not sources at all, and their
    absence from the STC is expected, not a gap.
    """
    have = (set(stc.vertices[0].tolist()), set(stc.vertices[1].tolist()))
    in_src = (set(src[0]['vertno'].tolist()), set(src[1]['vertno'].tolist()))
    missing = 0
    for _, _, members in units:
        lv, rv = unit_vertices(members)
        for hemi, v in ((0, lv), (1, rv)):
            need = set(v.tolist()) & in_src[hemi]
            missing += len(need - have[hemi])
    return missing == 0, missing


def stc_valid(stem):
    """
    Validate a saved .stc pair from its own header, without reading the data.

    Layout (big-endian): float32 tmin, float32 tstep, uint32 n_vert,
    uint32[n_vert] vertices, uint32 n_times, float32[n_vert*n_times].
    """
    for hemi in ('-lh.stc', '-rh.stc'):
        p = stem + hemi
        if not op.isfile(p):
            return False
        try:
            size = os.path.getsize(p)
            with open(p, 'rb') as f:
                head = f.read(12)
                if len(head) < 12:
                    return False
                _, _, n_vert = struct.unpack('>ffI', head)
                if not (0 < n_vert < 5_000_000):
                    return False
                f.seek(4 * n_vert, os.SEEK_CUR)
                raw_nt = f.read(4)
                if len(raw_nt) < 4:
                    return False
                n_times, = struct.unpack('>I', raw_nt)
            if size != 12 + 4 * n_vert + 4 + 4 * n_vert * n_times:
                log(f'  ! {op.basename(p)} truncated — recomputing')
                return False
            if n_times < 100:
                return False
        except Exception:
            return False
    return True


# ============================================================
# ROI COMPOSITION (HCP-MMP1)
# KK = Qin et al. 2020 three-level table (as used by Keskin et al. 2025)
# TB = nearest HCP-MMP1 parcel to Loescher/Tallon-Baudry 2025 MNI peaks
# ============================================================

ROI_COMPOSITION = {
    # ---- original KK / TB set ----
    'Primary_visual':                  ['L_V1_ROI-lh', 'R_V1_ROI-rh'],
    'Primary_auditive':                ['L_A1_ROI-lh', 'R_A1_ROI-rh'],
    'Primary_motor':                   ['L_4_ROI-lh', 'R_4_ROI-rh'],
    'Primary_sensorymotor':            ['L_3b_ROI-lh', 'R_3b_ROI-rh'],
    'Intero_R_Insula_KK':              ['R_FOP4_ROI-rh'],
    'Intero_L_dACC_KK':                ['L_24dd_ROI-lh'],
    'Intero_L_Insula_Posterior_KK':    ['L_PoI2_ROI-lh'],
    'Intero_R_MTL_KK':                 ['R_H_ROI-rh'],
    'Intero_L_MTL_KK': [
        'L_PeEc_ROI-lh',
        'L_EC_ROI-lh',
        'L_H_ROI-lh',
        'L_PIR_ROI-lh',
    ],
    'Intero_R_SMG_KK':                 ['R_PF_ROI-rh'],
    'Intero_R_SFG_KK':                 ['R_8BM_ROI-rh'],
    'Intero_L_STG_KK':                 ['L_6r_ROI-lh'],
    'Extero_R_Fusiform_KK':            ['R_PH_ROI-rh'],
    'Extero_L_Fusiform_KK':            ['L_PH_ROI-lh'],
    'Extero_R_IFG_KK':                 ['R_IFSa_ROI-rh'],
    'Extero_R_Insula_KK':              ['R_MI_ROI-rh'],
    'Extero_L_Insula_Anterior_KK':     ['L_AAIC_ROI-lh', 'L_MI_ROI-lh'],
    'Extero_R_SPL_KK':                 ['R_IP1_ROI-rh'],
    'Extero_R_Postcentral_KK':         ['R_PFt_ROI-rh'],
    'Extero_R_IPL_KK':                 ['R_LIPv_ROI-rh', 'R_7PC_ROI-rh'],
    'Extero_L_IPL_KK':                 ['L_AIP_ROI-lh'],
    'Extero_L_SPL_KK':                 ['L_MIP_ROI-lh'],
    'Extero_R_OccipitalInf_KK':        ['R_LO1_ROI-rh'],
    'Extero_R_Cingulate_KK':           ['R_p24pr_ROI-rh'],
    'Extero_L_mPFC_KK':                ['L_9m_ROI-lh'],
    'Self_L_ACC_KK':                   ['L_a24_ROI-lh'],
    'Self_L_PCC_KK':                   ['L_d23ab_ROI-lh'],
    'Self_L_Cingulate_mid_KK':         ['L_23d_ROI-lh'],
    'Self_L_MTG_KK':                   ['L_PGi_ROI-lh'],
    'Self_R_MTG_KK':                   ['R_PGi_ROI-rh'],
    'Self_L_ITG_KK':                   ['L_TE1a_ROI-lh'],
    'Self_R_Insula_KK':                ['R_TA2_ROI-rh'],
    'Self_L_SFG_dorsal_KK':            ['L_8Ad_ROI-lh'],
    'Self_L_SFG_anterior_KK':          ['L_9a_ROI-lh'],
    'InteroSelf_L_Insula_Anterior_KK': ['L_AVI_ROI-lh'],
    'ExteroSelf_R_Premotor_KK':        ['R_6r_ROI-rh', 'R_IFJp_ROI-rh'],
    'Right_Somatosensory_TB':          ['R_PFt_ROI-rh', 'R_2_ROI-rh'],
    'Right_mPFC_TB':                   ['R_10d_ROI-rh'],
    'Left_mPFC_TB':                    ['L_10d_ROI-lh'],
    'Left_Anterior_Temporal_TB':       ['L_TE2a_ROI-lh'],
    'Right_Anterior_Temporal_TB':      ['R_TE1a_ROI-rh'],
    'Left_Premotor_TB':                ['L_6d_ROI-lh', 'L_6a_ROI-lh'],
    'Left_PCC_TB':                     ['L_23c_ROI-lh', 'L_5mv_ROI-lh'],
    'Left_IPS_TB':                     ['L_AIP_ROI-lh'],

    # ---- EMO addendum, tier 1: strongest evidence base ----
    'EMO_L_pSTS':                      ['L_STSdp_ROI-lh', 'L_STSvp_ROI-lh'],
    'EMO_R_pSTS':                      ['R_STSdp_ROI-rh', 'R_STSvp_ROI-rh'],
    'EMO_L_aSTS':                      ['L_STSda_ROI-lh', 'L_STSva_ROI-lh', 'L_STGa_ROI-lh'],
    'EMO_R_aSTS':                      ['R_STSda_ROI-rh', 'R_STSva_ROI-rh', 'R_STGa_ROI-rh'],
    'EMO_L_TPJ':                       ['L_TPOJ1_ROI-lh', 'L_STV_ROI-lh', 'L_PSL_ROI-lh'],
    'EMO_R_TPJ':                       ['R_TPOJ1_ROI-rh', 'R_STV_ROI-rh', 'R_PSL_ROI-rh'],
    'EMO_L_Angular':                   ['L_PGs_ROI-lh', 'L_PFm_ROI-lh'],
    'EMO_R_Angular':                   ['R_PGs_ROI-rh', 'R_PFm_ROI-rh'],
    'EMO_L_Precuneus': [
        'L_7m_ROI-lh',
        'L_POS2_ROI-lh',
        'L_31pd_ROI-lh',
        'L_PCV_ROI-lh',
    ],
    'EMO_R_Precuneus': [
        'R_7m_ROI-rh',
        'R_POS2_ROI-rh',
        'R_31pd_ROI-rh',
        'R_PCV_ROI-rh',
    ],
    'EMO_L_TemporalPole':              ['L_TGd_ROI-lh', 'L_TGv_ROI-lh'],
    'EMO_R_TemporalPole':              ['R_TGd_ROI-rh', 'R_TGv_ROI-rh'],
    'EMO_L_dmPFC':                     ['L_8BM_ROI-lh', 'L_9m_ROI-lh', 'L_d32_ROI-lh'],
    'EMO_R_dmPFC':                     ['R_8BM_ROI-rh', 'R_9m_ROI-rh', 'R_d32_ROI-rh'],

    # ---- EMO addendum, tier 2 ----
    'EMO_L_vlPFC':                     ['L_45_ROI-lh', 'L_47l_ROI-lh', 'L_a47r_ROI-lh'],
    'EMO_R_vlPFC':                     ['R_45_ROI-rh', 'R_47l_ROI-rh', 'R_a47r_ROI-rh'],
    'EMO_L_dlPFC':                     ['L_46_ROI-lh', 'L_p9-46v_ROI-lh', 'L_9-46d_ROI-lh'],
    'EMO_R_dlPFC':                     ['R_46_ROI-rh', 'R_p9-46v_ROI-rh', 'R_9-46d_ROI-rh'],
    'EMO_L_preSMA':                    ['L_SCEF_ROI-lh', 'L_6ma_ROI-lh'],
    'EMO_R_preSMA':                    ['R_SCEF_ROI-rh', 'R_6ma_ROI-rh'],
    'EMO_L_vmPFC':                     ['L_10v_ROI-lh', 'L_10r_ROI-lh',
                                        'L_p32_ROI-lh', 'L_s32_ROI-lh'],
    'EMO_R_vmPFC':                     ['R_10v_ROI-rh', 'R_10r_ROI-rh',
                                        'R_p32_ROI-rh', 'R_s32_ROI-rh'],
    'EMO_L_PCC':                       ['L_23c_ROI-lh', 'L_d23ab_ROI-lh',
                                        'L_v23ab_ROI-lh', 'L_31pv_ROI-lh'],
    'EMO_R_PCC':                       ['R_23c_ROI-rh', 'R_d23ab_ROI-rh',
                                        'R_v23ab_ROI-rh', 'R_31pv_ROI-rh'],
}




def merge_identical_rois(roi_comp):
    """
    Collapse ROIs whose label sets are EXACTLY identical.

    The KK, TB and EMO definitions are independent readings of different
    papers, so they sometimes land on the same HCP-MMP1 parcel. Keeping both
    would produce two bit-identical timeseries under every extraction method,
    inflating the multiple-comparison count and looking like a replicated
    finding when it is one region counted twice.

    A merged ROI carries the suffixes of all its sources, in SUFFIX_ORDER
    (e.g. Extero_L_IPL_KK + Left_IPS_TB -> Extero_L_IPL_TB_KK), so the
    convergence stays visible in the name. The base name comes from the
    first source in that order.

    Only exact matches are merged. Nested ROIs (one a strict subset of
    another) are left alone: they are not the same region.
    """
    SUFFIX_ORDER = ('EMO', 'TB', 'KK')

    by_labels = {}
    for name, labels in roi_comp.items():
        by_labels.setdefault(frozenset(labels), []).append(name)

    merged, notes = {}, []
    for labels, names in by_labels.items():
        if len(names) == 1:
            merged[names[0]] = roi_comp[names[0]]
            continue
        suffixes, base = [], None
        for suf in SUFFIX_ORDER:
            for n in names:
                if n.endswith('_' + suf):
                    if suf not in suffixes:
                        suffixes.append(suf)
                    if base is None:
                        base = n[:-(len(suf) + 1)]
        if base is None:
            base = names[0]
        new = base + ''.join('_' + x for x in suffixes) if suffixes else base
        merged[new] = roi_comp[names[0]]
        notes.append((new, sorted(names), sorted(labels)))
    return merged, notes


def rename_bem_surfaces(subjects_dir):
    """watershed_bem writes {subj}_inner_skull_surface; MNE wants inner_skull.surf."""
    for sd in glob.glob(op.join(subjects_dir, 'sub*', 'bem')):
        for name in ('inner_skull', 'outer_skull', 'outer_skin', 'brain'):
            tgt = op.join(sd, f'{name}.surf')
            if op.exists(tgt):
                continue
            for cand in glob.glob(op.join(sd, f'*{name}*')):
                if op.isfile(cand) and not cand.endswith('.surf'):
                    try:
                        os.symlink(cand, tgt)
                    except OSError:
                        pass
                    break


def unit_vertices(labels):
    lh = [l.vertices for l in labels if l.hemi == 'lh']
    rh = [l.vertices for l in labels if l.hemi == 'rh']
    return (np.unique(np.concatenate(lh)) if lh else np.array([], int),
            np.unique(np.concatenate(rh)) if rh else np.array([], int))


def stc_rows_and_normals(stc, src, lh_v, rh_v):
    lh_all, rh_all = stc.vertices[0], stc.vertices[1]
    lh_mask, rh_mask = np.isin(lh_all, lh_v), np.isin(rh_all, rh_v)
    rows = np.concatenate([np.nonzero(lh_mask)[0],
                           np.nonzero(rh_mask)[0] + len(lh_all)]).astype(int)
    nn = []
    if lh_mask.any():
        nn.append(src[0]['nn'][lh_all[lh_mask]])
    if rh_mask.any():
        nn.append(src[1]['nn'][rh_all[rh_mask]])
    return rows, (np.vstack(nn) if nn else np.zeros((0, 3)))


def flip_from_normals(nn):
    """
    mean_flip signs from cortical geometry alone, plus a coherence score.

    At a fold the surface normals on the two banks point in near-opposite
    directions, so with pick_ori='normal' a synchronously active patch comes
    out with opposite signs either side and a plain average cancels it. The
    dominant orientation is the first right-singular vector of the normal
    matrix; coherence s0/s1 says how well-defined that direction is. Low
    coherence means the parcel is folded enough that the sign assignment is
    arbitrary and mean_flip should not be trusted there.
    """
    if len(nn) == 0:
        return np.array([]), np.nan
    _, s, Vh = np.linalg.svd(nn, full_matrices=False)
    signs = np.sign(nn @ Vh[0])
    signs[signs == 0] = 1.0
    return signs, (float(s[0] / s[1]) if len(s) > 1 and s[1] > 0 else np.inf)


def fit_pca_flip(data_cat, flip):
    """
    Joint pca_flip fit across concatenated blocks.

    Uses the Gram matrix (n_vert x n_vert) rather than a full SVD, because
    data_cat is n_vert x ~700k and the right-singular matrix would be
    gigabytes. Verified to reproduce MNE's pca_flip to ~1e-15: sign anchored
    to the mean_flip vector, power scaled to the average per-vertex power.
    """
    C = data_cat @ data_cat.T
    w, V = np.linalg.eigh(C)
    w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
    u0, s0, total = V[:, 0], float(np.sqrt(w[0])), float(w.sum())
    sign = np.sign(u0 @ flip) or 1.0
    return (u0, s0, float(np.sqrt(total) / np.sqrt(data_cat.shape[0])),
            float(sign), float(w[0] / total) if total > 0 else np.nan)


def get_vas_onsets_rel(raw, code=3, stim='STI101'):
    """
    VAS onset times in seconds, relative to the START of this raw segment.

    mne.find_events returns sample numbers in the continuous recording's
    absolute sample space. The STC is saved with tmin=0.0 (apply_inverse_raw
    resets the time axis to start at 0), so onset times must also be relative
    to sample 0 of the preproc file, i.e. (event_sample - first_samp) / sfreq.
    """
    try:
        ev = mne.find_events(raw, stim_channel=stim, shortest_event=1,
                             verbose=False)
    except Exception:
        return np.array([])
    on = (ev[ev[:, 2] == code, 0] - raw.first_samp) / raw.info['sfreq']
    return on


# ==================================================================
# main
# ==================================================================
def find_blocks(subject, version):
    pat = (f'{subject}_*_preproc_ecg.fif' if version == 'ecg'
           else f'{subject}_*_preproc.fif')
    files = [f for f in sorted(glob.glob(op.join(PREPROC, pat)))
             if not re.search(r'-\d+\.fif$', f)]
    if version == 'eog':
        files = [f for f in files if 'ecg' not in op.basename(f).lower()]
    out = []
    for f in files:
        b = op.basename(f)
        mb = re.search(r'_(\d+b)_(base|sad|fear)', b)
        if not mb:
            log(f'  ! cannot parse block/condition from {b} — skipped')
            continue
        out.append(dict(path=f, block=mb.group(1), cond=mb.group(2),
                        stem=f'{subject}_{mb.group(1)}_{mb.group(2)}_{version}'))
    return out


def read_block(path):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*naming conventions.*')
        try:
            return mne.io.read_raw_fif(path, preload=True, verbose=False)
        except Exception:
            return mne.io.read_raw_fif(path, preload=True,
                                       allow_maxshield=True, verbose=False)


def run(subject, version, overwrite, force):
    os.makedirs(TC_DIR, exist_ok=True)
    os.makedirs(COREG_DIR, exist_ok=True)
    os.makedirs(STC_DIR, exist_ok=True)

    out_path = op.join(TC_DIR, f'{subject}_{version}_label_tc.npz')

    # ---- stage 0: cheap pre-check --------------------------------
    # Full validation needs the unit count, which is only known after the
    # labels are read (stage 6). This first pass just avoids doing any work
    # at all for a job that is obviously already finished.
    if op.isfile(out_path) and not overwrite and 'extract' not in force:
        log(f'Output present: {out_path}')
        log('  validating before skipping...')
    elif op.isfile(out_path):
        log(f'Output present but will be recomputed: {out_path}')

    # ---- stage 1: blocks + empty room ---------------------------
    hdr('1. Inputs')
    blocks = find_blocks(subject, version)
    if not blocks:
        raise RuntimeError(f'No preproc blocks for {subject} {version}')
    for b in blocks:
        log(f'  {b["stem"]:<34} {op.basename(b["path"])}')

    em = [f for f in sorted(glob.glob(op.join(RAW_ROOT, subject, '*', 'em*tsss.fif')))
          if not re.search(r'-\d+\.fif$', f)]
    if not em:
        raise RuntimeError(f'No empty room file found for subject {subject}')
    log(f'  empty room: {op.basename(em[0])}')

    t1_id = t1_root_subject(subject)
    fs_subject = resolve_fs_subject(subject)
    if fs_subject is None:
        log(f'  ! No FreeSurfer recon for {subject} '
            f'(T1 override: {t1_id}) — nothing to do')
        return 0
    if t1_id != str(subject):
        log(f'  note: {subject} has a T1 override on file — '
            f'reusing anatomy from {fs_subject} (T1 code {t1_id})')
    log(f'  FreeSurfer: {fs_subject}')

    # BEM depends only on anatomy, so key it to the T1 code (t1_id, set
    # above) rather than the MEG subject code: subjects sharing a T1
    # override, or a subject with several repeat visits, share (and only
    # the first computes) one BEM solution. trans/fwd/cov stay keyed to
    # the MEG subject code below, since those depend on the actual
    # recording (head position, empty room).

    raw0 = read_block(blocks[0]['path'])
    log(f'  reference info from {blocks[0]["stem"]}: '
        f'sfreq={raw0.info["sfreq"]:.0f} Hz')

    trans_f = op.join(COREG_DIR, f'{subject}-trans.fif')
    bem_f   = op.join(COREG_DIR, f'{t1_id}_bem-sol.fif')
    fwd_f   = op.join(COREG_DIR, f'{subject}_fwd.fif')
    cov_f   = op.join(COREG_DIR, f'{subject}_noise-cov.fif')

    # ---- stage 2: coregistration --------------------------------
    hdr('2. Coregistration')
    if 'trans' not in force and valid_artifact(trans_f, 'trans'):
        log('  reusing existing trans')
        trans = mne.read_trans(trans_f)
    else:
        from mne.io.constants import FIFF
        dig = raw0.info['dig'] or []
        hsp = [p for p in dig if p['kind'] == FIFF.FIFFV_POINT_EXTRA]
        log(f'  {len(hsp)} headshape, '
            f'{len([p for p in dig if p["kind"] == FIFF.FIFFV_POINT_CARDINAL])}'
            f' fiducial points')
        rename_bem_surfaces(FS_DIR)
        try:
            fid, _ = mne.get_fiducials(subject=fs_subject,
                                       subjects_dir=FS_DIR, verbose=False)
        except Exception:
            fid = 'auto'
        cg = Coregistration(raw0.info, fs_subject, FS_DIR, fiducials=fid)
        cg.fit_fiducials(verbose=False)
        if hsp:
            cg.fit_icp(n_iterations=ICP_ITER, nasion_weight=ICP_NASION_W,
                       verbose=False)
            cg.omit_head_shape_points(distance=HSP_OUTLIER_M)
            cg.fit_icp(n_iterations=ICP_ITER, nasion_weight=ICP_NASION_W,
                       verbose=False)
        else:
            log('  WARNING: no headshape points — ICP skipped')
        trans = cg.trans
        atomic(trans_f, lambda p: mne.write_trans(p, trans, overwrite=True))
        if hsp:
            d = cg.compute_dig_mri_distances() * 1000
            log(f'  coreg quality — mean {d.mean():.2f} mm  max {d.max():.2f} mm')
            if d.mean() >= COREG_WARN_MM:
                log_error(subject, version, 'coreg_quality',
                          f'mean {d.mean():.2f} mm >= {COREG_WARN_MM} mm')
            np.savez(op.join(COREG_DIR, f'{subject}_coreg_distances.npz'),
                     distances_mm=d, mean_mm=d.mean(), max_mm=d.max(),
                     std_mm=d.std())
        log(f'  wrote {op.basename(trans_f)}')

    # ---- stages 3-4: BEM + forward ------------------------------
    hdr('3. BEM solution')
    if 'bem' not in force and valid_artifact(bem_f, 'bem'):
        log('  reusing existing BEM solution')
    else:
        model = mne.make_bem_model(subject=fs_subject, ico=BEM_ICO,
                                   subjects_dir=FS_DIR,
                                   conductivity=BEM_COND, verbose=False)
        sol = mne.make_bem_solution(model, verbose=False)
        atomic(bem_f, lambda p: mne.write_bem_solution(p, sol, overwrite=True))
        log(f'  wrote {op.basename(bem_f)}')

    hdr('4. Forward model')
    if 'fwd' not in force and valid_artifact(fwd_f, 'fwd'):
        log('  reusing existing forward solution')
        fwd = mne.read_forward_solution(fwd_f, verbose=False)
    else:
        log(f'  source space ({SPACING})...')
        src_ = mne.setup_source_space(fs_subject, subjects_dir=FS_DIR,
                                      spacing=SPACING, add_dist=True,
                                      verbose=False)
        bem_sol = mne.read_bem_solution(bem_f, verbose=False)
        fwd = mne.make_forward_solution(
            raw0.info, trans=trans, src=src_, bem=bem_sol, meg=True,
            eeg=False, mindist=MINDIST,
            n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)), verbose=False)
        atomic(fwd_f,
               lambda p: mne.write_forward_solution(p, fwd, overwrite=True))
        log(f'  wrote {op.basename(fwd_f)}')
    src = fwd['src']

    # ---- stage 5: noise covariance ------------------------------
    hdr('5. Noise covariance')
    if 'cov' not in force and valid_artifact(cov_f, 'cov'):
        log('  reusing existing noise covariance')
        noise_cov = mne.read_cov(cov_f, verbose=False)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                raw_em = mne.io.read_raw_fif(em[0], preload=True, verbose=False)
            except Exception:
                raw_em = mne.io.read_raw_fif(em[0], preload=True,
                                             allow_maxshield=True, verbose=False)
        raw_em.resample(sfreq=raw0.info['sfreq'], verbose=False)
        nyq = raw0.info['sfreq'] / 2
        notch = [f for f in np.arange(50, 250, 50) if f < nyq]
        raw_em.notch_filter(freqs=notch, filter_length='auto', phase='zero',
                            verbose=False)
        log(f'  notch {notch} Hz')
        noise_cov = mne.compute_raw_covariance(raw_em, tmin=0, tmax=None,
                                               verbose=False)
        atomic(cov_f, lambda p: mne.write_cov(p, noise_cov, overwrite=True))
        del raw_em
        log(f'  wrote {op.basename(cov_f)}')

    # ---- stage 6: labels and units ------------------------------
    hdr('6. Labels')
    roi_comp, merge_notes = merge_identical_rois(ROI_COMPOSITION)
    for new, olds, labels in merge_notes:
        log(f'  merged {" + ".join(olds)} -> {new}  ({len(labels)} label(s))')
    if merge_notes:
        log(f'  {len(ROI_COMPOSITION)} ROIs defined, {len(roi_comp)} after merging')
    labels = (read_labels_from_annot(fs_subject, parc=PARC, hemi='lh',
                                     subjects_dir=FS_DIR, surf_name='white',
                                     verbose=False) +
              read_labels_from_annot(fs_subject, parc=PARC, hemi='rh',
                                     subjects_dir=FS_DIR, surf_name='white',
                                     verbose=False))
    label_dict = {l.name: l for l in labels}

    units, seen, lh_u, rh_u = [], set(), None, None
    for roi, names in roi_comp.items():
        members = [label_dict[n] for n in names if n in label_dict]
        missing = [n for n in names if n not in label_dict]
        if missing:
            log(f'  ! {roi}: absent from annot: {missing}')
        if not members:
            log(f'  ! {roi}: no labels resolved — skipped')
            continue
        units.append(('roi', roi, members))
        for lab in members:
            if lab.name not in seen:
                seen.add(lab.name)
                units.append(('label', lab.name, [lab]))
            if lab.hemi == 'lh':
                lh_u = lab if lh_u is None else lh_u + lab
            else:
                rh_u = lab if rh_u is None else rh_u + lab
    roi_union = (lh_u + rh_u) if (lh_u is not None and rh_u is not None) \
        else (lh_u or rh_u)
    if roi_union is None:
        raise RuntimeError('No ROI labels resolved — check the parcellation')
    log(f'  {len(units)} units '
        f'({sum(u[0] == "roi" for u in units)} roi, '
        f'{sum(u[0] == "label" for u in units)} label)')

    # ---- stage 0 (deferred): validate the final output -----------
    # Now that the expected row count is known, the existing npz can be
    # checked properly rather than merely detected.
    if (not overwrite and 'extract' not in force
            and npz_valid(out_path, len(units), len(METHODS))):
        log(f'\nOutput exists and validates — nothing to do:\n  {out_path}')
        return 0

    # ---- stage 7: inverse per block -----------------------------
    hdr('7. Source estimates')
    per_block = []
    for bi, b in enumerate(blocks):
        stem = op.join(STC_DIR, f'{b["stem"]}_eloreta')
        raw = raw0 if bi == 0 else read_block(b['path'])
        sf = float(raw.info['sfreq'])
        vas = get_vas_onsets_rel(raw)

        reuse = ('stc' not in force) and stc_valid(stem)
        stc = None
        if reuse:
            stc = mne.read_source_estimate(stem)
            covered, n_missing = stc_covers_units(stc, src, units)
            if covered:
                log(f'  {b["stem"]}: reusing existing STC')
            else:
                log(f'  {b["stem"]}: existing STC predates the current ROI '
                    f'set ({n_missing} needed vertices absent) — recomputing')
                del stc
                stc = None
                reuse = False
        if not reuse:
            log(f'  {b["stem"]}: computing inverse...')
            inv = make_inverse_operator(raw.info, fwd, noise_cov, loose='auto',
                                        depth=None, fixed='auto', rank='info',
                                        verbose=False)
            stc = apply_inverse_raw(raw, inv, lambda2=1.0 / SNR ** 2,
                                    method=INV_METHOD, pick_ori=PICK_ORI,
                                    label=roi_union, verbose=False)
            stc.data = stc.data.astype(np.float32)
            stc.save(stem, overwrite=True)
            log(f'    saved STC {stc.data.shape}')
            del inv

        # rows/normals computed once, against the first STC
        if not per_block:
            unit_rows, unit_nn = [], []
            for _, _, members in units:
                lv, rv = unit_vertices(members)
                r, nn = stc_rows_and_normals(stc, src, lv, rv)
                unit_rows.append(r)
                unit_nn.append(nn)
            keep = np.unique(np.concatenate([r for r in unit_rows if len(r)]))
            remap = {int(g): i for i, g in enumerate(keep)}
            unit_rows = [np.array([remap[int(g)] for g in r], int)
                         for r in unit_rows]
            n_samp = int(round(SNIPPET_DUR_S * sf))
            log(f'  {len(keep)} vertices kept, snippet = {n_samp} samples')

        starts = np.round((vas - stc.tmin) * sf).astype(int) - n_samp
        good = [(s, t) for s, t in zip(starts, vas)
                if s >= 0 and s + n_samp <= stc.data.shape[1]]
        if len(good) < len(starts):
            log(f'    ! {len(starts) - len(good)} snippets out of range')
        cut = np.empty((len(keep), len(good), n_samp), np.float32)
        for k, (s, _) in enumerate(good):
            cut[:, k, :] = stc.data[keep, s:s + n_samp]
        per_block.append(dict(block=b['block'], cond=b['cond'],
                              vas=np.array([t for _, t in good]), data=cut))
        log(f'    {len(good)} snippets')
        del stc
        if bi > 0:
            del raw
    del raw0

    # ---- stage 8: joint extraction ------------------------------
    hdr('8. Joint mean_flip / pca_flip extraction')
    n_snip = sum(b['data'].shape[1] for b in per_block)
    if TARGET_SFREQ != sf:
        n_out = int(round(SNIPPET_DUR_S * TARGET_SFREQ))
        log(f'  resampling {sf:.0f} -> {TARGET_SFREQ:.0f} Hz')
    else:
        n_out = n_samp
        log(f'  keeping native {sf:.0f} Hz (no resampling)')

    out = np.full((len(units) * len(METHODS), n_snip, n_out), np.nan, np.float32)
    meta = {k: [] for k in ('unit_name', 'unit_level', 'method', 'n_vertices',
                            'pca_var', 'ori_coherence', 'zref_mean', 'zref_sd')}
    r = 0
    for (level, name, _), rows, nn in zip(units, unit_rows, unit_nn):
        flip, coh = flip_from_normals(nn)
        if len(rows) == 0:
            for m in METHODS:
                for k, v in zip(meta, [name, level, m, 0, np.nan, np.nan,
                                       np.nan, np.nan]):
                    meta[k].append(v)
                r += 1
            continue
        cat = np.concatenate([b['data'][rows].reshape(len(rows), -1)
                              for b in per_block], axis=1).astype(np.float64)
        u0, s0, scale, sign, pca_var = fit_pca_flip(cat, flip)
        del cat
        for m in METHODS:
            c0 = 0
            for b in per_block:
                d = b['data'][rows]
                tc = (np.mean(flip[:, None, None] * d, axis=0) if m == 'mean_flip'
                      else sign * scale * np.einsum('v,vst->st', u0, d) / s0)
                for k in range(tc.shape[0]):
                    y = tc[k].astype(np.float64)
                    if TARGET_SFREQ != sf:
                        y = mne.filter.resample(y, down=sf / TARGET_SFREQ,
                                                npad='auto')
                    out[r, c0 + k, :] = y.astype(np.float32)
                c0 += tc.shape[0]
            v = out[r][np.isfinite(out[r])]
            for k, val in zip(meta, [name, level, m, len(rows), pca_var, coh,
                                     float(v.mean()) if v.size else np.nan,
                                     float(v.std()) if v.size else np.nan]):
                meta[k].append(val)
            r += 1
    log(f'  {out.shape[0]} unit-method rows x {n_snip} snippets x {n_out} samples')

    # ---- stage 9: write -----------------------------------------
    hdr('9. Writing')
    sb, sc, sv = [], [], []
    for b in per_block:
        sb += [b['block']] * b['data'].shape[1]
        sc += [b['cond']] * b['data'].shape[1]
        sv += list(b['vas'])
    payload = dict(data=out, sfreq=TARGET_SFREQ, snippet_dur_s=SNIPPET_DUR_S,
                   subject=subject, version=version,
                   snip_block=np.array(sb), snip_condition=np.array(sc),
                   snip_vas_time=np.array(sv, float),
                   **{k: np.array(v) for k, v in meta.items()})
    atomic(out_path, lambda p: np.savez_compressed(p, **payload))
    log(f'  wrote {out_path}  ({out.nbytes / 1e6:.0f} MB uncompressed)')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subject')
    ap.add_argument('version', choices=['eog', 'ecg'])
    ap.add_argument('--overwrite', action='store_true',
                    help='redo everything, ignoring existing outputs')
    ap.add_argument('--force-stage', default='',
                    help=f'comma-separated stages to redo: {",".join(STAGES)}')
    a = ap.parse_args()
    force = {s.strip() for s in a.force_stage.split(',') if s.strip()}
    bad = force - set(STAGES)
    if bad:
        sys.exit(f'Unknown stage(s): {sorted(bad)}. Valid: {STAGES}')
    if a.overwrite:
        force = set(STAGES)

    log(f"{'=' * 62}\nsubject {a.subject}   version {a.version}"
        f"{'   force: ' + ','.join(sorted(force)) if force else ''}\n{'=' * 62}")
    try:
        return run(a.subject, a.version, a.overwrite, force)
    except Exception:
        log('FAILED:\n' + traceback.format_exc())
        log_error(a.subject, a.version, 'fatal', traceback.format_exc()[:500])
        return 1


if __name__ == '__main__':
    sys.exit(main())
