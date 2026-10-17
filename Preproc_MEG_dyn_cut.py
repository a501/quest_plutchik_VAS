# -*- coding: utf-8 -*-
"""
MEG Preprocessing Pipeline — corrected version
================================================
Steps (in correct order):
1.  Load raw data
2.  Downsample to 500 Hz
3.  High-pass filter at 0.5 Hz  (applied to saved data)
4.  Notch filter (50, 100, 150, 200 Hz)
5.  Detect and interpolate bad channels
6.  Run ICA on a 1 Hz high-pass filtered copy
7.  Detect EOG and ECG components BEFORE removing anything
8.  Save two output versions:
      {orig_name}_preproc.fif         — EOG components removed only
      {orig_name}_preproc_ecg.fif     — EOG + ECG components removed
9.  Save ICA solution
10. Save QC plots and processing summary

Input:  /home/ZhangJianfeng_04/emo_study/MEG/2_cut/{subject}_emo_rest_tsss_mc_{1b,2b,3b}.fif
Output: /home/ZhangJianfeng_04/emo_study/MEG/3_preproc/

Usage:
    python3 Preproc_MEG_dyn_cut.py

Processes every .fif in 2_cut/ in one run (no SLURM needed). A file is
SKIPPED if its three outputs (_preproc.fif, _preproc_ecg.fif, _ica.fif)
already exist in 3_preproc/. Set FORCE = True below to redo everything.
"""

from mne.preprocessing import ICA
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — required on Linux servers
import matplotlib.pyplot as plt
import os
import sys
import mne
import pandas as pd
import glob
import re
import traceback
from datetime import datetime

import warnings

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'downsample_freq'       : 500,
    'highpass_freq'         : 0.5,    # applied to the saved data
    'ica_highpass_freq'     : 1.0,    # applied only to the ICA fitting copy
    'ica_method'            : 'fastica',
    'ica_variance'          : 0.95,
    'flat_threshold'        : 1e-25,
    'flat_threshold_pct'    : 0.001,  # 0.1% of median variance
    'bad_channel_threshold' : 0.15,   # max fraction of bad channels before skipping interpolation
    'notch_freqs'           : np.arange(50, 250, 50),
    'eog_channels'          : ['EOG002', 'EOG003'],
    'eog_threshold'         : 2.0,
    'max_eog_components'    : 2,
    'max_ecg_components'    : 2,
}

# ============================================================
# PATHS
# ============================================================

raw_root   = '/home/ZhangJianfeng_04/emo_study/MEG/2_cut'
output_dir = '/home/ZhangJianfeng_04/emo_study/MEG/3_preproc'
os.makedirs(output_dir, exist_ok=True)

# Recompute files whose outputs already exist?
FORCE = False

# ============================================================
# COLLECT FILES
# ============================================================

# every .fif in 2_cut/. Split continuation parts (-1.fif, -2.fif) are
# read automatically together with their first part, so they are not
# listed separately.
fif_paths = sorted(p for p in glob.glob(os.path.join(raw_root, '*.fif'))
                   if not re.search(r'-\d+\.fif$', p))
print(f"Found {len(fif_paths)} .fif file(s) in {raw_root}")

MEG = []
for fif_path in fif_paths:
    fname = os.path.basename(fif_path)
    stem  = os.path.splitext(fname)[0]
    # Match subject ID (leading digits) and block+condition (e.g. 1b_base, 2b_sad, 3b_fear)
    m     = re.search(r'^(\d+)_.*_(\db(?:_\w+)?)$', stem)
    MEG.append({
        'subject'  : m.group(1) if m else 'unknown',
        'block'    : m.group(2) if m else 'unknown',
        'filepath' : fif_path,
        'filename' : fname,
    })

# ============================================================
# SUMMARY LOG
# ============================================================

summary_rows = []

# ============================================================
# MAIN LOOP
# ============================================================

for entry in MEG:

    subject  = entry['subject']
    block    = entry['block']
    rawfile  = entry['filepath']
    filename = entry['filename']

    # Output filenames: same stem as input + _preproc / _preproc_ecg
    stem             = os.path.splitext(filename)[0]
    out_eog          = os.path.join(output_dir, f'{stem}_preproc.fif')
    out_ecg          = os.path.join(output_dir, f'{stem}_preproc_ecg.fif')
    out_ica          = os.path.join(output_dir, f'{stem}_ica.fif')

    # ---- skip files that are already done ----
    done = all(os.path.isfile(f) for f in (out_eog, out_ecg, out_ica))
    if done and not FORCE:
        print(f"SKIP (already preprocessed): {filename}")
        continue

    log = {
        'Subject'        : subject,
        'Block'          : block,
        'File'           : filename,
        'Bad_Channels'   : 'N/A',
        'ICA_Components' : 'N/A',
        'EOG_Components' : 'N/A',
        'ECG_Components' : 'N/A',
        'Status'         : 'Unknown',
    }

    print(f"\n{'='*60}")
    print(f"Subject : {subject}")
    print(f"Block   : {block}")
    print(f"File    : {filename}")
    print(f"{'='*60}")

    try:

        # ----------------------------------------------------
        # 1. LOAD
        # ----------------------------------------------------
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*naming conventions.*', category=RuntimeWarning)
            raw = mne.io.read_raw_fif(rawfile, preload=True, verbose=False)
        print(f"Loaded  — sfreq: {raw.info['sfreq']} Hz | "
              f"duration: {raw.times[-1]:.1f} s | "
              f"channels: {len(raw.ch_names)}")
        print(f"Original filters — HP: {raw.info['highpass']} Hz  "
              f"LP: {raw.info['lowpass']} Hz")

        # ----------------------------------------------------
        # 2. DOWNSAMPLE
        # ----------------------------------------------------
        if raw.info['sfreq'] > CONFIG['downsample_freq']:
            raw.resample(sfreq=CONFIG['downsample_freq'], verbose=False)
            print(f"Downsampled to {raw.info['sfreq']} Hz")

        # ----------------------------------------------------
        # 3. HIGH-PASS FILTER (applied to data that will be saved)
        # ----------------------------------------------------
        print(f"Applying {CONFIG['highpass_freq']} Hz high-pass filter...")
        raw.filter(
            l_freq=CONFIG['highpass_freq'],
            h_freq=None,
            filter_length='auto',
            phase='zero',
            verbose=False
        )

        # ----------------------------------------------------
        # 4. NOTCH FILTER
        # ----------------------------------------------------
        nyquist = raw.info['sfreq'] / 2
        notch_freqs = [f for f in CONFIG['notch_freqs'] if f < nyquist]
        if notch_freqs:
            raw.notch_filter(
                freqs=notch_freqs,
                filter_length='auto',
                phase='zero',
                verbose=False
            )
            print(f"Notch filter applied at: {notch_freqs} Hz")

        # ----------------------------------------------------
        # 5. BAD CHANNEL DETECTION AND INTERPOLATION
        #    (done AFTER filtering so variance reflects clean signal)
        # ----------------------------------------------------
        print("\nDetecting bad channels...")
        all_bad_chans = []

        meg_picks    = mne.pick_types(raw.info, meg=True, eog=False, ecg=False, stim=False)
        meg_ch_names = [raw.ch_names[i] for i in meg_picks]

        if meg_picks.size > 0:
            data      = raw.get_data(picks=meg_picks)
            variances = np.var(data, axis=1)

            median_var    = np.median(variances)
            flat_threshold = min(CONFIG['flat_threshold'],
                                 median_var * CONFIG['flat_threshold_pct'])
            print(f"Median variance: {median_var:.2e}  |  Flat threshold: {flat_threshold:.2e}")

            # Flat channels
            flat_channels = [meg_ch_names[i]
                             for i, v in enumerate(variances)
                             if v < flat_threshold]
            if flat_channels:
                print(f"Flat channels ({len(flat_channels)}): {flat_channels}")
                all_bad_chans.extend(flat_channels)

            # Noisy channels (z-score on non-flat channels)
            non_flat_idx = [i for i, n in enumerate(meg_ch_names)
                            if n not in flat_channels]
            if non_flat_idx:
                non_flat_var = variances[non_flat_idx]
                mean_v, std_v = np.mean(non_flat_var), np.std(non_flat_var)
                if std_v > 0:
                    z_scores = (variances - mean_v) / std_v
                    noisy = [meg_ch_names[i]
                             for i in np.where(z_scores > 4)[0]
                             if meg_ch_names[i] not in flat_channels]
                    if noisy:
                        print(f"Noisy channels ({len(noisy)}): {noisy}")
                        all_bad_chans.extend(noisy)

        # Existing bads from file
        existing_bads = [ch for ch in raw.info['bads'] if ch not in all_bad_chans]
        if existing_bads:
            print(f"Pre-existing bad channels: {existing_bads}")
            all_bad_chans.extend(existing_bads)

        all_bad_chans = list(set(all_bad_chans))

        if all_bad_chans:
            raw.info['bads'] = all_bad_chans
            total_meg = len([c for c in raw.ch_names if c.startswith('MEG')])
            if len(all_bad_chans) < total_meg * CONFIG['bad_channel_threshold']:
                print(f"Interpolating {len(all_bad_chans)} bad channels...")
                raw.interpolate_bads(reset_bads=True, mode='accurate')
            else:
                print(f"WARNING: Too many bad channels "
                      f"({len(all_bad_chans)}/{total_meg}), skipping interpolation.")
        else:
            print("No bad channels detected.")

        log['Bad_Channels'] = ', '.join(all_bad_chans) if all_bad_chans else 'None'

        # ----------------------------------------------------
        # 6. ICA — fit on a 1 Hz high-pass filtered COPY
        #    The original 'raw' (0.5 Hz HP) is NOT modified here.
        # ----------------------------------------------------
        print(f"\nRunning ICA (fitting on {CONFIG['ica_highpass_freq']} Hz HP copy)...")

        raw_ica = raw.copy().filter(
            l_freq=CONFIG['ica_highpass_freq'],
            h_freq=None,
            filter_length='auto',
            phase='zero',
            verbose=False
        )

        ica = ICA(
            n_components  = CONFIG['ica_variance'],
            method        = CONFIG['ica_method'],
            random_state  = 42,
            max_iter      = 800,
            verbose       = False
        )
        ica.fit(raw_ica, verbose=False)
        n_components_fitted = ica.n_components_
        print(f"ICA fitted with {n_components_fitted} components")
        log['ICA_Components'] = n_components_fitted

        
        # Plot all ICA component topomaps  (plot_components returns a list of figs)
        figs = ica.plot_components(show=False)
        if not isinstance(figs, list):
            figs = [figs]
        for page_idx, fig in enumerate(figs):
            fig.suptitle(f'ICA Components — {stem} (page {page_idx+1})', y=1.02)
            fig.savefig(os.path.join(output_dir, f'{stem}_ica_components_p{page_idx+1}.png'),
                        dpi=100, bbox_inches='tight')
        plt.close('all')

        # ----------------------------------------------------
        # 7. DETECT EOG AND ECG ARTIFACTS (before any removal)
        #    We run detection on raw_ica (the HP-filtered copy)
        #    so detection is clean, but we apply to 'raw' (0.5 HP).
        # ----------------------------------------------------

        # --- EOG ---
        print("\nDetecting EOG artifacts...")
        all_eog_components = {}

        for eog_ch in CONFIG['eog_channels']:
            if eog_ch in raw_ica.ch_names:
                try:
                    eog_inds, scores = ica.find_bads_eog(
                        raw_ica,
                        ch_name=eog_ch,
                        threshold=CONFIG['eog_threshold'],
                        verbose=False
                    )
                    if eog_inds:
                        print(f"  {eog_ch}: components {eog_inds}")
                        for idx in eog_inds:
                            if idx not in all_eog_components:
                                all_eog_components[idx] = []
                            all_eog_components[idx].append(abs(scores[idx]))
                except Exception as e:
                    print(f"  Could not detect from {eog_ch}: {e}")

        # Rank by average score, keep top N
        eog_indices = []
        if all_eog_components:
            ranked = sorted(
                {idx: np.mean(sc) for idx, sc in all_eog_components.items()}.items(),
                key=lambda x: x[1], reverse=True
            )
            eog_indices = [idx for idx, _ in ranked[:CONFIG['max_eog_components']]]
            print(f"  Selected EOG components to remove: {eog_indices}")

            # Plot EOG components
            to_plot = sorted(list(all_eog_components.keys()))[:4]
            fig = ica.plot_components(picks=to_plot,
                                      title='EOG Components', show=False)
            fig = fig[0] if isinstance(fig, list) else fig
            fig.savefig(os.path.join(output_dir, f'{stem}_eog_spatial.png'),
                        dpi=100, bbox_inches='tight')
            plt.close('all')

            figs_prop = ica.plot_properties(raw_ica, picks=to_plot, show=False)
            figs_prop = figs_prop if isinstance(figs_prop, list) else [figs_prop]
            for prop_idx, fig in enumerate(figs_prop):
                fig.savefig(os.path.join(output_dir, f'{stem}_eog_components_ic{to_plot[prop_idx]}.png'),
                            dpi=100, bbox_inches='tight')
            plt.close('all')
        else:
            print("  No EOG artifacts detected.")

        log['EOG_Components'] = str(eog_indices) if eog_indices else 'None'

        # --- ECG ---
        print("\nDetecting ECG artifacts...")
        ecg_indices = []
        ecg_channels_in_data = [ch for ch in raw_ica.ch_names if 'ECG' in ch.upper()]

        try:
            if ecg_channels_in_data:
                ecg_inds, scores = ica.find_bads_ecg(
                    raw_ica,
                    ch_name=ecg_channels_in_data[0],
                    threshold='auto',
                    verbose=False
                )
            else:
                # Fallback: detect from MEG signal directly
                print("  No ECG channel found — detecting from MEG signal.")
                ecg_inds, scores = ica.find_bads_ecg(
                    raw_ica,
                    method='correlation',
                    threshold='auto',
                    verbose=False
                )

            if ecg_inds:
                # Rank by score, keep top N
                ranked_ecg = sorted(
                    [(idx, abs(scores[idx])) for idx in ecg_inds if idx < len(scores)],
                    key=lambda x: x[1], reverse=True
                )
                ecg_indices = [idx for idx, _ in ranked_ecg[:CONFIG['max_ecg_components']]]
                print(f"  Selected ECG components to remove: {ecg_indices}")

                # Plot ECG components
                to_plot_ecg = [idx for idx, _ in ranked_ecg[:4]]
                fig = ica.plot_components(picks=to_plot_ecg,
                                          title='ECG Components', show=False)
                fig = fig[0] if isinstance(fig, list) else fig
                fig.savefig(os.path.join(output_dir, f'{stem}_ecg_spatial.png'),
                            dpi=100, bbox_inches='tight')
                plt.close('all')

                figs_prop = ica.plot_properties(raw_ica, picks=to_plot_ecg, show=False)
                figs_prop = figs_prop if isinstance(figs_prop, list) else [figs_prop]
                for prop_idx, fig in enumerate(figs_prop):
                    fig.savefig(os.path.join(output_dir, f'{stem}_ecg_components_ic{to_plot_ecg[prop_idx]}.png'),
                                dpi=100, bbox_inches='tight')
                plt.close('all')
            else:
                print("  No ECG artifacts detected.")

        except Exception as e:
            print(f"  ECG detection failed: {e}")

        log['ECG_Components'] = str(ecg_indices) if ecg_indices else 'None'

        # ----------------------------------------------------
        # 8. APPLY ICA AND SAVE TWO VERSIONS
        #
        #   Version A (_preproc.fif):      EOG removed only
        #   Version B (_preproc_ecg.fif):  EOG + ECG removed
        #
        #   Both are applied to 'raw' (0.5 Hz HP, the version
        #   that will actually be used for analysis).
        # ----------------------------------------------------

        # --- Version A: EOG only ---
        print(f"\nSaving EOG-cleaned version...")
        raw_eog = raw.copy()
        ica_eog = ica.copy()
        ica_eog.exclude = list(set(eog_indices))
        if ica_eog.exclude:
            ica_eog.apply(raw_eog)
            print(f"  Applied EOG removal: components {sorted(ica_eog.exclude)}")
        else:
            print("  No EOG components to remove — saving as-is.")
        raw_eog.save(out_eog, overwrite=True, split_naming='neuromag', verbose=False)
        print(f"  Saved: {out_eog}")

        # PSD plot for EOG-cleaned
        psd = raw_eog.compute_psd(fmax=60)
        psd.plot(show=False)
        plt.title(f'Final PSD (EOG-cleaned) — {stem}')
        plt.savefig(os.path.join(output_dir, f'{stem}_final_psd_eog.png'),
                    dpi=100, bbox_inches='tight')
        plt.close('all')

        # --- Version B: EOG + ECG ---
        print(f"\nSaving EOG+ECG-cleaned version...")
        raw_ecg = raw.copy()
        ica_ecg = ica.copy()
        ica_ecg.exclude = list(set(eog_indices + ecg_indices))
        if ica_ecg.exclude:
            ica_ecg.apply(raw_ecg)
            print(f"  Applied EOG+ECG removal: components {sorted(ica_ecg.exclude)}")
        else:
            print("  No components to remove — saving as-is.")
        raw_ecg.save(out_ecg, overwrite=True, split_naming='neuromag', verbose=False)
        print(f"  Saved: {out_ecg}")

        # PSD plot for ECG-cleaned
        psd_ecg = raw_ecg.compute_psd(fmax=60)
        psd_ecg.plot(show=False)
        plt.title(f'Final PSD (EOG+ECG-cleaned) — {stem}')
        plt.savefig(os.path.join(output_dir, f'{stem}_final_psd_ecg.png'),
                    dpi=100, bbox_inches='tight')
        plt.close('all')

        # --- Save ICA solution ---
        ica.save(out_ica, overwrite=True, verbose=False)
        print(f"ICA solution saved: {out_ica}")

        # ---- Removal summary plot + print ----
        all_excluded = sorted(set(eog_indices + ecg_indices))
        print(f"\n{'─'*40}")
        print(f"REMOVED COMPONENTS SUMMARY — {stem}")
        print(f"  EOG : {sorted(eog_indices) if eog_indices else 'none'}")
        print(f"  ECG : {sorted(ecg_indices) if ecg_indices else 'none'}")
        print(f"  ALL : {all_excluded}")
        print(f"{'─'*40}\n")
        if all_excluded:
            figs_exc = ica.plot_components(picks=all_excluded, show=False)
            if not isinstance(figs_exc, list):
                figs_exc = [figs_exc]
            for fig in figs_exc:
                fig.suptitle(
                    f'Removed: EOG={sorted(eog_indices)} ECG={sorted(ecg_indices)} — {stem}',
                    y=1.02, fontsize=9
                )
            figs_exc[0].savefig(os.path.join(output_dir, f'{stem}_removed_components.png'),
                                dpi=100, bbox_inches='tight')
            plt.close('all')

        log['Status'] = 'Success'
        print(f"\n✓ Done: {stem}")
        
        
        
    except Exception as e:
        log['Status'] = f'Error: {str(e)[:150]}'
        print(f"\n✗ Error processing {filename}: {e}")
        traceback.print_exc()

    finally:
        summary_rows.append(log)
        plt.close('all')

# ============================================================
# SAVE SUMMARY
# ============================================================

# Merge this run's rows into the existing summary, so files skipped in
# this run keep the row from the run that processed them.
summary_file = os.path.join(output_dir, 'preprocessing_summary.csv')
new_rows = pd.DataFrame(summary_rows)
if os.path.isfile(summary_file):
    old_rows = pd.read_csv(summary_file)
    if len(new_rows):
        old_rows = old_rows[~old_rows['File'].isin(new_rows['File'])]
    new_rows = pd.concat([old_rows, new_rows], ignore_index=True)
if len(new_rows):
    new_rows.sort_values('File').to_csv(summary_file, index=False)

print(f"\n{'='*60}")
print(f"Processing complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
n_skip = len(MEG) - len(summary_rows)
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    n_ok  = (summary_df['Status'] == 'Success').sum()
    n_err = len(summary_df) - n_ok
    print(f"This run — processed: {len(summary_df)}  |  Success: {n_ok}  |  "
          f"Failed: {n_err}  |  Skipped (already done): {n_skip}")
    print(summary_df[['Subject', 'Block', 'File', 'Status']].to_string(index=False))
else:
    print(f"Nothing to do — all {n_skip} file(s) already processed.")
print(f"Summary (all runs): {summary_file}")
print(f"{'='*60}")
