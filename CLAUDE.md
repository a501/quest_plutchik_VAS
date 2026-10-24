# Conventions

- The `_python.py` suffix applies ONLY to the two dyn-task MEG scripts that
  run as plain Python (no SLURM): `cut_dyn_blocks_python.py` and
  `Preproc_MEG_dyn_cut_python.py`. Do not add it to other scripts
  (e.g. `cut_rest.py` keeps its name).
- MEG cutting scripts use tSSS files only (`*tsss*.fif`, `_tsss_mc` preferred
  over plain `_tsss`), search the subject folder in `MEG/1_raw/{subject}/` at
  any depth, and name outputs starting with the subject ID.
- Pipeline scripts skip outputs that already exist.
- VAS ratings: timeouts are ALWAYS used, in every analysis. A rating with
  status 'timeout' counts exactly like a confirmed one ('valid' = rating
  present); a 'timeout' column only marks them. Never filter on
  status == 'confirmed'.
