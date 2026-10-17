# Conventions

- Scripts that run as plain Python only (not via SLURM array jobs) carry the
  suffix `_python.py`, e.g. `cut_rest_python.py`, `cut_dyn_blocks_python.py`,
  `Preproc_MEG_dyn_cut_python.py`. Keep this suffix when creating or editing
  such scripts.
- MEG cutting scripts use tSSS files only (`*tsss*.fif`, `_tsss_mc` preferred
  over plain `_tsss`), search the subject folder in `MEG/1_raw/{subject}/` at
  any depth, and name outputs starting with the subject ID.
- Pipeline scripts skip outputs that already exist.
