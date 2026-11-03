# Working preferences

- "Copy-paste code" means a self-contained shell block that runs Python
  directly in the terminal on the cluster (fox-c18). Every such block starts
  with the working folder and the conda environment, then a heredoc:

  ```bash
  cd /home/ZhangJianfeng_04/emo_study/Behav/
  export PATH="/home/ZhangJianfeng_04/.conda/envs/MEG_env/bin:$PATH"
  python3 - <<'EOF'
  ...python code...
  EOF
  ```

  Use full cluster paths (e.g. /home/ZhangJianfeng_04/emo_study/...), no
  placeholders, and nothing that has to be edited before running.

- Every command that runs a script of this repository on the cluster also
  needs the two lines above first (cd + export PATH), e.g.

  ```bash
  cd /home/ZhangJianfeng_04/emo_study/Behav/
  export PATH="/home/ZhangJianfeng_04/.conda/envs/MEG_env/bin:$PATH"
  python3 10_vas_meg_networks.py
  ```

  Without the export, python3 is the system Python and mne is missing.

# Data conventions

- VAS ratings: it does NOT matter whether a participant pressed "confirm".
  An unconfirmed (timed-out) rating is a valid rating (the slider position
  at the timeout). Do not treat unconfirmed ratings as missing, do not
  exclude people for them, and do not use the confirm/timeout rate as a
  quality or drowsiness measure.
- Report mean_flip as the main source-extraction method; pca_flip results go
  to the supplement as a robustness check.

# Deferred analyses (noted, not done yet)

- HMM with real time between probes (13_meg_neural_patterns.py, part 4):
  the current Gaussian HMM treats every probe as one step, although probe
  intervals vary (20-60 s). Option for later: (1) test whether state
  switches are more likely after longer intervals; (2) if so, a
  continuous-time HMM (transition matrix exp(Q * dt) per interval).
