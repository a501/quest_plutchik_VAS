"""
make_emo_rest_triggers.py
Create {subj}_emo_rest_all_triggers.csv from the raw MEG files on the cluster:
    {SRC}/{subj}/**/*emo*tsss*.fif       (subject = the folder name)
File names with or without the subject-number prefix are both matched.

Same extraction as the Windows MEG_extract_trigger.py:
  - trigger code rebuilt from STI001-STI006 + STI008 (STI007 ignored)
  - MIN_DURATION_SAMPLES = 3 rejects 1-2 sample settling transients
  - output columns: order, sample, timestamp_ms, timestamp_s, trigger_code,
    time_from_start_s, iti_from_prev_ms, iti_from_prev_s

TWO KINDS OF SPLIT RECORDING, handled differently
  1. MNE auto-split      emo_rest_tsss_mc.fif + ...-1.fif + ...-2.fif
     MNE follows the links itself, so only the base file is opened and the
     -N.fif parts are skipped as separate inputs.
  2. SEPARATELY-NAMED PIECES, which MNE does NOT link, e.g.
        030_emo_rest.fif          + 030_emo_rest_1_2.fif
        031_emo_rest_baseline.fif + 031_emo_rest_3.fif
     These are continuation pieces of ONE recording. Their marker channels are
     reconstructed separately and CONCATENATED in order into one continuous
     timeline before events are detected, so the CSV covers the whole task.
     Without this, only the first piece is read: a subject whose baseline piece
     is separate then has no video triggers (codes 7/8) at all, which is what
     produced the ORDER CONFLICT and low trig_match flags in the build step.

PIECE ORDER (from the Windows script)
  - bare / unsuffixed piece -> first
  - "baseline"              -> next
  - everything else         -> natural sort: 1 < 1_2 < 2 < 3 < "foo"
  The order used is printed per subject -- CHECK IT. If a subject's pieces are
  named so the rule orders them wrongly, the concatenated timeline is wrong.

  Several versions of ONE piece (e.g. emo_rest_tsss.fif and
  emo_rest_tsss_mc.fif) are not pieces: only one is used, _tsss_mc
  preferred (same rule as cut_dyn_blocks_python.py). Using both would put
  the whole session into the CSV twice.

  Pieces must share one sampling rate; a mismatch is refused, not guessed.
  The subject is also refused if a piece is AMBIGUOUS, i.e. the file name
  gives no way to order it. Nothing is written in either case.

Skips subjects whose CSV already exists (unless --overwrite).

Usage:
    python make_emo_rest_triggers.py                      # all new subjects
    python make_emo_rest_triggers.py --subjects 030 031
    python make_emo_rest_triggers.py --subjects 030 --overwrite
    python make_emo_rest_triggers.py --dry-run            # list pieces only
"""
import argparse
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# ---- paths ------------------------------------------------------------------
DEFAULT_SRC = "/home/ZhangJianfeng_04/emo_study/MEG/1_raw"    # {subj}/**/*.fif
DEFAULT_OUT = "/home/ZhangJianfeng_04/emo_study/Behav/MEG_trigger_files"

# ---- extraction settings ----------------------------------------------------
STI_CHANNELS = ["STI001", "STI002", "STI003", "STI004", "STI005", "STI006", "STI008"]
THRESHOLD = 1.0
MIN_DURATION_SAMPLES = 3

# The old Windows script removed consecutive identical codes, which also drops
# genuine repeats (two left presses in a row, 11 then 11). False = keep every
# real event. Set True only to reproduce the old CSVs byte for byte.
DEDUP_CONSECUTIVE = False

FILE_GLOB = "*emo*.fif"         # matches emo_rest_*.fif and 031_emo_rest_*.fif alike
REQUIRE_TSSS = True             # keep only names containing 'tsss' (set False if
                                # some pieces are not maxfiltered)

# -N.fif are MNE auto-split continuations: loaded automatically, never alone.
SKIP = [re.compile(r"empty_room"), re.compile(r"-\d+\.fif$")]

# tokens stripped from a file name before what remains is read as the piece id
NOISE_TOKENS = {"tsss", "mc", "sss", "raw", "meg", "fif"}
RECORDING_PREFIXES = ["emo_rest", "emo"]


# ==========================================================================
# piece parsing / ordering
# ==========================================================================
def piece_token(fname, subj):
    """
    Piece id from a file name, or None for the bare recording.
        emo_rest_tsss_mc.fif            -> None
        030_emo_rest_1_2_tsss_mc.fif    -> '1_2'
        031_emo_rest_baseline_tsss.fif  -> 'baseline'
    """
    base = fname[:-4] if fname.lower().endswith(".fif") else fname
    base = re.sub(rf"^0*{int(subj)}_|^{re.escape(subj)}_", "", base)
    for pre in RECORDING_PREFIXES:
        if base == pre:
            return None
        if base.startswith(pre + "_"):
            base = base[len(pre) + 1:]
            break
    parts = [p for p in base.split("_") if p and p.lower() not in NOISE_TOKENS]
    return "_".join(parts) if parts else None


def piece_sort_key(piece):
    """bare -> first, 'baseline' -> next, then natural order."""
    if piece is None:
        return (0, ())
    p = str(piece).strip().lower()
    if p == "baseline":
        return (1, ())
    chunks = [c for c in re.split(r"(\d+)", p) if c]
    return (2, tuple((0, int(c)) if c.isdigit() else (1, c) for c in chunks))


def ambiguous(pieces):
    """Piece ids that carry no number and are not 'baseline' cannot be ordered."""
    bad = []
    for p in pieces:
        if p is None or str(p).strip().lower() == "baseline":
            continue
        if not re.search(r"\d", str(p)):
            bad.append(p)
    return bad


# ==========================================================================
# extraction
# ==========================================================================
def reconstruct_markers(path):
    """Per-sample composite marker value from the STI bit channels."""
    raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
    sfreq = raw.info["sfreq"]
    chans = [c for c in STI_CHANNELS if c in raw.ch_names]
    if not chans:
        raise RuntimeError(f"none of {STI_CHANNELS} present; "
                           f"first channels: {raw.ch_names[:20]}")
    sti, _ = raw[chans]
    binary = (sti > THRESHOLD).astype(int)
    marker = np.zeros(sti.shape[1], dtype=int)
    for i, ch in enumerate(chans):
        ch_num = int(ch[3:6])
        bit = 7 if ch_num > 6 else ch_num - 1
        marker += binary[i] * (2 ** bit)
    return marker, sfreq, len(raw.filenames)


def build_trigger_df(marker, sfreq):
    events = []
    prev_stable = 0
    i, n = 0, len(marker)
    while i < n:
        cur = marker[i]
        if cur != prev_stable and cur > 0:
            j = i + 1
            while j < n and marker[j] == cur:
                j += 1
            if j - i >= MIN_DURATION_SAMPLES:
                events.append((i, cur))
                prev_stable = cur
            i = j
            continue
        elif cur == 0:
            prev_stable = 0
        i += 1

    if not events:
        raise RuntimeError("no trigger events found")

    if DEDUP_CONSECUTIVE:
        kept = [events[0]]
        for ev in events[1:]:
            if ev[1] != kept[-1][1]:
                kept.append(ev)
        events = kept

    samples = np.array([e[0] for e in events], dtype=int)
    codes = np.array([e[1] for e in events], dtype=int)
    t_s = np.round(samples / sfreq, 6)
    t_ms = np.round(t_s * 1000, 3)
    df = pd.DataFrame({"order": np.arange(1, len(events) + 1), "sample": samples,
                       "timestamp_ms": t_ms, "timestamp_s": t_s, "trigger_code": codes})
    df["time_from_start_s"] = ((df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000).round(3)
    df["iti_from_prev_ms"] = df["timestamp_ms"].diff().round(1)
    df["iti_from_prev_s"] = (df["iti_from_prev_ms"] / 1000).round(4)
    return df


def extract_subject(ordered):
    """Reconstruct every piece, concatenate in order, then detect events once."""
    markers, sfreqs, info = [], [], []
    for piece, path in ordered:
        m, sf, nparts = reconstruct_markers(path)
        markers.append(m)
        sfreqs.append(sf)
        info.append(f"        {path.name}: {nparts} auto-split part(s), "
                    f"{len(m) / sf / 60:.1f} min")
    for line in info:
        print(line)
    if len(set(sfreqs)) > 1:
        raise RuntimeError(f"sfreq differs across pieces {sfreqs}; "
                           "the concatenated timeline would be wrong")
    combined = np.concatenate(markers)
    if len(markers) > 1:
        print(f"        concatenated -> {len(combined) / sfreqs[0] / 60:.1f} min total")
    return build_trigger_df(combined, sfreqs[0]), sfreqs[0], len(markers)


# ==========================================================================
# discovery
# ==========================================================================
def find_files(src, subjects=None):
    found = {}
    for d in sorted(Path(src).iterdir()):
        if not (d.is_dir() and re.fullmatch(r"\d+", d.name)):
            continue
        if subjects and d.name not in subjects:
            continue
        for p in sorted(d.rglob(FILE_GLOB)):
            if any(s.search(p.name) for s in SKIP):
                continue
            if REQUIRE_TSSS and "tsss" not in p.name.lower():
                continue
            found.setdefault(d.name, []).append((piece_token(p.name, d.name), p))
    return {s: sorted(one_version_per_piece(s, v), key=lambda pf: piece_sort_key(pf[0]))
            for s, v in found.items()}


def version_rank(path):
    """Lower = preferred. _tsss_mc over plain _tsss, as in
    cut_dyn_blocks_python.py. Maxfiltering does not change the trigger
    channels, so either version gives the same triggers - but only ONE may
    be used, otherwise the session is concatenated twice."""
    n = path.name.lower()
    if re.search(r"_tsss_mc\.fif$", n):
        return 0
    if re.search(r"_tsss\.fif$", n):
        return 1
    return 2


def one_version_per_piece(subj, pieces):
    """Several files with the same piece id (e.g. emo_rest_tsss.fif and
    emo_rest_tsss_mc.fif, both 'bare') are versions of ONE recording, not
    pieces: keep the preferred one."""
    by_piece = {}
    for piece, path in pieces:
        by_piece.setdefault(piece, []).append(path)
    kept = []
    for piece, paths in by_piece.items():
        paths = sorted(paths, key=lambda q: (version_rank(q), str(q)))
        for q in paths[1:]:
            print(f"[{subj}] ignoring {q.name} - same recording as {paths[0].name}")
        kept.append((piece, paths[0]))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--subjects", nargs="*", help="e.g. 030 031 (default: all new)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the pieces and their order, write nothing")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    subjects = [s.zfill(3) for s in a.subjects] if a.subjects else None

    files = find_files(a.src, subjects)
    print(f"Found {len(files)} subject folder(s) under {a.src}\n")

    rows = []
    for subj, ordered in files.items():
        out_csv = out / f"{subj}_emo_rest_all_triggers.csv"
        if out_csv.exists() and not a.overwrite and not a.dry_run:
            print(f"[{subj}] exists, skipping")
            continue

        names = [p if p is not None else "(bare)" for p, _ in ordered]
        print(f"[{subj}] {len(ordered)} piece(s) in order: {names}")

        bad = ambiguous([p for p, _ in ordered])
        if bad and len(ordered) > 1:
            print(f"        REFUSED: cannot order piece(s) {bad} from the file name. "
                  f"Rename them or pass --subjects {subj} after fixing.")
            rows.append(dict(subject=subj, status="ambiguous piece order",
                             detail=";".join(map(str, bad))))
            continue
        if a.dry_run:
            rows.append(dict(subject=subj, status="dry-run", detail=";".join(names)))
            continue

        try:
            df, sfreq, npieces = extract_subject(ordered)
        except Exception as e:
            print(f"        FAILED: {type(e).__name__}: {e}")
            rows.append(dict(subject=subj, status=f"FAILED {type(e).__name__}", detail=str(e)))
            continue

        df.to_csv(out_csv, index=False)
        counts = df["trigger_code"].value_counts().sort_index().to_dict()
        n_sad, n_fear = counts.get(7, 0), counts.get(8, 0)
        n_rest = counts.get(1, 0)
        print(f"        {len(df)} triggers, sfreq {sfreq} Hz -> {out_csv.name}")
        print(f"        code counts: {counts}")
        flags = []
        if n_sad != 1 or n_fear != 1:
            flags.append(f"video triggers sad={n_sad} fear={n_fear} (expect 1 each)")
        if n_rest < 3:
            flags.append(f"only {n_rest} rest_onset (expect 3, or 4+ with a false start)")
        for f in flags:
            print(f"        !! {f}")
        rows.append(dict(subject=subj, status="ok", n_pieces=npieces, n_triggers=len(df),
                         sfreq=sfreq, n_rest_onset=n_rest, n_video_sad=n_sad,
                         n_video_fear=n_fear, detail=str(counts),
                         flag="; ".join(flags)))

    S = pd.DataFrame(rows)
    if len(S) and not a.dry_run:
        S.to_csv(out / "ALL_meg_trigger_summary.csv", index=False)
    print("\nDone.")
    if len(S):
        cols = [c for c in ("subject", "status", "n_pieces", "n_rest_onset",
                            "n_video_sad", "n_video_fear", "flag") if c in S]
        print(S[cols].to_string(index=False))


if __name__ == "__main__":
    main()
