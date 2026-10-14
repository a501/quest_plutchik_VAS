# -*- coding: utf-8 -*-
"""
5_heatmaps_to_pptx.py
=====================
Collects the heatmap PNG figures produced by scripts 3 and 4 and
assembles them into one PowerPoint file, one figure per slide.
Does NOT recompute anything - purely a packaging step.

Layout:
  - One slide per PNG, image fills the full 13.33 x 7.5 in canvas
  - Section divider slides (dark background, large text) separate each
    figure group so the deck is navigable
  - The filename (without extension) is printed as a speaker note on
    each slide so you can trace back to the source file

Sources collected (in order):
  1. Script 3 output: corr_with_dynamics/figures/  (all *.png)
  2. Script 4 output: spearman/figures_by_measure/
                      spearman/figures/
                      pearson/figures_by_measure/
                      pearson/figures/

Usage:
    python3 5_heatmaps_to_pptx.py
    python3 5_heatmaps_to_pptx.py spearman   # only spearman figures from s4
    python3 5_heatmaps_to_pptx.py pearson    # only pearson figures from s4
    python3 5_heatmaps_to_pptx.py both       # both (default)
"""

import sys
from pathlib import Path

# ===================================================================
# CONFIG
# ===================================================================

BASE = Path("/home/ZhangJianfeng_04/emo_study/Behav")

# Script 3 output
S3_DIR = BASE / "quest/analysis/corr_with_dynamics/figures"

# Script 4 output root (contains spearman/ and pearson/ subfolders)
S4_DIR = BASE / "analysis/corr_all_modalities"

OUT_PPTX = BASE / "analysis/heatmaps_all.pptx"

# slide dimensions match the figures exactly (13.33 x 7.5 in = PPT widescreen)
SLIDE_W_IN = 13.33
SLIDE_H_IN = 7.5

# divider slide colours
DIV_BG = "1E2761"      # navy
DIV_FG = "FFFFFF"      # white


# ===================================================================
# COLLECT FILES
# ===================================================================

def collect_groups(methods=("spearman", "pearson")):
    """
    Returns a list of (group_label, [png_paths]) tuples in display order.
    Figures within each group are sorted by filename so page numbers run
    in order.
    """
    groups = []

    # --- script 3 ---
    if S3_DIR.exists():
        pngs = sorted(S3_DIR.glob("corr_*_spearman*.png"))
        pngs += sorted(S3_DIR.glob("corr_*_pearson*.png"))
        pngs = sorted(set(pngs), key=lambda p: p.name)
        # exclude n-csv files that accidentally match (shouldn't happen but safe)
        pngs = [p for p in pngs if p.suffix == ".png"]
        if pngs:
            groups.append(("Script 3 – Questionnaires × EMA dynamics", pngs))
    else:
        print(f"  [skip] Script 3 figures not found: {S3_DIR}")

    # --- script 4 ---
    for method in methods:
        m_dir = S4_DIR / method
        if not m_dir.exists():
            print(f"  [skip] Script 4 {method} folder not found: {m_dir}")
            continue

        bym_dir = m_dir / "figures_by_measure"
        fig_dir = m_dir / "figures"

        if bym_dir.exists():
            pngs = sorted(p for p in bym_dir.glob("*.png"))
            if pngs:
                groups.append((
                    f"Script 4 ({method}) – per-measure heatmaps", pngs))

        if fig_dir.exists():
            pngs = sorted(p for p in fig_dir.glob("*.png")
                          if not p.name.startswith("topcorr"))
            top = sorted(p for p in fig_dir.glob("topcorr*.png"))
            if pngs:
                groups.append((
                    f"Script 4 ({method}) – family heatmaps", pngs))
            if top:
                groups.append((
                    f"Script 4 ({method}) – top correlations", top))

    total = sum(len(g[1]) for g in groups)
    print(f"Collected {total} figures across {len(groups)} group(s):")
    for label, pngs in groups:
        print(f"  {label}: {len(pngs)} figure(s)")
    return groups


# ===================================================================
# BUILD PPTX
# ===================================================================

def build_pptx(groups):
    try:
        import pptxgenjs as pptxgen
        _use_pptxgen = True
    except ImportError:
        _use_pptxgen = False

    if not _use_pptxgen:
        # fall back to python-pptx, which is always available in the env
        _build_with_python_pptx(groups)
        return

    prs = pptxgen()
    prs.layout = "LAYOUT_WIDE"  # 13.33 x 7.5 in

    def add_divider(label):
        slide = prs.addSlide()
        slide.addShape({
            "type": "rect",
            "x": 0, "y": 0, "w": "100%", "h": "100%",
            "fill": {"color": DIV_BG},
            "line": {"color": DIV_BG},
        })
        slide.addText(label, {
            "x": 0.5, "y": 2.5, "w": 12.33, "h": 2.5,
            "fontSize": 32, "bold": True, "color": DIV_FG,
            "align": "center", "valign": "middle",
            "isTextBox": True, "margin": 0,
        })

    def add_figure(png_path):
        slide = prs.addSlide()
        slide.addImage({
            "path": str(png_path),
            "x": 0, "y": 0, "w": SLIDE_W_IN, "h": SLIDE_H_IN,
            "sizing": {"type": "contain", "align": "center", "valign": "top"},
        })
        slide.addNotes(png_path.name)

    for label, pngs in groups:
        add_divider(label)
        for png in pngs:
            add_figure(png)

    prs.writeFile(str(OUT_PPTX))
    print(f"\n[OK] -> {OUT_PPTX}")


def _build_with_python_pptx(groups):
    """Fallback using python-pptx (always available, no npm needed)."""
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    W = Inches(SLIDE_W_IN)
    H = Inches(SLIDE_H_IN)

    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    blank_layout = prs.slide_layouts[6]   # blank

    def hex_to_rgb(h):
        return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def add_divider(label):
        slide = prs.slides.add_slide(blank_layout)
        bg = slide.shapes.add_shape(
            1,  # MSO_SHAPE_TYPE.RECTANGLE
            0, 0, W, H)
        bg.fill.solid()
        bg.fill.fore_color.rgb = hex_to_rgb(DIV_BG)
        bg.line.fill.background()

        txb = slide.shapes.add_textbox(
            Inches(0.5), Inches(2.5), Inches(12.33), Inches(2.5))
        tf = txb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = label
        run.font.size = Pt(32)
        run.font.bold = True
        run.font.color.rgb = hex_to_rgb(DIV_FG)

    def add_figure(png_path):
        slide = prs.slides.add_slide(blank_layout)
        slide.shapes.add_picture(
            str(png_path), 0, 0, W, H)
        # speaker note
        notes_slide = slide.notes_slide
        notes_slide.notes_text_frame.text = png_path.name

    total = sum(len(g[1]) for g in groups)
    done = 0
    for label, pngs in groups:
        add_divider(label)
        for png in pngs:
            add_figure(png)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{total} slides added...")

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    print(f"\n[OK] -> {OUT_PPTX}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else None
    if arg in ("spearman", "pearson"):
        methods = (arg,)
    elif arg in (None, "both", "all"):
        methods = ("spearman", "pearson")
    else:
        raise SystemExit(f"Unknown argument {arg!r}. "
                         f"Use: spearman | pearson | both (default)")

    print(f"Building heatmap deck  (method(s): {', '.join(methods)})")
    print(f"Output: {OUT_PPTX}\n")

    groups = collect_groups(methods=methods)
    if not any(pngs for _, pngs in groups):
        raise SystemExit(
            "No PNG figures found. Run scripts 3 and/or 4 first.")

    build_pptx(groups)


if __name__ == "__main__":
    main()
