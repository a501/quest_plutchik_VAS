// Deck: neural patterns (13_meg_neural_patterns.py outputs, folder vas_meg_patterns)
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/72d3cb23-3a85-47df-bb97-0a67fd28eed7_70099548-fde8-425a-bdbf-0a6eabfccad5/pptx/scripts/apply_theme.js");
const OUT = process.argv[2] || "patterns.pptx";

const HEX = {
  ink: "1F2433", navy: "232B45", slate: "5A6478", mist: "EEF1F6", white: "FFFFFF",
  f1: "2A9D8F", f2: "6C5B9B", base: "8A93A6", warn: "B5473A", grid: "D9DDE5", sad: "3B5BA5", fear: "D9822B",
};
const THEME = {
  name: "Neural Patterns", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: HEX.ink, lt1: HEX.white, dk2: HEX.navy, lt2: HEX.mist,
    accent1: HEX.f1, accent2: HEX.f2, accent3: HEX.base, accent4: HEX.sad, accent5: HEX.warn,
    accent6: HEX.fear, hlink: HEX.sad, folHlink: HEX.f2 },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Neural patterns at rest";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 1.6, fontFace: THEME.headFontFace, fontSize: 42, bold: true, color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.0, h: 1.8, fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontFace: THEME.headFontFace, fontSize: 28, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "MEG neural patterns · 13_meg_neural_patterns.py", options: { x: 0.6, y: 7.05, w: 7, h: 0.3, fontSize: 10, color: C.accent3, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: C.accent3, align: "right" },
});

const axisBase = () => ({
  catAxisLabelColor: HEX.slate, valAxisLabelColor: HEX.slate,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" },
  legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.slate,
  titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: HEX.ink,
  valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt",
  valAxisTitleFontSize: 11, catAxisTitleFontSize: 11,
  valAxisTitleColor: HEX.slate, catAxisTitleColor: HEX.slate,
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10, dataLabelColor: HEX.ink,
});
function content(title, section, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function card(s, x, y, w, h, head, items, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { type: "none" }, objectName: "card " + head });
  s.addText(head, { x: x + 0.25, y: y + 0.18, w: w - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    const o = { bullet: true, paraSpaceAfter: 6, breakLine: !last };
    if (Array.isArray(it)) {
      runs.push({ text: it[0], options: { ...o, bold: true, breakLine: false } });
      runs.push({ text: it[1], options: { breakLine: !last } });
    } else runs.push({ text: it, options: o });
  });
  s.addText(runs, { x: x + 0.25, y: y + 0.65, w: w - 0.5, h: h - 0.8, fontSize: opts.fontSize || 14, color: C.text1, valign: "top", margin: 0, isTextBox: true });
}
function caption(s, text, y = 6.55) {
  s.addText(text, { x: 0.6, y, w: 12.1, h: 0.4, fontSize: 11, italic: true, color: C.accent3, margin: 0, valign: "top", isTextBox: true });
}
function stat(s, x, y, w, big, label, color) {
  s.addText(big, { x, y, w, h: 0.85, fontSize: 40, bold: true, color: color || C.accent1, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
  s.addText(label, { x, y: y + 0.85, w, h: 0.7, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true });
}
function table(s, rows, x, y, w, colW, opts = {}) {
  const hdr = rows[0].map((t) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: HEX.navy }, fontSize: 12 } }));
  const body = rows.slice(1).map((r, i) => r.map((t) => (typeof t === "object" ? t : { text: String(t), options: { fontSize: 12, color: HEX.ink, fill: { color: i % 2 ? HEX.white : HEX.mist } } })));
  s.addTable([hdr, ...body], { x, y, w, colW, border: { type: "solid", color: HEX.grid, pt: 0.5 }, fontFace: "Calibri", valign: "middle", margin: 0.06, rowH: opts.rowH || 0.36 });
}

// ===================================================================== 1
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Neural patterns at rest and emotion dynamics", { placeholder: "title" });
s.addText([
  { text: "Which patterns do the source-space MEG dynamics form, and how do they relate to the VAS ratings?", options: { breakLine: true } },
  { text: "47 participants · 12 MEG measures × 4 networks × 2 source methods · baseline, sad and fear rest blocks", options: { fontSize: 16, color: C.accent3 } },
], { placeholder: "body" });
s.addNotes("Results of 13_meg_neural_patterns.py (output folder vas_meg_patterns). MEG: static dynamics measures of the 19 s snippets before each VAS probe, averaged per subject x block (parts 1-3) or kept per snippet (part 4). Methods mean_flip and pca_flip are the two ways of extracting one time series per ROI. Behaviour: per subject x block from scripts 5 and 7 (mean of the Sadness and Fear scales). All tests uncorrected and rank-based; this is a descriptive, exploratory analysis.");

// ===================================================================== 2
s = content("Four questions about the neural patterns", "Introduction",
  "Part 1 uses rotated factor analysis on the 12 measures averaged over networks (47 subjects are enough for 12 variables but not for 48). Part 2 splits every network value into whole-brain level + network deviation. Part 3 is a latent profile analysis (Gaussian mixtures). Part 4 is a hidden Markov model on the snippet-level factor scores within each block.");
const qs = [
  ["1", "Patterns among the measures", "Do the 12 MEG measures reduce to a few underlying patterns?", "Rotated factor analysis"],
  ["2", "Network-specific patterns", "Is there anything beyond the whole-brain level, e.g. in the DMN?", "Whole brain + network deviation"],
  ["3", "Types of people", "Do people fall into distinct neural profiles?", "Latent profile analysis"],
  ["4", "Brain states over time", "Does the brain switch between states during rest?", "Hidden Markov model"],
];
qs.forEach((q, i) => {
  const x = 0.6 + i * 3.075, y = 1.55, w = 2.85, h = 4.6;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { type: "none" }, objectName: "question " + q[0] });
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.3, w: 0.7, h: 0.7, fill: { color: i % 2 ? C.accent2 : C.accent1 }, line: { type: "none" }, objectName: "badge " + q[0] });
  s.addText(q[0], { x: x + 0.25, y: y + 0.3, w: 0.7, h: 0.7, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(q[1], { x: x + 0.25, y: y + 1.2, w: w - 0.5, h: 0.8, fontSize: 17, bold: true, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  s.addText(q[2], { x: x + 0.25, y: y + 2.1, w: w - 0.5, h: 1.3, fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  s.addText(q[3], { x: x + 0.25, y: y + 3.7, w: w - 0.5, h: 0.6, fontSize: 13, italic: true, color: C.accent3, valign: "top", margin: 0, isTextBox: true });
});
caption(s, "Behaviour per person and block: mean level, share of non-zero ratings, SD, noise-corrected fluctuation SD (latent SD, partialled on the mean) and first-probe elevation.", 6.4);

// ===================================================================== 3
pres.addSection({ title: "Measures" });
s = content("The MEG measures are skewed, so they were rank-normalised first", "Measures",
  "P0_distributions.csv, baseline, subject means over networks, mean_flip. Raw data: 10 of 12 measures fail Shapiro-Wilk (p < .05); oscillatory power has skew 5.7 and an outlier at |z| = 6.4; the low/high 8 Hz ratio has skew 2.0, multiscale entropy -1.9. Factor analysis, Gaussian mixtures and HMMs assume normal data, so all measures were rank-based inverse-normal transformed (pooled over subjects, blocks and networks per method x measure). After the transform: |skew| <= 0.54, only ACW50 still fails Shapiro (p = .011), largest |z| = 3.0. pca_flip looks the same.");
const meas = ["IRASA_oscpower", "Spec_lowhigh_8Hz", "MSE_costa_AUC", "SampEn", "IRASA_slope", "SpectralSlope", "LZC", "SpectralEntropy", "ACW50", "ACW0", "ACF_area", "MedianFreq"];
const skRaw = [5.68, 1.97, 1.92, 1.56, 1.01, 1.05, 0.98, 0.94, 0.91, 0.83, 0.45, 0.06];
const skTr = [0.43, 0.07, 0.43, 0.54, 0.19, 0.27, 0.37, 0.32, 0.18, 0.16, 0.21, 0.20];
s.addChart(pres.charts.BAR, [
  { name: "Raw", labels: meas, values: skRaw },
  { name: "Rank-normalised", labels: meas, values: skTr },
], {
  x: 0.6, y: 1.45, w: 7.8, h: 5.0, barDir: "bar", chartColors: [HEX.warn, HEX.f1], barGapWidthPct: 40,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 9, showLegend: true, legendPos: "t",
  showTitle: true, title: "Absolute skew of each measure (baseline, mean_flip)", valAxisMinVal: 0, valAxisMaxVal: 6.5, catAxisOrientation: "maxMin",
  ...axisBase(),
});
stat(s, 8.9, 1.6, 3.8, "10 → 1", "of 12 measures fail the normality test, before → after the transform", HEX.f1);
stat(s, 8.9, 3.3, 3.8, "6.4 → 3.0", "largest |z| of a person on any measure", HEX.f2);
s.addText("All Gaussian models (factor analysis, profiles, HMM) use the transformed data. Behaviour tests are rank-based throughout.", { x: 8.9, y: 5.0, w: 3.8, h: 1.2, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });

// ===================================================================== 4
pres.addSection({ title: "Part 1 · Factors" });
s = content("Twelve measures reduce to two patterns", "Part 1 · Factors",
  "P1_parallel_analysis.csv. Parallel analysis keeps components whose eigenvalue is above the 95th percentile of random data with the same size (n = 47, 12 variables). mean_flip: eigenvalues 5.80, 4.51, 0.72 vs random 2.20, 1.82, 1.59 -> 2 factors; pca_flip: 6.30, 4.15, 0.69 -> 2 factors. The two factors carry 86% (mean_flip) and 87% (pca_flip) of the variance. P1_congruence.csv: refitting the factor analysis in the sad and fear blocks gives Tucker congruence 0.97-0.99 with the baseline factors, and 0.998 between the two methods (> .95 = same factor).");
const comp = ["1", "2", "3", "4", "5", "6"];
s.addChart(pres.charts.BAR, [
  { name: "Observed eigenvalue", labels: comp, values: [5.80, 4.51, 0.72, 0.52, 0.19, 0.10] },
  { name: "Random data (95th percentile)", labels: comp, values: [2.20, 1.82, 1.59, 1.40, 1.24, 1.09] },
], {
  x: 0.6, y: 1.45, w: 7.4, h: 5.0, barDir: "col", chartColors: [HEX.f1, "C9CED8"], barGapWidthPct: 50,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: true, legendPos: "t",
  showTitle: true, title: "Parallel analysis (baseline, mean_flip)", valAxisMinVal: 0, valAxisMaxVal: 7,
  showCatAxisTitle: true, catAxisTitle: "Component", ...axisBase(),
});
stat(s, 8.5, 1.6, 4.2, "2 factors", "explain 86–87% of the variance of the 12 measures", HEX.f1);
stat(s, 8.5, 3.3, 4.2, "0.97–0.99", "similarity of the factors when refitted in the sad and fear blocks; 0.998 between the two methods", HEX.f2);

// ===================================================================== 5
s = content("Factor 1 = complexity, Factor 2 = spectral speed", "Part 1 · Factors",
  "P1_loadings.csv, mean_flip (pca_flip is the same, congruence 0.998). Varimax rotation. F1: LZC +0.98, SampEn +0.97, SpectralSlope -0.87, IRASA_slope -0.85, ACW50 -0.79, IRASA_oscpower -0.67, SpectralEntropy +0.49. High F1 = complex, irregular signal with a flatter aperiodic spectrum and less oscillatory power. F2: MedianFreq +0.96, Spec_lowhigh_8Hz -0.95, MSE_AUC +0.87, SpectralEntropy +0.82, ACF_area_to_zero -0.82, ACW0 -0.65. High F2 = power shifted to higher frequencies, broadband, short autocorrelation; low F2 = slow, alpha-dominated, long autocorrelation. The slope sign: the slope measures are exponents (positive numbers), so a negative loading means a flatter spectrum goes with high F1.");
const mL = ["LZC", "SampEn", "SpectralSlope", "IRASA_slope", "ACW50", "IRASA_oscpower", "SpectralEntropy", "ACW0", "MedianFreq", "Spec_lowhigh_8Hz", "MSE_costa_AUC", "ACF_area"];
const l1 = [0.98, 0.97, -0.87, -0.85, -0.79, -0.67, 0.49, 0.31, 0.01, -0.12, 0.00, -0.02];
const l2 = [-0.04, -0.06, -0.19, -0.23, -0.49, 0.30, 0.82, -0.65, 0.96, -0.95, 0.87, -0.82];
s.addChart(pres.charts.BAR, [
  { name: "Factor 1", labels: mL, values: l1 },
  { name: "Factor 2", labels: mL, values: l2 },
], {
  x: 0.6, y: 1.45, w: 7.6, h: 5.05, barDir: "bar", chartColors: [HEX.f1, HEX.f2], barGapWidthPct: 35,
  showValue: false, showLegend: true, legendPos: "t", catAxisOrientation: "maxMin",
  showTitle: true, title: "Factor loadings (baseline, mean_flip)", valAxisMinVal: -1, valAxisMaxVal: 1, valAxisMajorUnit: 0.5, catAxisLabelPos: "low",
  valAxisLabelFormatCode: "0.0", ...axisBase(),
});
card(s, 8.6, 1.55, 4.1, 2.35, "Factor 1 · complexity", [
  ["High: ", "complex, irregular signal (LZC, sample entropy), flatter spectrum."],
  ["Low: ", "steep aperiodic slope, more oscillatory power, longer ACW50."],
], { fontSize: 13 });
card(s, 8.6, 4.1, 4.1, 2.35, "Factor 2 · spectral speed", [
  ["High: ", "power at higher frequencies, broadband, short autocorrelation."],
  ["Low: ", "slow, alpha-dominated activity, long autocorrelation."],
], { fontSize: 13 });

// ===================================================================== 6
s = content("Both patterns are stable traits of a person", "Part 1 · Factors",
  "P1_factor_scores.csv. Spearman correlation of each person's factor score between blocks (scores in all blocks computed with the baseline model). F1: baseline-sad .82, baseline-fear .91 (mean_flip); .81 and .92 (pca_flip). F2: .67 / .73 (mean_flip); .69 / .76 (pca_flip). Mean shift: F1 is about 0.16 SD higher in both video blocks than at baseline; F2 +0.06-0.08 SD. Block order is fixed (baseline always first), so the shift can be time-on-task as much as the videos.");
table(s, [
  ["Factor", "Baseline ↔ sad", "Baseline ↔ fear", "Shift vs baseline"],
  ["F1 complexity (mean_flip)", "0.82", "0.91", "+0.16 SD"],
  ["F1 complexity (pca_flip)", "0.81", "0.92", "+0.15 SD"],
  ["F2 spectral speed (mean_flip)", "0.67", "0.73", "+0.06 SD"],
  ["F2 spectral speed (pca_flip)", "0.69", "0.76", "+0.07 SD"],
], 0.6, 1.6, 7.8, [3.3, 1.5, 1.5, 1.5], { rowH: 0.5 });
card(s, 8.8, 1.6, 3.9, 4.6, "Reading", [
  ["Trait-like: ", "a person's factor scores are very similar in all three blocks, especially F1."],
  ["Small state shift: ", "F1 is slightly higher after the videos (+0.16 SD)."],
  ["Order caveat: ", "baseline is always first, so the shift may be time in the session rather than emotion."],
]);
caption(s, "Spearman correlation of the factor scores between blocks; shift = mean factor score in the video blocks minus baseline (factor scores are z-scaled at baseline).", 4.8);

// ===================================================================== 7
s = content("Lower brain complexity goes with larger emotion fluctuations", "Part 1 · Factors",
  "P1_factor_vs_behaviour.csv. Partial Spearman of the factor score with the noise-corrected fluctuation SD (latent SD, script 5), controlling for the mean rating level; and plain Spearman with the observed SD. F1 baseline: latent SD rho = -0.36 (p = .017) mean_flip, -0.39 (p = .009) pca_flip; SD -0.32 (p = .034) / -0.30 (p = .044). Sad block: latent SD -0.29 (p = .055) / -0.30 (p = .047); SD -0.31 (p = .035) / -0.30 (p = .042). Fear block: about -0.1, n.s. F2: no significant link (largest: fear block latent SD +0.28, p = .065). 7 of 60 tests p < .05 (expected 3), all of them F1 with fluctuation size. Uncorrected; the confound check (script 12: head position, noise, heart rate, blink rate, missed ratings) is the next step.");
const cats = ["Baseline", "Sad block", "Fear block"];
s.addChart(pres.charts.BAR, [
  { name: "mean_flip", labels: cats, values: [-0.36, -0.29, -0.09] },
  { name: "pca_flip", labels: cats, values: [-0.39, -0.30, -0.06] },
], {
  x: 0.6, y: 1.45, w: 7.4, h: 5.0, barDir: "col", chartColors: [HEX.f1, "8FCFC6"], barGapWidthPct: 50,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: true, legendPos: "t",
  showTitle: true, title: "Factor 1 (complexity) vs emotion fluctuation (latent SD, mean held constant)",
  valAxisMinVal: -0.5, valAxisMaxVal: 0.1, valAxisLabelFormatCode: "0.0", showValAxisTitle: true, valAxisTitle: "Partial Spearman ρ", ...axisBase(),
});
card(s, 8.4, 1.55, 4.3, 4.9, "Result", [
  ["Baseline: ", "ρ = −0.36 / −0.39 (p = .017 / .009)."],
  ["Sad block: ", "same direction, ρ ≈ −0.30 (p ≈ .05)."],
  ["Fear block: ", "no link (ρ ≈ −0.1)."],
  ["Factor 2: ", "no link with any behaviour."],
  ["Meaning: ", "a less complex, steeper-spectrum brain at rest goes with larger emotion fluctuations."],
]);
caption(s, "Uncorrected; 7 of 60 factor × behaviour tests reach p < .05 (3 expected), all F1 with fluctuation size. Next: rule out head position, noise and drowsiness (12_meg_pc_confounds.py).");

// ===================================================================== 8
pres.addSection({ title: "Part 2 · Networks" });
s = content("Networks add little beyond the whole brain", "Part 2 · Networks",
  "P2_network_specific_variance.csv: share of each measure's variance that is specific to one network (value minus whole-brain mean): 5-12%, median 9%. P2_network_deviation_stability.csv: the network deviations of a person correlate rho = 0.60-0.90 (median 0.79) between blocks, so they are stable and not noise. P2_network_deviation_vs_behaviour.csv: 15 of 240 tests p < .05 (12 expected). One recurring pattern: the DMN deviation on F2 (DMN relatively faster/less alpha than the rest of the brain) goes with a higher first rating of the block (first minus rest): positive in all 6 block x method combinations, rho 0.26-0.35, p < .05 in 3 (baseline mean_flip .031, baseline pca_flip .018, fear pca_flip .018).");
stat(s, 0.6, 1.6, 3.8, "9%", "of a measure's variance is network-specific (5–12%); the rest is whole-brain", HEX.f1);
stat(s, 4.75, 1.6, 3.8, "ρ ≈ 0.79", "stability of a person's network deviations between blocks: small but real", HEX.f2);
stat(s, 8.9, 1.6, 3.8, "15 / 240", "network-deviation × behaviour tests at p < .05 (12 expected by chance)", HEX.warn);
const six = ["Baseline · mean", "Baseline · pca", "Sad · mean", "Sad · pca", "Fear · mean", "Fear · pca"];
s.addChart(pres.charts.BAR, [{ name: "ρ", labels: six, values: [0.33, 0.35, 0.26, 0.29, 0.27, 0.35] }], {
  x: 0.6, y: 3.55, w: 8.0, h: 2.95, barDir: "col", chartColors: [HEX.f2], barGapWidthPct: 50,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: false,
  showTitle: true, title: "One recurring lead: DMN deviation on Factor 2 vs first-probe elevation",
  valAxisMinVal: 0, valAxisMaxVal: 0.45, valAxisLabelFormatCode: "0.0", ...axisBase(),
});
s.addText("Positive in all 6 block × method combinations, p < .05 in 3. Whole-brain score held constant. Exploratory.", { x: 8.9, y: 4.0, w: 3.8, h: 2.0, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });

// ===================================================================== 9
pres.addSection({ title: "Part 3 · Types" });
s = content("No neural types of people: the profiles form a continuum", "Part 3 · Types",
  "P3_profile_selection.csv. Gaussian mixtures with 1-5 classes on the baseline factor scores (F1, F2). BIC is lowest for one class in both methods (mean_flip 284.6 vs 297.0 for two; pca_flip 284.4 vs 289.7). Two-class solutions are also unstable on bootstrap resampling (adjusted Rand 0.49 and 0.65; criterion >= 0.60 plus BIC) - so one continuous distribution, no types. Therefore no comparison with the behavioural fluctuation types of script 5 (P3_profiles_vs_behaviour.csv is empty).");
const ks = ["1", "2", "3", "4", "5"];
s.addChart(pres.charts.LINE, [
  { name: "mean_flip", labels: ks, values: [284.6, 297.0, 311.1, 309.5, 316.2] },
  { name: "pca_flip", labels: ks, values: [284.4, 289.7, 303.0, 316.4, 317.2] },
], {
  x: 0.6, y: 1.45, w: 7.4, h: 5.0, chartColors: [HEX.f1, HEX.f2], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 7,
  showLegend: true, legendPos: "t", showTitle: true, title: "BIC by number of profiles (lower = better)",
  valAxisMinVal: 270, valAxisMaxVal: 325, showCatAxisTitle: true, catAxisTitle: "Number of profiles", ...axisBase(),
});
card(s, 8.4, 1.55, 4.3, 4.9, "Result", [
  ["BIC: ", "one profile is best in both methods."],
  ["Stability: ", "two-profile splits are not reproducible (adjusted Rand 0.49 / 0.65)."],
  ["Meaning: ", "people differ along the two factors continuously, not in distinct neural types."],
  ["Consequence: ", "use the factor scores as continuous measures, not groups."],
]);

// ===================================================================== 10
pres.addSection({ title: "Part 4 · States" });
s = content("Brain states: no fast switching at the 40-second resolution", "Part 4 · States",
  "P4_hmm_selection.csv: Gaussian HMM on the snippet-level factor scores (one snippet per VAS probe, about every 40 s), centred within subject x block. Held-out log-likelihood (subjects in 5 folds) is highest at K = 5 among solutions where every state holds >= 5% of snippets (mean_flip; pca_flip K = 5 as well; K = 7-8 fail the occupancy rule). P4_hmm_states.csv: state means differ by at most 0.44 SD. P4_hmm_transmat.csv: staying in a state has probability 0.72-0.99. P4_hmm_subject_block.csv: 34% (mean_flip) / 50% (pca_flip) of blocks never switch state; median switch rate 0.08 / 0.02 per snippet; median dwell 8 / 15 snippets. So the states mostly mark blocks that differ in how variable they are, not transient states. P4_hmm_vs_behaviour.csv: 6 of 210 tests p < .05 (10 expected). P4_state_locked_ratings.csv: 3 of 40 p < .05; the strongest, pca_flip state 2 -> larger next Fear rating change (p = .0003, n = 23), has no counterpart in mean_flip (states are not matched between methods).");
s.addChart(pres.charts.LINE, [
  { name: "mean_flip", labels: ["2", "3", "4", "5", "6", "7", "8"], values: [-0.656, -0.638, -0.633, -0.627, -0.632, -0.633, -0.632] },
  { name: "pca_flip", labels: ["2", "3", "4", "5", "6", "7", "8"], values: [-0.560, -0.545, -0.515, -0.504, -0.515, -0.487, -0.488] },
], {
  x: 0.6, y: 1.45, w: 6.4, h: 5.0, chartColors: [HEX.f1, HEX.f2], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 7,
  showLegend: true, legendPos: "t", showTitle: true, title: "Held-out log-likelihood per snippet by number of states",
  valAxisMinVal: -0.7, valAxisMaxVal: -0.45, valAxisLabelFormatCode: "0.00", showCatAxisTitle: true, catAxisTitle: "Number of states", ...axisBase(),
});
stat(s, 7.4, 1.6, 2.6, "34–50%", "of blocks never switch state", HEX.warn);
stat(s, 10.2, 1.6, 2.6, "0.72–0.99", "probability of staying in a state", HEX.f2);
card(s, 7.4, 3.45, 5.3, 3.0, "Result", [
  ["States: ", "5, but they differ by ≤ 0.44 SD and rarely change."],
  ["Behaviour: ", "6 of 210 tests p < .05 (10 expected); state-locked ratings 3 of 40."],
  ["Meaning: ", "no transient brain states detectable with one snippet every ~40 s."],
], { fontSize: 13 });
caption(s, "K chosen by held-out likelihood among solutions with every state ≥ 5% of snippets. One lead (pca_flip state 2 → larger next Fear change, p = .0003) has no counterpart in mean_flip.");

// ===================================================================== 11
pres.addSection({ title: "Summary" });
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Summary" });
s.addText("Take-home", { x: 0.8, y: 0.5, w: 11.7, h: 0.9, fontSize: 36, bold: true, color: C.background1, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
const take = [
  ["Two whole-brain patterns.", "Complexity vs steep slope, and fast vs slow (alpha) activity; robust across blocks and methods, and stable within a person."],
  ["Complexity tracks emotion fluctuation.", "Less complex, steeper-spectrum brains go with larger fluctuations at baseline (ρ ≈ −0.37) and in the sad block, not in the fear block."],
  ["Networks barely differ.", "About 9% network-specific variance; one lead: DMN spectral speed with the first-probe elevation."],
  ["No types, no fast states.", "People vary continuously; the HMM finds no switching states at a 40-s resolution."],
];
take.forEach((t, i) => {
  const y = 1.6 + i * 1.05;
  s.addShape(pres.shapes.OVAL, { x: 0.8, y, w: 0.55, h: 0.55, fill: { color: i % 2 ? C.accent2 : C.accent1 }, line: { type: "none" }, objectName: "take-home " + (i + 1) });
  s.addText(String(i + 1), { x: 0.8, y, w: 0.55, h: 0.55, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText([{ text: t[0] + " ", options: { bold: true, color: HEX.white } }, { text: t[1], options: { color: "C9D0DE" } }], { x: 1.6, y: y - 0.12, w: 11.0, h: 0.85, fontSize: 16, valign: "middle", margin: 0, isTextBox: true });
});
s.addText("Next: confound check of Factor 1 (head position, noise, heart rate, blinks, missed ratings; 12_meg_pc_confounds.py) · add 052 and 036's sad block", { x: 0.8, y: 6.05, w: 11.7, h: 0.8, fontSize: 14, italic: true, color: "AEB6C7", margin: 0, isTextBox: true });
s.addNotes("All results are uncorrected and exploratory. The F1-fluctuation link is the one result that is consistent across methods and appears in two blocks; it needs the confound check before interpretation.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
