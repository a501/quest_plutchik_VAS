// Build "How do emotions move at rest?" deck from 2_behav_dynamics.py outputs
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const { applyTheme } = require("/root/.claude/skills/synced/72d3cb23-3a85-47df-bb97-0a67fd28eed7_70099548-fde8-425a-bdbf-0a6eabfccad5/pptx/scripts/apply_theme.js");
const curves = JSON.parse(fs.readFileSync(__dirname + "/dynamics_curves.json", "utf8"));
const OUT = process.argv[2] || "deck.pptx";

const HEX = {
  ink: "1F2433", navy: "232B45", slate: "5A6478", mist: "EEF1F6", white: "FFFFFF",
  sad: "3B5BA5", fear: "D9822B", base: "8A93A6", ok: "2A9D8F", warn: "B5473A", grid: "D9DDE5",
};
const THEME = {
  name: "Emotion Dynamics", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: HEX.ink, lt1: HEX.white, dk2: HEX.navy, lt2: HEX.mist,
    accent1: HEX.sad, accent2: HEX.fear, accent3: HEX.base, accent4: HEX.ok, accent5: HEX.warn,
    accent6: "6C5B9B", hlink: HEX.sad, folHlink: "6C5B9B" },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "How do emotions move at rest?";
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 1.6, fontFace: THEME.headFontFace, fontSize: 44, bold: true, color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.0, h: 1.6, fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontFace: THEME.headFontFace, fontSize: 28, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "VAS dynamics at rest · 2_behav_dynamics.py", options: { x: 0.6, y: 7.05, w: 7, h: 0.3, fontSize: 10, color: C.accent3, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: C.accent3, align: "right" },
});

// ---------- helpers ----------
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

// side card: header + bullet list; items: strings or [bold, rest]
function card(s, x, y, w, h, head, items, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: opts.fill || C.background2 }, line: { type: "none" }, objectName: "card " + head });
  s.addText(head, { x: x + 0.25, y: y + 0.18, w: w - 0.5, h: 0.4, fontSize: 16, bold: true, color: opts.headColor || C.text2, margin: 0, isTextBox: true });
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
  s.addText(label, { x, y: y + 0.85, w, h: 0.65, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true });
}

function table(s, rows, x, y, w, colW, opts = {}) {
  const hdr = rows[0].map((t) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: HEX.navy }, fontSize: 12 } }));
  const body = rows.slice(1).map((r, i) => r.map((t) => (typeof t === "object" ? t : { text: String(t), options: { fontSize: 12, color: HEX.ink, fill: { color: i % 2 ? HEX.white : HEX.mist } } })));
  s.addTable([hdr, ...body], { x, y, w, colW, border: { type: "solid", color: HEX.grid, pt: 0.5 }, fontFace: "Calibri", valign: "middle", margin: 0.06, rowH: opts.rowH || 0.36 });
}
const hl = (t, color) => ({ text: t, options: { fontSize: 12, bold: true, color: color || HEX.ok, fill: { color: HEX.white } } });

// =====================================================================
// 1 Title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("How do emotions move at rest?", { placeholder: "title" });
s.addText([
  { text: "Dynamics of the within-rest VAS ratings (Sadness, Fear) in the baseline, sad and fear rest blocks", options: { breakLine: true } },
  { text: "Follow-up to “Do emotions fluctuate at rest?” · 50 participants · 300 rating series", options: { fontSize: 16, color: C.accent3 } },
], { placeholder: "body" });
s.addNotes("Follow-up of the 'Do emotions fluctuate at rest?' deck. That deck showed the rest ratings are not flat. This one asks how they move: recovery after the video, the time scale of fluctuations, coupling between the two scales, carry-over, rating behaviour, reliability and redundancy. All numbers come from 2_behav_dynamics.py (sections A-F, H-J). The HMM (section G) outputs were not part of this upload and are not shown.");

// 2 Roadmap
s = content("From “do they move?” to “how do they move?”", "Introduction",
  "Previous result: within-rest VAS ratings fluctuate beyond rating noise. Here we describe the shape of that movement. Each card is one analysis section of 2_behav_dynamics.py. D is a prerequisite for C (the variogram uses the uneven probe spacing as a built-in experiment).");
s.addText("Previous deck: within-rest ratings are not flat; they move by more than rating noise. This deck describes how they move.", { x: 0.6, y: 1.35, w: 12.1, h: 0.5, fontSize: 16, color: C.text1, margin: 0, isTextBox: true });
const qs = [
  ["A", "Induction", "Are the two videos equally specific?"],
  ["D", "Probe timing", "Are probe intervals random, so lag can be studied?"],
  ["B", "Recovery", "How fast does the induced emotion return to a plateau?"],
  ["C", "Variogram", "Do fluctuations level off (regulated) or keep drifting?"],
  ["E", "Coupling", "Do Sadness and Fear move together?"],
  ["F", "Carry-over", "Does one block’s level carry into the next?"],
  ["H", "Rating behaviour", "How much back-and-forth on the slider?"],
  ["I · J", "Reliability & redundancy", "Which parameters are stable and not just mean/SD?"],
];
qs.forEach((q, i) => {
  const col = i % 4, row = Math.floor(i / 4);
  const x = 0.6 + col * 3.075, y = 2.15 + row * 2.25, w = 2.85, h = 2.0;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { type: "none" }, objectName: "question card " + q[0] });
  s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.2, w: 0.62, h: 0.62, fill: { color: i % 2 ? C.accent2 : C.accent1 }, line: { type: "none" }, objectName: "badge " + q[0] });
  s.addText(q[0], { x: x + 0.2, y: y + 0.2, w: 0.62, h: 0.62, fontSize: q[0].length > 2 ? 11 : 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(q[1], { x: x + 0.95, y: y + 0.2, w: w - 1.1, h: 0.62, fontSize: 15, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
  s.addText(q[2], { x: x + 0.2, y: y + 0.95, w: w - 0.4, h: 0.95, fontSize: 13, color: C.text1, valign: "top", margin: 0, isTextBox: true });
});

// 3 Design & data
s = content("Data: 50 people, three 20-min rest blocks, ~25 probes each", "Introduction",
  "Each block: 20 min rest with probes about every 40 s; at each probe the participant rates Sadness and Fear on a 0-100 VAS (5-point steps). Block 1 = baseline rest, then video 1 + rest, video 2 + rest. Video order counterbalanced: 26 sad-first, 24 fear-first. A series = subject x block x emotion. Usable = at least 18 ratings and not flat (MIN_VALID, same floor as the RT analysis). Timeout ratings are kept, as in the script.");
stat(s, 0.6, 1.6, 2.9, "50", "participants (ID < 200); subj 010 has no baseline, 054 baseline only");
stat(s, 3.7, 1.6, 2.9, "~25", "probes per 20-min block, each rating Sadness and Fear");
stat(s, 6.8, 1.6, 2.9, "300", "rating series (subject × block × emotion)");
stat(s, 9.9, 1.6, 2.9, "264", "usable series: ≥ 18 ratings and not flat", HEX.ok);
// design timeline
const steps = [["Baseline rest", "20 min", HEX.base], ["Video 1", "sad or fear", HEX.slate], ["Rest", "20 min", HEX.sad], ["Video 2", "the other", HEX.slate], ["Rest", "20 min", HEX.fear]];
steps.forEach((st, i) => {
  const x = 0.6 + i * 2.45, w = st[0].startsWith("Video") ? 1.9 : 2.2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.0, w, h: 1.0, rectRadius: 0.1, fill: { color: st[2] }, line: { type: "none" }, objectName: "design step " + (i + 1) });
  s.addText([{ text: st[0], options: { bold: true, breakLine: true } }, { text: st[1], options: { fontSize: 12 } }], { x, y: 4.0, w, h: 1.0, fontSize: 15, color: HEX.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
});
s.addText("Video order counterbalanced: 26 sad-first, 24 fear-first. Post-video ratings taken right after each video.", { x: 0.6, y: 5.2, w: 12.1, h: 0.4, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
caption(s, "Flat series (all ratings identical, n = 32) carry no dynamics and are excluded from per-series fits; most are flat at 0. Ratings on a 0–100 VAS in 5-point steps.", 5.8);

// =====================================================================
pres.addSection({ title: "Prerequisites" });
// 4 Interval randomness (D)
s = content("Probe intervals are random (20–60 s) and unrelated to time in block", "Prerequisites",
  "Section D. Interval = fixation interval from triggers (or elapsed gap when triggers missing, 7 subjects). Per subject x block: Spearman rho between interval and elapsed time; group Wilcoxon against 0. Sequence check: intervals rounded to 10 s; 99 distinct sequences out of 150 subject-block series. So the schedule is not one fixed list for everyone, but several sequences are shared. The 70 s bin (n = 69) comes from the elapsed-gap subjects whose gaps include rating time.");
s.addChart(pres.charts.BAR, [{ name: "Probes", labels: ["20 s", "30 s", "40 s", "50 s", "60 s", "70 s"], values: [657, 712, 699, 718, 715, 69] }], {
  x: 0.6, y: 1.5, w: 7.4, h: 4.9, barDir: "col", chartColors: [HEX.slate], barGapWidthPct: 40,
  showValue: true, dataLabelPosition: "outEnd", showLegend: false,
  showTitle: true, title: "Inter-probe interval, all probes (rounded to 10 s)",
  showValAxisTitle: true, valAxisTitle: "Number of probes", ...axisBase(),
});
stat(s, 8.5, 1.6, 4.2, "ρ = −0.01", "mean correlation of interval with time in block (Wilcoxon vs 0, p = .09)", HEX.ok);
stat(s, 8.5, 3.2, 4.2, "99 / 150", "distinct interval sequences across subject × block series", HEX.sad);
s.addText("Short and long lags occur all through the block, so the variogram (C) can compare lags without confounding them with time since rest onset.", { x: 8.5, y: 4.85, w: 4.2, h: 1.3, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
caption(s, "Caveat: sequences are partly shared between participants, so a fixed-schedule effect cannot be fully ruled out. The 70 s bin is mostly gap-based timing from 7 participants without trigger files.");

// 5 Induction (A)
s = content("Both videos induce their target emotion with equal specificity", "Prerequisites",
  "Section A. Paired tests on raw post-video ratings (n = 50). Target vs non-target: sad video dz = 0.81, fear video dz = 0.90, both p < 1e-6. Cross-contamination (Fear after sad video vs Sadness after fear video): 23.1 vs 23.7, p = .86. Specificity (target minus non-target): 20.8 vs 29.1, p = .13. Target intensity: Fear after fear video slightly higher than Sadness after sad video, 52.8 vs 43.9, p = .036 (t), p = .058 (Wilcoxon). At the first in-block probe the pattern holds (dz 0.85 and 0.63), and specificity is still equal (p = .58). End-of-baseline ratings are shown for reference.");
s.addChart(pres.charts.BAR, [
  { name: "Sadness", labels: ["End of baseline", "After sad video", "After fear video"], values: [20.5, 43.9, 23.7] },
  { name: "Fear", labels: ["End of baseline", "After sad video", "After fear video"], values: [16.8, 23.1, 52.8] },
], {
  x: 0.6, y: 1.5, w: 6.6, h: 4.9, barDir: "col", chartColors: [HEX.sad, HEX.fear], barGapWidthPct: 60,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", showLegend: true, legendPos: "t",
  showTitle: true, title: "Post-video rating (mean, n = 50)", valAxisMinVal: 0, valAxisMaxVal: 60,
  showValAxisTitle: true, valAxisTitle: "VAS (0–100)", ...axisBase(),
});
table(s, [
  ["Test (post-video)", "Result", "p"],
  ["Sad video: Sadness > Fear", "dz = 0.81", "< .001"],
  ["Fear video: Fear > Sadness", "dz = 0.90", "< .001"],
  ["Cross-contamination", "23.1 vs 23.7", hl(".86", HEX.slate)],
  ["Specificity (target − other)", "20.8 vs 29.1", hl(".13", HEX.slate)],
  ["Target intensity", "43.9 vs 52.8", hl(".036", HEX.warn)],
  ["First probe in rest: specificity", "13.8 vs 16.2", hl(".58", HEX.slate)],
], 7.6, 1.6, 5.1, [2.6, 1.4, 1.1], { rowH: 0.42 });
s.addText("Fear is induced a little more strongly, but the gap between target and non-target emotion is the same for both videos. Recovery can be compared between them.", { x: 7.6, y: 4.85, w: 5.1, h: 1.4, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });

// =====================================================================
pres.addSection({ title: "Recovery" });
// 6 Recovery curves
const mins = Array.from({ length: 20 }, (_, i) => String(i + 1));
const lineOpts = (title) => ({
  ...axisBase(), showTitle: true, title, valAxisMinVal: 0, valAxisMaxVal: 40, valAxisMajorUnit: 10,
  showLegend: true, legendPos: "b", lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 5,
  showValAxisTitle: true, valAxisTitle: "VAS (mean over subjects)", showCatAxisTitle: true, catAxisTitle: "Minute of rest block",
  catAxisLabelFrequency: 2,
});
s = content("The induced emotion settles within ~3–4 minutes; the other stays flat", "Recovery",
  "Section B. Group time courses: ratings averaged per subject in 1-min bins, then across subjects. Pooled group exponential fits (B_recovery_group): sad block Sadness tau = 3.2 min (bootstrap 95% CI 1.2-6.6), half-life 2.2 min, plateau 16.5. Fear block Fear tau = 3.7 min (2.4-5.9), half-life 2.6 min, plateau 15.1. Non-target emotion: constant model best in both blocks. Note the baseline block also drops during the first minute (group tau about 0.6 min, CI up to the 60-min bound) - a settling-in effect at every rest onset - and drifts up again in the last 5 minutes.");
s.addChart(pres.charts.LINE, [
  { name: "Sadness", labels: mins, values: curves["sad_Sadness"].mean },
  { name: "Fear", labels: mins, values: curves["sad_Fear"].mean },
  { name: "Sadness at baseline", labels: mins, values: curves["baseline_Sadness"].mean },
], { x: 0.6, y: 1.45, w: 6.0, h: 4.6, chartColors: [HEX.sad, HEX.fear, HEX.base], ...lineOpts("Rest after the SAD video") });
s.addChart(pres.charts.LINE, [
  { name: "Fear", labels: mins, values: curves["fear_Fear"].mean },
  { name: "Sadness", labels: mins, values: curves["fear_Sadness"].mean },
  { name: "Fear at baseline", labels: mins, values: curves["baseline_Fear"].mean },
], { x: 6.7, y: 1.45, w: 6.0, h: 4.6, chartColors: [HEX.fear, HEX.sad, HEX.base], ...lineOpts("Rest after the FEAR video") });
caption(s, "Group exponential fit: Sadness after sad video τ = 3.2 min (95% CI 1.2–6.6), Fear after fear video τ = 3.7 min (2.4–5.9); both settle at ≈ 15–17, close to the baseline level. The baseline block also drops in minute 1 and rises again after minute 14.", 6.15);

// 7 Model comparison
s = content("Most people recover exponentially, but exp vs power is not decisive", "Recovery",
  "Per usable series: constant, linear, exponential and power models compared by AICc. Bars count which model wins. In the induced target cells (sad-Sadness, fear-Fear) the exponential wins most often (31/47, 25/47). Median individual tau 1.9 min (IQR 0.9-8.8) and 3.1 min (IQR 1.5-7.3), excluding taus at the 0.1/60-min bounds. Group level: exp beats power in 80% of bootstraps for Fear but only 36% for Sadness, i.e. the Sadness curve may have a long tail (power-law-like). Baseline cells: exp also often wins because of the first-minute drop.");
const cells = ["Baseline · Sadness", "Baseline · Fear", "Sad block · Sadness", "Sad block · Fear", "Fear block · Sadness", "Fear block · Fear"];
s.addChart(pres.charts.BAR, [
  { name: "Exponential", labels: cells, values: [18, 25, 31, 15, 12, 25] },
  { name: "Power", labels: cells, values: [1, 1, 0, 0, 1, 4] },
  { name: "Linear", labels: cells, values: [18, 8, 10, 14, 19, 9] },
  { name: "Constant", labels: cells, values: [8, 6, 6, 12, 12, 9] },
], {
  x: 0.6, y: 1.45, w: 7.3, h: 5.0, barDir: "bar", barGrouping: "stacked", chartColors: [HEX.ok, "6C5B9B", HEX.base, "C9CED8"],
  showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: "0;;;", showLegend: true, legendPos: "b",
  showTitle: true, title: "Best model per series (AICc), usable series", catAxisOrientation: "maxMin", ...axisBase(), dataLabelColor: HEX.white,
});
table(s, [
  ["Group fit", "τ (min)", "95% CI", "Exp > power"],
  ["Sad block · Sadness", "3.2", "1.2–6.6", hl("36%", HEX.warn)],
  ["Fear block · Fear", "3.7", "2.4–5.9", hl("80%")],
  ["Sad block · Fear", "flat", "—", "—"],
  ["Fear block · Sadness", "flat", "—", "—"],
  ["Baseline · Sadness", "0.6", "0.3–60*", "79%"],
  ["Baseline · Fear", "0.8", "0.3–60*", "93%"],
], 8.2, 1.55, 4.5, [1.9, 0.75, 0.95, 0.9], { rowH: 0.4 });
s.addText([
  { text: "Individual τ (median): ", options: { bold: true } },
  { text: "1.9 min for Sadness, 3.1 min for Fear. Split-half reliability of τ is high in these cells (0.93, 0.90).", options: {} },
], { x: 8.2, y: 4.6, w: 4.5, h: 1.2, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
caption(s, "* Upper CI at the 60-min fitting bound: the baseline drop is too short and shallow to pin τ down. Exp > power = share of 1000 subject bootstraps in which the exponential has the lower AICc.", 6.55);

// =====================================================================
pres.addSection({ title: "Fluctuations" });
// 8 Variogram curves
const lagLbl = ["30", "50", "75", "105", "150", "210", "300", "420", "595", "875"];
const vg = {
  det: { bS: [40.5, 34.6, 36.2, 39.2, 47.0, 54.9, 49.5, 57.9, 57.6, 50.5], bF: [38.5, 41.5, 37.3, 45.0, 47.3, 45.5, 54.1, 55.8, 50.1, 49.8], sS: [28.7, 21.9, 25.9, 32.2, 35.4, 38.4, 40.8, 39.0, 34.4, 30.2], fF: [37.2, 35.9, 43.7, 46.4, 52.6, 53.9, 55.1, 54.1, 51.2, 45.1] },
  raw: { bS: [42.5, 34.9, 45.2, 48.1, 56.7, 70.1, 69.8, 88.7, 101.7, 144.0], bF: [40.4, 43.6, 47.0, 52.0, 60.3, 66.7, 81.3, 90.4, 114.3, 168.3], sS: [31.9, 23.6, 32.6, 40.1, 50.8, 66.3, 84.7, 116.1, 169.9, 279.0], fF: [40.0, 40.2, 54.5, 58.9, 83.6, 111.0, 143.5, 197.3, 266.2, 472.6] },
};
const vgSeries = (v) => [
  { name: "Sad block · Sadness", labels: lagLbl, values: v.sS },
  { name: "Fear block · Fear", labels: lagLbl, values: v.fF },
  { name: "Baseline · Sadness", labels: lagLbl, values: v.bS },
  { name: "Baseline · Fear", labels: lagLbl, values: v.bF },
];
const vgOpts = (title, max) => ({
  ...axisBase(), showTitle: true, title, valAxisMinVal: 0, valAxisMaxVal: max, chartColors: [HEX.sad, HEX.fear, "9DB0D8", "EDBE8E"],
  showLegend: true, legendPos: "b", lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 5,
  showValAxisTitle: true, valAxisTitle: "Semivariance γ", showCatAxisTitle: true, catAxisTitle: "Time between two ratings (s)",
});
s = content("Once the trend is removed, fluctuations level off within ~1–2 minutes", "Fluctuations",
  "Section C. Variogram: for every pair of ratings in a series, half the squared difference vs the time between them; binned, averaged within subject, then across subjects. Detrended = residuals after the per-series recovery curve; raw = series minus its mean. A curve that levels off means deviations are pulled back (regulated, OU-like); one that keeps rising means free drift. Detrended: all 6 cells best fit by the OU model (levels off). Raw: keeps rising (linear or power best), because the slow recovery trend and the late-block drift are still in the data. Caveat from the script: detrending removes slow variance, so the detrended tau is biased short - read the shape, not the exact tau.");
s.addChart(pres.charts.LINE, vgSeries(vg.det), { x: 0.6, y: 1.45, w: 6.0, h: 4.7, ...vgOpts("Detrended: levels off (regulated)", 70) });
s.addChart(pres.charts.LINE, vgSeries(vg.raw), { x: 6.7, y: 1.45, w: 6.0, h: 4.7, ...vgOpts("Raw: keeps rising (slow drift)", 500) });
caption(s, "γ = ½ (rating difference)². A curve that flattens means deviations are pulled back; one that keeps rising means the level wanders. Note the different y-axis ranges. Bins are labelled by their mean lag.", 6.25);

// 9 Variogram model table
s = content("About half of the short-lag variance is nugget: fast change or noise", "Fluctuations",
  "Group variogram fits (weighted by n subjects per bin), AICc. Detrended: OU best in all 6 cells; tau 0.6-2.5 min; nugget share = nugget / (nugget + sill) = 43-64% except fear-block Sadness where the fit puts nugget at 0 (likely a fit artifact; that cell has a noisy first bin). Nugget = variance at zero lag: rating noise plus change faster than the shortest interval (~20-30 s). Raw: linear free drift best in baseline and sad-Fear; power (H about 1.1-1.4, i.e. faster than linear growth) in sad-Sadness, fear-Fear, fear-Sadness, reflecting the recovery trend. Per-subject OU tau hits the fitting bound in 18-50% of series and has low reliability, so it is not used as an individual parameter. Residual lag-1 autocorrelation is near 0 (median -0.02 to 0.17).");
table(s, [
  ["Cell", "Detrended: best", "τ (min)", "Nugget share", "Raw: best"],
  ["Baseline · Sadness", hl("OU (levels off)"), "2.4", "53%", "Linear drift"],
  ["Baseline · Fear", hl("OU (levels off)"), "2.1", "64%", "Linear drift"],
  ["Sad block · Sadness", hl("OU (levels off)"), "1.3", "48%", "Power (H = 1.4)"],
  ["Sad block · Fear", hl("OU (levels off)"), "1.6", "58%", "Linear drift"],
  ["Fear block · Sadness", hl("OU (levels off)"), "0.6", "0%*", "Power (H = 1.2)"],
  ["Fear block · Fear", hl("OU (levels off)"), "1.0", "43%", "Power (H = 1.4)"],
], 0.6, 1.55, 7.6, [2.1, 1.75, 0.95, 1.3, 1.5], { rowH: 0.45 });
card(s, 8.6, 1.55, 4.1, 4.7, "How to read it", [
  ["Levels off: ", "after the trend is removed, a rating pulled away from its level comes back within 1–2 min."],
  ["Nugget ≈ ½: ", "half of the variance is faster than one probe interval, or is rating noise."],
  ["Raw drift: ", "on top, the level slowly wanders over the 20 min."],
  ["τ is a lower bound: ", "detrending removes slow variance."],
], { fontSize: 14 });
caption(s, "* Fit sets the nugget to 0 in this cell, probably an artifact of a noisy first lag bin. Per-subject OU τ hits the fitting bound in 18–50% of series, so it is not used as an individual measure.", 6.55);

// 10 Coupling
s = content("Sadness and Fear move together, even with the trend removed", "Fluctuations",
  "Section E. Per subject x block: correlation of Sadness and Fear across probes. Raw = ratings as given; residual = after removing each scale's own recovery trend. Fisher-z mean vs 0, n = 41-43 (series with a flat scale excluded). Raw r = .67 / .64 / .51 (baseline / sad / fear), residual r = .38 / .31 / .29, all p < .006. Spearman gives the same picture (residual rho .46 / .52 / .36). No condition differences (paired Fisher-z, all p > .19). Interpretation: a shared component (general negative affect, attention, or response style) drives both scales probe by probe. This fits with the dissociation analysis in the previous deck: the scales are separable but not independent.");
s.addChart(pres.charts.BAR, [
  { name: "Raw ratings", labels: ["Baseline", "After sad video", "After fear video"], values: [0.67, 0.64, 0.51] },
  { name: "Residuals (own trend removed)", labels: ["Baseline", "After sad video", "After fear video"], values: [0.38, 0.31, 0.29] },
], {
  x: 0.6, y: 1.45, w: 7.0, h: 5.0, barDir: "col", chartColors: ["6C5B9B", "B9AED6"], barGapWidthPct: 60,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: true, legendPos: "t",
  showTitle: true, title: "Sadness–Fear correlation within a block (mean Pearson r)", valAxisMinVal: 0, valAxisMaxVal: 0.8,
  ...axisBase(),
});
card(s, 8.0, 1.55, 4.7, 4.9, "Result", [
  ["All > 0: ", "p < .006 in every block, raw and residual (n = 41–43)."],
  ["Same in every block: ", "no difference between baseline, sad and fear (all p > .19)."],
  ["About half shared: ", "removing the recovery trend halves the coupling, but r ≈ .3 remains."],
  ["Reading: ", "a common factor (general negative affect or response style) moves both scales together."],
]);

// 11 Carry-over
s = content("Levels carry over between blocks, Sadness more than Fear", "Fluctuations",
  "Section F, exploratory. Start level = mean of first 3 probes in a block. Compare a block that came first (after baseline) vs second (after the other video), between subjects (Welch / Mann-Whitney). Only sad-block Sadness differs: 34.6 after the fear block vs 21.2 after baseline, Hedges d = 0.59, p = .043 (Welch), p = .061 (MW). Within subjects, the end of block 2 (last 3 probes) predicts the start of block 3: Sadness rho = .67, p < .001; Fear rho = .32, p = .026. Confounded with video sequence and fatigue; part of this is simply stable individual level (trait-like).");
s.addChart(pres.charts.BAR, [
  { name: "Block came 1st (after baseline)", labels: ["Sad block · Sadness", "Sad block · Fear", "Fear block · Sadness", "Fear block · Fear"], values: [21.2, 13.9, 26.0, 31.4] },
  { name: "Block came 2nd (after other video)", labels: ["Sad block · Sadness", "Sad block · Fear", "Fear block · Sadness", "Fear block · Fear"], values: [34.6, 17.8, 16.8, 34.8] },
], {
  x: 0.6, y: 1.45, w: 7.4, h: 5.0, barDir: "col", chartColors: [HEX.base, HEX.navy], barGapWidthPct: 60,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", showLegend: true, legendPos: "t",
  showTitle: true, title: "Start level of each block (mean of first 3 probes)", valAxisMinVal: 0, valAxisMaxVal: 45,
  showValAxisTitle: true, valAxisTitle: "VAS", ...axisBase(),
});
stat(s, 8.4, 1.6, 4.3, "d = 0.59", "Sadness starts higher in the sad block when it follows the fear block (p = .043; others n.s.)", HEX.sad);
stat(s, 8.4, 3.3, 4.3, "ρ = .67 / .32", "end of block 2 → start of block 3, Sadness / Fear (p < .001 / .026)", "6C5B9B");
caption(s, "Exploratory: block position is confounded with video sequence and fatigue, and the between-block correlation also reflects stable individual rating levels.", 6.55);

// =====================================================================
pres.addSection({ title: "Measurement quality" });
// 12 Rating behaviour
s = content("Ratings are given directly, with few corrections on the slider", "Measurement quality",
  "Section H. From button presses per rating screen (screens matched to triggers). Excess presses = presses beyond the minimum needed to reach the final value; reversal = change of direction; overshoot = how far past the final value. Means over 272 subject x block x emotion series. Very similar across conditions (excess 0.30-0.43). Split-half reliability of excess presses 0.65-0.80, and mean/SD explain little of it (R2 0.04-0.22): it is a stable, separate person trait (rating style), not emotion dynamics.");
stat(s, 0.6, 1.6, 3.8, "0.37", "excess button presses per rating (median 0.32)", HEX.sad);
stat(s, 4.75, 1.6, 3.8, "7%", "of rating screens contain a reversal (back-and-forth)", HEX.fear);
stat(s, 8.9, 1.6, 3.8, "0.65–0.80", "split-half reliability of excess presses: a stable rating style", HEX.ok);
s.addChart(pres.charts.BAR, [{ name: "Excess presses", labels: cells, values: [0.38, 0.43, 0.38, 0.30, 0.38, 0.37] }], {
  x: 0.6, y: 3.45, w: 12.1, h: 3.0, barDir: "col", chartColors: [HEX.slate], barGapWidthPct: 50,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: false,
  showTitle: true, title: "Excess presses per rating, by block and scale", valAxisMinVal: 0, valAxisMaxVal: 0.6, ...axisBase(),
});

// 13 Reliability
s = content("Recovery τ is reliable only where there is something to recover from", "Measurement quality",
  "Section I. Odd vs even probes within a series, each half fitted separately, Pearson r, Spearman-Brown corrected. Values averaged over the two cells of each group: baseline = baseline Sadness + Fear; induced target = sad-Sadness + fear-Fear; induced non-target = sad-Fear + fear-Sadness. Mean level: about 0.99 everywhere. SD: 0.94 in induced target cells, 0.67 at baseline. Exp tau: 0.92 in target cells, 0.48 at baseline (fewer valid taus, n = 21-23). Residual SD: 0.60-0.71. Excess presses: 0.69-0.73.");
s.addChart(pres.charts.BAR, [
  { name: "Baseline", labels: ["Mean", "SD", "Recovery τ", "Residual SD", "Excess presses"], values: [0.995, 0.665, 0.481, 0.638, 0.725] },
  { name: "Induced, target emotion", labels: ["Mean", "SD", "Recovery τ", "Residual SD", "Excess presses"], values: [0.996, 0.941, 0.916, 0.711, 0.711] },
  { name: "Induced, other emotion", labels: ["Mean", "SD", "Recovery τ", "Residual SD", "Excess presses"], values: [0.997, 0.828, 0.685, 0.600, 0.685] },
], {
  x: 0.6, y: 1.45, w: 8.2, h: 5.0, barDir: "col", chartColors: [HEX.base, HEX.ok, "9FD3CC"], barGapWidthPct: 50,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 9, showLegend: true, legendPos: "t",
  showTitle: true, title: "Split-half reliability (Spearman–Brown), odd vs even probes", valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0",
  ...axisBase(),
});
card(s, 9.1, 1.55, 3.6, 4.9, "Result", [
  ["Mean: ", "≈ 0.99 everywhere."],
  ["τ: ", "0.92 after the target video, 0.48 at baseline."],
  ["Residual SD: ", "0.60–0.71; usable but noisier."],
  ["Use: ", "τ as an individual measure only in the induced target blocks."],
]);

// 14 Redundancy
s = content("Most dynamic parameters are more than mean and SD in disguise", "Measurement quality",
  "Section J, following Dejonckheere et al. (2019): for each dynamic parameter, R2 explained by the series mean + SD across subjects, per cell (usable series; taus at bound excluded). Bars show the mean over the 6 cells; range in brackets. Residual SD is largely redundant (R2 0.40-0.73) - it mostly re-expresses SD. The linear slope is redundant in the induced target cells (R2 about 0.71, through SD) but not elsewhere. Coupling, variogram sill/tau, rating-behaviour measures, recovery tau and AC1 share at most ~20% with mean/SD on average.");
const red = [["Sad–Fear coupling (resid.)", 0.09], ["Variogram sill", 0.10], ["Slider reversals", 0.10], ["Excess presses", 0.10], ["Variogram τ", 0.12], ["Recovery τ", 0.18], ["Residual AC1", 0.20], ["Variogram nugget", 0.21], ["Recovery amplitude", 0.22], ["Linear slope", 0.33], ["Residual SD", 0.57]];
s.addChart(pres.charts.BAR, [{ name: "R²", labels: red.map((r) => r[0]), values: red.map((r) => r[1]) }], {
  x: 0.6, y: 1.45, w: 7.8, h: 5.0, barDir: "bar", chartColors: red.map((r) => (r[1] >= 0.3 ? HEX.warn : HEX.ok)), barGapWidthPct: 35,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: false,
  showTitle: true, title: "Variance explained by series mean + SD (R², mean of 6 cells)", valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisLabelFormatCode: "0.0",
  ...axisBase(),
});
card(s, 8.8, 1.55, 3.9, 4.9, "Result", [
  ["Redundant: ", "residual SD (R² 0.40–0.73); linear slope in the induced target blocks (≈ 0.71)."],
  ["Mostly new information: ", "coupling, recovery τ, variogram parameters, slider behaviour (R² ≤ 0.22 on average)."],
  ["For 3_ and 4_: ", "enter mean and SD as covariates when testing a dynamic parameter."],
]);

// =====================================================================
pres.addSection({ title: "Summary" });
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Summary" });
s.addText("Take-home", { x: 0.8, y: 0.5, w: 11.7, h: 0.9, fontSize: 36, bold: true, color: C.background1, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
const take = [
  ["Induction works and is balanced.", "Both videos are equally specific; Fear is slightly more intense."],
  ["Recovery takes about 3–4 min.", "The induced emotion settles to a plateau near baseline level (τ ≈ 3–4 min); the other emotion stays flat."],
  ["After that, ratings still move.", "Deviations are pulled back within ~1–2 min, about half the variance is fast or noise, and the level slowly drifts over the block."],
  ["Sadness and Fear move together.", "r ≈ .3 even with the trend removed, the same in every block."],
  ["Usable individual parameters:", "recovery τ in the induced target blocks, coupling and rating style are reliable and not just mean/SD."],
];
take.forEach((t, i) => {
  const y = 1.6 + i * 0.95;
  s.addShape(pres.shapes.OVAL, { x: 0.8, y, w: 0.55, h: 0.55, fill: { color: i % 2 ? C.accent2 : C.accent1 }, line: { type: "none" }, objectName: "take-home " + (i + 1) });
  s.addText(String(i + 1), { x: 0.8, y, w: 0.55, h: 0.55, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText([{ text: t[0] + " ", options: { bold: true, color: HEX.white } }, { text: t[1], options: { color: "C9D0DE" } }], { x: 1.6, y: y - 0.1, w: 11.0, h: 0.8, fontSize: 16, valign: "middle", margin: 0, isTextBox: true });
});
s.addText("Next: mixed models (3_behav_models.R) · MEG link (4_meg_behav_link.py) · HMM states (section G, not shown) · check shared interval sequences", { x: 0.8, y: 6.5, w: 11.7, h: 0.5, fontSize: 14, italic: true, color: "AEB6C7", margin: 0, isTextBox: true });
s.addNotes("Summary. Open points: the HMM results (section G) were not in this upload; the shared interval sequences (99 unique of 150) should be checked before interpreting probe reactivity in 3_; the plateau vs baseline comparison should be tested formally in the mixed model.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
