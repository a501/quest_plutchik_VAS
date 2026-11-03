// Neural results write-up (Nature Human Behaviour style): main findings, confirmatory,
// complementary; supplementary tables from data.json (prep.py)
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, BorderStyle, PageBreak, LevelFormat, Footer, PageNumber,
} = require("docx");
const D = JSON.parse(fs.readFileSync(__dirname + "/data.json", "utf8"));
const OUT = process.argv[2] || __dirname + "/Intrinsic_emotions_at_rest_neural_results.docx";

const FONT = "Arial";
const W = 9360; // content width (DXA), US Letter with 1" margins

// ---------- text helpers ----------
// t("plain **bold** *italic*") -> runs
function runs(s, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(new TextRun({ text: s.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), italics: true, ...base }));
    last = m.index + tok.length;
  }
  if (last < s.length) out.push(new TextRun({ text: s.slice(last), ...base }));
  return out;
}
const P = (s, opts = {}) => new Paragraph({ children: runs(s), spacing: { after: 140, line: 300 }, ...opts });
const H1 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(s)] });
const H2 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(s)] });
const H3 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(s)] });
const BUL = (s) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(s), spacing: { after: 80, line: 280 } });
const CAP = (s) => new Paragraph({ children: runs(s, { size: 18 }), spacing: { before: 60, after: 200 } });
const BREAK = () => new Paragraph({ children: [new PageBreak()] });

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFC5CF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: head ? { fill: "232B45", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ children: [new TextRun({ text: txt === null || txt === undefined ? "" : String(txt), size: 17, bold: !!head, color: head ? "FFFFFF" : "1F2433", font: FONT })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((v, i) => cell(v, i, false)) }))],
  });
}
const fmt = (x) => (x === null || x === undefined ? "" : (typeof x === "number" ? x.toFixed(2).replace(/^-?0\./, (m) => m.replace("0.", ".")) : x));
const r2 = (x) => (x === null || x === undefined ? "" : x.toFixed(2));
const MEAS = { LZC: "LZC", SampEn_m2: "Sample entropy", SpectralSlope: "Spectral slope", IRASA_slope: "IRASA slope", ACW50: "ACW50", IRASA_oscpower: "Oscillatory power", SpectralEntropy: "Spectral entropy", ACW0: "ACW0", MedianFreq: "Median frequency", Spec_lowhigh_8Hz: "Low/high power (8 Hz)", MSE_costa_m2_AUC: "Multiscale entropy", ACF_area_to_zero: "ACF area" };
const NET = { DMN: "DMN", Frontoparietal: "Frontoparietal", Salience_Interoceptive: "Salience/interoceptive", Sensory_control: "Sensory/control" };
const BLK = { baseline: "Baseline", sad: "Sad block", fear: "Fear block" };
const MET = { mean_flip: "mean_flip", pca_flip: "pca_flip" };

// ---------- numbers used in the text ----------
const adj = Object.fromEntries(D.confadj.map((r) => [r[0] + "|" + r[1], r]));
const fb = (meth, cond, fac, beh) => D.fb.find((r) => r[0] === meth && r[1] === cond && r[2] === fac && r[3] === beh);
const f1b = fb("mean_flip", "baseline", "F1", "Latent SD (partial on mean)");
const f1s = fb("mean_flip", "sad", "F1", "Latent SD (partial on mean)");
const f1f = fb("mean_flip", "fear", "F1", "Latent SD (partial on mean)");
const f1bp = fb("pca_flip", "baseline", "F1", "Latent SD (partial on mean)");
const f1sp = fb("pca_flip", "sad", "F1", "Latent SD (partial on mean)");
const sdb = fb("mean_flip", "baseline", "F1", "SD");
const sds = fb("mean_flip", "sad", "F1", "SD");
const st = Object.fromEntries(D.stability.map((r) => [r[0] + "|" + r[1], r]));
const rho = (x) => "ρ = " + (x < 0 ? "−" : "") + Math.abs(x).toFixed(2);
const p = (s) => (s.startsWith("<") ? "P " + s.replace("<", "< 0") : "P = 0" + s);

// =====================================================================
const body = [];
body.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Intrinsic emotions at rest: neural dynamics")] }));
body.push(P("*Results, part 2 (MEG). Continuation of the behavioural results. Draft for Nature Human Behaviour; numbers from scripts 4, 10, 12 and 13 (mean_flip source extraction; pca_flip in the Supplement).*"));

body.push(H1("Results"));
body.push(P("The behavioural results showed that self-reported Sadness and Fear are not flat at rest: they fluctuate beyond rating noise, and after an emotional film the induced emotion decays to its resting level within minutes (Fear: τ = 5.2 min, 95% CI 3.2–8.5; Sadness: τ = 10.3 min, 3.5–24.9), whereas the non-induced emotion does not change. Here we asked whether these spontaneous emotional dynamics are reflected in the intrinsic dynamics of the brain. We summarised source-reconstructed MEG during the same rest periods (19-s windows before each rating) with 12 established measures of neural dynamics — intrinsic timescales, aperiodic (1/f) slope, signal complexity and spectral composition — in four large-scale networks, and related them to the emotional dynamics of the same participants (n = 47 with MEG; 45–46 per block with both measures)."));

body.push(H2("Main findings"));
body.push(H3("Resting brain dynamics form two stable, whole-brain dimensions"));
body.push(P(`The 12 measures reduced to two dimensions (parallel analysis: eigenvalues 5.80 and 4.51 versus 2.20 and 1.82 expected from random data; third 0.72 versus 1.59), which together explained 86% of the between-person variance (Supplementary Table S3). After varimax rotation, **Factor 1 (neural complexity)** contrasted an irregular, complex signal with a flat aperiodic spectrum (Lempel–Ziv complexity loading 0.98, sample entropy 0.97) against a steep aperiodic slope (spectral slope −0.87, IRASA slope −0.85), long autocorrelation half-width (−0.79) and strong oscillatory power (−0.67). **Factor 2 (spectral speed)** contrasted power at higher frequencies and broadband activity (median frequency 0.96, multiscale entropy 0.87, spectral entropy 0.82) against slow, alpha-dominated activity with long autocorrelation (low/high power ratio −0.95, ACF area −0.82, ACW0 −0.65). Both factors were nearly identical when refitted in the two film blocks (Tucker congruence 0.97–0.99) and with the alternative source-extraction method (0.998). They were also stable traits of a person: factor scores correlated ρ = ${r2(st["mean_flip|F1"][2])}–${r2(st["mean_flip|F1"][3])} (Factor 1) and ρ = ${r2(st["mean_flip|F2"][2])}–${r2(st["mean_flip|F2"][3])} (Factor 2) between the baseline and the film blocks. People did not fall into distinct neural types; the factor scores formed a single continuum (Gaussian mixtures: one class preferred by BIC; two-class solutions not reproducible, adjusted Rand index 0.49).`));

body.push(H3("Lower neural complexity accompanies larger spontaneous emotional fluctuations"));
body.push(P(`People whose resting brain activity was less complex and had a steeper aperiodic slope (low Factor 1) showed larger spontaneous fluctuations of their emotional ratings. At baseline, before any film, Factor 1 correlated with the noise-corrected fluctuation amplitude of the ratings, controlling for the mean rating level (partial ${rho(f1b[5])}, ${p(f1b[6])}, n = ${f1b[4]}; observed SD: ${rho(sdb[5])}, ${p(sdb[6])}). The association was present in the sad block (${rho(f1s[5])}, ${p(f1s[6])}; SD: ${rho(sds[5])}, ${p(sds[6])}) but not in the fear block (${rho(f1f[5])}, ${p(f1f[6])}). Factor 2 was not associated with any emotional measure (all P > 0.06; Supplementary Table S5). The same association appeared measure by measure in every network: fluctuation amplitude increased with steeper aperiodic slopes (IRASA slope ρ = 0.46–0.49, spectral slope ρ = 0.40–0.46, all P ≤ 0.006) and decreased with signal complexity (sample entropy ρ = −0.34 to −0.41, Lempel–Ziv complexity ρ = −0.33 to −0.37; Supplementary Table S7).`));

body.push(H3("Neural change after the film tracks how long the induced emotion persists"));
body.push(P(`In an exploratory analysis we related the change of each neural measure from baseline to the film blocks to the decline of the induced emotion within the block (from the exponential recovery model). People whose aperiodic slope became steeper after the film showed a smaller decline of the induced emotion, that is, the emotion persisted longer (IRASA slope change: ρ = −0.35 in the DMN and the salience/interoceptive network, −0.35 in the sensory/control network, P = 0.015–0.017), and an increase in complexity and spectral entropy went with a larger decline (ρ = 0.33–0.37). ${D.v2_H4_count.mean_flip + D.v2_H4_count.pca_flip} of 192 network × measure × method tests reached P < 0.05 (10 expected by chance). This pattern mirrors the trait association: a less complex, steeper-sloped brain state goes with more lasting emotional states. The pre-specified test for this hypothesis (salience/interoceptive ACW0) was not significant (ρ = −0.18, P = 0.22), so this finding requires confirmation.`));

body.push(H3("Each rest period begins in a distinct neural state"));
body.push(P(`Behaviourally, the first rating of every rest period was elevated — after the films for the induced emotion over 4–5 ratings (about 3 min), and at baseline for both emotions for one rating only (Sadness 27.8 versus 16.1, d_z = 0.77; Fear 25.5 versus 14.2, d_z = 0.64). The brain showed a parallel start state. In the 19 s before the first rating of the baseline block, oscillatory power was lower and complexity higher than later in the block, in all four networks (oscillatory power, sample entropy, multiscale entropy and ACW50 in 4 of 4 networks; Lempel–Ziv complexity and spectral entropy in 3 of 4; ${D.first_counts.baseline} of 48 tests significant after FDR correction). The same, weaker pattern was present at the start of the sad (${D.first_counts.sad} of 48) and fear blocks (${D.first_counts.fear} of 48). However, the size of the neural start effect did not predict the size of the behavioural first-rating elevation across people (pre-specified test ρ = −0.04, P = 0.80; 0 of 96 exploratory tests at P < 0.05), suggesting that both reflect a general orienting state at the onset of rest rather than a shared emotional process.`));

body.push(H2("Confirmatory analyses"));
body.push(H3("Robustness to non-normal data"));
body.push(P("Ten of the 12 MEG measures deviated from normality (Shapiro–Wilk P < 0.05; skew up to 5.7, single participants up to 6.4 s.d. from the mean). All Gaussian models therefore used rank-based inverse-normal transformed measures (afterwards one measure deviated, largest |z| 3.0; Supplementary Table S2), and all brain–behaviour tests were rank-based. The factor structure and the complexity–fluctuation association were unchanged with log-transformed or untransformed data (congruence ≥ 0.97; ρ = −0.35 to −0.46)."));
body.push(H3("Robustness to the source-extraction method"));
body.push(P(`Results replicated with the alternative extraction of regional time series (first principal component, pca_flip; main text: mean of sign-aligned vertices, mean_flip). Across all analyses the two methods produced similar numbers of significant results (for example ${D.v2_H2_count.mean_flip} versus ${D.v2_H2_count.pca_flip} for fluctuation amplitude) and near-identical effect sizes (correlation of test statistics between methods r = 0.98). The complexity–fluctuation association with pca_flip was ${rho(f1bp[5])} (${p(f1bp[6])}) at baseline and ${rho(f1sp[5])} (${p(f1sp[6])}) in the sad block (Supplementary Tables S5, S13).`));
body.push(H3("The association is not explained by non-neural factors"));
body.push(P(`We tested whether head position in the helmet, overall signal size, heart rate (from the ECG), blink rate (from the EOG) or the number of ICA components explained the association. Factor 1 was strongly related to overall signal size (magnetometers ρ = ${fmt(D.confcorr[4][1])}, P < 0.001) and moderately to the number of ICA components (ρ = ${fmt(D.confcorr[8][1])}), but not to head position, head–helmet distance (ρ = ${fmt(D.confcorr[3][1])}), heart rate or blink rate (Supplementary Table S6). Controlling for all of these together left the complexity–fluctuation association intact or stronger (baseline: ${rho(adj["mean_flip|baseline"][6])}, ${p(adj["mean_flip|baseline"][7])}; sad block: ${rho(adj["mean_flip|sad"][6])}, ${p(adj["mean_flip|sad"][7])}; same n = 45). Because the link between low complexity and large signal amplitude is expected physiologically (strong slow oscillations increase amplitude and reduce complexity) and is unrelated to head–sensor distance, we interpret Factor 1 as a property of brain activity rather than of the recording.`));
body.push(H3("Pre-specified tests"));
body.push(P(`Of seven pre-specified hypotheses (one network × measure each, Holm-corrected), only the fluctuation-amplitude hypothesis reached nominal significance (DMN Lempel–Ziv complexity, ρ = −0.33, P = 0.026, P_Holm = 0.18), in the same direction as the factor result; the hypotheses on rating presence, fluctuation types, induction, start-of-rest, Sadness–Fear coupling and within-person coupling were not supported (Supplementary Table S8). The pre-registered link between the DMN intrinsic timescale (ACW0) at baseline and the recovery time constant of the induced emotion was not supported (r = −0.05 to 0.00, n = 39; Supplementary Table S12).`));

body.push(H2("Complementary analyses"));
body.push(H3("Network-specific dynamics"));
body.push(P(`About 9% of each measure's between-person variance was specific to one network (range 5–12% across measures; Supplementary Table S9); the rest was shared across the whole brain. These network-specific deviations were stable within a person across blocks (ρ = 0.60–0.90, median 0.79), indicating individual network profiles rather than noise. Their relation to the ratings beyond the whole-brain level was weak (${D.nethit_count[0]} of ${D.nethit_count[1]} tests at P < 0.05, about 12 expected). Two patterns recurred: a relatively faster, less alpha-dominated DMN (Factor 2) went with a larger first-rating elevation (positive in all six block × method combinations, ρ = 0.26–0.35, P < 0.05 in three), and a more complex sensory/control network (Factor 1) with a higher rating level in the sad block (ρ = 0.30–0.39). Both are exploratory.`));
body.push(H3("No transient brain states at the 40-s resolution"));
body.push(P(`A hidden Markov model on the window-by-window factor scores (centred within each block) favoured five states by held-out likelihood. The states differed little (≤ 0.44 s.d.), were persistent (probability of remaining in a state ${r2(D.hmmstay[0])}–${r2(D.hmmstay[1])}), and ${D.hmmsw.mean_flip[0]}% of blocks contained at least one state change (median three changes in those blocks). State occupancy and switching were unrelated to the emotional dynamics (${D.hmmbeh[0]} of ${D.hmmbeh[1]} tests at P < 0.05), and the state before a rating did not predict that rating (${D.hmmbeh[2]} of ${D.hmmbeh[3]}). With one window every 20–60 s, the model captures slow shifts rather than fast state transitions.`));
body.push(H3("Window-by-window prediction of the next rating"));
body.push(P(`Within persons, the MEG window before a probe did not reliably predict the change of the next rating (network-level models: ${D.m2_counts[0]} of ${D.m2_counts[1]} at P < 0.05, smallest FDR-corrected P = ${fmt(D.m2_counts[2])}; the most consistent signal was lower DMN multiscale entropy before a rise in Fear, β = −0.32, P = 0.002, uncorrected). A multivariate decoder using all regions and measures did not exceed chance (out-of-sample r ≤ 0.04, permutation P ≥ 0.08; Supplementary Table S12). The neural correlates of emotional dynamics are therefore found between people (trait-like), not from moment to moment at this temporal resolution.`));

body.push(H2("Summary"));
body.push(BUL("Resting MEG dynamics organise into two robust, trait-like whole-brain dimensions: complexity versus steep aperiodic slope, and spectral speed."));
body.push(BUL("Lower complexity and steeper aperiodic slopes accompany larger spontaneous emotional fluctuations at rest and after a sad film, independent of head position, signal size, heart rate and blinks."));
body.push(BUL("After the films, a shift towards this low-complexity state goes with a more persistent induced emotion (exploratory)."));
body.push(BUL("Rest periods begin in an alert, desynchronised neural state, in parallel with — but not proportional to — the first-rating elevation."));
body.push(BUL("About 9% of the variance is network-specific and stable; network-specific links to emotion are exploratory. No neural types, no fast state switching, no moment-to-moment prediction of the next rating."));

body.push(H2("Open points before submission"));
body.push(BUL("Three levels of self (interoceptive, exteroceptive, mental-self regions): analysis implemented (10_ and 13_ with --roi-set self), not yet run."));
body.push(BUL("Add participant 052 (triggers recovered) and the incomplete sad block of 036; rerun scripts 4, 10, 12, 13."));
body.push(BUL("Decide the main extraction method a priori (currently mean_flip; pca_flip in the Supplement)."));
body.push(BUL("Continuous-time hidden Markov model with the actual probe intervals (deferred)."));
body.push(BUL("Multiple comparisons: main findings are uncorrected exploratory associations except where stated; confirm in an independent sample or by a pre-registered replication."));

// =====================================================================
const supp = [BREAK(), H1("Supplementary Information")];
supp.push(H2("Supplementary Methods (summary)"));
supp.push(P("**Data.** 47 participants with source-reconstructed MEG (baseline block; 46 in each film block). Each block contained about 25 VAS probes (Sadness and Fear); the MEG window was the 19 s before each probe. Snippet k of a block was aligned to probe k + 1 of the same block after numbering snippets within block (alignment check: snippet and probe counts equal; blocks failing the check were excluded from window-level analyses)."));
supp.push(P("**MEG measures.** Intrinsic timescale (ACW0, ACW50, ACF area to zero), aperiodic slope (IRASA slope, spectral slope), complexity (Lempel–Ziv complexity, sample entropy, multiscale entropy AUC) and spectral composition (low/high power ratio at 8 Hz, median frequency, IRASA oscillatory power, spectral entropy). Values were averaged over regions within four networks (DMN, salience/interoceptive, frontoparietal, sensory/control) and over windows within a block; for factor analysis, also over networks."));
supp.push(P("**Source extraction.** mean_flip: mean of all vertices of a region after sign alignment by the cortical surface normals. pca_flip: first spatial principal component of the region, fitted jointly across the three blocks, sign-anchored to mean_flip and scaled to the average vertex power."));
supp.push(P("**Behaviour.** Per participant and block, mean of the two scales: noise-corrected fluctuation amplitude (latent SD), observed SD, mean level, share of non-zero ratings, first-rating elevation (first minus rest of block) and the decline of the induced emotion (exponential recovery model). Unconfirmed (timed-out) ratings are valid ratings."));
supp.push(P("**Statistics.** Measures were rank-based inverse-normal transformed before Gaussian models. Number of factors by parallel analysis (95th percentile of 1,000 random data sets), varimax rotation; film blocks scored with the baseline model. Brain–behaviour associations: Spearman or partial Spearman correlations (ranks residualised on covariates); fluctuation amplitude always controlled for the mean level. Confounds: head position (device–head transform), signal size (median RMS over the middle 60 s), heart rate (R-peaks from the ECG per minute; values outside 40–150 bpm set missing), blink rate (EOG), number of ICA components. Latent profiles: Gaussian mixtures with 1–5 classes, BIC, minimum class size 5, bootstrap stability (adjusted Rand index ≥ 0.60). Hidden Markov model: Gaussian emissions, number of states by held-out log-likelihood (participants in five folds) with every state ≥ 5% of windows. Exploratory grids are reported uncorrected; pre-specified tests Holm-corrected; group-level window tests FDR-corrected."));

supp.push(H2("Table S2. Distribution of the MEG measures (baseline, mean_flip)"));
supp.push(table(["Measure", "Skew (raw)", "Shapiro P (raw)", "Max |z| (raw)", "Skew (transf.)", "Shapiro P (transf.)", "Max |z| (transf.)"],
  D.dist.map((r) => [MEAS[r[0]], fmt(r[1]), r[2], r[3], fmt(r[4]), r[5], r[6]]), [2100, 1180, 1240, 1180, 1180, 1300, 1180]));
supp.push(CAP("Rank-based inverse-normal transform pooled over participants, blocks and networks per method × measure."));

supp.push(H2("Table S3. Parallel analysis and factor loadings"));
supp.push(table(["Component", "Eigenvalue (mean_flip)", "Random 95%", "Eigenvalue (pca_flip)", "Random 95%"],
  D.parallel.map((r, i) => [r[0], r[1], r[2], D.parallel_pca[i][1], D.parallel_pca[i][2]]), [1500, 2000, 1900, 2000, 1960]));
supp.push(new Paragraph({ spacing: { after: 120 } }));
supp.push(table(["Measure", "F1 mean_flip", "F2 mean_flip", "F1 pca_flip", "F2 pca_flip"],
  D.loadings.map((r) => [MEAS[r[0]], fmt(r[1]), fmt(r[2]), fmt(r[3]), fmt(r[4])]), [2760, 1650, 1650, 1650, 1650]));
supp.push(CAP("Varimax-rotated loadings. F1, neural complexity; F2, spectral speed. Congruence with refits in the sad and fear blocks: " + D.congruence.filter((r) => r[2] !== "pca_flip").map((r) => `${r[0]} ${r[1]} ${r[2].replace(" block refit", "")} ${r[3]}`).join("; ") + "; between methods 0.998."));

supp.push(H2("Table S4. Stability and block shift of the factor scores"));
supp.push(table(["Method", "Factor", "ρ baseline–sad", "ρ baseline–fear", "Shift sad (s.d.)", "P", "Shift fear (s.d.)", "P"],
  D.stability.map((r) => [r[0], r[1], fmt(r[2]), fmt(r[3]), fmt(r[4]), r[5], fmt(r[6]), r[7]]), [1300, 800, 1250, 1250, 1250, 900, 1250, 1360]));
supp.push(CAP("Shift = film-block minus baseline factor score (Wilcoxon). The Factor 1 shift was the same whether the film block came second or third (P > 0.2) and was unrelated to the strength of the induced emotion (first rating, ρ = 0.10–0.20, n.s.)."));

supp.push(H2("Table S5. Factor scores and emotional dynamics"));
supp.push(table(["Method", "Block", "Factor", "Behaviour", "n", "ρ", "P"],
  D.fb.map((r) => [r[0], BLK[r[1]], r[2], r[3], r[4], fmt(r[5]), r[6]]), [1250, 1250, 800, 2860, 700, 1000, 1500]));
supp.push(CAP(`Spearman (partial Spearman for latent SD, controlling the mean level). P < 0.05: ${D.fb_count.mean_flip[0]}/${D.fb_count.mean_flip[1]} (mean_flip), ${D.fb_count.pca_flip[0]}/${D.fb_count.pca_flip[1]} (pca_flip), all Factor 1 with fluctuation amplitude.`));

supp.push(H2("Table S6. Confounds"));
supp.push(table(["Confound", "Median", "Min", "Max", "n blocks"], D.confdesc.map((r) => [r[0], r[1], r[2], r[3], r[4]]), [3360, 1500, 1500, 1500, 1500]));
supp.push(new Paragraph({ spacing: { after: 120 } }));
supp.push(table(["Confound (baseline)", "ρ with F1", "P", "ρ with F2", "P"], D.confcorr.map((r) => [r[0], fmt(r[1]), r[2], fmt(r[3]), r[4]]), [3360, 1500, 1500, 1500, 1500]));
supp.push(new Paragraph({ spacing: { after: 120 } }));
supp.push(table(["Method", "Block", "n", "F1 × latent SD", "P", "n", "+ all confounds", "P"],
  D.confadj.map((r) => [r[0], BLK[r[1]], r[2], fmt(r[3]), r[4], r[5], fmt(r[6]), r[7]]), [1200, 1300, 700, 1500, 1000, 700, 1660, 1300]));
supp.push(CAP("Partial Spearman controlling the mean level, then additionally head position (x, y, z, distance), signal size (magnetometers, gradiometers), heart rate, blink rate and number of ICA components. Heart rate of one participant (failed R-peak detection) and of one participant without ECG set missing."));

for (const [h, title] of [["H2", "Table S7a. Single measures: fluctuation amplitude at baseline (latent SD, partial on mean; mean_flip)"], ["H4", "Table S7b. Single measures: change from baseline vs decline of the induced emotion (mean_flip)"]]) {
  const T = D["v2_" + h];
  supp.push(H2(title));
  supp.push(table(["Measure", ...T.groups.map((g) => NET[g])], T.rows.map((r) => [MEAS[r[0]], ...r.slice(1).map((c) => `${fmt(c[0])} (${c[1]})`)]), [2160, 1800, 1800, 1800, 1800]));
  supp.push(CAP(`ρ (P). P < 0.05: ${D["v2_" + h + "_count"].mean_flip}/48 mean_flip, ${D["v2_" + h + "_count"].pca_flip}/48 pca_flip.`));
}

supp.push(H2("Table S8. Pre-specified hypotheses"));
supp.push(table(["Hypothesis", "Description", "Network", "Measure", "n", "Statistic", "P", "P Holm"],
  D.v1.map((r) => [r[0], r[1], NET[r[2]] || r[2], MEAS[r[3]] || r[3], r[4], fmt(r[5]), r[6], r[7]]), [700, 3000, 1300, 1260, 500, 900, 800, 900]));
supp.push(CAP("Exploratory grids (96 tests each), P < 0.05: " + Object.entries(D.v2_counts).map(([h, v]) => `${h} ${v[0]}/${v[1]}`).join(", ") + ". H3 statistic: AUC-type Mann–Whitney; H7: repeated-measures correlation."));

supp.push(H2("Table S9. Network-specific variance, stability and associations"));
supp.push(table(["Measure", "Network-specific variance, mean_flip (%)", "pca_flip (%)"], D.netvar.map((r) => [MEAS[r[0]], r[1], r[2]]), [3360, 3000, 3000]));
supp.push(new Paragraph({ spacing: { after: 120 } }));
supp.push(table(["Network", "Factor", "Stability ρ min", "max", "median"], D.netstab.map((r) => [NET[r[0]], r[1], fmt(r[2]), fmt(r[3]), fmt(r[4])]), [2760, 1200, 1800, 1800, 1800]));
supp.push(new Paragraph({ spacing: { after: 120 } }));
supp.push(table(["Method", "Block", "Network", "Factor", "Behaviour", "ρ", "P"], D.nethits.map((r) => [r[0], BLK[r[1]], NET[r[2]], r[3], r[4], fmt(r[5]), r[6]]), [1200, 1200, 2000, 800, 2160, 900, 1100]));
supp.push(CAP(`Network deviation (network minus whole-brain factor score) vs behaviour, controlling for the whole-brain score; all ${D.nethit_count[0]} of ${D.nethit_count[1]} tests with P < 0.05.`));

supp.push(H2("Table S10. Latent profiles"));
supp.push(table(["Method", "Classes", "BIC", "Smallest class", "Stability (ARI)", "Passes"], D.profiles.map((r) => [r[0], r[1], r[2], r[3], r[4] === null ? "" : fmt(r[4]), r[5]]), [1600, 1100, 1500, 1700, 1800, 1660]));

supp.push(H2("Table S11. Hidden Markov model"));
supp.push(table(["Method", "States", "BIC", "Held-out log-lik. per window", "Smallest occupancy", "Selected"], D.hmmsel.map((r) => [r[0], r[1], r[2], r[3], fmt(r[4]), r[5]]), [1500, 900, 1300, 2500, 1900, 1260]));
supp.push(CAP(`Blocks with ≥ 1 state change: ${D.hmmsw.mean_flip[0]}% (mean_flip), ${D.hmmsw.pca_flip[0]}% (pca_flip); median dwell ${D.hmmsw.mean_flip[1]} and ${D.hmmsw.pca_flip[1]} windows. The model treats windows as equally spaced although probe intervals vary (20–60 s).`));

supp.push(H2("Table S12. Neural timescale and recovery (M1), window-level prediction (M2) and decoding (M3)"));
supp.push(table(["Film block", "Emotion", "Method", "n", "r (DMN ACW0 × log τ)", "P"], D.m1.map((r) => [r[0], r[1], r[2], r[3], fmt(r[4]), r[5]]), [1500, 1300, 1500, 900, 2600, 1560]));
supp.push(CAP(`Pre-registered primary test. All ${D.m1_counts[1]} between-person tests: ${D.m1_counts[0]} at P < 0.05, smallest FDR-corrected P = ${D.m1_counts[2]}.`));
supp.push(table(["Emotion", "Method", "Network", "Measure", "β", "P", "P FDR"], D.m2_top.map((r) => [r[0], r[1], NET[r[2]], r[3], fmt(r[4]), r[5], fmt(r[6])]), [1100, 1200, 1700, 2000, 1000, 1100, 1260]));
supp.push(CAP(`Six strongest of ${D.m2_counts[1]} within-person models (rating change ~ preceding MEG window, previous rating, time, block; participant-clustered errors); ${D.m2_counts[0]} at P < 0.05.`));
supp.push(table(["Method", "Emotion", "Windows", "Participants", "r out-of-sample", "Null 95%", "P perm"], D.m3.map((r) => [r[0], r[1], r[2], r[3], r[4], r[5], r[6]]), [1300, 1200, 1300, 1400, 1600, 1260, 1300]));
supp.push(CAP("Ridge regression on all regions × measures, 10-fold cross-validation by participant, 200 within-participant permutations."));

supp.push(H2("Table S13. Robustness: pca_flip"));
supp.push(P("All tables above report both extraction methods where applicable; pca_flip reproduced the factor structure (congruence 0.998), the complexity–fluctuation association (Table S5, S6) and the start-of-rest effects (sad block 14, fear block 8 of 48 FDR-significant)."));

const doc = new Document({
  creator: "quest_plutchik_VAS", title: "Intrinsic emotions at rest: neural dynamics",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 36, bold: true, color: "232B45" }, paragraph: { spacing: { before: 0, after: 160 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 30, bold: true, color: "232B45" }, paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 25, bold: true, color: "2A9D8F" }, paragraph: { spacing: { before: 260, after: 110 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 22, bold: true, color: "1F2433" }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "8A93A6" })] })] }) },
    children: [...body, ...supp],
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT); });
