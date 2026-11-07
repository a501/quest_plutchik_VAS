# Pending instructions for the next update of the results document

Apply ONLY when the user explicitly says to update the results doc
("Intrinsic_emotions_at_rest_neural_results.docx"). Recorded 2026-11-07.

1. **Self-contained supplementary legends.** Every Supplementary table gets an
   extensive legend: what the rows/columns are, the data (block, extraction
   method, n), the statistic (Spearman / partial Spearman and covariates),
   transforms, sign conventions (e.g. slope sign: larger = steeper), abbreviations,
   and the multiple-comparison status, so a reader understands the table from the legend alone.
2. **Label every subtable.** Tables with several parts (S3, S6, S9, S12, ...)
   get lettered subtables with their own subtitle (e.g. Table S3a Parallel
   analysis; Table S3b Varimax factor loadings) AND their own legend.
3. **Factor loadings table** stays in the tables (currently the unlabelled 2nd part of S3 -> becomes S3b).
4. **Nature Human Behaviour style**: house writing style and formatting
   (table layout, colours, headings, legends), per NHB author guidelines.
5. **Figures at 300 dpi**, so that subfigures are sharp too.
6. Text fixes found during review (2026-11-07):
   - Latent profiles: "two-class solutions not reproducible (ARI 0.49)" holds
     for mean_flip only; pca_flip 2-class passed stability (ARI 0.65) but BIC
     still preferred 1 class. Reword; explain "continuum" (unimodal, graded
     differences, no subtypes).
   - Network grouping (4_meg_behav_link.py ROI_GROUPS regex): 5 ROIs assigned
     to two networks, 27/67 ROIs unassigned (incl. Primary_motor/auditive,
     all EMO_ temporal/TPJ/vlPFC ROIs). Fix before rerun / describe in Methods.
   - ACW50 loads on F1 with the aperiodic slope; only ACW0 loads on F2. Say so
     explicitly in the text (users may read "timescale" as one thing).
