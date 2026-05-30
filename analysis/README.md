# Analysis Map

This folder collects the paper-analysis code and organizes it by function.
The goal is to keep the original figure logic intact while making the important entrypoints easier to find.

## Paper Figure Map

- Figures 1–2: graphical abstract and study overview (produced externally, not scripted)
- Figure 3a: `subgroups/analyze_subgroups.py --mode age_bmi`
- Figure 3b: `subgroups/analyze_subgroups.py --mode sex`
- Figure 3c: `subgroups/analyze_subgroups.py --mode contrast`
- Figure 3d: `subgroups/analyze_subgroups.py --mode orientation`
- Figure 3e-f: `figures/plot_totalseg_comparison.py`
- Figure 4a-b: `figures/plot_dose_scatter.py`
- Figure 4c-d: `dose/analyze_dvh_errors.py`
- Figure 5a,c,e: `figures/plot_dose_case_lung.py`
- Figure 5b,d,f: `figures/plot_dvh_lung.py`

## Structure

- `metrics/`: segmentation scoring utilities and CSV generation
- `dose/`: dose-metric computation, QA, and statistical analysis
- `subgroups/`: robustness analyses for age/BMI, sex, contrast, breast orientation, and multi-model contrast comparisons
- `figures/`: figure-specific plotting scripts that stay split when panel logic is highly customized
- `stats/`: summary comparison and ablation scripts used for paper-level plots and tables
- `metadata/`: local metadata inputs used by subgroup and paper-analysis scripts
- `conversion/`: generic DICOM/RTSTRUCT conversion helpers
- `results/`: local archived analysis outputs, intentionally gitignored
  This also includes generated inference folders and Platipy outputs such as `analysis/results/platipy_results/`.

## Notes

- The figure scripts remain split where panel composition is highly case-specific; this preserves the original paper logic.
- Some case-visualization scripts include placeholder case IDs and local result names; update these values to match your local dataset before running them.
- The subgroup and scoring scripts were consolidated because they were structurally duplicated and differed mainly by metadata or path configuration.
- Analysis scripts assume the centralized dataset root at `data/AllDatasets`.
- The consolidated subgroup defaults follow the paper framing and default to the Balanced model (`run1_plus_cnc64_bnorm`).
- `txseg_mrn_redaction_map.md` is a local-only internal mapping file; it is gitignored and should not be pushed to GitHub.
