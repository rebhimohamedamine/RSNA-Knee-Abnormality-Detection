# Knee MRI Multi-Label Classification

Research-grade, MRNet-inspired, hierarchical (slice → series → study) deep
learning system for predicting 12 knee-MRI abnormalities from a full study
(multiple series, variable slice counts, series metadata). Report text is
used only as auxiliary weak supervision during training; the model runs on
MRI + series metadata alone at inference.

## Status

Training and inference run on Kaggle GPU, per the competition. The local
machine has no CUDA and no real Kaggle DICOM data, so it's used only for
writing and unit-testing code (per the spec's local-vs-Kaggle split) — the
whole pipeline is verified locally against small synthetic DICOM series
generated in `tests/` before ever touching Kaggle. No Kaggle run, no
training-to-convergence, and no ablation numbers have happened yet; those
follow once this pass's `pytest -q` is green and the code is pushed to a
Kaggle notebook.

## Targets

12 independent sigmoid outputs (multi-label, not multi-class): ACL, MCL,
Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion,
Synovitis, Baker's, Contusion, Fracture. Primary metric: macro ROC-AUC
(mean of the 12 per-target AUCs).

## Data reality

Only 58 of 4407 studies in the local `data/train.csv` carry ground-truth
labels; the rest are all-NaN across the 12 columns and carry only a free-text
`Report`. Loss and metrics are mask-aware per label — a missing label is
excluded from that row's loss/metric, never treated as a confident negative.
This is also why report weak-supervision (see `src/reports/`) matters: it's
close to the only training signal for ~99% of local rows.

A separate, pre-existing notebook (`rsna-knee-read-the-report-then-the-knee.ipynb`)
lives at the repo root with its own report-centric approach; it predates this
project pass, is left untouched, and the code here is implemented
independently of it.

## Setup

```bash
pip install -r requirements.txt
pytest -q
```

`pytest -q` is the smoke test: it builds tiny synthetic DICOM studies on the
fly and runs the full preprocess → train → evaluate → predict pipeline on
CPU in well under a minute. No real data or GPU required.

## Layout

```
src/data/        DICOM loading, preprocessing, caching, study Dataset
src/models/      slice encoder -> slice attention -> series encoder (+metadata)
                 -> cross-series attention -> study encoder -> pathology
                 adapters -> 12 task heads, plus the auxiliary report branch
src/training/    trainer, losses, metrics, optimizer/scheduler, callbacks
src/reports/     report weak-label extraction (synonym mapping + negation)
src/inference/   MRI+metadata-only inference and submission-file writing
src/utils/       seeding, config loading (YAML with _base_ inheritance), logging
configs/         one YAML per experiment (M1 baseline .. M7 full model)
scripts/         thin CLIs: preprocess.py / train.py / evaluate.py / predict.py
tests/           unit + end-to-end smoke tests against synthetic DICOM data
```

## Experiment progression (M1 → M7)

One shared model (`src/models/knee_model.py`), config-flagged, not seven
separate architectures. See `configs/base.yaml` for the full flag set and
`configs/m2_slice_attention.yaml` .. `configs/final.yaml` for each step:

| Config | Slice Attn | Metadata | Cross-Series | Pathology | Report |
|---|---|---|---|---|---|
| `baseline.yaml` (M1) | ✗ | ✗ | ✗ | ✗ | ✗ |
| `m2_slice_attention.yaml` | ✓ | ✗ | ✗ | ✗ | ✗ |
| `m3_metadata.yaml` | ✓ | ✓ | ✗ | ✗ | ✗ |
| `m4_cross_series.yaml` | ✓ | ✓ | ✓ | ✗ | ✗ |
| `m5_pathology.yaml` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `m6_report_weak.yaml` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `final.yaml` (M7, + distillation) | ✓ | ✓ | ✓ | ✓ | ✓ |

`scripts/evaluate.py` appends one row per run to
`outputs/metrics/ablation_table.csv`, read straight off the config flags, so
this table can be regenerated from saved runs without rerunning anything.

## Known limitations

- Report weak-label synonym/negation coverage is English plus partial
  Spanish/Dutch inferred from the reports actually present in
  `data/train.csv` — not exhaustive multilingual coverage. Unmapped
  languages degrade recall rather than crashing.
- Not yet run against real DICOM data or on GPU; Kaggle orchestration
  (`kernel-metadata.json`) is not yet updated for this codebase.
