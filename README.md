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

## Running on Kaggle

This is [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
(RSNA, $77,000, ROC-AUC evaluation — matches this repo's primary metric),
verified via the Kaggle API: it's a **code competition**
(`is_kernels_submissions_only`) with a max of 5 daily submissions, and its
real DICOM layout is confirmed to exactly match what `src/data/` already
assumes: `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`.
The train split alone has 4,407 studies / 24,371 series.

Attach the competition to a Kaggle notebook with a GPU, then:

```bash
!git clone https://github.com/rebhimohamedamine/RSNA-Knee-Abnormality-Detection.git
%cd RSNA-Knee-Abnormality-Detection
!pip install -r requirements.txt -q

# 1) Sanity check against real data first: 150 real studies, M1, a few epochs.
!python scripts/preprocess.py --config configs/kaggle_smoke.yaml --split both
!python scripts/train.py --config configs/kaggle_smoke.yaml
!python scripts/evaluate.py --config configs/kaggle_smoke.yaml --checkpoint checkpoints/kaggle_smoke/best.pt
!python scripts/predict.py --config configs/kaggle_smoke.yaml --checkpoint checkpoints/kaggle_smoke/best.pt --out /kaggle/working/submission.csv

# 2) Once that's green, move to a real run -- baseline first, per the M1..M7 progression.
!python scripts/preprocess.py --config configs/kaggle_baseline.yaml --split both
!python scripts/train.py --config configs/kaggle_baseline.yaml
!python scripts/evaluate.py --config configs/kaggle_baseline.yaml --checkpoint checkpoints/kaggle_m1_baseline/best.pt
!python scripts/predict.py --config configs/kaggle_baseline.yaml --checkpoint checkpoints/kaggle_m1_baseline/best.pt --out /kaggle/working/submission.csv
```

`configs/kaggle_smoke.yaml` / `kaggle_baseline.yaml` / `kaggle_final.yaml`
point `paths.*` at the real Kaggle mount (`/kaggle/input/competitions/rsna-knee-abnormality-detection/...`)
and `/kaggle/working/{cache,checkpoints,outputs}`, and set `data.max_studies`
to a deterministic subset (see `src/data/dataset.py::_subsample_studies`) so
you're never forced to run against all 24,371 series just to validate the
pipeline. Set `data.max_studies: null` for a full-data run once you're ready.

**Disk**: the series-tensor cache is `~series_count * max_slices * image_size²
* 4 bytes`. Caching every series in the full training set at the default
`image_size: 256` (~153 GB) will not fit a standard Kaggle instance --
`kaggle_baseline.yaml`/`kaggle_final.yaml` use `image_size: 160, max_slices: 20`
instead (~31 GB full-dataset cache). Only `checkpoints/` and `outputs/` (a
few MB of weights/JSON/CSV) need to survive a "Save Version" — `cache/` is
disposable; delete it or point `cache_root` elsewhere if you're short on the
output quota.

**Before your first *real* submission** (not just interactive development),
check this competition's Code Requirements tab for whether internet access
is disabled during the scored run — code competitions commonly require it.
If so, the workflow above (which needs internet for `git clone`/`pip
install`) is for *training* only: train interactively with internet on,
save the resulting checkpoint as a Kaggle Dataset, then use a second,
minimal, internet-off notebook that installs nothing beyond what's
preinstalled, loads that checkpoint, and runs `scripts/predict.py` to
produce the graded submission.

## Known limitations

- Report weak-label synonym/negation coverage is English plus partial
  Spanish/Dutch inferred from the reports actually present in
  `data/train.csv` — not exhaustive multilingual coverage. Unmapped
  languages degrade recall rather than crashing.
- Verified against real Kaggle DICOM structure via the Kaggle API (see
  "Running on Kaggle" above) but not yet actually executed on Kaggle/GPU;
  `kernel-metadata.json` at the repo root belongs to the separate
  pre-existing notebook, not this codebase.
