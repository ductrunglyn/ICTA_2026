# TransVal-Dep

**Transferable-Validity Depression detection** — a multi-corpus,
confound-controlled, calibration-first benchmark for depression detection from
clinical interviews.

This repository is the production-grade implementation of the design document
[`TransValDep_Design.md`](TransValDep_Design.md). It asks: *which depression
signals are genuinely about depression (not general distress) and genuinely
transfer across corpora/languages?*

## Five contributions

| # | Contribution | Where |
|---|---|---|
| NV1 | Transferable-validity benchmark (leakage-free, participant-level CV; pooled + LOCO) | `src/data/splitter.py`, `src/train/cv_runner.py` |
| NV2 | Confound-controlled evaluation (group AUC, residualized AUC, specificity-gap) | `src/eval/confound_eval.py` |
| NV3 | Invariant representation learning (DANN adversary + Group-DRO + optional IRM) | `src/models/`, `src/losses/` |
| NV4 | Calibration-first + selective prediction (Platt/Isotonic, risk-coverage, AURC) | `src/calibration/` |
| NV5 | Sufficiently-powered statistics (multi-seed bootstrap CI, TOST equivalence) | `src/eval/stats.py` |

Plus a revived **question-type validity probe** (`src/eval/probe.py`).

## Layout

```
configs/      default + per-corpus + per-experiment configs, prompt2qtype.yaml
src/
  data/       segmentation, features, confounds, MIL dataset, leakage-free splitter, bag builder
  models/     encoders, GRL, domain adversary, heads, TransValNet
  losses/     Group-DRO, IRM penalty, intermodal consistency
  calibration/calibrators (Platt/Isotonic), selective prediction
  train/      per-fold trainer, multi-seed CV runner
  eval/       metrics, confound evaluator, statistics, question-type probe
  utils/      seeding, logging, config, registry
scripts/      00_build_manifests, 01_extract_features, 02_run_cv, 03_make_report
tests/        splitter, GRL, calibrator, model-forward smoke tests
```

## Pipeline

```bash
# 0) Build the participant-level manifest from corpus label files.
python scripts/00_build_manifests.py --corpora configs/corpora.yaml

# 1) Segment + extract & cache per-segment features.
python scripts/01_extract_features.py --manifest data/manifests/all.csv

# 2) Run an experiment (5-fold x 5-seed leakage-free CV).
python scripts/02_run_cv.py --experiment configs/experiments/E2_corpus_adv.yaml

# 3) Build the EvaluationCard (+ optional TOST vs a baseline).
python scripts/03_make_report.py --exp E2_corpus_adv --baseline E0_acoustic_only
```

## Tests

The model/training logic is verified without external data:

```bash
pip install -r requirements.txt
pytest -q
```

> **External data note.** DAIC-WOZ / E-DAIC / EATD / Androids corpora and the
> frozen backbones (XLS-R, XLM-R) are **not** distributed here. Feature
> extraction (`scripts/01`) is guarded so it degrades gracefully (missing
> modalities are masked, never imputed). Places needing the actual corpora are
> marked with `TODO(external-data)`.
