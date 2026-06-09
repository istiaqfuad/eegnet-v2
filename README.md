# Aligned EEGNet + IM-TTA — BCI Competition IV-2a (4-class Motor Imagery)

Honest, **leak-free** motor-imagery decoding on BCI IV-2a (BNCI2014-001, 9 subjects, 22 ch).
Model selection / early stopping use a held-out validation split; the test set is evaluated
**exactly once**.

## Results (honest)

| Protocol | Method | Accuracy |
|---|---|---|
| Within-subject (session 1 → 2) | EA + UnifiedEEGNet | **82.7% ± 0.6** (+IM-TTA: 83.3) |
| Cross-subject (LOSO) | RA + UnifiedEEGNet + IM-TTA | **67.8% ± 0.4** (baseline 59.1, **+8.7 pp**) |

Reference SOTA (within, 4-class): CTNet 82.52 · MSCARNet 82.66 · EEGEncoder 86.46.

## Method (one architecture, three stages)

1. **Alignment** — label-free covariance whitening (Euclidean `ea` or Riemannian/Centroid `ra`).
2. **Backbone** — `UnifiedEEGNet`, an EEGNet-style CNN (34k params).
3. **IM-TTA** — novel source-free Information-Maximization test-time adaptation on the target
   subject's unlabelled trials (entropy + anti-collapse diversity). Biggest lever cross-subject.

Full description in **[ARCHITECTURE.md](ARCHITECTURE.md)**; all tables, ablations and source
CSVs in **[RESULTS.md](RESULTS.md)**.

## Reproduce

```bash
uv sync
# within-subject (EA)
python main.py --model unified --protocol within --align ea --seed 42
# cross-subject (RA + IM-TTA) — best LOSO
python main.py --model unified --protocol loso --align ra --tta_steps 5 --tta_div 1.0 --seed 42
```

Key flags: `--protocol {within,loso}` · `--align {none,ea,ra}` · `--tta_steps N` ·
`--tta_div {1=IM,0=plain Tent}` · `--seed`.

## Files

- `model.py` — `UnifiedEEGNet` (backbone) + alignment-model variants
- `dataset.py` — MOABB loader, leak-free splits, EA/RA alignment
- `train.py` — training loop, val-based selection, `tent_adapt` (IM-TTA)
- `main.py` — entry point (protocols, alignment, TTA, multi-seed)
- `ARCHITECTURE.md` / `RESULTS.md` — method + results
- `results_archive/` — all ablation CSVs + training logs (preserved for the paper; do not delete)

## Eval integrity

Session-based within-subject (train session 1, test session 2) and leave-one-subject-out
cross-subject. Validation split carved from training data drives all model selection; test is
seen once. Alignment references are computed without labels (transductive, no label leakage).
