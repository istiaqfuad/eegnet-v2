# Aligned EEGNet + IM-TTA — Motor Imagery decoding (BCI IV-2a & IV-2b)

A compact pipeline for motor-imagery EEG: align each recording's covariance, run it through a
small EEGNet-style network, then adapt the model to the target subject at test time. It runs on
two BCI Competition IV datasets — IV-2a (4-class, 22 channels) and IV-2b (2-class, 3 channels),
9 subjects each — using the usual session-holdout and leave-one-subject-out (LOSO) protocols.

## Results

| Method | IV-2a within | IV-2a LOSO | IV-2b within | IV-2b LOSO |
|---|---|---|---|---|
| baseline (no alignment) | 75.6 | 59.1 | 85.7 | 76.1 |
| + RA alignment | 81.9 | 64.8 | **87.2** | 76.6 |
| **+ RA + IM-TTA** | **84.9 ± 0.8** | **67.8 ± 0.4** | 86.1 ± 0.2 | **80.2 ± 0.2** |

Within-subject uses the official session split (IV-2a: train session 1, test session 2; IV-2b:
train sessions 1–3, test sessions 4–5). LOSO trains on eight subjects and tests on the held-out
ninth. A validation split carved from the training data picks the checkpoint; the test set is
scored once at the end.

Alignment helps in every setting. The test-time adaptation step is where the cross-subject gains
come from — roughly +3 points on both datasets' LOSO. Within-subject it helps IV-2a and comes out
about even with plain alignment on IV-2b. For context, recent within-subject IV-2a results:
CTNet 82.5, MSCARNet 82.7, EEGEncoder 86.5. Full tables and ablations are in
[RESULTS.md](RESULTS.md).

## How it works

1. **Alignment** — whiten each recording by its mean spatial covariance, either Euclidean (`ea`)
   or Riemannian/centroid (`ra`). It's computed from the signals alone, so it applies the same way
   to training and test data.
2. **Backbone** — `UnifiedEEGNet`, a small EEGNet-style CNN (~34k parameters).
3. **IM-TTA** — at inference, the model adapts to the target subject's unlabelled trials by
   information maximization: it sharpens each prediction while keeping the overall class mix
   balanced, which stops it from collapsing onto a single class. This is the main driver of the
   cross-subject improvement.

[ARCHITECTURE.md](ARCHITECTURE.md) walks through the whole pipeline in detail.

## Reproduce

```bash
uv sync
# both protocols at once, the RA + IM-TTA setup
python main.py --dataset iv2a --protocol both --align ra --tta_steps 5 --tta_div 1.0 --seed 42
python main.py --dataset iv2b --protocol both --align ra --tta_steps 5 --tta_div 1.0 --seed 42
```

Flags: `--dataset {iv2a,iv2b}` · `--protocol {within,loso,both}` · `--align {none,ea,ra}` ·
`--tta_steps N` · `--tta_div {1 = info-max, 0 = entropy-only}` · `--seed`.

## Layout

- `model.py` — `UnifiedEEGNet` and the alignment-model variants
- `dataset.py` — MOABB loader, dataset registry (iv2a/iv2b), session-holdout splits, EA/RA alignment
- `train.py` — training loop, validation-based checkpointing, `tent_adapt` (the test-time adaptation)
- `main.py` — entry point: datasets, protocols, alignment, TTA, multi-seed runs
- `ARCHITECTURE.md` / `RESULTS.md` — method writeup and full results
- `results/` — per-run CSVs and training logs (see `results/README.md`)
