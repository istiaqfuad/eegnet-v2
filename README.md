# Aligned EEGNet + IM-TTA — Motor Imagery (BCI IV-2a & IV-2b)

Honest, **leak-free** motor-imagery decoding. Standard session-holdout protocols (no session
pooling); model selection on a held-out validation split; test evaluated **exactly once**.
Evaluated on BCI IV-2a (BNCI2014-001, 4-class, 22 ch) and IV-2b (BNCI2014-004, 2-class, 3 ch),
9 subjects each.

## Results (honest, leak-free)

| Method | IV-2a within | IV-2a LOSO | IV-2b within | IV-2b LOSO |
|---|---|---|---|---|
| baseline (no align) | 75.6 | 59.1 | 85.7 | 76.1 |
| + RA alignment | 81.9 | 64.8 | **87.2** | 76.6 |
| **+ RA + IM-TTA** | **84.9 ± 0.8** | **67.8 ± 0.4** | 86.1 ± 0.2 | **80.2 ± 0.2** |

Within = session holdout (IV-2a 1→2; IV-2b 1–3→4–5). LOSO = leave-one-subject-out.
**IM-TTA gives a consistent cross-subject gain** (LOSO +3.0 / +3.6); within-subject it helps
IV-2a (+3.0) and is neutral on IV-2b (RA-alone best). Alignment helps everywhere.
Reference SOTA (IV-2a within, 4-class): CTNet 82.52 · MSCARNet 82.66 · EEGEncoder 86.46.
Full tables, ablations (align×TTA, Tent-vs-IM, steps) and source CSVs: **[RESULTS.md](RESULTS.md)**.

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
# within + cross-subject, RA + IM-TTA (best), both protocols at once
python main.py --dataset iv2a --protocol both --align ra --tta_steps 5 --tta_div 1.0 --seed 42
python main.py --dataset iv2b --protocol both --align ra --tta_steps 5 --tta_div 1.0 --seed 42
```

Key flags: `--dataset {iv2a,iv2b}` · `--protocol {within,loso,both}` · `--align {none,ea,ra}` ·
`--tta_steps N` · `--tta_div {1=IM,0=plain Tent}` · `--seed`.

## Files

- `model.py` — `UnifiedEEGNet` (backbone) + alignment-model variants
- `dataset.py` — MOABB loader + dataset registry (iv2a/iv2b), leak-free session-holdout splits, EA/RA alignment
- `train.py` — training loop, val-based selection, `tent_adapt` (IM-TTA)
- `main.py` — entry point (datasets, protocols, alignment, TTA, multi-seed)
- `ARCHITECTURE.md` / `RESULTS.md` — method + full results/ablations
- `results/` — all ablation CSVs + training logs, namespaced by box (preserved for the paper; see `results/README.md`)

## Eval integrity

Session-holdout within-subject (IV-2a: train session 1 / test session 2; IV-2b: train sessions
1–3 / test sessions 4–5 — sessions never pooled) and leave-one-subject-out cross-subject. A
validation split carved from training data drives all model selection; test is seen once.
Alignment references are computed without labels (transductive, no label leakage).
