# Results — BCI Competition IV-2a (4-class Motor Imagery)

All numbers are **honest, leak-free** test accuracies. Model selection and early stopping
use a held-out **validation** split carved from the training data; the **test set is
evaluated exactly once** on the val-selected checkpoint (`train.py::train_model`). Source
CSVs for every table are listed and archived under `results_archive/{fedora,hcilab}/`.
**Do not delete any `results_*.csv` — they are the paper's ablation evidence.**

Backbone: `UnifiedEEGNet` (EEGNet-style, 34,132 params). Dataset: BNCI2014-001, 9 subjects,
22 channels, 1000 samples @ 250 Hz, broadband 0.5–100 Hz (MOABB `MotorImagery`).

---

## 1. Headline

| Setting | Protocol | Accuracy |
|---|---|---|
| **Within-subject** | session 1 → train/val, session 2 → test | **82.67% ± 0.63** (EA, 6 seeds) |
| **Cross-subject** | leave-one-subject-out (LOSO) | **67.83% ± 0.42** (RA + IM-TTA, 3 seeds) |

Cross-subject novelty: **source-free Information-Maximization Test-Time Adaptation (IM-TTA)**
on alignment-whitened EEGNet. Lifts LOSO from 59.1 → 67.8 (**+8.7 pp**).

---

## 2. Within-subject (session-to-session)

Train = session 1 (stratified 80/20 train/val), Test = session 2. Cross-subject supervised
pretraining on session 1 of all subjects (no test-session leakage).

| Config | Mean | Source |
|---|---|---|
| Baseline UnifiedEEGNet (no alignment) | 75.6 (seed42); ~74.8 multi-seed | `*_base9*.csv` |
| **+ Euclidean Alignment (EA)** | **82.67 ± 0.63** (6 seeds) | `*_ea9*.csv` |
| + Riemannian/Centroid Alignment (RA) | 81.87 (seed42) | `*_ra9.csv` |

EA per-subject (seed 42): S1 86.8 · S2 71.5 · S3 95.8 · S4 84.0 · S5 78.5 · S6 67.0 · S7 84.4 · S8 88.5 · S9 87.5.

Reference SOTA (same 4-class protocol): CTNet 82.52 · MSCARNet 82.66 · EEGEncoder 86.46.
EA-EEGNet (82.67) is competitive and leak-free.

### 2.1 Within-subject ablations (negative / neutral — justify "EA, not X")
Probe on subjects {2,3,5}; baseline (no align) on those = 62.5 / 89.2 / 67.0.

| Lever | S2 | S3 | S5 | Verdict |
|---|---|---|---|---|
| band-pass 4–40 Hz | 52.1 | 91.0 | 55.6 | hurts (broadband best) |
| band-pass 8–30 Hz | 48.6 | 86.5 | 42.4 | hurts more |
| aug-expand ×3 | 60.1 | 89.2 | 57.6 | hurts |
| refit on train+val | — | — | — | hurts (exp1) |
| learnable spatial align (on EA) | 74.3 | 94.8 | 76.4 | neutral vs EA (71.5/95.8/78.5) |
| per-trial adaptive align (on EA) | 67.4 | 93.8 | 70.1 | hurts |
| dual EA+RA fusion | 74.7 | 94.8 | 72.6 | hurts vs EA |
| FreqAware gated-freq-attn + EA | negative | | | hurts |

Conclusion: within-subject is saturated at ~82.7%; EA is the decisive lever, learnable/adaptive
spatial-alignment and frequency tricks do not improve on it.

---

## 3. Cross-subject (LOSO) — main contribution

Train on 8 subjects (both sessions, stratified val), test on the held-out subject (both
sessions), per-subject alignment, trained from scratch.

### 3.1 Alignment × TTA matrix (full-9)
| Alignment | no TTA | + IM-TTA |
|---|---|---|
| none | 59.12 | 65.53 (+6.4) |
| EA | 64.58 | 66.68 ± 0.30 (3 seeds) |
| **RA** | 64.12–64.80 | **67.83 ± 0.42 (3 seeds)** |

- Alignment alone: +5.5 (EA) / +5.7 (RA) over no-align.
- IM-TTA alone (no align): +6.4.
- Combined (RA + IM-TTA): **+8.7** over baseline — the two contributions are complementary.

RA+IM-TTA seeds: 67.28 / 67.96 / 68.25. Per-subject (seed 42, RA → RA+IM-TTA):
S1 74.7→79.5 · S2 50.5→49.3 · S3 85.9→86.3 · S4 53.1→57.3 · S5 50.2→58.0 · S6 57.3→58.0 ·
S7 67.4→71.9 · S8 74.8→78.0 · S9 63.2→67.4. IM-TTA improves 8/9 subjects.

Sources: `results_loso_unified.csv` (none), `*_lea9/_leatta9*` (EA), `*_lra9/_lratta9*` (RA),
`*_lnonetta9` (none+IM-TTA).

### 3.2 TTA-design ablation — the diversity term is essential
IM-TTA loss = E[H(p)] − div·H(E[p]); `div=0` is plain Tent (entropy-only).

| TTA loss | LOSO (EA) | Source |
|---|---|---|
| no TTA | 64.58 | `*_lea9` |
| plain Tent (div=0) | **64.51** | `*_leatent9` |
| IM-TTA (div=1) | **66.68** | `*_leatta9*` |

Plain entropy-min ≈ no TTA (the anti-collapse diversity term is what converts TTA into a
+2.1 pp gain; without it, prediction collapse on hard subjects cancels the benefit).

### 3.3 TTA-steps sensitivity (RA + IM-TTA, seed 42)
| steps | 1 | 5 | 10 | 20 |
|---|---|---|---|---|
| LOSO | 66.78 | 67.28 | 68.19 | 67.98 |

Accuracy rises gently with the adaptation budget and plateaus around steps 10–20; the method
is not sensitive to the exact step count (steps=5 used throughout; 10–20 marginally best).
Sources: `_rs1`, `_lratta9` (steps=5), `_rs10`, `_rs20`.

---

## 4. Method (novelty)

`tent_adapt` (`train.py`): label-free, source-free test-time adaptation. After training on the
aligned source subjects, adapt **only the BatchNorm affine parameters** on the held-out
subject's **unlabelled** trials (BN on batch statistics) by minimising
**L = E[H(p)] − H(E[p])** (Information Maximization / SHOT-style). The marginal-entropy term
prevents the collapse that plain Tent suffers on hard MI subjects. No labels are used →
transductive and honest; natural for LOSO where the target subject's unlabelled EEG is
available. First application of source-free IM test-time adaptation to cross-subject MI-BCI,
combined with Euclidean/Riemannian alignment.

CLI: `--align {none,ea,ra}` · `--tta_steps N` · `--tta_div {1=IM,0=Tent}`.

---

## 5. Reproduce

```bash
# within-subject (EA), multi-seed
python main.py --model unified --protocol within --align ea --seed 42
# cross-subject best (RA + IM-TTA)
python main.py --model unified --protocol loso --align ra --tta_steps 5 --tta_div 1.0 --seed 42
# ablations
python main.py --model unified --protocol loso --align none --tta_steps 5            # none+IM-TTA
python main.py --model unified --protocol loso --align ea --tta_steps 5 --tta_div 0  # plain Tent
```

Seeds 42/1/2 used for multi-seed means. Hardware: RTX 2070 SUPER (fedora), GTX 1050 Ti (hci-lab);
EA removes cross-GPU variance (results agree across boxes).
