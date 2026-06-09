# Architecture & Novelty

This document describes the actual model and method in `model.py` / `dataset.py` / `train.py`.
(The previous version of this file described an obsolete `MTSEEGFormer` that no longer exists.)

Task: 4-class motor-imagery decoding on BCI Competition IV-2a (BNCI2014-001), 9 subjects,
22 EEG channels, 1000 samples @ 250 Hz, broadband 0.5–100 Hz.

The system has three parts, applied in order:

```
 raw EEG [B,22,1000]
   │
   ▼  (1) ALIGNMENT  — label-free covariance whitening (EA or RA/Centroid)
 aligned EEG [B,22,1000]
   │
   ▼  (2) BACKBONE   — UnifiedEEGNet (EEGNet-style CNN, 34k params)
 logits [B,4]
   │
   ▼  (3) IM-TTA     — source-free test-time adaptation on the target subject (cross-subject only)
 prediction
```

---

## 1. Alignment front-end (label-free, `dataset.py`)

EEG covariance drifts across sessions and subjects — the dominant source of error in
cross-session/cross-subject MI. We whiten each recording by a covariance **reference** so its
trials are mapped toward a common geometry. Reference is computed **without labels**, so it is
honest (no test-label leakage) and applies equally to train, validation and test.

For a set of trials, compute the per-channel reference whitener `M` and apply `X' = M · X`:

- **Euclidean Alignment (EA)** — `M = R^{-1/2}`, `R = mean_n (X_n X_nᵀ / T)` (arithmetic mean of
  per-trial spatial covariances). Maps the arithmetic-mean covariance to the identity.
- **Riemannian / Centroid Alignment (RA)** — `M = invsqrtm( mean_riemann(cov_n) )`, the SPD
  geometric (Fréchet) mean of OAS-shrunk covariances (`pyriemann`). Respects the curved geometry
  of the SPD manifold; better conditioned when covariances are heterogeneous.

Where the reference is fit (always label-free):
- **Within-subject**: session-1 reference fit on the training split only (applied to train+val);
  session-2 (test) reference fit on its own trials.
- **Cross-subject (LOSO)**: each subject whitened by its own reference (its centroid → identity).

Effect (measured): within-subject **+7 pp** (75.6 → 82.7); cross-subject **+5.5 pp** (59.1 → 64.6).
Alignment is the single largest honest lever and the foundation the rest builds on.

---

## 2. Backbone — `UnifiedEEGNet` (`model.py`, 34,132 params)

A compact EEGNet-style CNN — deliberately small for the ~288 trials/subject regime.

```
Input [B,1,22,1000]
  Conv2d(1→8, kernel=(1,64), pad=(0,31))      temporal filters (~mu/beta cycles @250Hz)
  BatchNorm2d(8)
  DepthwiseConv2d(8→16, kernel=(22,1), groups=8)   spatial filters over channels (CSP-like)
  BatchNorm2d(16) → ELU → AvgPool(1,4) → Dropout(0.35)
  Conv2d(16→32, (1,16), pad=(0,7)) → BN → ELU → AvgPool(1,8) → Dropout(0.35)
  Conv2d(32→32, (1,8),  pad=(0,3)) → BN → ELU → AvgPool(1,4) → Dropout(0.35)
  AdaptiveAvgPool(1,8) → Flatten(256) → Linear(256→64) → GELU → Dropout(0.5) → Linear(64→4)
```

Why it works **with** alignment: the depthwise spatial conv learns CSP-like spatial filters; on
EA/RA-whitened input these operate in a whitened space, so they behave like CSP-after-whitening —
the classical recipe, now learned end-to-end. Training: AdamW, OneCycleLR, label smoothing 0.1,
MixUp, S&R + signal-space augmentation, **checkpoint/early-stop on a held-out validation split**,
**test evaluated exactly once** (leak-free).

Cross-subject supervised pretraining (session 1 of all subjects) initialises the within-subject
fine-tuning; for LOSO the model is trained from scratch on the 8 source subjects.

---

## 3. Novel module — Information-Maximization Test-Time Adaptation (`tent_adapt`, `train.py`)

**The contribution.** Alignment removes covariance shift but a residual subject gap remains —
strongest on hard subjects. After the model is trained on the (aligned) source subjects, we adapt
it to the held-out target subject using **only that subject's unlabelled trials** (which are
legitimately available in the LOSO setting — transductive, no labels touched).

Adapt **only the BatchNorm affine parameters** (γ, β); BN uses **batch statistics** (running stats
disabled). For each pass over the target's unlabelled trials, minimise the **Information
Maximization** objective:

```
  L  =  E[ H(p) ]            (conditional entropy — sharpen each prediction)
      −  H( E[p] )           (marginal entropy   — keep classes balanced)
```

where `p = softmax(f(x))`, `E[·]` over the batch.

**Why both terms (the key design):** plain entropy minimisation (Tent) collapses on hard MI
subjects — predictions pile onto one class (confirmation bias), and the gain vanishes. The second
term **maximises the batch marginal entropy**, forbidding collapse to a single class. The
`--tta_div` weight toggles it; the ablation is decisive:

| TTA on aligned EEGNet | LOSO acc |
|---|---|
| no TTA | 64.6 |
| plain Tent (entropy only, div=0) | 64.5  ← no gain (collapse) |
| **IM-TTA (entropy + diversity, div=1)** | **66.7** |

Cost: a few gradient steps on a handful of BN parameters; one forward model at test time
(no ensembling). Helps **8/9** subjects.

---

## 4. Full pipelines

One unified pipeline — **RA-align → UnifiedEEGNet → IM-TTA** — is best on both protocols.

**Within-subject** (per subject): RA-align → pretrained UnifiedEEGNet → fine-tune (val-selected) →
IM-TTA on session-2 unlabelled trials → single test eval. **84.88% ± 0.83** (3 seeds).
(EA alignment without TTA gives 82.7; IM-TTA adds +3.0.)

**Cross-subject (LOSO)**: RA-align all subjects → train UnifiedEEGNet on 8 source subjects
(val-selected) → IM-TTA on the held-out subject's unlabelled trials → single test eval.
**67.8% ± 0.4** (3 seeds); baseline → here = **+8.7 pp** (59.1 → 67.8).

---

## 5. Novelty statement (honest)

- **EEGNet backbone** — standard (Lawhern 2018); not claimed as novel.
- **EA / RA alignment** — established transfer-learning tools (He & Wu 2020; Riemannian/Centroid
  alignment); not claimed as novel — used as a strong, principled baseline lever.
- **Information-Maximization test-time adaptation for cross-subject MI-BCI** — the contribution.
  IM/entropy adaptation exists in unsupervised domain adaptation (Tent, Wang 2021; SHOT,
  Liang 2020). The novelty here is **(a)** the first source-free IM test-time adaptation applied to
  cross-subject motor-imagery BCI, **(b)** combined with Euclidean/Riemannian alignment as a two-stage
  align-then-adapt pipeline, and **(c)** the empirical finding that the anti-collapse diversity term
  is *necessary* for hard MI subjects (plain Tent gives no gain), with a full align×TTA ablation
  showing the two stages are complementary (+5.7 align, +6.4 TTA-alone, +8.7 combined).

Framing: an **applied / methods** contribution (new method combination + mechanism finding for a
domain), not a brand-new optimisation objective. All evaluation is leak-free and reported honestly;
see `RESULTS.md` for tables and source CSVs.
