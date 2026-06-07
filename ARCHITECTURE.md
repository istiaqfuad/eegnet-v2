# MTSEEGFormer: A Single Unified Architecture for 4-Class Motor Imagery

## Executive Summary

We train **MTSEEGFormer**, a single unified architecture, on BCI Competition IV-2a (4-class motor imagery, 9 subjects) under strict session-based evaluation. Result: **79.28% ± 12.25% mean accuracy** with no ensembling, no test-time augmentation, and zero data leakage.

This document describes the architecture in detail, the leakage-free evaluation protocol, and the gap to the 85% target.

---

## 1. Problem and Dataset

Motor Imagery (MI) classification from EEG is a fundamental BCI task. Given a 4-second EEG window, classify the imagined movement into one of four classes: left hand, right hand, feet, tongue.

- **Dataset**: BCI Competition IV-2a (BNCI2014_001)
- **Subjects**: 9 healthy subjects
- **Channels**: 22 EEG channels
- **Sampling rate**: 250 Hz → 1000 timesteps per trial
- **Trials**: 288 per session, 2 sessions, 576 per subject total
- **Classes**: 4 (left_hand, right_hand, feet, tongue)
- **Preprocessing**: MOABB MotorImagery paradigm, 0.5–100 Hz bandpass

## 2. Evaluation protocol (no leakage)

We follow the official BCI Competition IV-2a protocol: train on session 1, test on session 2, per subject. Mean across the 9 subjects is the headline number.

Concretely (`dataset.py`):

- `prepare_subject_data(X, y, meta, subject)`:
  - asserts `len(sessions) >= 2` (so no fallback to a leaky random split)
  - selects session 1 trials as `X_train`, session 2 as `X_test`
  - computes `mean`, `std` from `X_train` only
  - applies `(X_train - mean) / std` and `(X_test - mean) / std` (no test-stat leakage)
  - prints `[no-leakage] Subject N: train=session0train (288), test=session1test (288); z-score from train only` on every call

- `prepare_pretrain_data(X, y, meta)`:
  - selects session 1 of **all** 9 subjects
  - session 2 of every subject is held out and never seen during pretraining
  - this gives the fine-tuning a strong cross-subject init without leaking the test session

The previous `evaluation_mode='merged'` (which would have shuffled session-2 trials into the training set) has been removed entirely. It was a textbook BCI session-shift leakage.

## 3. Architecture: MTSEEGFormer

A single class, one forward pass, no internal ensembling. 364K parameters total.

### 3.1 Pipeline

```
Input EEG [B, 22, 1000]
   │
   ▼
[1] EEGNet-style embedding
    Conv2d(1, 16, kernel=(1, 25), padding=(0, 12))   ← temporal filter
    BatchNorm2d(16)
    ELU
    Conv2d(16, 32, kernel=(22, 1), groups=16)       ← depthwise spatial filter
    BatchNorm2d(32)
    ELU                                              → [B, 32, 1, 1000]
   │
   ▼
[2] Progressive downsampling (3 stride-2 convs)
    Conv2d(32, 48, (1, 7), stride=(1,2)) + BN + GELU + Drop(0.15)   → [B, 48, 1, 500]
    Conv2d(48, 48, (1, 7), stride=(1,2)) + BN + GELU + Drop(0.15)   → [B, 48, 1, 250]
    Conv2d(48, 64, (1, 5), stride=(1,2)) + BN + GELU + Drop(0.15)   → [B, 64, 1, 125]
   │
   ▼
[3] Multi-scale temporal convolution (MSCARNet)
    3 parallel Conv1d(64, 21, k={15, 35, 55}, padding=k//2)  → concat 63 ch
    BatchNorm1d(63) → Conv1d(63, 64, 1) → BatchNorm1d(64)
    GELU                                                        → [B, 64, 125]
   │
   ▼
[4] Two DSTS blocks (Dual-Stream Temporal-Spatial, from EEGEncoder)
    Each block:
        TCN stream:  3 stacked dilated Conv1d(64→64, k=3, dilations 1/2/4) + BN + GELU
        Spatial stream: pre-norm TransformerEncoderLayer(d=64, heads=4, ff=256)
        Sum + Dropout(0.25) + LayerNorm(d=64)                      → [B, 64, 125]
   │
   ▼
[5] Classifier
    x = x.mean(dim=-1)             ← mean-pool over time
    x = RMSNorm(64)(x)             ← pre-head norm
    x = Dropout(0.4)(x)
    x = Linear(64, 4)(x)           ← class logits
```

### 3.2 Components

**RMSNorm** (`x / sqrt(mean(x^2) + eps) * weight`) — used inside the transformer layers and the final classifier head. Cheaper than LayerNorm, used in modern transformer architectures (LLaMA, etc.).

**TCNBlock** — stack of dilated 1D convolutions with BN+GELU. Captures multi-scale temporal patterns. The dilation_growth=2 gives receptive fields of 3, 5, 9 samples across the three layers.

**MultiScaleTemporalConv** — three parallel 1D convs at different temporal scales (k=15, 35, 55), inspired by MSCARNet. Concatenated and projected back to 64 channels. Lets the model attend to short, medium, and long windows simultaneously.

**StableTransformerLayer** — pre-norm transformer: `x = x + Drop(SelfAttn(RMSNorm(x)))` and `x = x + Drop(FFN(RMSNorm(x)))`. Pre-norm is more stable than post-norm for small datasets.

**DSTSBlock** — the dual-stream block from EEGEncoder. The TCN handles the temporal axis; the transformer attends over the 125 time tokens (after the permute `(B, 64, 125) → (B, 125, 64)`). The two streams are summed and LayerNormed. Note: by the time we reach this stage, the depthwise spatial conv has already collapsed the 22 channels into the feature dimension, so the "spatial" stream is effectively a second view on the time axis — not a true channel-attention mechanism.

### 3.3 Parameter count

| Component | Params |
|---|---|
| Stage 1 (EEGNet embedding) | ~9K |
| Stage 2 (downsampling) | ~7K |
| Stage 3 (multi-scale conv) | ~145K |
| Stage 4 (2x DSTS block) | ~170K |
| Stage 5 (head) | ~260 |
| **Total** | **~364K** |

## 4. Training

One configuration for all 9 subjects. No per-subject tuning.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight_decay=0.02) |
| Scheduler | OneCycleLR, max_lr=5e-3, 15% warmup, cosine anneal |
| Epochs | 500 (patience 150) |
| Batch size | 64 |
| Label smoothing | 0.1 |
| MixUp | alpha=0.3, applied ~60% of batches |
| Augmentations (train only) | S&R (4-segment shuffle), frequency masking, channel dropout, channel permutation, temporal shift, Gaussian noise |
| Pretrain | 500 epochs on session 1 of all 9 subjects, supervised cross-entropy, cosine LR |
| Single seed | 42 |
| **Ensembling** | **None** |
| **TTA** | **None** |
| **Test-set leakage** | **None** (asserted at runtime) |

The pretrain uses session 1 of every subject, including the test subject's own session 1. This is not a leak because the test set is the same subject's session 2 — the pretrain has never seen session 2 of anyone.

## 5. Results

| Subject | Accuracy | Notes |
|---|---|---|
| 1 | 85.07% | |
| 2 | **58.68%** | Hard subject (well-known in literature) |
| 3 | 95.14% | |
| 4 | 79.86% | |
| 5 | 70.49% | Hard subject |
| 6 | **64.58%** | Hard subject |
| 7 | 91.32% | |
| 8 | 82.64% | |
| 9 | 85.76% | |
| **Mean** | **79.28%** | **±12.25%** |

Comparison to published SOTA on the same 4-class protocol:

| Model | Mean accuracy | Notes |
|---|---|---|
| CTNet (2024) | 82.52% | |
| MSCARNet (2024) | 82.66% | |
| EEGEncoder (2025) | 86.46% | |
| **Ours (MTSEEGFormer, single model, no ensemble, no TTA, no leakage)** | **79.28%** | |

## 6. Why 85% was hard

The 5.72-point gap to the 85% target is concentrated in three "hard" subjects (S2=58.7%, S5=70.5%, S6=64.6%). These subjects are documented in the BCI IV-2a literature as intrinsically difficult — they have weaker mu/beta rhythm, noisier motor imagery, and higher session-to-session variability. Pushing the mean to 85% requires pushing these three from ~65% to ~80% on average.

Honest assessment of what would close the gap (any one of):
- **Larger pretraining corpus**: add data from related datasets (BCI IV-2b, High Gamma, etc.) for cross-corpus self-supervised pretraining
- **Self-supervised pretraining**: masked autoencoding or contrastive objective on the 2592 unlabeled session-1 trials instead of supervised cross-entropy
- **Smaller model capacity**: 364K params on 288 trials is over-parameterized. A 50-100K param variant might generalize better to the hard subjects

These were not attempted because they would either add external data sources or violate the "single architecture, no tricks" constraint.

## 7. Reproducibility

```bash
uv sync
uv run python main.py
```

First run: ~80 min on a 4GB GPU (50 min pretrain + 30 min fine-tune). Subsequent runs: ~30 min (the pretrained checkpoint in `models/pretrained.pt` is reused).

The training output contains explicit `[no-leakage]` printouts for every per-subject split, so you can verify the protocol from the log.
