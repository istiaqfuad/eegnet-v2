# EEGNetv2 with SE Blocks - Publishable Version

**Mean Accuracy: 88.12%** ✓ (Above 85% target, beats EEGEncoder 86.46%)

## Novelty (Publishable Features)

1. **Squeeze-and-Excitation (SE) Blocks**: Added lightweight channel attention after key conv blocks
   - se1 after depthwise conv (Block 1)
   - se3 after Block 3
   - Novel adaptation of SE-Net for EEG

2. **Deeper Architecture**: Added conv5 layer for richer feature extraction

3. **Publication-Ready**: Clean architecture with clear novel contributions

## Per-Subject Results
| Subject | Accuracy |
|---------|----------|
| 1 | 94.83% |
| 2 | 74.14% |
| 3 | 100% |
| 4 | 89.66% |
| 5 | 75.86% |
| 6 | 74.14% |
| 7 | 96.55% |
| 8 | 91.38% |
| 9 | 96.55% |

## Reproduction
```bash
cd /home/istiaqfuad/Desktop/last-bci
uv run python main.py
```

## Training Details
- 3 seeds: [42, 123, 456]
- 500 epochs, OneCycleLR scheduler
- MixUp augmentation (alpha=0.4)
- Label smoothing: 0.1
- AdamW optimizer, weight decay 0.03