# NovelEEGNet v1 - Channel Attention + Multi-scale Temporal Conv

**Mean Accuracy: 80.95%**

## Novelty
- Added Channel Attention modules after each conv block
- Added Multi-scale Temporal Convolutions (kernels 3 and 7)
- Simplified architecture from original EEGNet

## Per-Subject Results
| Subject | Accuracy |
|---------|----------|
| 1 | 93.10% |
| 2 | 72.41% |
| 3 | 96.55% |
| 4 | 81.03% |
| 5 | 56.90% |
| 6 | 65.52% |
| 7 | 96.55% |
| 8 | 91.38% |
| 9 | 84.48% |

## Reproduction
```bash
cd /home/istiaqfuad/Desktop/last-bci
uv run python main.py
```

## Notes
- Uses single seed (42) for faster testing
- 500 epochs, OneCycleLR scheduler
- MixUp augmentation with alpha=0.4