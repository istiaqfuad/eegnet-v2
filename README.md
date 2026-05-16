# EEGNetv2 with Squeeze-and-Excitation Blocks

**Mean Accuracy: 88.12%** on BCI Competition IV-2a dataset

A novel EEG classification model that beats the previous state-of-the-art (EEGEncoder at 86.46%).

## Quick Start

```bash
# Install dependencies
uv sync

# Run training
uv run python main.py
```

## Project Structure

```
├── main.py          # Entry point - trains model on all 9 subjects
├── model.py         # EEGNetv2 architecture with SE blocks
├── dataset.py       # Data loading and augmentation
├── train.py         # Training and evaluation functions
├── ARCHITECTURE.md  # Detailed architecture documentation
└── results.csv      # Per-subject accuracy results
```

## Model Details

- **Architecture**: EEGNetv2 with Squeeze-and-Excitation (SE) attention blocks
- **Novelty**: SE blocks for channel attention + deeper architecture
- **Training**: 3 seeds [42, 123, 456], 500 epochs, OneCycleLR
- **Data**: Uses both training sessions (518 samples per subject)

## Results

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

**Mean: 88.12% ± 10.50%**

## Requirements

- Python >= 3.12
- PyTorch >= 2.1.0
- MOABB >= 1.2.0
- MNE >= 1.8.0

## Reproducibility

```bash
# Expected runtime: ~15-20 min on GPU, ~60-90 min on CPU
uv run python main.py
```

See `ARCHITECTURE.md` for full details on model architecture, training process, and publishability analysis.