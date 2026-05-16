# Graph Report - /home/istiaqfuad/Desktop/last-bci  (2026-05-16)

## Corpus Check
- 8 files · ~50,035 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 34 nodes · 36 edges · 9 communities (5 shown, 4 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.83)
- Token cost: 24,000 input · 8,500 output

## Community Hubs (Navigation)
- [[_COMMUNITY_EEGEncoder Transformer|EEGEncoder Transformer]]
- [[_COMMUNITY_BCI Graz Dataset|BCI Graz Dataset]]
- [[_COMMUNITY_GDC-Net Deep Learning|GDC-Net Deep Learning]]
- [[_COMMUNITY_CTNet Architecture|CTNet Architecture]]
- [[_COMMUNITY_Claude Settings|Claude Settings]]
- [[_COMMUNITY_Main Function|Main Function]]
- [[_COMMUNITY_Local Settings|Local Settings]]
- [[_COMMUNITY_EEG Fundamentals|EEG Fundamentals]]

## God Nodes (most connected - your core abstractions)
1. `CTNet` - 9 edges
2. `EEGEncoder` - 6 edges
3. `GDC-Net` - 6 edges
4. `BCI Competition 2008 Graz Dataset` - 6 edges
5. `Stable Transformer` - 5 edges
6. `Dual-Stream Temporal-Spatial Block` - 3 edges
7. `Temporal Convolutional Network` - 3 edges
8. `BCI Competition IV-2a Dataset` - 3 edges
9. `BCI Competition IV-2b Dataset` - 3 edges
10. `CNN-Transformer Hybrid Architecture` - 3 edges

## Surprising Connections (you probably didn't know these)
- `EEGEncoder` --semantically_similar_to--> `CTNet`  [INFERRED] [semantically similar]
  papers/transformer_mi_eeg_paper.md → papers/CTNet_paper.md
- `GDC-Net` --semantically_similar_to--> `CTNet`  [INFERRED] [semantically similar]
  papers/EEG Augmented Deep Learning.md → papers/CTNet_paper.md
- `Temporal Convolutional Network` --conceptually_related_to--> `CNN-Transformer Hybrid Architecture`  [INFERRED]
  papers/transformer_mi_eeg_paper.md → papers/CTNet_paper.md
- `CNN-LSTM Hybrid` --semantically_similar_to--> `CNN-Transformer Hybrid Architecture`  [INFERRED] [semantically similar]
  papers/EEG Augmented Deep Learning.md → papers/CTNet_paper.md
- `BCI Competition IV-2a Dataset` --references--> `BCI Competition 2008 Graz Dataset`  [EXTRACTED]
  papers/transformer_mi_eeg_paper.md → papers/bci_dataset_description.md

## Hyperedges (group relationships)
- **BCI Competition IV Datasets** — transformer_mi_eeg_paper_bci_competition_iv_2a, eeg_augmented_deep_learning_bci_competition_iv_2b, bci_dataset_description_graz_dataset, bci_dataset_description_9_subjects, bci_dataset_description_4_classes, bci_dataset_description_22_channels [EXTRACTED 1.00]
- **Transformer-based BCI Models** — transformer_mi_eeg_paper_eegencoder, ctnet_paper_ctnet, transformer_mi_eeg_paper_stabletransformer, transformer_mi_eeg_paper_tcn, ctnet_paper_multi_head_attention [EXTRACTED 1.00]
- **Deep Learning MI Classification Approaches** — transformer_mi_eeg_paper_eegencoder, eeg_augmented_deep_learning_gdcnet, ctnet_paper_ctnet, eeg_augmented_deep_learning_cnn_lstm, ctnet_paper_cnn_transformer_hybrid [EXTRACTED 1.00]

## Communities (9 total, 4 thin omitted)

### Community 0 - "EEGEncoder Transformer"
Cohesion: 0.28
Nodes (9): BCI Competition IV-2a Dataset, Dual-Stream Temporal-Spatial Block, EEGEncoder, Motor Imagery, Pre-normalization, RMSNorm, Stable Transformer, SwiGLU (+1 more)

### Community 1 - "BCI Graz Dataset"
Cohesion: 0.33
Nodes (6): 22 EEG Channels, 250 Hz Sampling Rate, 4 MI Classes, 9 Subjects, BCI Competition 2008 Graz Dataset, BCI Competition IV-2b Dataset

### Community 2 - "GDC-Net Deep Learning"
Cohesion: 0.4
Nodes (6): CNN-Transformer Hybrid Architecture, CNN-LSTM Hybrid, DCGAN, GDC-Net, Generalized Morse Wavelet Transform, Left/Right Hand Motor Imagery

### Community 3 - "CTNet Architecture"
Cohesion: 0.4
Nodes (5): Cross-Subject Classification, CTNet, EEGNet, Multi-Head Attention, Subject-Specific Classification

## Knowledge Gaps
- **17 isolated node(s):** `allow`, `Main Function`, `Local Settings`, `RMSNorm`, `SwiGLU` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CTNet` connect `CTNet Architecture` to `EEGEncoder Transformer`, `BCI Graz Dataset`, `GDC-Net Deep Learning`?**
  _High betweenness centrality (0.282) - this node is a cross-community bridge._
- **Why does `EEGEncoder` connect `EEGEncoder Transformer` to `CTNet Architecture`?**
  _High betweenness centrality (0.231) - this node is a cross-community bridge._
- **Why does `BCI Competition 2008 Graz Dataset` connect `BCI Graz Dataset` to `EEGEncoder Transformer`?**
  _High betweenness centrality (0.171) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `CTNet` (e.g. with `EEGEncoder` and `GDC-Net`) actually correct?**
  _`CTNet` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `allow`, `Main Function`, `Local Settings` to the rest of the system?**
  _17 weakly-connected nodes found - possible documentation gaps or missing edges._