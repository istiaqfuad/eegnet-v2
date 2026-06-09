# Graph Report - .  (2026-06-09)

## Corpus Check
- 11 files · ~64,617 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 161 nodes · 249 edges · 11 communities (9 shown, 2 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 20 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Learnable Alignment Models|Learnable Alignment Models]]
- [[_COMMUNITY_Covariance Alignment (EARA)|Covariance Alignment (EA/RA)]]
- [[_COMMUNITY_Training & IM-TTA|Training & IM-TTA]]
- [[_COMMUNITY_Data Augmentation|Data Augmentation]]
- [[_COMMUNITY_MI-EEG Domain & Baselines|MI-EEG Domain & Baselines]]
- [[_COMMUNITY_Experiment Orchestration|Experiment Orchestration]]
- [[_COMMUNITY_Window & S&R Augmentation|Window & S&R Augmentation]]
- [[_COMMUNITY_Dataset Loading|Dataset Loading]]
- [[_COMMUNITY_Local Settings|Local Settings]]
- [[_COMMUNITY_EEG Domain|EEG Domain]]

## God Nodes (most connected - your core abstractions)
1. `prepare_subject_data()` - 12 edges
2. `train_model()` - 12 edges
3. `_whitener()` - 11 edges
4. `BCIDataset` - 10 edges
5. `prepare_subject_data_cv()` - 10 edges
6. `prepare_pretrain_data()` - 9 edges
7. `run_within()` - 9 edges
8. `prepare_loso_data()` - 8 edges
9. `SlidingWindowDataset` - 7 edges
10. `_ea_reference()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `MSCARNet (multi-scale temporal conv for MI-EEG)` --semantically_similar_to--> `FreqAwareEEGNet`  [INFERRED] [semantically similar]
  papers/MSCARNet_Paper.md → model.py
- `EEG augmentation & deep-learning reference (S&R, mixup, freq-mask)` --conceptually_related_to--> `prepare_subject_data()`  [INFERRED]
  EEG_Augmented_Deep_Learning.md → dataset.py
- `Leave-one-subject-out (LOSO) cross-subject protocol` --rationale_for--> `prepare_loso_data()`  [EXTRACTED]
  ARCHITECTURE.md → dataset.py
- `Learnable spatial alignment refinement` --rationale_for--> `SpatialAlign`  [EXTRACTED]
  ARCHITECTURE.md → model.py
- `Learnable spatial alignment refinement` --rationale_for--> `AdaptiveSpatialAlign`  [EXTRACTED]
  ARCHITECTURE.md → model.py

## Hyperedges (group relationships)
- **Transformer-based BCI Models** — transformer_mi_eeg_paper_eegencoder, ctnet_paper_ctnet, transformer_mi_eeg_paper_stabletransformer, transformer_mi_eeg_paper_tcn, ctnet_paper_multi_head_attention [EXTRACTED 1.00]
- **Deep Learning MI Classification Approaches** — transformer_mi_eeg_paper_eegencoder, eeg_augmented_deep_learning_gdcnet, ctnet_paper_ctnet, eeg_augmented_deep_learning_cnn_lstm, ctnet_paper_cnn_transformer_hybrid [EXTRACTED 1.00]
- **BCI Competition IV Datasets** — transformer_mi_eeg_paper_bci_competition_iv_2a, eeg_augmented_deep_learning_bci_competition_iv_2b, bci_dataset_description_graz_dataset, bci_dataset_description_9_subjects, bci_dataset_description_4_classes, bci_dataset_description_22_channels [EXTRACTED 1.00]
- **Covariance alignment family** — last_bci_architecture_euclidean_alignment, last_bci_architecture_riemannian_alignment, last_bci_architecture_dual_fusion [INFERRED 0.85]
- **RA + EEGNet + IM-TTA pipeline** — last_bci_architecture_riemannian_alignment, last_bci_model_unifiedeegnet, last_bci_architecture_im_tta [EXTRACTED 1.00]

## Communities (11 total, 2 thin omitted)

### Community 0 - "Learnable Alignment Models"
Cohesion: 0.09
Nodes (15): Learnable spatial alignment refinement, AdaptiveAlignEEGNet, AdaptiveSpatialAlign, AlignedEEGNet, DualAlignEEGNet, FreqAttn, FreqAwareEEGNet, Static learnable spatial alignment: x' = (I + dW) x, dW init 0.     Weight decay (+7 more)

### Community 1 - "Covariance Alignment (EA/RA)"
Cohesion: 0.13
Nodes (30): Dual-geometry alignment fusion (EA+RA), Euclidean Alignment (EA), Riemannian / Centroid Alignment (RA), Session-holdout within-subject protocol, _ca_whitener(), _dual_views(), _ea_apply(), _ea_reference() (+22 more)

### Community 2 - "Training & IM-TTA"
Cohesion: 0.11
Nodes (21): Anti-collapse diversity term (marginal-entropy maximization), Information-Maximization Test-Time Adaptation (IM-TTA), Leak-free evaluation (val-based selection, test once), Leave-one-subject-out (LOSO) cross-subject protocol, BCIDataset, BCI Competition IV-2a Dataset with comprehensive augmentation pipeline, _build_optim_sched(), evaluate() (+13 more)

### Community 3 - "Data Augmentation"
Cohesion: 0.1
Nodes (10): ChannelDropout, ChannelPermutation, FrequencyMasking, GaussianNoise, Random frequency masking - masks random frequency bands     Simulates band-stop, Randomly zero out channels (keeps same size), Add Gaussian noise for robustness, Random temporal shift (+2 more)

### Community 4 - "MI-EEG Domain & Baselines"
Cohesion: 0.12
Nodes (20): 22 EEG Channels, 250 Hz Sampling Rate, 4 MI Classes, 9 Subjects, BCI Competition 2008 Graz Dataset, CNN-Transformer Hybrid Architecture, Cross-Subject Classification, CTNet (+12 more)

### Community 5 - "Experiment Orchestration"
Cohesion: 0.4
Nodes (10): assert_no_leakage(), _fit(), main(), Within-subject k-fold CV (sessions pooled). Uniform across datasets.      Each s, Leave-one-subject-out cross-subject, trained from scratch on the 8-subject pool., Within-subject (session 1 -> train/val, session 2 -> test), leak-free., run_loso(), run_within() (+2 more)

### Community 6 - "Window & S&R Augmentation"
Cohesion: 0.2
Nodes (5): Dataset, Sliding window augmentation - from MSCARNet paper (+8.82% improvement)     Split, Segmentation and Recombination (S&R) augmentation - critical for CTNet     From, SlidingWindowDataset, SRSegment

### Community 7 - "Dataset Loading"
Cohesion: 0.33
Nodes (6): load_bci_iva_dataset(), load_dataset(), Load a MOABB MotorImagery dataset by registry name.      Band-pass [fmin, fmax], Backwards-compatible alias for the IV-2a loader (returns X, y, meta)., BCI Competition IV-2a (4-class, 22ch), BCI Competition IV-2b (2-class, 3ch)

## Knowledge Gaps
- **54 isolated node(s):** `Static learnable spatial alignment: x' = (I + dW) x, dW init 0.     Weight decay`, `Input-adaptive (per-trial) spatial alignment — the novel contribution.      EA a`, `UnifiedEEGNet + static learnable spatial alignment front-end.`, `UnifiedEEGNet + per-trial input-adaptive spatial alignment front-end.`, `Dual-geometry alignment fusion — the novel contribution.      Input is the chann` (+49 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Dual-geometry alignment fusion (EA+RA)` connect `Covariance Alignment (EA/RA)` to `Learnable Alignment Models`?**
  _High betweenness centrality (0.264) - this node is a cross-community bridge._
- **Why does `DualAlignEEGNet` connect `Learnable Alignment Models` to `Covariance Alignment (EA/RA)`?**
  _High betweenness centrality (0.263) - this node is a cross-community bridge._
- **Why does `BCIDataset` connect `Training & IM-TTA` to `Covariance Alignment (EA/RA)`, `Data Augmentation`, `Window & S&R Augmentation`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `prepare_subject_data()` (e.g. with `run_within()` and `EEG augmentation & deep-learning reference (S&R, mixup, freq-mask)`) actually correct?**
  _`prepare_subject_data()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `train_model()` (e.g. with `_fit()` and `BCIDataset`) actually correct?**
  _`train_model()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `_whitener()` (e.g. with `Euclidean Alignment (EA)` and `Riemannian / Centroid Alignment (RA)`) actually correct?**
  _`_whitener()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `BCIDataset` (e.g. with `ModelEma` and `FocalLoss`) actually correct?**
  _`BCIDataset` has 4 INFERRED edges - model-reasoned connections that need verification._