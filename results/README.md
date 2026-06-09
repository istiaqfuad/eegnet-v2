# Results archive

All raw per-subject result CSVs and training logs, preserved for paper tables/figures.
Namespaced by the GPU box that produced them (`fedora`, `hcilab`) so identical filenames
from different boxes never collide. Live runs write loose `results_*.csv` to the repo root
(git-ignored); these curated copies are the kept record.

```
results/
  fedora/   *.csv   per-subject test accuracies produced on fedora (RTX 2070S)
  hcilab/   *.csv   per-subject test accuracies produced on hci-lab (GTX 1050Ti)
  logs/
    fedora/ *.log   per-epoch TrLoss/ValAcc/ValLoss + per-fold "Test (single eval) | +TTA"
    hcilab/ *.log
```

CSV columns: `subject, test_acc, model`.

## Filename key
`results_<protocol>_<dataset>_<model>_<tag>.csv`
- protocol: `within_honest` (session holdout) · `loso` (cross-subject) · `within_cv` (k-fold, unused/non-standard)
- dataset: `iv2a` (4-class, 22ch) · `iv2b` (2-class, 3ch) · (older IV-2a files predate the dataset tag)
- tag encodes the config/seed, e.g. `iv2bratta42` = IV-2b RA+IM-TTA seed 42; `ea9s1` = EA seed 1;
  `leatta9` = LOSO EA+IM-TTA; `lratta9` = LOSO RA+IM-TTA; `base9` = baseline; `leatent9` = EA + plain-Tent (div=0).

## Map to the main tables (see ../RESULTS.md)
- Multi-dataset table → `*_iv2a_*` and `*_iv2b_*` (RA / RA+IM-TTA / baseline).
- IV-2a within EA 6-seed → `results_within_honest_unified_ea9*.csv`.
- IV-2a LOSO align×TTA matrix → `results_loso_unified_{loso_unified(none),lea9(EA),lratta9(RA+IMTTA),leatent9(Tent),lnonetta9(none+IMTTA)}`.
- TTA-steps sweep → `results_loso_unified_{rs1,lratta9,rs10,rs20}.csv`.
- Per-epoch / pre-vs-post-TTA curves → corresponding `logs/*/w_*.log`.
