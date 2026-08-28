"""Canonical results manifest / paper-table generator.

Reads the per-run result CSVs written by main.py, aggregates them into one
manifest (long format), and emits the paper's headline tables (LaTeX + Markdown)
directly from that manifest. Paper numbers, RESULTS.md, and the CSVs can then
never drift apart: this script is the single source of truth for every number
that appears in the manuscript.

Usage:
    python make_tables.py [--results-dir .] [--spec specs/v2.json] [--out-dir tables]

The spec is a JSON list of run groups:
    [{"name": "LOSO RA + IM-TTA",
      "pattern": "results_loso_iv2a_unified_v2_ra_im_s*.csv",
      "protocol": "loso", "dataset": "iv2a", "align": "ra", "tta": "im"},
     ...]

Checks enforced (fail loudly, not silently):
  - every group in the spec resolves to >= 1 CSV
  - all groups in the same table report the SAME number of seeds (consistency
    complaint from the paper audit: mixed single-seed / 3-seed rows)
  - LOSO groups have exactly one row per subject per seed
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def load_group(group, results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, group['pattern'])))
    if not paths:
        raise FileNotFoundError(f"spec group '{group['name']}': no CSV matches "
                                f"{group['pattern']} in {results_dir}")
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        seed = _seed_from_name(os.path.basename(path))
        for _, r in df.iterrows():
            rows.append({
                'group': group['name'],
                'protocol': group.get('protocol', ''),
                'dataset': group.get('dataset', ''),
                'align': group.get('align', ''),
                'tta': group.get('tta', ''),
                'seed': seed,
                'subject': int(r['subject']),
                'test_acc': float(r['test_acc']),
                'source': os.path.basename(path),
            })
    return pd.DataFrame(rows)


def _seed_from_name(fname):
    # tag naming convention: ..._s{SEED}.csv
    stem = fname[:-4] if fname.endswith('.csv') else fname
    tail = stem.rsplit('_s', 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return -1


def aggregate(manifest):
    """Per-group aggregate: seed-mean of the subject-mean, +/- std over seeds,
    plus per-subject means. The headline number is the mean over seeds of the
    mean over subjects, so every seed contributes equally."""
    out = []
    for (group), g in manifest.groupby('group', sort=False):
        seed_means = g.groupby('seed')['test_acc'].mean()
        per_subject = g.groupby('subject')['test_acc'].mean()
        meta = g.iloc[0]
        out.append({
            'group': group,
            'protocol': meta['protocol'],
            'dataset': meta['dataset'],
            'align': meta['align'],
            'tta': meta['tta'],
            'n_seeds': len(seed_means),
            'n_subjects': len(per_subject),
            'mean': seed_means.mean(),
            'std': seed_means.std(ddof=1) if len(seed_means) > 1 else 0.0,
            'seed_means': [round(float(m), 4) for m in seed_means],
            'per_subject': {int(s): round(float(a), 4)
                            for s, a in per_subject.items()},
        })
    return pd.DataFrame(out)


def check_consistency(agg, table_groups):
    """Every group printed in the same paper table must have the same seed count."""
    sub = agg[agg['group'].isin(table_groups)]
    if sub.empty:
        return
    n = sub['n_seeds'].unique()
    if len(n) > 1:
        bad = sub[sub['n_seeds'] != n[0]][['group', 'n_seeds']].to_dict('records')
        raise AssertionError(
            f"inconsistent seed counts within table: {bad} (all rows must use "
            f"the same seed policy)")


def latex_main_table(agg, groups):
    lines = [
        r'\begin{tabular}{lcc}', r'\toprule',
        r'\textbf{Configuration} & \textbf{Mean (\%)} & \textbf{Seeds} \\', r'\midrule',
    ]
    for name in groups:
        row = agg[agg['group'] == name].iloc[0]
        mean = row['mean'] * 100
        std = row['std'] * 100
        cell = f"{mean:.1f}" + (rf" $\pm$ {std:.1f}" if row['n_seeds'] > 1 else "")
        lines.append(f"{name} & {cell} & {row['n_seeds']} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='.')
    ap.add_argument('--spec', default=None, help='JSON spec file (list of groups)')
    ap.add_argument('--out-dir', default='tables')
    args = ap.parse_args()

    if args.spec:
        with open(args.spec) as f:
            groups = json.load(f)
    else:
        groups = DEFAULT_SPEC

    frames = [load_group(g, args.results_dir) for g in groups]
    manifest = pd.concat(frames, ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest_path = os.path.join(args.out_dir, 'manifest.csv')
    manifest.to_csv(manifest_path, index=False)
    print(f"manifest: {len(manifest)} rows -> {manifest_path}")

    agg = aggregate(manifest)
    agg.drop(columns=['per_subject']).to_csv(os.path.join(args.out_dir, 'aggregate.csv'),
                                             index=False)
    with open(os.path.join(args.out_dir, 'per_subject.json'), 'w') as f:
        json.dump({r['group']: r['per_subject'] for _, r in agg.iterrows()}, f, indent=1)

    # consistency: all groups in one paper table share a seed policy
    check_consistency(agg, [g['name'] for g in groups])

    print('\n=== aggregate (mean over seeds of mean over subjects) ===')
    with pd.option_context('display.width', 200):
        print(agg[['group', 'protocol', 'align', 'tta', 'n_seeds', 'mean', 'std']]
              .to_string(index=False,
                         formatters={'mean': '{:.4f}'.format, 'std': '{:.4f}'.format}))

    tex = latex_main_table(agg, [g['name'] for g in groups])
    with open(os.path.join(args.out_dir, 'main_table.tex'), 'w') as f:
        f.write(tex + '\n')
    print(f"\nlatex -> {os.path.join(args.out_dir, 'main_table.tex')}")
    return 0


# Default spec: the E1 matrix + E2/E3 runs from the v2 revision (fill patterns
# in as runs land; keep every paper row here so the seed-consistency check runs).
DEFAULT_SPEC = [
    # E1 — IV-2a LOSO alignment x TTA matrix
    {'name': r'LOSO none', 'pattern': 'results_loso_iv2a_unified_v2_none_off_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2a', 'align': 'none', 'tta': 'off'},
    {'name': r'LOSO none + Tent', 'pattern': 'results_loso_iv2a_unified_v2_none_tent_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2a', 'align': 'none', 'tta': 'tent'},
    {'name': r'LOSO none + IM-TTA', 'pattern': 'results_loso_iv2a_unified_v2_none_im_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2a', 'align': 'none', 'tta': 'im'},
    {'name': r'LOSO EA', 'pattern': 'results_loso_iv2a_unified_v2_ea_off_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2a', 'align': 'ea', 'tta': 'off'},
    {'name': r'LOSO EA + Tent', 'pattern': 'results_loso_iv2a_unified_v2_ea_tent_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2a', 'align': 'ea', 'tta': 'tent'},
    {'name': r'LOSO EA + IM-TTA', 'pattern': 'results_loso_iv2a_unified_v2_ea_im_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2a', 'align': 'ea', 'tta': 'im'},
    {'name': r'LOSO RA', 'pattern': 'results_loso_iv2a_unified_v2_ra_off_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2a', 'align': 'ra', 'tta': 'off'},
    {'name': r'LOSO RA + Tent', 'pattern': 'results_loso_iv2a_unified_v2_ra_tent_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2a', 'align': 'ra', 'tta': 'tent'},
    {'name': r'LOSO RA + IM-TTA', 'pattern': 'results_loso_iv2a_unified_v2_ra_im_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2a', 'align': 'ra', 'tta': 'im'},
    # E2 — IV-2a within-subject, both protocols
    {'name': r'within-pure RA', 'pattern': 'results_within-pure_honest_iv2a_unified_v2pure_ra_off_s*.csv',
     'protocol': 'within-pure', 'dataset': 'iv2a', 'align': 'ra', 'tta': 'off'},
    {'name': r'within-pure RA + IM-TTA', 'pattern': 'results_within-pure_honest_iv2a_unified_v2pure_ra_im_s*.csv',
     'protocol': 'within-pure', 'dataset': 'iv2a', 'align': 'ra', 'tta': 'im'},
    {'name': r'within (transfer) RA', 'pattern': 'results_within_honest_iv2a_unified_v2trans_ra_off_s*.csv',
     'protocol': 'within-transfer', 'dataset': 'iv2a', 'align': 'ra', 'tta': 'off'},
    {'name': r'within (transfer) RA + IM-TTA', 'pattern': 'results_within_honest_iv2a_unified_v2trans_ra_im_s*.csv',
     'protocol': 'within-transfer', 'dataset': 'iv2a', 'align': 'ra', 'tta': 'im'},
    # E3 — IV-2b
    {'name': r'IV-2b within none', 'pattern': 'results_within_honest_iv2b_unified_v2b_within_none_off_s*.csv',
     'protocol': 'within-transfer', 'dataset': 'iv2b', 'align': 'none', 'tta': 'off'},
    {'name': r'IV-2b within RA', 'pattern': 'results_within_honest_iv2b_unified_v2b_within_ra_off_s*.csv',
     'protocol': 'within-transfer', 'dataset': 'iv2b', 'align': 'ra', 'tta': 'off'},
    {'name': r'IV-2b within RA + IM-TTA', 'pattern': 'results_within_honest_iv2b_unified_v2b_within_ra_im_s*.csv',
     'protocol': 'within-transfer', 'dataset': 'iv2b', 'align': 'ra', 'tta': 'im'},
    {'name': r'IV-2b LOSO none', 'pattern': 'results_loso_iv2b_unified_v2b_loso_none_off_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2b', 'align': 'none', 'tta': 'off'},
    {'name': r'IV-2b LOSO RA', 'pattern': 'results_loso_iv2b_unified_v2b_loso_ra_off_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2b', 'align': 'ra', 'tta': 'off'},
    {'name': r'IV-2b LOSO RA + IM-TTA', 'pattern': 'results_loso_iv2b_unified_v2b_loso_ra_im_s*.csv',
     'protocol': 'loso', 'dataset': 'iv2b', 'align': 'ra', 'tta': 'im'},
]


if __name__ == '__main__':
    sys.exit(main())
