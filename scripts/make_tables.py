#!/usr/bin/env python3
"""Aggregate queue results into paper-ready tables.

- 5-seed mean ± std per (model, dataset, metric) from a results JSONL
- optional comparison against results/paper_reference.json
- per-user paired t-test (DGMRec vs best baseline per dataset/metric) from
  logs/user_metrics/<job_id>.npz files

Usage:
    python make_tables.py --results results/stageB.jsonl \
        --out results/tables/stageB --reference results/paper_reference.json
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRICS = ['recall@20', 'recall@50', 'ndcg@20', 'ndcg@50']
DATASET_ORDER = ['baby', 'sports', 'clothing', 'tiktok', 'elec']


def canon(job_id):
    """job_id format: <stage>-<Model>-<dataset>-seed<k>."""
    m = re.match(r'^[^-]+-(.+)-([a-z]+)-seed(\d+)$', job_id)
    return m.group(1), m.group(2), int(m.group(3))


def load_results(path):
    per = defaultdict(dict)  # (model, dataset) -> {seed: best_test}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get('status') != 'ok':
                continue
            model, dataset, seed = canon(rec['job_id'])
            per[(model, dataset)][seed] = rec['best_test']
    return per


def paired_ttest(a, b):
    """Paired t-test on per-user vectors; returns (t, p). SciPy if available."""
    try:
        from scipy import stats
        t, p = stats.ttest_rel(a, b)
        return float(t), float(p)
    except ImportError:
        d = np.asarray(a) - np.asarray(b)
        n = len(d)
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
        return float(t), float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference', default=os.path.join(ROOT, 'results', 'paper_reference.json'))
    ap.add_argument('--main_model', default='DGMRec')
    ap.add_argument('--user_metrics_dir', default=os.path.join(ROOT, 'logs', 'user_metrics'))
    args = ap.parse_args()

    per = load_results(args.results)
    models = sorted({m for m, _ in per})
    datasets = [d for d in DATASET_ORDER if any(ds == d for _, ds in per)]

    os.makedirs(args.out, exist_ok=True)

    # ---------------- aggregate mean ± std ----------------
    agg = {}
    for (model, dataset), seed_map in per.items():
        vals = {k: [] for k in METRICS}
        for seed, test in sorted(seed_map.items()):
            for k in METRICS:
                if k in test:
                    vals[k].append(test[k])
        agg[(model, dataset)] = {
            k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
            for k, v in vals.items() if v
        }

    ref = None
    if args.reference and os.path.exists(args.reference):
        with open(args.reference) as f:
            ref = json.load(f)

    lines = ['# Aggregated results (mean ± std over seeds)\n']
    for dataset in datasets:
        lines.append(f'\n## {dataset}\n')
        header = '| Model | ' + ' | '.join(METRICS) + ' | seeds |'
        lines.append(header)
        lines.append('|' + '---|' * (len(METRICS) + 2))
        for model in models:
            if (model, dataset) not in agg:
                continue
            row = [model]
            n_seeds = 0
            for k in METRICS:
                if k in agg[(model, dataset)]:
                    mu, sd, n = agg[(model, dataset)][k]
                    n_seeds = n
                    cell = f'{mu:.4f}±{sd:.4f}'
                    if ref and k in ref and dataset in ref[k] and model in ref[k][dataset]:
                        diff = mu - ref[k][dataset][model]
                        cell += f' (paper {ref[k][dataset][model]:.4f}, Δ{diff:+.4f})'
                    row.append(cell)
                else:
                    row.append('-')
            row.append(str(n_seeds))
            lines.append('| ' + ' | '.join(row) + ' |')

    # ---------------- paired t-tests ----------------
    stage = None
    with open(args.results) as f:
        for line in f:
            if line.strip():
                stage = json.loads(line)['job_id'].split('-')[0]
                break

    tt_lines = ['\n# Paired t-tests (per-user, DGMRec vs best baseline)\n',
                'Per-user vectors are averaged across seeds per user before testing.\n']
    for dataset in datasets:
        base_models = [m for m in models if m != args.main_model and (m, dataset) in agg]
        if not base_models or (args.main_model, dataset) not in agg:
            continue
        for metric in ['recall@20', 'ndcg@20', 'recall@50', 'ndcg@50']:
            cand = [(m, agg[(m, dataset)][metric][0]) for m in base_models
                    if metric in agg[(m, dataset)]]
            if not cand:
                continue
            best_base = max(cand, key=lambda x: x[1])[0]

            def user_vec(model):
                pats = glob.glob(os.path.join(
                    args.user_metrics_dir, f'{stage}-{model}-{dataset}-seed*.npz'))
                if not pats:
                    return None
                acc, users0 = None, None
                for p in sorted(pats):
                    z = np.load(p)
                    if users0 is None:
                        users0 = z['users']
                        acc = z[metric].astype(np.float64)
                    else:
                        if not np.array_equal(users0, z['users']):
                            return None
                        acc = acc + z[metric]
                return acc / len(pats)

            a = user_vec(args.main_model)
            b = user_vec(best_base)
            if a is None or b is None or len(a) != len(b):
                tt_lines.append(f'- {dataset} {metric}: user metrics unavailable '
                                f'({args.main_model} vs {best_base})')
                continue
            t, p = paired_ttest(a, b)
            mark = '**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.')
            tt_lines.append(f'- {dataset} {metric}: {args.main_model} vs {best_base}: '
                            f't={t:.3f}, p={p:.2e} [{mark}]')

    with open(os.path.join(args.out, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines + tt_lines) + '\n')
    with open(os.path.join(args.out, 'agg.json'), 'w') as f:
        json.dump({f'{m}|{d}': v for (m, d), v in agg.items()}, f, indent=1)
    print('\n'.join(lines + tt_lines))
    print(f"\nwrote {args.out}/summary.md and agg.json")


if __name__ == '__main__':
    main()
