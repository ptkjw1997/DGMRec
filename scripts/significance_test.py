import argparse
import glob
import os

import numpy as np
from scipy import stats


def load(path):
    d = np.load(path)
    order = np.argsort(d['users'])
    return {k: d[k][order] for k in d.files}


def seed_mean(metrics_dir, model, dataset, seeds):
    runs = [load(os.path.join(metrics_dir, f'{model}_{dataset}_{s}.npz')) for s in seeds]
    users = runs[0]['users']
    assert all(np.array_equal(users, r['users']) for r in runs)
    return {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0] if k != 'users'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--metrics_dir', default='user_metrics')
    parser.add_argument('--model', default='DGMRec')
    parser.add_argument('--seeds', nargs='+', default=['999', '42', '2023', '2024', '2025'])
    parser.add_argument('--metrics', nargs='+', default=['recall@20', 'recall@50', 'ndcg@20', 'ndcg@50'])
    args = parser.parse_args()

    suffix = f'_{args.dataset}_{args.seeds[0]}.npz'
    baselines = sorted(os.path.basename(p)[:-len(suffix)] for p in glob.glob(os.path.join(args.metrics_dir, '*' + suffix)))
    baselines = [b for b in baselines if b != args.model]

    ours = seed_mean(args.metrics_dir, args.model, args.dataset, args.seeds)
    others = {b: seed_mean(args.metrics_dir, b, args.dataset, args.seeds) for b in baselines}

    print('| Dataset | Metric | Strongest baseline | t | p | Marker |')
    print('|---|---|---|---|---|---|')
    for m in args.metrics:
        best = max(others, key=lambda b: others[b][m].mean())
        t, p = stats.ttest_rel(ours[m], others[best][m])
        marker = '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f'| {args.dataset} | {m} | {best} | {t:.3f} | {p:.2e} | {marker} |')


if __name__ == '__main__':
    main()
