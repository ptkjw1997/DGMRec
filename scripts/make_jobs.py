#!/usr/bin/env python3
"""Generate job JSONL files for run_queue.py.

Examples:
  # verification runs (seed 999, best configs)
  python make_jobs.py --stage verify --models DGMRec,LGMRec,GUME,DAMRS \
      --datasets baby,sports,clothing --seeds 999 --out jobs/verify.jsonl

  # Stage B: 5-seed final runs
  python make_jobs.py --stage stageB --models all --datasets baby,sports,clothing \
      --seeds 999,42,2023,2024,2025 --out jobs/stageB.jsonl
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST = os.path.join(ROOT, 'src', 'configs', 'best')

ALL_MODELS = ['DGMRec', 'LGMRec', 'GUME', 'DAMRS', 'MGCN', 'BM3', 'LATTICE', 'SLMRec',
              'GRCN', 'MMGCN', 'VBPR', 'MFBPR', 'NGCF', 'SGL', 'SimGCL', 'LightGCN',
              'MILK', 'SIBRAR', 'CI2MG']


def best_override(model, dataset):
    p = os.path.join(BEST, f'{model}_{dataset}.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--datasets', required=True)
    ap.add_argument('--seeds', default='999')
    ap.add_argument('--missing_ratio', default='0.666')
    ap.add_argument('--missing_imputation', default='[1]')
    ap.add_argument('--save_best_model', type=int, default=0)
    ap.add_argument('--extra_override', default=None, help='JSON merged into every job')
    ap.add_argument('--user_metrics', type=int, default=0,
                    help='save per-user test metrics (for paired t-tests)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    models = ALL_MODELS if args.models == 'all' else args.models.split(',')
    datasets = args.datasets.split(',')
    seeds = [s.strip() for s in args.seeds.split(',')]
    extra = json.loads(args.extra_override) if args.extra_override else {}

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    n = 0
    with open(args.out, 'w') as f:
        for m in models:
            for d in datasets:
                override = best_override(m, d)
                override.update(extra)
                for s in seeds:
                    if args.user_metrics:
                        override = dict(override)
                        override['save_user_metrics'] = True
                        override['user_metrics_path'] = os.path.join(
                            ROOT, 'logs', 'user_metrics', f'{args.stage}-{m}-{d}-seed{s}.npz')
                    job = {
                        'job_id': f'{args.stage}-{m}-{d}-seed{s}',
                        'model': m, 'dataset': d,
                        'missing_modal': 1,
                        'missing_ratio': args.missing_ratio,
                        'missing_imputation': args.missing_imputation,
                        'seed': f'[{s}]',
                        'config_override': override,
                        'save_best_model': args.save_best_model,
                    }
                    if d == 'tiktok':
                        # TikTok (3 modalities) uses its own missing-item
                        # masks (ratio 0.6)
                        job['missing_ratio'] = '0.6'
                    f.write(json.dumps(job) + '\n')
                    n += 1
    print(f'wrote {n} jobs to {args.out}')


if __name__ == '__main__':
    main()
