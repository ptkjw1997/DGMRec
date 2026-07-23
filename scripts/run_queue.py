#!/usr/bin/env python3
"""GPU job-queue runner for DGMRec_refactor experiments.

Reads a JSONL job file where each line is a job:
    {"job_id": "stageB-DGMRec-baby-seed999",       # unique; used for resume
     "model": "DGMRec", "dataset": "baby",
     "missing_modal": 1, "missing_ratio": "0.666", "missing_imputation": "[1]",
     "seed": "[999]",
     "config_override": {"n_mm_layers": 1},         # optional
     "save_best_model": 0}                          # optional

Usage:
    python run_queue.py --jobs jobs/stageB.jsonl --gpus 0,1,2,3,4,5,6,7,8 \
        --results results/stageB.jsonl

- One job per GPU at a time; a GPU is freed when its job exits.
- Resume: job_ids already present in the results file are skipped.
- Each job's stdout/stderr goes to logs/queue/<job_id>.out; the model's own
  training log lands in src/log/<MODEL>/ as usual; parsed metrics are read from
  a per-job result JSON written by quick_start (config key `result_json`).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

MIN_FREE_MB = 8000


def gpu_free_mb():
    """physical-gpu-index -> free MiB, via nvidia-smi (empty dict on failure)."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=20)
        return {int(l.split(',')[0]): int(l.split(',')[1]) for l in out.strip().splitlines()}
    except Exception:
        return {}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')


def load_done(results_path):
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('status') == 'ok':
                    done.add(rec['job_id'])
    return done


def build_cmd(job, gpu, result_json):
    cmd = [sys.executable, 'main.py',
           '-m', job['model'], '-d', job['dataset'], '-g', str(gpu),
           '--missing_modal', str(job.get('missing_modal', 1)),
           '--missing_ratio', str(job.get('missing_ratio', '0.666')),
           '--missing_imputation', str(job.get('missing_imputation', '[1]')),
           '--new_items', str(job.get('new_items', 0)),
           '--save_best_model', str(job.get('save_best_model', 0))]
    if job.get('seed') is not None:
        cmd += ['--seed', str(job['seed'])]
    override = dict(job.get('config_override') or {})
    override['result_json'] = result_json
    cmd += ['--config_override', json.dumps(override)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7,8')
    ap.add_argument('--per_gpu', type=int, default=1, help='concurrent jobs per GPU')
    ap.add_argument('--poll', type=float, default=20.0)
    args = ap.parse_args()

    with open(args.jobs) as f:
        jobs = [json.loads(l) for l in f if l.strip()]
    seen = set()
    for j in jobs:
        if j['job_id'] in seen:
            raise SystemExit(f"duplicate job_id: {j['job_id']}")
        seen.add(j['job_id'])

    done = load_done(args.results)
    pending = [j for j in jobs if j['job_id'] not in done]
    print(f'{len(jobs)} jobs total, {len(done)} already done, {len(pending)} to run', flush=True)

    slots = []
    for g in args.gpus.split(','):
        slots += [g.strip()] * args.per_gpu

    qlog_dir = os.path.join(ROOT, 'logs', 'queue')
    rj_dir = os.path.join(ROOT, 'logs', 'result_json')
    os.makedirs(qlog_dir, exist_ok=True)
    os.makedirs(rj_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results) or '.', exist_ok=True)

    running = {}  # slot_idx -> (proc, job, result_json, out_path, t0)
    free = list(range(len(slots)))
    idx = 0

    def record(rec):
        with open(args.results, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    while idx < len(pending) or running:
        free_mb = gpu_free_mb()
        while free and idx < len(pending):
            # skip slots whose GPU is currently hogged (e.g. by other users)
            usable = [s for s in free if free_mb.get(int(slots[s]), MIN_FREE_MB) >= MIN_FREE_MB]
            if not usable:
                break
            slot = usable[0]
            free.remove(slot)
            job = pending[idx]
            idx += 1
            rj = os.path.join(rj_dir, job['job_id'] + '.json')
            if os.path.exists(rj):
                os.remove(rj)
            out_path = os.path.join(qlog_dir, job['job_id'] + '.out')
            cmd = build_cmd(job, slots[slot], rj)
            outf = open(out_path, 'w')
            outf.write(' '.join(shlex.quote(c) for c in cmd) + '\n')
            outf.flush()
            proc = subprocess.Popen(cmd, cwd=SRC, stdout=outf, stderr=subprocess.STDOUT)
            running[slot] = (proc, job, rj, out_path, time.time())
            print(f"[{datetime.now():%H:%M:%S}] START gpu{slots[slot]} {job['job_id']}", flush=True)

        time.sleep(args.poll)
        for slot in list(running):
            proc, job, rj, out_path, t0 = running[slot]
            if proc.poll() is None:
                continue
            del running[slot]
            free.append(slot)
            dur = round(time.time() - t0, 1)
            rec = {'job_id': job['job_id'], 'job': job, 'gpu_log': out_path,
                   'duration_sec': dur, 'finished_at': datetime.now().isoformat()}
            if proc.returncode == 0 and os.path.exists(rj):
                with open(rj) as f:
                    rec.update(json.load(f))
                rec['status'] = 'ok'
                r20 = rec.get('best_test', {}).get('recall@20')
                print(f"[{datetime.now():%H:%M:%S}] DONE  {job['job_id']} ({dur}s) R@20={r20}", flush=True)
            else:
                rec['status'] = 'fail'
                rec['returncode'] = proc.returncode
                print(f"[{datetime.now():%H:%M:%S}] FAIL  {job['job_id']} rc={proc.returncode} see {out_path}", flush=True)
            record(rec)

    n_fail = sum(1 for l in open(args.results) if '"status": "fail"' in l)
    print(f'queue finished; failures in results file: {n_fail}', flush=True)


if __name__ == '__main__':
    main()
