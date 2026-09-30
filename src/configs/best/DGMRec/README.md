# DGMRec configurations

`<dataset>.json` holds the configuration of the full model on each dataset.

For ablation, we set the following weight(s) to 0 and keep all other values unchanged.

| Variant | Weight(s) set to 0 |
|---|---|
| w/o Disentangle | `sampler`, `interModal` |
| &nbsp;&nbsp;w/o CLUB | `sampler` |
| &nbsp;&nbsp;w/o InfoNCE | `interModal` |
| w/o Generation | `additive`, `recon` |
| &nbsp;&nbsp;w/o Recon Loss | `recon` |
| &nbsp;&nbsp;w/o Gen Loss | `additive` |
| w/o Alignment | `intraModal`, `alignBM` |
| &nbsp;&nbsp;w/o UI-align | `intraModal` |
| &nbsp;&nbsp;w/o BM-align | `alignBM` |

Example (w/o CLUB on TikTok):

    python main.py --model DGMRec --dataset tiktok \
        --config_override "$(python -c "import json;d=json.load(open('configs/best/DGMRec/tiktok.json'));d['sampler']=0;print(json.dumps(d))")"
