# DGMRec
The official source code for [**DMGRec: Disentangling and Generating Modalities for Recommendation in Missing Modality Scenarios**](https://arxiv.org/abs/2504.16352), accepted at **SIGIR 2025**.

## Overview
Multi-modal recommender systems (MRSs) have demonstrated significant success in improving personalization by leveraging diverse modalities such as images, text, and audio. However, they face two critical challenges: (1) addressing missing modality scenarios and (2) effectively disentangling shared and unique characteristics of modalities, leading to severe performance degradation.
To overcome these challenges, we propose **D**isentangling and **G**enerating **M**odality **Rec**ommender (DGMRec), a novel framework designed for missing modality scenarios.
DGMRec disentangles modality features into general and specific modality features from an information perspective to achieve better representations for recommendation.
Building on this, DGMRec generates missing modality features by integrating aligned features from other modalities and leveraging modality preferences, enabling the accurate reconstruction of missing modalities.
Extensive experiments demonstrate that DGMRec consistently outperforms state-of-the-art MRSs in challenging scenarios, including missing modalities and new item settings as well as diverse missing ratios and varying levels of missing modalities.
Beyond recommendation tasks, DGMRec's generation-based method enables cross-modal retrieval, which is inapplicable for existing MRSs, demonstrating its adaptability and potential for real-world applications.

![architecture](./img/architecture.png)

## Environment
    conda create -n [env name] python=3.9
    conda activate [env name]
    pip install -r requirements.txt

## Dataset
Download from Google Drive: [Baby/Sports/Clothing](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing) from [MMRec](https://github.com/enoche/MMRec).

The data already contains text and image features extracted from Sentence-Transformers and CNN, which is provided by [MMRec](https://github.com/enoche/MMRec).
Please move your downloaded data into the folder for model training.

## Missing Modality Setting
    cd data
    python preprocess_missing_modality.py --dataset [dataset]

After running the code, it will produce `missing_items_[missing_ratio].npy` in `./data/[dataset]/` directory.

## New Item Setting
    cd data
    python preprocess_new_items.py --dataset [dataset]

After running the code, it will produce `[dataset]_del.inter` in `./data/[dataset]/` directory.

## Unified `src/` (Missing-only + New-item)

The previous repository shipped **two separate code trees** — `src/` for the
missing-modality setting and `src_new_item/` for the missing-modality + new-item
setting — that duplicated most of their logic and could drift out of sync. The
new `src/` here merges them into a **single tree** that picks the right code
path from a `--new_items` flag, so both experiments are reproducible from one
entry point.

### Training / Test for Missing Modality only (default)

    cd src
    python main.py --dataset [dataset] \
        --missing_modal 1 \
        --new_items 0

### Training / Test for Missing Modality + New Item

    cd src
    python main.py --dataset [dataset] \
        --missing_modal 1 \
        --new_items 1

This is equivalent to running the legacy `src_new_item/main.py`.

### What changed in `src/`

| File | Change |
|---|---|
| `main.py` | Adds `--new_items` argument (default `0` = missing-only behavior). |
| `utils/quick_start.py` | If `new_items=1`, swap in `<dataset>_del.inter` and use the new-item test split. |
| `utils/dataset.py` | If `new_items=1`, additionally build a new-item-only test set and return `(splits, new_df)`; otherwise return the legacy 3-tuple. |
| `utils/dataloader.py` | If `new_items=1`, exclude new items from negative sampling. |
| `common/trainer.py` | If `new_items=1` and `missing_modal=1`, call `generate_missing_modal_infer()` / `update_adj_infer()` before validation. |
| `common/loss.py` | `MSELoss` now takes a `weight` argument (default `0.05`); pass `0.1` for new-item mode. |
| `utils/configurator.py` | Added a `.get(key, default)` accessor on the `Config` class (dict-style fallback used by the unified entry points). |
| `models/dgmrec.py` | Gates the new-item-specific tensors (`image_adj_infer`, `text_adj_infer`, generation/update hooks) behind `if self.new_items`. With `new_items=0` the inference adjacency aliases the training adjacency, reproducing the legacy `src/` behavior. The MSE weight is read from `config['mse_loss_weight']` and defaults to `0.05` when `new_items=0`, `0.1` when `new_items=1`. Also fixes a latent typo in `generate_missing_modal_infer` that referenced `image_g_filter_trans`/`text_g_filter_trans` (undefined); these are renamed to `image_preference_`/`text_preference_` to match the model's actual attributes. |

### Backup

The pre-unification code is preserved verbatim under `src_backup/` as a
single tree (the legacy `src_new_item/` snapshot, which is a strict
superset of the legacy `src/` — the only difference between the two old
trees was new-item support):

    src_backup/
    ├── common/
    ├── configs/
    ├── main.py
    ├── models/
    └── utils/

If anything goes wrong with the unified `src/`, `cd src_backup` and run
`python main.py --dataset [dataset] --new_items 0` (missing-only) or
`python main.py --dataset [dataset] --new_items 1` (missing + new-item)
exactly as the old README described.
