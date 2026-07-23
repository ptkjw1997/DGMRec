#!/bin/bash
# Download and verify the public multimodal datasets used in the paper.
#
# Interaction files (.inter), split labels, and multimodal features
# (image_feat.npy / text_feat.npy) for Amazon Baby / Sports / Clothing /
# Electronics follow the MMRec data release:
#   https://github.com/enoche/MMRec  (see its data/README for the Google
#   Drive links; features are the standard 4096-d CNN + 384-d sentence
#   transformer embeddings)
#
# TikTok is NOT downloadable — its original distribution channel has been
# discontinued. See the reproducibility statement in the paper: we release
# the interaction file, split labels, and missing-item masks; the multimodal
# features are available from the authors upon reasonable request.
#
# Usage: ./download_data.sh <dataset> [target_dir]
set -e

DS=${1:?usage: download_data.sh <baby|sports|clothing|elec> [target_dir]}
TARGET=${2:-data}

command -v gdown >/dev/null || pip install gdown

declare -A FOLDER  # MMRec Google-Drive folder ids per dataset
# All four datasets are distributed in one MMRec folder (see README):
FOLDER[baby]="13cBy1EA_saTUuXxVllKgtfci2A09jyaG"
FOLDER[sports]="13cBy1EA_saTUuXxVllKgtfci2A09jyaG"
FOLDER[clothing]="13cBy1EA_saTUuXxVllKgtfci2A09jyaG"
FOLDER[elec]="13cBy1EA_saTUuXxVllKgtfci2A09jyaG"

mkdir -p "$TARGET/$DS"
echo ">> downloading $DS features into $TARGET/$DS"
gdown --folder "https://drive.google.com/drive/folders/${FOLDER[$DS]}" -O "$TARGET/$DS"

echo ">> generating missing-modality masks (fixed seed; identical to the paper's masks)"
python "$TARGET/preprocess_missing_modality.py" --dataset "$DS"

echo "done. Note: the repo already ships the exact missing_items_*.npy masks"
echo "used in the paper under data/masks/${DS}/ — preprocessing is only a fallback."
