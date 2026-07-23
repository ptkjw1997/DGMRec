# DGMRec (TORS) — reproduction image
# Build:  docker build -t dgmrec:tors .
# Run:    docker run --gpus all -v /path/to/data:/workspace/data dgmrec:tors <MODEL> <DATASET> [SEED]
#         e.g. docker run --gpus all -v $PWD/data:/workspace/data dgmrec:tors DGMRec baby 999
#
# The data volume must contain <dataset>/{<dataset>.inter, image_feat.npy,
# text_feat.npy, missing_items_0.666.npy, ...}; see data/README_DATA.md and
# scripts/download_data.sh for how to obtain and preprocess the public
# Amazon datasets. TikTok features are NOT publicly available (see the
# reproducibility statement in the paper) and are excluded from this image.

FROM docker.io/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir \
    torch-geometric==2.6.1 \
    numpy==1.26.4 \
    scipy==1.13.1 \
    pandas==2.2.3 \
    PyYAML==6.0.2 \
    tqdm==4.67.1 \
    lmdb==1.4.1 \
    scikit-learn==1.5.2 \
    matplotlib==3.9.2

COPY src /workspace/src
COPY scripts /workspace/scripts
COPY run.sh /workspace/run.sh
RUN chmod +x /workspace/run.sh

# fixed seeds + hardcoded best configurations live in src/configs/best/
ENTRYPOINT ["/workspace/run.sh"]
CMD ["DGMRec", "baby", "999"]
