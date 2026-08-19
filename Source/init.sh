#!/usr/bin/env bash

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SOURCE_DIR/.venv"
NANOVLM_DIR="$SOURCE_DIR/nanoVLM"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install \
    numpy \
    torch \
    psutil \
    flask \
    gunicorn \
    transformers \
    torchvision \
    einops \
    requests \
    datasets

if [ ! -d "$NANOVLM_DIR" ]; then
    git clone https://github.com/huggingface/nanoVLM.git "$NANOVLM_DIR"
fi
