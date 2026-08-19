#!/usr/bin/env bash
###
 # @Description: 
 # @Date: 2026-08-19 01:34:15
 # @Author: Yaoquan Ma
### 

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SOURCE_DIR/.venv"
NANOVLM_DIR="$SOURCE_DIR/nanoVLM"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

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
    git clone https://github.com/YMa10-ACS/nanoVLM.git "$NANOVLM_DIR"
fi
