# Datasets

This directory contains dataset acquisition scripts and local image data used by the VLM experiments.

## Contents

- download_hf_images.py: downloads image samples through the Hugging Face Datasets library.
- ai2d/: contains AI2D diagram images used by batch experiments.
- coco_sample/: contains the local COCO-derived sample.

## Download Images

Use download_hf_images.py to download images from a Hugging Face dataset:

```bash
cd Source/datasets
python3 download_hf_images.py
```

By default, the script downloads 20 images from the ai2d subset of HuggingFaceM4/the_cauldron and saves them in the ai2d/ directory.
