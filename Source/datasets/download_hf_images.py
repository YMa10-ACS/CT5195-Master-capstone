#!/usr/bin/env python
'''
Description: 
Date: 2026-06-10 09:12:01
Author: Yaoquan Ma
'''

import argparse
from pathlib import Path

# import load_dataest function from hugging face datasets library
from datasets import load_dataset
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export images from a Hugging Face Dataset for split/non-split output comparison."
    )
    parser.add_argument("--dataset", default="HuggingFaceM4/the_cauldron")
    parser.add_argument("--name", default="ai2d", help="Dataset config/subset name.")
    parser.add_argument("--split", default="train") # Use train sub dataset
    parser.add_argument("--prefix", default="./")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=1000)
    return parser.parse_args()


def iter_images(value):
    if value is None:
        return
    if isinstance(value, list):
        yield from value
    else:
        yield value


def save_image(image, path):
    if not isinstance(image, Image.Image):
        return False
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, quality=95)
    return True


def main():
    args = parse_args()

    # Convert output parameter into an absolute path.
    output_dir = (Path(args.prefix) / args.name).resolve()
    print(f"Download image into directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset,
        args.name,
        split=args.split,
        streaming=False,
    )

    saved = 0
    scanned = 0
    done = False
    for sample_index, sample in enumerate(dataset):
        if sample_index >= args.max_samples:
            break
        scanned = sample_index + 1

        for image_index, image in enumerate(iter_images(sample.get("images"))):
            path = output_dir / f"{sample_index:06d}_{image_index:02d}.jpg"
            if save_image(image, path):
                saved += 1
                print(path)

            if saved >= args.limit:
                done = True
                break

        if done:
            break

    print(
        f"Saved {saved} images to {output_dir}. "
        f"Stopped after scanning {scanned} samples."
    )


if __name__ == "__main__":
    main()
    print(f"Return from main function")
