#!/usr/bin/env python3
"""
Summarize image size distribution and percent meeting various min-side thresholds.

Usage:
python eval/summarize_image_sizes.py --source "/Users/alext/Downloads/Datasets/wiki_crop/01"
python eval/summarize_image_sizes.py --source "/path/to/images" --thresholds 128 160 224 256 320 384 448 512 640 720 1024
"""

import argparse
import os
from typing import List, Tuple
from PIL import Image
import statistics


def scan_sizes(source: str) -> List[Tuple[int, int]]:
    sizes: List[Tuple[int, int]] = []
    for root, _dirs, files in os.walk(source):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, name)
                try:
                    with Image.open(path) as im:
                        w, h = im.size
                        sizes.append((int(w), int(h)))
                except Exception:
                    continue
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize image sizes and threshold coverage")
    parser.add_argument("--source", required=True, help="Source image directory")
    parser.add_argument("--thresholds", nargs="*", type=int, default=[128,160,224,256,320,384,448,512,640,720,1024], help="Min-side thresholds to evaluate")
    args = parser.parse_args()

    sizes = scan_sizes(args.source)
    total = len(sizes)
    if total == 0:
        print("No images found")
        return

    min_sides = [min(w, h) for (w, h) in sizes]
    max_sides = [max(w, h) for (w, h) in sizes]

    print(f"Total images: {total}")
    print(f"Min-side: min={min(min_sides)}, p25={int(statistics.quantiles(min_sides, n=4)[0])}, median={int(statistics.median(min_sides))}, p75={int(statistics.quantiles(min_sides, n=4)[2])}, max={max(min_sides)}")
    print(f"Max-side: min={min(max_sides)}, p25={int(statistics.quantiles(max_sides, n=4)[0])}, median={int(statistics.median(max_sides))}, p75={int(statistics.quantiles(max_sides, n=4)[2])}, max={max(max_sides)}")

    print("\nCoverage by min-side threshold:")
    for t in sorted(args.thresholds):
        count = sum(1 for s in min_sides if s >= t)
        pct = 100.0 * count / total
        print(f"  >= {t:>4}px: {count:>6} / {total} ({pct:5.1f}%)")


if __name__ == "__main__":
    main() 