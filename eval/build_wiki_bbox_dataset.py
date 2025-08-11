#!/usr/bin/env python3
"""
Build a dataset from wiki_crop using only wiki.mat metadata:
- single face (second_face_score <= 0 or NaN)
- face_score > threshold (default: 2.0)
- min face bbox (min(width, height) from face_location) > threshold (default: 128 px)

This does not run any models and preserves relative paths.

Example:
python eval/build_wiki_bbox_dataset.py \
  --source "/Users/alext/Downloads/Datasets/wiki_crop" \
  --dest "/Users/alext/Downloads/Datasets/wiki_fs_gt2_mfb_gt128_single" \
  --face-score-gt 2.0 --min-face-bbox-gt 128
"""

import argparse
import csv
import json
import os
import shutil
from typing import List, Dict

import numpy as np
import scipy.io as sio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curate wiki dataset by face_score and min face bbox (single-face only)")
    p.add_argument("--source", required=True, help="Path to wiki_crop root (contains wiki.mat and folders 00..99)")
    p.add_argument("--dest", required=True, help="Destination dir to copy selected images")
    p.add_argument("--face-score-gt", type=float, default=2.0, help="Keep items with face_score strictly greater than this")
    p.add_argument("--min-face-bbox-gt", type=float, default=128.0, help="Keep items with min face bbox strictly greater than this (pixels)")
    p.add_argument("--limit", type=int, default=0, help="Optional limit on number of selected items (0 = no limit)")
    return p.parse_args()


def load_wiki(source: str):
    mat = sio.loadmat(os.path.join(source, 'wiki.mat'))
    wiki = mat['wiki'][0][0]
    # Fields: ('dob','photo_taken','full_path','gender','name','face_location','face_score','second_face_score')
    full_path = wiki[2].squeeze()
    gender = wiki[3].squeeze().astype(float)
    face_location = wiki[5].squeeze()
    face_score = wiki[6].squeeze().astype(float)
    second_face_score = wiki[7].squeeze()
    try:
        second_face_score = second_face_score.astype(float)
    except Exception:
        pass
    return full_path, gender, face_location, face_score, second_face_score


def get_item(arr, i):
    return arr[i] if arr.ndim == 1 else arr[0, i]


def get_rel_path(full_path, i) -> str:
    p = get_item(full_path, i)
    if isinstance(p, np.ndarray):
        p = p.item()
    if isinstance(p, bytes):
        p = p.decode('utf-8', errors='ignore')
    return str(p)


def compute_min_face_bbox(face_location, i) -> float:
    try:
        fl = get_item(face_location, i)
        arr = np.array(fl).astype(float).reshape(-1)
        if arr.size >= 4:
            x1, y1, x2, y2 = arr[:4]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            return float(min(w, h))
    except Exception:
        return float('nan')
    return float('nan')


def main():
    args = parse_args()
    full_path, gender, face_location, face_score, second_face_score = load_wiki(args.source)

    # Determine N robustly
    N = full_path.shape[0] if full_path.ndim == 1 else full_path.shape[1]

    # Single-face mask
    sfs = second_face_score if second_face_score.ndim == 1 else second_face_score.reshape(-1)
    single_mask = np.isnan(sfs) | (sfs <= 0)

    # Face score mask
    fs = face_score if face_score.ndim == 1 else face_score.reshape(-1)
    fs_mask = fs > args.face_score_gt

    # Iterate and select
    os.makedirs(args.dest, exist_ok=True)
    manifest_rows: List[Dict] = []

    counts = {
        'total': int(N),
        'single_face': int(np.sum(single_mask)),
        'face_score_gt': args.face_score_gt,
        'min_face_bbox_gt': args.min_face_bbox_gt,
        'selected': 0,
        'gender': {'male': 0, 'female': 0, 'unknown': 0}
    }

    selected = 0
    for i in range(N):
        if not single_mask[i]:
            continue
        if not fs_mask[i]:
            continue
        mfb = compute_min_face_bbox(face_location, i)
        if not (mfb > args.min_face_bbox_gt):
            continue

        rel = get_rel_path(full_path, i)
        src_path = os.path.join(args.source, rel)
        if not os.path.exists(src_path):
            continue

        dst_path = os.path.join(args.dest, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            shutil.copy2(src_path, dst_path)
        except Exception:
            continue

        g = gender[i]
        if np.isnan(g):
            counts['gender']['unknown'] += 1
            g_str = 'unknown'
        elif int(g) == 1:
            counts['gender']['male'] += 1
            g_str = 'male'
        else:
            counts['gender']['female'] += 1
            g_str = 'female'

        manifest_rows.append({
            'rel_path': rel,
            'abs_path': dst_path,
            'face_score': float(fs[i]),
            'min_face_bbox': float(mfb),
            'gender': g_str,
        })
        selected += 1
        counts['selected'] = selected

        if args.limit and selected >= args.limit:
            break

    # Write manifest and summary
    manifest_path = os.path.join(args.dest, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        fieldnames = ['rel_path', 'abs_path', 'face_score', 'min_face_bbox', 'gender']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    summary_path = os.path.join(args.dest, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(counts, f, indent=2)

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main() 