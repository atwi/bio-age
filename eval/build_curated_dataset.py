#!/usr/bin/env python3
"""
Build a curated dataset from a source image directory.

Two modes:
- detect (default): local face detection (MTCNN) to enforce one high-confidence face
- wiki: use IMDB-WIKI metadata (wiki.mat) to enforce one face and confidence, plus recency

Common filters:
- minimum image side length (default 224 px)
- photo taken within the last N years (default 20)
- copy selected images to destination directory preserving relative paths
- write manifest.csv and summary.json

Examples:
# Use wiki metadata only (no detectors/models), recommended for wiki_crop
python eval/build_curated_dataset.py \
  --mode wiki \
  --source "/Users/alext/Downloads/Datasets/wiki_crop" \
  --wiki-mat "/Users/alext/Downloads/Datasets/wiki_crop/wiki.mat" \
  --dest "/Users/alext/Downloads/Datasets/wiki_curated_20y_faceScore1p0_single_min224" \
  --years 20 --face-score-min 1.0 --min-side 224

# Use local detection if metadata unavailable (slower)
python eval/build_curated_dataset.py \
  --mode detect \
  --source "/path/to/images" \
  --dest "/path/to/curated" \
  --min-conf 0.95 --years 20 --min-side 224
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from PIL import Image

try:
    import numpy as np
except Exception:
    np = None


@dataclass
class SelectionCounts:
    total_seen: int = 0
    exists_count: int = 0
    one_face_count: int = 0
    confidence_ok_count: int = 0
    recent_ok_count: int = 0
    size_ok_count: int = 0
    selected_count: int = 0


def min_side_of_image(path: str) -> Optional[int]:
    try:
        with Image.open(path) as im:
            w, h = im.size
            return min(w, h)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curate dataset from images using detection or wiki metadata")
    p.add_argument("--mode", choices=["detect", "wiki"], default="detect", help="Filtering mode")
    p.add_argument("--source", required=True, help="Source root directory (e.g., wiki_crop)")
    p.add_argument("--dest", required=True, help="Destination root to copy selected images")

    # Shared filters
    p.add_argument("--min-side", type=int, default=224, help="Minimum image min-side in pixels")
    p.add_argument("--years", type=int, default=20, help="Keep photos taken within last N years")
    p.add_argument("--limit", type=int, default=0, help="Stop after selecting/copying this many (0 = no limit)")

    # Detect mode options
    p.add_argument("--min-conf", type=float, default=0.95, help="Min face confidence (detect mode)")

    # Wiki mode options
    p.add_argument("--wiki-mat", default=None, help="Path to wiki.mat (default: <source>/wiki.mat)")
    p.add_argument("--face-score-min", type=float, default=1.0, help="Min face_score threshold (wiki mode)")

    return p.parse_args()


# -------- detect mode (kept from previous version) --------

def ensure_mtcnn():
    from mtcnn import MTCNN  # type: ignore
    return MTCNN()


def detect_mode_select(source: str, dest: str, min_side: int, years: int, min_conf: float, limit: int) -> Tuple[SelectionCounts, List[Dict]]:
    import re
    from datetime import datetime

    detector = ensure_mtcnn()
    counts = SelectionCounts()
    selected_rows: List[Dict] = []

    # Heuristic to infer photo year from wiki filename if present: *_YYYY.* or *_YYYY-..*
    year_pattern = re.compile(r"_(\d{4})(?:\D|$)")
    cutoff_year = date.today().year - years

    for root, _dirs, files in os.walk(source):
        for name in files:
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            counts.total_seen += 1
            src_path = os.path.join(root, name)
            if not os.path.exists(src_path):
                continue
            counts.exists_count += 1

            # Size filter
            side = min_side_of_image(src_path)
            if side is None:
                continue
            if side >= min_side:
                counts.size_ok_count += 1
            else:
                continue

            # Year filter from filename if available
            m = year_pattern.search(name)
            photo_year_ok = True
            photo_year_val: Optional[int] = None
            if m:
                try:
                    photo_year_val = int(m.group(1))
                    photo_year_ok = photo_year_val >= cutoff_year
                except Exception:
                    photo_year_ok = True
            if photo_year_ok:
                counts.recent_ok_count += 1
            else:
                continue

            # Face detection
            try:
                with Image.open(src_path) as im:
                    im_rgb = im.convert("RGB")
                dets = detector.detect_faces(np.array(im_rgb))
            except Exception:
                continue

            if len(dets) == 1:
                counts.one_face_count += 1
            else:
                continue

            conf = float(dets[0].get("confidence", 0.0)) if dets else 0.0
            if conf >= min_conf:
                counts.confidence_ok_count += 1
            else:
                continue

            # Passed all filters
            rel_path = os.path.relpath(src_path, start=source)
            dst_path = os.path.join(dest, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            try:
                shutil.copy2(src_path, dst_path)
            except Exception:
                continue

            selected_rows.append({
                "rel_path": rel_path,
                "abs_path": dst_path,
                "min_side": side,
                "photo_year": photo_year_val,
                "mode": "detect",
                "face_confidence": conf,
            })
            counts.selected_count += 1

            if limit and counts.selected_count >= limit:
                return counts, selected_rows

    return counts, selected_rows


# -------- wiki mode --------

def wiki_mode_select(source: str, dest: str, wiki_mat_path: str, min_side: int, years: int, face_score_min: float, limit: int) -> Tuple[SelectionCounts, List[Dict]]:
    if np is None:
        raise RuntimeError("NumPy is required for wiki mode")

    try:
        import scipy.io as sio
    except Exception as e:
        raise RuntimeError("scipy is required to read wiki.mat") from e

    counts = SelectionCounts()
    selected_rows: List[Dict] = []

    if not os.path.exists(wiki_mat_path):
        raise FileNotFoundError(f"wiki.mat not found at {wiki_mat_path}")

    mat = sio.loadmat(wiki_mat_path)
    wiki = mat['wiki'][0][0]
    # Field order: ['dob','photo_taken','full_path','face_score','second_face_score','gender']
    dob = wiki[0].squeeze()
    photo_taken = wiki[1].squeeze()
    full_path = wiki[2].squeeze()
    face_score = wiki[3].squeeze().astype(float)
    second_face_score = wiki[4].squeeze()

    # Coerce second_face_score to float array if possible; otherwise treat as NaN array
    try:
        second_face_score = second_face_score.astype(float)
    except Exception:
        second_face_score = np.full_like(face_score, np.nan)

    n = face_score.shape[0]
    cutoff_year = date.today().year - years

    def is_single_face(second_val: float) -> bool:
        # NaN → OK (no second face). <= 0 → OK. > 0 → more than one face
        return (math.isnan(second_val)) or (second_val <= 0.0)

    for i in range(n):
        counts.total_seen += 1
        # Build absolute path from full_path entry
        try:
            rel = full_path[i]
            if isinstance(rel, np.ndarray):
                rel = rel.item()
            if isinstance(rel, bytes):
                rel = rel.decode('utf-8', errors='ignore')
            rel_str: str = str(rel)
        except Exception:
            continue

        src_path = os.path.join(source, rel_str)
        if not os.path.exists(src_path):
            continue
        counts.exists_count += 1

        # Face count and confidence using metadata
        fs = float(face_score[i]) if not math.isnan(face_score[i]) else -math.inf
        sfs = float(second_face_score[i]) if not math.isnan(second_face_score[i]) else float('nan')
        if not is_single_face(sfs):
            continue
        counts.one_face_count += 1

        if fs >= face_score_min:
            counts.confidence_ok_count += 1
        else:
            continue

        # Recency filter using photo_taken year
        try:
            py = int(photo_taken[i])
        except Exception:
            py = None
        if py is None or py < cutoff_year:
            continue
        counts.recent_ok_count += 1

        # Size filter (min side)
        side = min_side_of_image(src_path)
        if side is None:
            continue
        if side >= min_side:
            counts.size_ok_count += 1
        else:
            continue

        # Passed all filters → copy
        dst_path = os.path.join(dest, rel_str)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            shutil.copy2(src_path, dst_path)
        except Exception:
            continue

        selected_rows.append({
            "rel_path": rel_str,
            "abs_path": dst_path,
            "min_side": side,
            "photo_year": py,
            "mode": "wiki",
            "face_score": fs,
            "second_face_score": sfs,
        })
        counts.selected_count += 1

        if limit and counts.selected_count >= limit:
            break

    return counts, selected_rows


def write_outputs(dest_root: str, counts: SelectionCounts, rows: List[Dict]) -> None:
    os.makedirs(dest_root, exist_ok=True)
    manifest_path = os.path.join(dest_root, "manifest.csv")
    summary_path = os.path.join(dest_root, "summary.json")

    # Write manifest
    if rows:
        fieldnames = sorted(rows[0].keys())
        with open(manifest_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    # Write summary
    with open(summary_path, "w") as f:
        json.dump({
            "total": counts.total_seen,
            "exists": counts.exists_count,
            "one_face": counts.one_face_count,
            "confidence_ok_count": counts.confidence_ok_count,
            "recent_ok_count": counts.recent_ok_count,
            "size_ok_count": counts.size_ok_count,
            "selected": counts.selected_count,
        }, f, indent=2)

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved summary: {summary_path}")


def main() -> None:
    args = parse_args()

    if args.mode == "wiki":
        wiki_mat = args.wiki_mat or os.path.join(args.source, "wiki.mat")
        print(f"Mode=wiki | source={args.source} | wiki_mat={wiki_mat} | dest={args.dest}")
        print(f"Filters: face_score_min={args.face_score_min} one_face=True years={args.years} min_side={args.min_side} limit={args.limit}")
        counts, rows = wiki_mode_select(
            source=args.source,
            dest=args.dest,
            wiki_mat_path=wiki_mat,
            min_side=args.min_side,
            years=args.years,
            face_score_min=args.face_score_min,
            limit=args.limit,
        )
    else:
        print(f"Mode=detect | source={args.source} | dest={args.dest}")
        print(f"Filters: min_conf={args.min_conf} one_face=True years={args.years} min_side={args.min_side} limit={args.limit}")
        counts, rows = detect_mode_select(
            source=args.source,
            dest=args.dest,
            min_side=args.min_side,
            years=args.years,
            min_conf=args.min_conf,
            limit=args.limit,
        )

    print("Counts:")
    print(f"  total={counts.total_seen}")
    print(f"  exists={counts.exists_count}")
    print(f"  one_face={counts.one_face_count}")
    print(f"  confidence_ok_count={counts.confidence_ok_count}")
    print(f"  recent_ok_count={counts.recent_ok_count}")
    print(f"  size_ok_count={counts.size_ok_count}")
    print(f"  selected={counts.selected_count}")

    write_outputs(args.dest, counts, rows)


if __name__ == "__main__":
    main() 