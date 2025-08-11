#!/usr/bin/env python3
"""
Run API-based evaluation against the live backend to ensure identical preprocessing and model behavior.

Usage examples:

# Evaluate using a CSV manifest with columns: image_path, age
python eval/run_api_eval.py --manifest /path/to/manifest.csv --output predictions.csv

# Evaluate UTKFace by scanning a directory and parsing age from filename
python eval/run_api_eval.py --data-root /path/to/UTKFace --parse utk --output predictions.csv --limit 500

# Evaluate Wiki-like dataset (birth year/date and photo year in filename)
python eval/run_api_eval.py --data-root "/path/to/filtered_wiki/batch 1/folder 1" --parse wiki --output predictions.csv

# Faster stratified sample across age buckets (e.g., 20 per bucket)
python eval/run_api_eval.py --data-root "/path/to/wiki_crop/01" --parse wiki --stratify-per-bucket 20 --shuffle --output predictions.csv

# Customize API base (defaults to http://localhost:8001/api)
python eval/run_api_eval.py --base-url http://localhost:8001/api --manifest my.csv --output out.csv
"""

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image


DEFAULT_BASE_URL = "http://localhost:8001/api"
DISABLE_CHATGPT_HEADERS = {"x-disable-chatgpt": "true"}


@dataclass
class EvalItem:
    image_path: str
    actual_age: Optional[float]


def read_manifest(manifest_path: str) -> List[EvalItem]:
    items: List[EvalItem] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "age"}
        if not required.issubset(reader.fieldnames or {}):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}; found {reader.fieldnames}")
        for row in reader:
            image_path = row["image_path"].strip()
            age_raw = row["age"].strip()
            age_value: Optional[float] = None
            if age_raw != "":
                try:
                    age_value = float(age_raw)
                except ValueError:
                    age_value = None
            items.append(EvalItem(image_path=image_path, actual_age=age_value))
    return items


def parse_age_from_utk_filename(filename: str) -> Optional[float]:
    base = os.path.basename(filename)
    m = re.match(r"^(\d+)_", base)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def parse_age_from_wiki_filename(filename: str) -> Optional[float]:
    # Heuristic: find 4-digit years in the filename; assume first is birth year, last is photo year
    years = re.findall(r"(?<!\d)(\d{4})(?!\d)", filename)
    if len(years) < 2:
        return None
    try:
        birth_year = int(years[0])
        photo_year = int(years[-1])
        if 1800 <= birth_year <= 2025 and 1800 <= photo_year <= 2025 and photo_year >= birth_year:
            age = photo_year - birth_year
            if 0 <= age <= 120:
                return float(age)
    except Exception:
        return None
    return None


def scan_utkface(data_root: str) -> List[EvalItem]:
    items: List[EvalItem] = []
    for root, _dirs, files in os.walk(data_root):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, name)
                items.append(EvalItem(image_path=path, actual_age=parse_age_from_utk_filename(name)))
    return items


def scan_wiki(data_root: str) -> List[EvalItem]:
    items: List[EvalItem] = []
    for root, _dirs, files in os.walk(data_root):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, name)
                items.append(EvalItem(image_path=path, actual_age=parse_age_from_wiki_filename(name)))
    return items


def warmup_server(base_url: str, timeout_s: int = 5) -> None:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout_s)
        if resp.ok:
            data = resp.json()
            print(f"Health: {json.dumps(data)}")
        else:
            print(f"Health check returned status {resp.status_code}")
    except Exception as e:
        print(f"Health check failed (continuing anyway): {e}")


@dataclass
class Prediction:
    image_path: str
    actual_age: Optional[float]
    harvard_age: Optional[float]
    deepface_age: Optional[float]
    chatgpt_age: Optional[float]
    face_confidence: Optional[float]
    num_faces: int
    error: Optional[str]
    image_width: Optional[int]
    image_height: Optional[int]


def call_analyze_face(base_url: str, image_path: str, timeout_s: int = 300) -> Tuple[Optional[dict], Optional[str]]:
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            resp = requests.post(f"{base_url}/analyze-face", files=files, headers=DISABLE_CHATGPT_HEADERS, timeout=timeout_s)
        if not resp.ok:
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        return resp.json(), None
    except Exception as e:
        return None, str(e)


def extract_prediction(json_payload: dict, image_path: str, actual_age: Optional[float], img_wh: Tuple[Optional[int], Optional[int]]) -> Prediction:
    w, h = img_wh
    try:
        faces = json_payload.get("faces", []) if isinstance(json_payload, dict) else []
        num_faces = len(faces)
        if num_faces == 0:
            return Prediction(
                image_path=image_path,
                actual_age=actual_age,
                harvard_age=None,
                deepface_age=None,
                chatgpt_age=None,
                face_confidence=None,
                num_faces=0,
                error="no_face",
                image_width=w,
                image_height=h,
            )
        best = max(faces, key=lambda f: f.get("confidence", 0.0))
        return Prediction(
            image_path=image_path,
            actual_age=actual_age,
            harvard_age=best.get("age_harvard"),
            deepface_age=best.get("age_deepface"),
            chatgpt_age=best.get("age_chatgpt"),
            face_confidence=best.get("confidence"),
            num_faces=num_faces,
            error=None,
            image_width=w,
            image_height=h,
        )
    except Exception as e:
        return Prediction(
            image_path=image_path,
            actual_age=actual_age,
            harvard_age=None,
            deepface_age=None,
            chatgpt_age=None,
            face_confidence=None,
            num_faces=0,
            error=f"parse_error: {e}",
            image_width=w,
            image_height=h,
        )


def compute_mae_rmse(pairs: Iterable[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    diffs: List[float] = []
    for gt, pred in pairs:
        if gt is None or pred is None:
            continue
        try:
            diffs.append(float(gt) - float(pred))
        except Exception:
            continue
    if not diffs:
        return None, None
    abs_errors = [abs(d) for d in diffs]
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    return mae, rmse


def compute_extended_stats(pairs: Iterable[Tuple[float, float]]):
    diffs: List[float] = []
    for gt, pred in pairs:
        if gt is None or pred is None:
            continue
        try:
            diffs.append(float(gt) - float(pred))
        except Exception:
            continue
    if not diffs:
        return {"n": 0, "mae": None, "rmse": None, "median_abs": None, "p90_abs": None}
    abs_errors = [abs(d) for d in diffs]
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    median_abs = statistics.median(abs_errors)
    p90_abs = sorted(abs_errors)[int(0.9 * (len(abs_errors) - 1))]
    return {"n": len(diffs), "mae": mae, "rmse": rmse, "median_abs": median_abs, "p90_abs": p90_abs}


def age_bucket(age: float) -> str:
    if age is None:
        return "unknown"
    try:
        a = int(round(float(age)))
    except Exception:
        return "unknown"
    if a < 0:
        return "<0"
    if a >= 90:
        return "90+"
    lo = (a // 10) * 10
    hi = lo + 9
    return f"{lo}-{hi}"


def grouped_stats(predictions: List[Prediction], model: str) -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    buckets: Dict[str, List[Tuple[float, float]]] = {}
    for p in predictions:
        if p.actual_age is None:
            continue
        if model == "harvard":
            pred_val = p.harvard_age
        else:
            pred_val = p.deepface_age
        b = age_bucket(p.actual_age)
        buckets.setdefault(b, []).append((p.actual_age, pred_val))
    for b, pairs in buckets.items():
        stats[b] = compute_extended_stats(pairs)
    return stats


def pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def stratified_sample(items: List[EvalItem], per_bucket: int, seed: int) -> List[EvalItem]:
    if per_bucket <= 0:
        return items
    rng = random.Random(seed)
    buckets: Dict[str, List[EvalItem]] = {}
    for it in items:
        b = age_bucket(it.actual_age) if it.actual_age is not None else "unknown"
        buckets.setdefault(b, []).append(it)
    sampled: List[EvalItem] = []
    for b, group in buckets.items():
        if len(group) <= per_bucket:
            sampled.extend(group)
        else:
            sampled.extend(rng.sample(group, per_bucket))
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Harvard/DeepFace via live API")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL (default: http://localhost:8001/api)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", help="CSV with columns: image_path, age")
    group.add_argument("--data-root", help="Directory to scan")
    parser.add_argument("--parse", choices=["utk", "wiki"], default="utk", help="How to parse ages from filenames when using --data-root")
    parser.add_argument("--output", required=True, help="Output CSV for predictions")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to evaluate")
    parser.add_argument("--stratify-per-bucket", type=int, default=0, help="If >0, sample up to N items per age bucket")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle input set before limiting")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle/stratify seed")
    args = parser.parse_args()

    # Prepare items
    if args.manifest:
        items = read_manifest(args.manifest)
    else:
        items = scan_wiki(args.data_root) if args.parse == "wiki" else scan_utkface(args.data_root)

    # Stratified sampling across age buckets
    if args.stratify_per_bucket and args.stratify_per_bucket > 0:
        items = stratified_sample(items, per_bucket=args.stratify_per_bucket, seed=args.seed)

    if args.shuffle:
        random.Random(args.seed).shuffle(items)

    if args.limit is not None:
        items = items[: args.limit]

    print(f"Loaded {len(items)} items")

    # Warm up server (non-fatal)
    warmup_server(args.base_url, timeout_s=5)

    predictions: List[Prediction] = []

    start_time = time.time()
    for idx, item in enumerate(items, start=1):
        if idx % 10 == 1:
            elapsed = time.time() - start_time
            print(f"[{idx}/{len(items)}] elapsed={elapsed:.1f}s")
        # Read image dimensions
        img_wh: Tuple[Optional[int], Optional[int]]
        try:
            with Image.open(item.image_path) as im:
                w, h = im.size
                img_wh = (int(w), int(h))
        except Exception:
            img_wh = (None, None)
        payload, err = call_analyze_face(args.base_url, item.image_path)
        if err is not None:
            predictions.append(Prediction(
                image_path=item.image_path,
                actual_age=item.actual_age,
                harvard_age=None,
                deepface_age=None,
                chatgpt_age=None,
                face_confidence=None,
                num_faces=0,
                error=err,
                image_width=img_wh[0],
                image_height=img_wh[1],
            ))
            continue
        predictions.append(extract_prediction(payload, item.image_path, item.actual_age, img_wh))

    # Save CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "actual_age",
            "harvard_age",
            "deepface_age",
            "chatgpt_age",
            "face_confidence",
            "num_faces",
            "image_width",
            "image_height",
            "error"
        ])
        for p in predictions:
            writer.writerow([
                p.image_path,
                p.actual_age if p.actual_age is not None else "",
                p.harvard_age if p.harvard_age is not None else "",
                p.deepface_age if p.deepface_age is not None else "",
                p.chatgpt_age if p.chatgpt_age is not None else "",
                f"{p.face_confidence:.4f}" if isinstance(p.face_confidence, (int, float)) else "",
                p.num_faces,
                p.image_width if p.image_width is not None else "",
                p.image_height if p.image_height is not None else "",
                p.error or ""
            ])

    # Metrics
    harvard_pairs = [
        (p.actual_age, p.harvard_age)
        for p in predictions
        if p.actual_age is not None and p.harvard_age is not None
    ]
    deepface_pairs = [
        (p.actual_age, p.deepface_age)
        for p in predictions
        if p.actual_age is not None and p.deepface_age is not None
    ]

    harvard_mae, harvard_rmse = compute_mae_rmse(harvard_pairs)
    deepface_mae, deepface_rmse = compute_mae_rmse(deepface_pairs)

    print("\nSaved:", args.output)
    print("Counts:")
    print("  total:", len(predictions))
    print("  with_actual_age:", sum(1 for p in predictions if p.actual_age is not None))
    print("  harvard_pred:", sum(1 for p in predictions if p.harvard_age is not None))
    print("  deepface_pred:", sum(1 for p in predictions if p.deepface_age is not None))

    if harvard_mae is not None:
        print(f"Harvard   MAE={harvard_mae:.3f}  RMSE={harvard_rmse:.3f}  (n={len(harvard_pairs)})")
    else:
        print("Harvard   MAE/RMSE: n/a")
    if deepface_mae is not None:
        print(f"DeepFace  MAE={deepface_mae:.3f}  RMSE={deepface_rmse:.3f}  (n={len(deepface_pairs)})")
    else:
        print("DeepFace  MAE/RMSE: n/a")

    # Per-bucket stats
    per_bucket_h = grouped_stats(predictions, model="harvard")
    per_bucket_d = grouped_stats(predictions, model="deepface")

    # Coverage by bucket
    bucket_counts: Dict[str, int] = {}
    pred_counts_h: Dict[str, int] = {}
    pred_counts_d: Dict[str, int] = {}
    for p in predictions:
        b = age_bucket(p.actual_age) if p.actual_age is not None else "unknown"
        bucket_counts[b] = bucket_counts.get(b, 0) + (1 if p.actual_age is not None else 0)
        if p.actual_age is not None and p.harvard_age is not None:
            pred_counts_h[b] = pred_counts_h.get(b, 0) + 1
        if p.actual_age is not None and p.deepface_age is not None:
            pred_counts_d[b] = pred_counts_d.get(b, 0) + 1

    # Correlations: |error| vs age, |error| vs min(image side)
    def build_corr_lists(model: str) -> Tuple[List[float], List[float], List[float]]:
        abs_errs: List[float] = []
        ages: List[float] = []
        sizes: List[float] = []
        for p in predictions:
            if model == "harvard":
                pred = p.harvard_age
            else:
                pred = p.deepface_age
            if pred is None or p.actual_age is None:
                continue
            try:
                abs_errs.append(abs(float(pred) - float(p.actual_age)))
                ages.append(float(p.actual_age))
                if p.image_width and p.image_height:
                    sizes.append(float(min(p.image_width, p.image_height)))
                else:
                    sizes.append(float('nan'))
            except Exception:
                continue
        # filter Nans in sizes
        filtered = [(e, a, s) for (e, a, s) in zip(abs_errs, ages, sizes) if not math.isnan(s)]
        if filtered:
            abs_errs2, ages2, sizes2 = zip(*filtered)
            return list(abs_errs2), list(ages2), list(sizes2)
        else:
            return abs_errs, ages, []

    h_abs, h_age, h_size = build_corr_lists("harvard")
    d_abs, d_age, d_size = build_corr_lists("deepface")

    corr_h_age = pearson_corr(h_abs, h_age) if h_abs and h_age else None
    corr_d_age = pearson_corr(d_abs, d_age) if d_abs and d_age else None
    corr_h_size = pearson_corr(h_abs, h_size) if h_abs and h_size else None
    corr_d_size = pearson_corr(d_abs, d_size) if d_abs and d_size else None

    # Save metrics JSON
    metrics_path = os.path.splitext(args.output)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump({
            "overall": {
                "harvard": {"mae": harvard_mae, "rmse": harvard_rmse, "n": len(harvard_pairs)},
                "deepface": {"mae": deepface_mae, "rmse": deepface_rmse, "n": len(deepface_pairs)}
            },
            "per_bucket": {
                "harvard": per_bucket_h,
                "deepface": per_bucket_d
            },
            "coverage": {
                "bucket_counts": bucket_counts,
                "harvard_pred_counts": pred_counts_h,
                "deepface_pred_counts": pred_counts_d
            },
            "correlations": {
                "harvard": {"abs_error_vs_age": corr_h_age, "abs_error_vs_min_image_side": corr_h_size},
                "deepface": {"abs_error_vs_age": corr_d_age, "abs_error_vs_min_image_side": corr_d_size}
            }
        }, mf, indent=2)
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main() 