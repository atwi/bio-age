#!/usr/bin/env python3
"""
Fit robust monotonic (isotonic) calibration for Harvard predictions with a young-focused map and a global map, blended smoothly.
- Train on WIKI adults train predictions (CSV with columns: image_path, actual_age, harvard_age)
- Validate on WIKI adults val predictions
- Outputs:
  - Prints pre/post MAE overall and by age buckets (esp. <40)
  - Saves JSON with piecewise-linear maps and chosen blend window

Usage:
  python eval/fit_isotonic_calibration.py \
    --train evaluation_results/wiki_train_calib.csv \
    --val   evaluation_results/wiki_val_calib.csv \
    --out   calibration/harvard_isotonic_young_global.json
"""
import argparse
import csv
import json
import math
import os
from typing import List, Tuple, Dict

import numpy as np


def read_preds(path: str) -> List[Tuple[float, float]]:
    pairs = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                y_true = float(row.get('actual_age'))
                y_pred = float(row.get('harvard_age'))
            except Exception:
                continue
            if not (0 <= y_true <= 120):
                continue
            if not (0 <= y_pred <= 120):
                continue
            pairs.append((y_pred, y_true))
    return pairs


def unique_by_x_sorted(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sort by x and average duplicates
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    xs = []
    ys = []
    ws = []
    i = 0
    n = len(x_sorted)
    while i < n:
        j = i + 1
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        xs.append(float(x_sorted[i]))
        ys.append(float(y_sorted[i:j].mean()))
        ws.append(float(j - i))
        i = j
    return np.array(xs, dtype=float), np.array(ys, dtype=float), np.array(ws, dtype=float)


def pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    # Pool Adjacent Violators Algorithm for isotonic regression (increasing)
    n = len(y)
    g = y.astype(float).copy()
    w = w.astype(float).copy()
    idx = list(range(n))
    # Each block has (start, end, value, weight)
    starts = list(range(n))
    ends = list(range(n))
    vals = g.tolist()
    wgts = w.tolist()

    k = 0
    while k < len(vals) - 1:
        if vals[k] <= vals[k + 1]:
            k += 1
            continue
        # Merge blocks k and k+1
        new_w = wgts[k] + wgts[k + 1]
        new_v = (wgts[k] * vals[k] + wgts[k + 1] * vals[k + 1]) / new_w
        wgts[k] = new_w
        vals[k] = new_v
        ends[k] = ends[k + 1]
        del wgts[k + 1]
        del vals[k + 1]
        del starts[k + 1]
        del ends[k + 1]
        # Step back if possible to enforce monotonicity
        k = max(k - 1, 0)
    # Expand block values back to pointwise fitted values
    fitted = np.empty(n, dtype=float)
    for v, s, e in zip(vals, starts, ends):
        fitted[s:e + 1] = v
    return fitted


def fit_isotonic_map(pairs: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    # pairs: (x = y_pred, y = y_true)
    if not pairs:
        return np.array([0.0, 120.0]), np.array([0.0, 120.0])
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    x_u, y_u, w_u = unique_by_x_sorted(x, y)
    y_fit = pava(y_u, w_u)
    # Ensure strictly within [0,120]
    y_fit = np.clip(y_fit, 0.0, 120.0)
    # Remove any NaNs or degenerate
    if len(x_u) == 1:
        return np.array([0.0, 120.0]), np.array([y_fit[0], y_fit[0]])
    return x_u, y_fit


def apply_iso(x_thr: np.ndarray, y_thr: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    # Piecewise linear interpolation with clipping
    x_min = float(x_thr[0])
    x_max = float(x_thr[-1])
    x_clipped = np.clip(x_new, x_min, x_max)
    return np.interp(x_clipped, x_thr, y_thr)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if len(a) else float('nan')


def bucket(age: float) -> str:
    a = int(round(age))
    if a < 18:
        return '<18'
    if a <= 24:
        return '18-24'
    if a <= 29:
        return '25-29'
    if a <= 34:
        return '30-34'
    if a <= 39:
        return '35-39'
    if a <= 49:
        return '40-49'
    if a <= 59:
        return '50-59'
    if a <= 69:
        return '60-69'
    return '70+'


def eval_metrics(pairs_val: List[Tuple[float, float]], xg: np.ndarray, yg: np.ndarray, xy: np.ndarray, yy: np.ndarray, t0: float, t1: float) -> Dict[str, float]:
    yp = np.array([p[0] for p in pairs_val], dtype=float)
    yt = np.array([p[1] for p in pairs_val], dtype=float)
    g = apply_iso(xg, yg, yp)
    y = apply_iso(xy, yy, yp)
    # Blend by prediction with linear weight in [t0,t1]
    w = np.clip((yp - t0) / max(t1 - t0, 1e-6), 0.0, 1.0)
    yb = (1 - w) * y + w * g
    base_mae = mae(yp, yt)
    cal_mae = mae(yb, yt)
    # Bucket MAEs
    buckets = {}
    for b in ['18-24', '25-29', '30-34', '35-39', '40-49', '50-59', '60-69', '70+']:
        idx = [i for i in range(len(yt)) if bucket(yt[i]) == b]
        if not idx:
            continue
        idx = np.array(idx, dtype=int)
        buckets[b] = {
            'base_mae': mae(yp[idx], yt[idx]),
            'cal_mae': mae(yb[idx], yt[idx]),
            'n': int(len(idx)),
        }
    return {
        'base_mae': base_mae,
        'cal_mae': cal_mae,
        'improvement_pct': 100.0 * (base_mae - cal_mae) / max(base_mae, 1e-6),
        'buckets': buckets,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    train_pairs = read_preds(args.train)
    val_pairs = read_preds(args.val)

    # Fit global and young (18-39 by y_true) on train
    train_pairs_sorted = sorted(train_pairs, key=lambda p: p[0])
    xg, yg = fit_isotonic_map(train_pairs_sorted)
    train_young = [p for p in train_pairs_sorted if 18.0 <= p[1] <= 39.0]
    xy, yy = fit_isotonic_map(train_young)

    # Candidate blend windows (by prediction)
    candidates = [(36.0, 41.0), (37.0, 42.0), (38.0, 43.0), (39.0, 44.0), (35.0, 40.0)]
    best = None
    best_metrics = None
    for (t0, t1) in candidates:
        m = eval_metrics(val_pairs, xg, yg, xy, yy, t0, t1)
        if best is None or m['cal_mae'] < best_metrics['cal_mae']:
            best = (t0, t1)
            best_metrics = m

    # Report
    print(f"Base MAE (val): {best_metrics['base_mae']:.3f}")
    print(f"Calibrated MAE (val): {best_metrics['cal_mae']:.3f}  (improvement {best_metrics['improvement_pct']:.1f}%)")
    print(f"Blend window (pred): [{best[0]}, {best[1]}]")
    print("Per-bucket MAE (val):")
    for b, s in best_metrics['buckets'].items():
        print(f"  {b:>6}  base={s['base_mae']:.2f}  cal={s['cal_mae']:.2f}  n={s['n']}")

    # Save JSON mapping
    payload = {
        'f_global': {
            'x_thresholds': xg.tolist(),
            'y_values': yg.tolist(),
        },
        'f_young': {
            'x_thresholds': xy.tolist(),
            'y_values': yy.tolist(),
        },
        'blend_window_pred': {'t0': best[0], 't1': best[1]},
        'val_metrics': best_metrics,
        'meta': {
            'train_size': len(train_pairs_sorted),
            'train_young_size': len(train_young),
            'val_size': len(val_pairs),
        },
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved calibration to {args.out}")


if __name__ == '__main__':
    main() 