"""
barcode_extent.py - find the FULL extent of a 1D barcode in a rectified mirror image and
reject look-alikes (ruler ticks, printed text). Companion to barcode_region_detector.py:
that script proposes seed regions; this one measures the true barcode box and scores it with
interpretable features instead of a fixed 0.85.

How it works
  A 1D barcode is a STACK of parallel bars with irregular thickness and spacing.
  1. dark mask = pixels darker than their local mean
  2. close along the bar direction so each bar becomes one solid blob
  3. keep blobs that are long and thin; group blobs that line up (same x-range)
  4. the barcode = the largest group; its union is the box
  Tried for both bar orientations (bars horizontal / bars vertical).

Feature scores (each 0..1, printed so you can see WHY a region ranks high)
  count      enough bars (>= 8)
  irregular  bar thickness / gap spacing is NOT periodic (rulers are periodic -> low score)
  classes    several distinct bar thicknesses (barcodes have 2-4 widths; ruler ticks mostly 1-2)
  contrast   bars are really dark against a bright background

The overall score is a heuristic ranking, NOT a probability.

Usage
  python barcode_extent.py IMG_50_left_rectified.jpg --side left --out out.png
  python barcode_extent.py rect.jpg --side right --ignore-below-mm 124   (optional ruler band)
Rectified images are 10 px/mm: hinge-edge distance = (width - x)/10 for the LEFT mirror, x/10 for the RIGHT.
"""

import argparse
import json
import sys

import cv2
import numpy as np


def _blobs(gray, bar_axis):
    """Return list of (x, y, w, h) blobs for bars lying along `bar_axis` ('h' or 'v')."""
    g = gray if bar_axis == "h" else gray.T
    loc = cv2.GaussianBlur(g, (0, 0), 25)
    dark = (g.astype(np.int16) < loc.astype(np.int16) - 16).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    n, _, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w >= 25 and h <= 45 and w / max(h, 1) >= 2.5 and a >= 0.5 * w * h:
            out.append((int(x), int(y), int(w), int(h)))
    return out


def _stacks(blobs):
    """All distinct candidate stacks: blobs that line up (same x-range, same length), as vertical chains."""
    seen, out = set(), []
    for b in blobs:
        grp = [c for c in blobs
               if min(b[0] + b[2], c[0] + c[2]) - max(b[0], c[0]) >= 0.8 * max(b[2], c[2])
               and abs(c[2] - b[2]) <= 0.2 * max(b[2], c[2])
               and abs(c[0] - b[0]) <= 0.15 * max(b[2], c[2])]
        grp.sort(key=lambda c: c[1])
        chain = [grp[0]] if grp else []
        chains = []
        for c in grp[1:]:
            prev = chain[-1]
            if c[1] - (prev[1] + prev[3]) <= 3 * max(prev[3], 6) + 30:
                chain.append(c)
            else:
                chains.append(chain)
                chain = [c]
        chains.append(chain)
        for ch in chains:
            key = tuple(sorted(ch))
            if len(ch) >= 6 and key not in seen:
                seen.add(key)
                out.append(ch)
    return out


def _features(stack, gray_t):
    th = np.array([b[3] for b in stack], float)                 # bar thicknesses
    ys = np.array([b[1] for b in stack], float)
    gaps = ys[1:] - (ys[:-1] + th[:-1])
    pitch = ys[1:] - ys[:-1]
    f = {}
    f["count"] = float(np.clip((len(stack) - 5) / 10.0, 0, 1))
    ln = np.array([b[2] for b in stack], float)                  # bar LENGTH (barcode: all equal)
    f["uniform"] = float(np.clip(1.0 - (np.std(ln) / (np.mean(ln) + 1e-6)) / 0.15, 0, 1))
    cv_pitch = float(np.std(pitch) / (np.mean(pitch) + 1e-6)) if len(pitch) > 2 else 0.0
    f["irregular"] = float(np.clip(cv_pitch / 0.5, 0, 1))        # periodic ruler -> small CV
    q = np.round(th / max(np.min(th), 1.0))
    f["classes"] = float(np.clip((len(np.unique(np.clip(q, 1, 8))) - 1) / 3.0, 0, 1))
    x0 = min(b[0] for b in stack)
    x1 = max(b[0] + b[2] for b in stack)
    y0, y1 = int(ys.min()), int((ys + th).max())
    roi = gray_t[y0:y1, x0:x1]
    if roi.size:
        lo, hi = np.percentile(roi, 10), np.percentile(roi, 90)
        f["contrast"] = float(np.clip((hi - lo) / 110.0, 0, 1))
    else:
        f["contrast"] = 0.0
    base = float(np.mean([f["count"], f["irregular"], f["classes"], f["contrast"], f["uniform"]]))
    gate = 1.0 if (f["uniform"] > 0.5 and f["irregular"] > 0.25 and f["classes"] > 0.3) else 0.3
    f["score"] = round(base * gate, 3)
    return {k: round(v, 3) for k, v in f.items()}


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx)); ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def find_barcodes(img, ignore_below_px=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if ignore_below_px is not None:
        gray = gray.copy()
        gray[int(ignore_below_px):, :] = 255
    cands = []
    for axis in ("h", "v"):
        gt = gray if axis == "h" else gray.T
        for stack in _stacks(_blobs(gray, axis)):
            feats = _features(stack, gt)
            x0 = min(b[0] for b in stack); x1 = max(b[0] + b[2] for b in stack)
            y0 = min(b[1] for b in stack); y1 = max(b[1] + b[3] for b in stack)
            box = (x0, y0, x1 - x0, y1 - y0) if axis == "h" else (y0, x0, y1 - y0, x1 - x0)
            cands.append({"bar_axis": "horizontal_bars" if axis == "h" else "vertical_bars",
                          "n_bars": len(stack), "box_xywh": [int(v) for v in box],
                          "features": feats, "score": feats["score"]})
    cands.sort(key=lambda r: -r["score"])
    kept = []
    for c in cands:                      # non-maximum suppression
        if all(_iou(c["box_xywh"], k["box_xywh"]) < 0.3 for k in kept):
            kept.append(c)
    return kept[:5]


def hinge_edge_mm(box, side, width_px=1000, px_per_mm=10.0):
    x, y, w, h = box
    if side == "left":
        near, far = (width_px - (x + w)) / px_per_mm, (width_px - x) / px_per_mm
    else:
        near, far = x / px_per_mm, (x + w) / px_per_mm
    return {"near": round(near, 1), "far": round(far, 1), "centre": round((near + far) / 2, 1),
            "height_range_mm_from_top": [round(y / px_per_mm, 1), round((y + h) / px_per_mm, 1)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--side", choices=("left", "right"), required=True)
    ap.add_argument("--ignore-below-mm", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        sys.exit("cannot read image")
    r = find_barcodes(img, None if a.ignore_below_mm is None else a.ignore_below_mm * 10)
    if not r:
        print("no barcode-like stack found")
    for k, d in enumerate(r):
        d["hinge_edge_mm"] = hinge_edge_mm(d["box_xywh"], a.side, img.shape[1])
        print(k, json.dumps(d))
    if a.out:
        vis = img.copy()
        for d in r:
            x, y, w, h = d["box_xywh"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(vis, f"{d['score']:.2f}", (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite(a.out, vis)


if __name__ == "__main__":
    main()