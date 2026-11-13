"""
barcode_region_detector.py  --  Phase 5: automatic barcode localisation
=======================================================================

Finds barcode-like regions in a full camera photo of the flask + two mirrors,
WITHOUT any clicking, and reports each region in the coordinates the Phase 3 /
Phase 4 code already uses.

Where it searches
  * LEFT mirror   : photo is rectified with the Phase 3 homography (10 px/mm)
  * RIGHT mirror  : same, with the right-mirror homography
  * DIRECT view   : the strip between the two mirrors (or --direct-roi, or the
                    whole photo when no calibration is supplied)

What it reports per region
  * bounding box in photo pixels (and in rectified px / mm for mirrors)
  * for mirrors: near/far/centre distance from the HINGE-SIDE glass edge in mm
      left mirror  -> hinge edge is its RIGHT edge
      right mirror -> hinge edge is its LEFT edge
    (same convention as phase4_batch_measure.py, so values feed straight into
     reflection_model_2d_hinge_reference.solve_barcode_position / solve_phi;
     near/far give the barcode BOUNDARIES, not just a midpoint)
  * orientation of the bars (vertical / horizontal in that view)
  * a heuristic proposal score in 0..1.  THIS IS NOT A PROBABILITY.
  * optional decode attempt (OpenCV / zxing-cpp / pyzbar if installed)

Usage
  python barcode_region_detector.py photo.jpg --calib calibration_record_1_.json
  python barcode_region_detector.py photo.jpg --calib mirror_homographies.json --out-dir det
  python barcode_region_detector.py photo.jpg            # no calibration: whole-image detection

Outputs (in --out-dir): <name>_detections.json, <name>_annotated.jpg,
and crops of each region.
"""

import argparse
import json
import os

import cv2
import numpy as np

MIRROR_W_MM, MIRROR_H_MM, PX_PER_MM = 100.0, 150.0, 10.0
EXTENT_MIN_CONTRAST = 0.33  # set by --extent-contrast
MAX_HINGE_MM, MIN_HINGE_MM = 93.5, 10.0  # outer glass post / hinge zone: a tube cannot be here
DEBUG = False               # set by --debug


# ----------------------------------------------------------------- calibration
def load_calibration(path):
    """Accepts calibration_record_N.json (Phase 3/4.1) or mirror_homographies.json."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    cal = {"px_per_mm": float(d.get("px_per_mm", PX_PER_MM)),
           "width_mm": MIRROR_W_MM, "height_mm": MIRROR_H_MM}
    if "mirrors" in d:  # calibration record
        cal["width_mm"] = float(d.get("mirror_visible_glass_mm", {}).get("width", MIRROR_W_MM))
        cal["height_mm"] = float(d.get("mirror_visible_glass_mm", {}).get("height", MIRROR_H_MM))
        for s in ("left", "right"):
            m = d["mirrors"][s]
            cal[s] = {"H": np.array(m.get("homography_photo_to_rectified"), float),
                      "corners": np.array(m.get("corners_photo_px_TL_TR_BR_BL"), float)}
    else:  # mirror_homographies.json
        cal["width_mm"] = float(d.get("mirror_width_mm", MIRROR_W_MM))
        cal["height_mm"] = float(d.get("mirror_height_mm", MIRROR_H_MM))
        for s in ("left", "right"):
            cal[s] = {"H": np.array(d[f"{s}_photo_to_rectified"], float),
                      "corners": np.array(d[f"{s}_corners_photo_px"], float)}
    ow, oh = int(round(cal["width_mm"] * cal["px_per_mm"])), int(round(cal["height_mm"] * cal["px_per_mm"]))
    for sd in ("left", "right"):
        H = np.asarray(cal[sd]["H"], float)
        if H.size == 9:
            cal[sd]["H"] = H.reshape(3, 3)
            continue
        C = np.asarray(cal[sd]["corners"], float)
        if C.shape != (4, 2):
            raise ValueError(f"{sd} mirror in {path} has no homography and no corners. "
                             "Re-run calibration and rectify BOTH mirrors before pressing S.")
        dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
        cal[sd]["H"] = cv2.getPerspectiveTransform(C.astype(np.float32), dst).astype(float)
        print(f"NOTE: {sd} homography missing in {path}; rebuilt from saved mirror corners.")
    cal["out_size"] = (int(round(cal["width_mm"] * cal["px_per_mm"])),
                       int(round(cal["height_mm"] * cal["px_per_mm"])))
    return cal


def direct_roi_from_calibration(cal, img_shape, margin_px=40):
    """Strip between the mirrors' hinge-side edges, padded by a margin."""
    h, w = img_shape[:2]
    L, R = cal["left"]["corners"], cal["right"]["corners"]
    x0 = max(L[1][0], L[2][0]) - margin_px          # left mirror TR/BR
    x1 = min(R[0][0], R[3][0]) + margin_px          # right mirror TL/BL
    y0 = min(L[0][1], L[1][1], R[0][1], R[1][1])
    y1 = max(L[2][1], L[3][1], R[2][1], R[3][1])
    x0, x1 = int(max(0, x0)), int(min(w, x1))
    y0, y1 = int(max(0, y0)), int(min(h, y1))
    return (x0, y0, x1 - x0, y1 - y0)


# ------------------------------------------------------------------- detection
def _transitions_per_row(gray_box):
    """Average black/white transitions per row after Otsu: barcodes have many."""
    if gray_box.size == 0:
        return 0.0
    g = cv2.GaussianBlur(gray_box, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = np.abs(np.diff(b.astype(np.int8), axis=1)).sum(axis=1)
    return float(np.median(t))


def _gradient_candidates_vertical_bars(gray, min_area_frac=0.0004):
    """Boxes where gradients are strongly horizontal (i.e. vertical bars)."""
    h, w = gray.shape
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    ax, ay = np.abs(gx), np.abs(gy)
    resp = np.clip(ax - ay, 0, None)
    resp = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    k = max(7, int(0.012 * min(h, w)))
    resp = cv2.blur(resp, (k, k))
    _, bw = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, 3 * k), max(5, k // 2)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_k)
    bw = cv2.erode(bw, None, iterations=3)
    bw = cv2.dilate(bw, None, iterations=4)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in cnts:
        x, y, bw_, bh_ = cv2.boundingRect(c)
        if bw_ * bh_ < min_area_frac * h * w:
            continue
        ar = bw_ / float(bh_)
        if not (0.25 < ar < 12):               # reject extreme slivers (glass edges, rulers)
            continue
        box_gray = gray[y:y + bh_, x:x + bw_]
        sx, sy = ax[y:y + bh_, x:x + bw_], ay[y:y + bh_, x:x + bw_]
        coherence = float(np.clip(sx - sy, 0, None).sum() / (sx.sum() + sy.sum() + 1e-6))
        trans = _transitions_per_row(box_gray)
        if trans < 8:                           # fewer than ~4 bars: not a barcode
            continue
        score = float(np.clip(coherence * 1.6, 0, 1) * min(1.0, trans / 30.0))
        out.append({"box": (x, y, bw_, bh_), "score": score,
                    "transitions_per_row": trans, "method": "gradient"})
    return out


def _opencv_barcode_candidates(gray):
    """cv2.barcode detector (OpenCV >= 4.8), if available. Returns boxes only."""
    out = []
    if not hasattr(cv2, "barcode"):
        return out
    try:
        det = cv2.barcode.BarcodeDetector()
        res = det.detectMulti(gray)
        ok, pts = res[0], res[1]
        if ok and pts is not None:
            for quad in np.asarray(pts).reshape(-1, 4, 2):
                x, y, w, h = cv2.boundingRect(quad.astype(np.float32))
                if w * h > 100:
                    out.append({"box": (x, y, w, h), "score": 0.85,
                                "transitions_per_row": None, "method": "opencv_barcode"})
    except Exception:
        pass
    return out


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-9)


def _nms(cands, thr=0.3):
    cands = sorted(cands, key=lambda c: -c["score"])
    keep = []
    for c in cands:
        if all(_iou(c["box"], k["box"]) < thr for k in keep):
            keep.append(c)
    return keep


def detect_regions(img_bgr, border_px=0, max_regions=5):
    """Detect barcode-like regions in one image. Tries bars vertical AND horizontal."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    h, w = gray.shape
    gray_eq = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

    cands = []
    for c in _gradient_candidates_vertical_bars(gray_eq):
        c["orientation"] = "vertical_bars"
        cands.append(c)
    rot = cv2.rotate(gray_eq, cv2.ROTATE_90_CLOCKWISE)           # (x,y)->(h-1-y, x)
    for c in _gradient_candidates_vertical_bars(rot):
        x, y, bw_, bh_ = c["box"]
        # rotate box back: rotated coords (xr,yr) -> orig x = yr, orig y = h-1-xr
        c["box"] = (y, h - (x + bw_), bh_, bw_)
        c["orientation"] = "horizontal_bars"
        cands.append(c)
    for c in _opencv_barcode_candidates(gray):
        c["orientation"] = "opencv"
        cands.append(c)

    if border_px > 0:  # drop boxes that hug the glass border (edge artefacts)
        cands = [c for c in cands
                 if c["box"][0] > border_px and c["box"][1] > border_px
                 and c["box"][0] + c["box"][2] < w - border_px
                 and c["box"][1] + c["box"][3] < h - border_px]
    return _nms(cands)[:max_regions]


# ---------------------------------------------------------------------- decode
def try_decode(crop_bgr):
    """Best-effort decode of a crop. Detection != decoding; failure is normal for
    partial / curved / mirrored views. Tries the crop and its mirror image."""
    results = []
    variants = {"as_is": crop_bgr, "flipped": cv2.flip(crop_bgr, 1)}
    for name, im in variants.items():
        im = cv2.copyMakeBorder(im, 30, 30, 60, 60, cv2.BORDER_REPLICATE)
        try:
            import zxingcpp
            for r in zxingcpp.read_barcodes(im):
                results.append({"decoder": "zxing-cpp", "view": name,
                                "format": str(r.format), "text": r.text})
        except Exception:
            pass
        try:
            from pyzbar import pyzbar
            for r in pyzbar.decode(im):
                results.append({"decoder": "pyzbar", "view": name,
                                "format": r.type, "text": r.data.decode("utf-8", "replace")})
        except Exception:
            pass
        if hasattr(cv2, "barcode"):
            try:
                det = cv2.barcode.BarcodeDetector()
                res = det.detectAndDecodeWithType(im)
                ok, texts = res[0], res[1]
                if ok and texts is not None:
                    for t in ([texts] if isinstance(texts, str) else texts):
                        if t:
                            results.append({"decoder": "opencv", "view": name, "text": t})
            except Exception:
                pass
    return results


# ------------------------------------------------------------------- pipelines
def _box_to_photo_poly(box, H):
    """Rectified box -> 4 photo-pixel corners via inverse homography."""
    x, y, w, h = box
    quad = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], np.float32)
    return cv2.perspectiveTransform(quad, np.linalg.inv(H)).reshape(-1, 2)


def refine_with_extent(rect, regions, ignore_below_px=None):
    """Grow each seed box to the full barcode using barcode_extent.py (same folder).
    Only extent candidates that cover >=50% of a seed are accepted, so look-alikes
    elsewhere (frames, rulers, text) are not promoted."""
    try:
        import barcode_extent as be
    except Exception:
        return regions
    cands = be.find_barcodes(rect, ignore_below_px)
    # second pass on a CLAHE-enhanced + lightly sharpened copy: narrow, foreshortened barcodes
    # are sensitive to blur, and a single pass can miss them
    g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY) if rect.ndim == 3 else rect
    enh = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    enh = cv2.addWeighted(enh, 1.6, cv2.GaussianBlur(enh, (0, 0), 2), -0.6, 0)
    cands = cands + be.find_barcodes(cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR), ignore_below_px)
    cands.sort(key=lambda c: -c["score"])
    kept = []
    for c in cands:
        if all(_iou(c["box_xywh"], k["box_xywh"]) < 0.3 for k in kept):
            kept.append(c)
    cands = kept
    if DEBUG:
        for c in cands:
            print("   extent cand", c["box_xywh"], "score", c["score"], "contrast(enh)", c["features"]["contrast"], "contrast(orig)", round(c.get("orig_contrast", -1), 3))
    out = []
    for r in regions:
        sx, sy, sw, sh = r["box"]
        best = None
        for c in cands:
            x, y, w, h = c["box_xywh"]
            ix = max(0, min(sx + sw, x + w) - max(sx, x)); iy = max(0, min(sy + sh, y + h) - max(sy, y))
            if ix * iy >= 0.5 * sw * sh and (best is None or c["score"] > best["score"]):
                best = c
        if best:
            r = dict(r, box=tuple(best["box_xywh"]), method=r["method"] + "+extent",
                     score=max(r["score"], best["score"]), extent_features=best["features"],
                     seed_box=r["box"])
        out.append(r)
    for c in cands:  # contrast measured on the ORIGINAL image (enhanced copy inflates it)
        x, y, w, h = c["box_xywh"]
        roi = g[max(0, y):y + h, max(0, x):x + w]
        c["orig_contrast"] = float(np.clip((np.percentile(roi, 90) - np.percentile(roi, 10)) / 110.0, 0, 1)) if roi.size else 0.0
    out = _nms(out)  # seeds that grew into the same box collapse to one
    # Extent-only hits: narrow/foreshortened barcodes that OpenCV cannot seed. Require a
    # confident score AND high bar contrast (separates labels from frames/edges).
    for c in cands:
        f = c["features"]
        if c["score"] >= 0.6 and c["orig_contrast"] >= EXTENT_MIN_CONTRAST and c["box_xywh"][3] >= 80 and c["n_bars"] >= 8 and c["box_xywh"][0] > 15 and c["box_xywh"][0] + c["box_xywh"][2] < rect.shape[1] - 15 and all(_iou(c["box_xywh"], o["box"]) < 0.3 for o in out):
            out.append({"box": tuple(c["box_xywh"]), "score": c["score"], "method": "extent_only",
                        "orientation": c["bar_axis"], "transitions_per_row": None,
                        "extent_features": f})
    keep = [o for o in out if o["method"] != "extent_only"]
    return keep + _merge_stacked([o for o in out if o["method"] == "extent_only"])


def _merge_stacked(regs):
    """Union boxes that share the same x-range and touch/overlap vertically (pieces of one barcode)."""
    regs = sorted(regs, key=lambda r: r["box"][1])
    merged = []
    for r in regs:
        x, y, w, h = r["box"]
        for m in merged:
            mx, my, mw, mh = m["box"]
            xo = min(x + w, mx + mw) - max(x, mx)
            if xo >= 0.8 * min(w, mw) and y <= my + mh + 20:
                nx, ny = min(x, mx), min(y, my)
                m["box"] = (nx, ny, max(x + w, mx + mw) - nx, max(y + h, my + mh) - ny)
                m["score"] = max(m["score"], r["score"])
                break
        else:
            merged.append(dict(r))
    return merged


MIN_PURITY = 0.0  # set by --min-purity (off by default: too strict for small/soft tubes)


def row_purity(gray_box, orientation="unknown"):
    """Barcode rows (or columns) are all-dark or all-light across the label; text rows are only
    partly dark. 1.0 = perfectly barcode-like. Measured ~0.5-0.7 for barcodes, ~0.3-0.45 for text."""
    if gray_box.ndim == 3:
        gray_box = cv2.cvtColor(gray_box, cv2.COLOR_BGR2GRAY)
    if gray_box.size < 50:
        return 0.0
    def one(g):
        g = cv2.GaussianBlur(g, (3, 3), 0)
        t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frac = (g < t).astype(float).mean(axis=1)
        return float(np.mean(np.abs(2 * frac - 1)))
    if "vertical" in orientation:
        return one(gray_box.T.copy())
    if "horizontal" in orientation:
        return one(gray_box)
    return max(one(gray_box), one(gray_box.T.copy()))


def detect_mirror(img, cal, side, border_mm=2.0, ignore_below_mm=None):
    H = cal[side]["H"]
    ppm, wmm = cal["px_per_mm"], cal["width_mm"]
    rect = cv2.warpPerspective(img, H, cal["out_size"])
    regions = detect_regions(rect, border_px=int(border_mm * ppm))
    if ignore_below_mm is not None:  # e.g. physical ruler along the bottom of the glass
        regions = [r for r in regions if (r["box"][1] + r["box"][3] / 2) / ppm < ignore_below_mm]
    regions = refine_with_extent(rect, regions,
                                 None if ignore_below_mm is None else ignore_below_mm * ppm)
    dets = []
    for r in regions:
        x, y, w, h = r["box"]
        x_mm0, x_mm1 = x / ppm, (x + w) / ppm
        if side == "left":      # hinge edge = RIGHT edge of rectified left mirror
            near, far = wmm - x_mm1, wmm - x_mm0
        else:                   # hinge edge = LEFT edge of rectified right mirror
            near, far = x_mm0, x_mm1
        if far > MAX_HINGE_MM or near < MIN_HINGE_MM:
            continue  # white holder post at the outer glass edge, or the hinge zone
        pur = row_purity(rect[y:y + h, x:x + w], r["orientation"])
        if pur < MIN_PURITY:
            continue  # printed text / edges: rows only partly dark
        dets.append({
            "view": side, "score": round(r["score"], 3), "method": r["method"],
            "orientation": r["orientation"],
            "transitions_per_row": r["transitions_per_row"],
            "row_purity": round(pur, 2),
            "extent_features": r.get("extent_features"),
            "box_rectified_px": [int(x), int(y), int(w), int(h)],
            "box_rectified_mm": [round(x / ppm, 2), round(y / ppm, 2),
                                 round(w / ppm, 2), round(h / ppm, 2)],
            "hinge_edge_distance_mm": {"near": round(near, 2), "far": round(far, 2),
                                       "centre": round((near + far) / 2, 2)},
            "height_range_mm_from_top": [round(y / ppm, 2), round((y + h) / ppm, 2)],
            "photo_polygon_px": _box_to_photo_poly(r["box"], H).round(1).tolist(),
            "_crop": rect[y:y + h, x:x + w],
        })
    return dets, rect


def detect_direct(img, roi):
    x0, y0, w0, h0 = roi
    sub = img[y0:y0 + h0, x0:x0 + w0]
    regions = detect_regions(sub, border_px=0)
    dets = []
    for r in regions:
        x, y, w, h = r["box"]
        pur = row_purity(sub[y:y + h, x:x + w], r["orientation"])
        if pur < MIN_PURITY:
            continue
        dets.append({
            "row_purity": round(pur, 2),
            "view": "direct", "score": round(r["score"], 3), "method": r["method"],
            "orientation": r["orientation"],
            "transitions_per_row": r["transitions_per_row"],
            "box_photo_px": [int(x + x0), int(y + y0), int(w), int(h)],
            "_crop": sub[y:y + h, x:x + w],
        })
    return dets


def annotate(img, cal, all_dets, direct_roi):
    vis = img.copy()
    colours = {"left": (0, 200, 0), "right": (255, 120, 0), "direct": (0, 0, 255)}
    if direct_roi is not None:
        x, y, w, h = direct_roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (150, 150, 150), 2)
    for side in ("left", "right"):
        if cal and side in cal:
            cv2.polylines(vis, [cal[side]["corners"].astype(np.int32)], True, (200, 200, 0), 2)
    t = max(2, img.shape[1] // 600)
    for i, d in enumerate(all_dets):
        c = colours[d["view"]]
        if "photo_polygon_px" in d:
            cv2.polylines(vis, [np.array(d["photo_polygon_px"], np.int32)], True, c, t + 1)
            x, y = map(int, d["photo_polygon_px"][0])
        else:
            x, y, w, h = d["box_photo_px"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), c, t + 1)
        cv2.putText(vis, f"{d['view']}#{i} s={d['score']:.2f}", (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, t * 0.5, c, t, cv2.LINE_AA)
    return vis


def run(image_path, calib_path=None, out_dir="detections", direct_roi=None,
        decode=True, margin_px=40, min_score=0.5, ignore_below_mm=None, top_per_view=1):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    cal, dets, roi = None, [], direct_roi
    if calib_path:
        cal = load_calibration(calib_path)
        for side in ("left", "right"):
            d, rect = detect_mirror(img, cal, side, ignore_below_mm=ignore_below_mm)
            cv2.imwrite(os.path.join(out_dir, f"{stem}_{side}_rectified.jpg"), rect)
            dets += d
        if roi is None:
            roi = direct_roi_from_calibration(cal, img.shape, margin_px)
    else:
        roi = roi or (0, 0, img.shape[1], img.shape[0])
    dets += detect_direct(img, roi)

    for i, d in enumerate(dets):
        crop = d.pop("_crop")
        p = os.path.join(out_dir, f"{stem}_{d['view']}_{i}.png")
        cv2.imwrite(p, crop)
        d["crop_file"] = os.path.basename(p)
        d["decode_attempts"] = try_decode(crop) if decode else []
    dets = [d for d in dets if d["score"] >= min_score]
    dets.sort(key=lambda d: -d["score"])
    if top_per_view:  # keep only the strongest region(s) per view: one interval per mirror
        seen, kept = {}, []
        for d in dets:
            seen[d["view"]] = seen.get(d["view"], 0) + 1
            if seen[d["view"]] <= top_per_view:
                kept.append(d)
        dets = kept

    geo = None
    if cal is not None:
        L, R = cal["left"]["corners"], cal["right"]["corners"]
        geo = {"axis_x_px": round(float(((L[1][0] + L[2][0]) / 2 + (R[0][0] + R[3][0]) / 2) / 2), 1),
               "edge_height_px": round(float(((L[2][1] - L[1][1]) + (R[3][1] - R[0][1])) / 2), 1),
               "note": "tube axis assumed on the hinge line; glass edge height is 150 mm"}
    result = {"direct_geometry": geo, "image": image_path, "image_size_px": [img.shape[1], img.shape[0]],
              "calibration": calib_path, "direct_roi_px": list(roi),
              "score_note": "Heuristic proposal score 0..1. NOT a calibrated probability.",
              "detections": dets}
    with open(os.path.join(out_dir, f"{stem}_detections.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_annotated.jpg"), annotate(img, cal, dets, roi))
    return result


def main():
    ap = argparse.ArgumentParser(description="Automatic barcode-region detection (Phase 5)")
    ap.add_argument("image")
    ap.add_argument("--calib", help="calibration_record_N.json or mirror_homographies.json")
    ap.add_argument("--out-dir", default="detections")
    ap.add_argument("--direct-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--ignore-below-mm", type=float, default=124.0,
                    help="rig setting: ignore mirror regions lower than this y (mm from top) = ruler band; 0 disables")
    ap.add_argument("--extent-contrast", type=float, default=0.33,
                    help="min bar contrast for extent-only hits; lower finds fainter barcodes but admits frames")
    ap.add_argument("--debug", action="store_true", help="print extent-finder candidates per mirror")
    ap.add_argument("--top-per-view", type=int, default=1, help="max regions kept per view (0 = keep all)")
    ap.add_argument("--max-hinge-mm", type=float, default=93.5, help="ignore mirror regions farther than this from the hinge edge (outer post)")
    ap.add_argument("--min-purity", type=float, default=0.0, help="min row purity (barcode-vs-text gate); 0 disables")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.5, help="drop weak proposals (heuristic score)")
    a = ap.parse_args()
    global EXTENT_MIN_CONTRAST, DEBUG, MAX_HINGE_MM, MIN_PURITY
    MIN_PURITY = a.min_purity
    MAX_HINGE_MM = a.max_hinge_mm
    DEBUG = a.debug
    EXTENT_MIN_CONTRAST = a.extent_contrast
    res = run(a.image, a.calib, a.out_dir, tuple(a.direct_roi) if a.direct_roi else None,
              not a.no_decode, min_score=a.min_score,
              ignore_below_mm=a.ignore_below_mm or None, top_per_view=a.top_per_view)
    print(f"{len(res['detections'])} candidate region(s)")
    for i, d in enumerate(res["detections"]):
        extra = ""
        if "hinge_edge_distance_mm" in d:
            h = d["hinge_edge_distance_mm"]
            extra = f" hinge-edge near/far/centre = {h['near']}/{h['far']}/{h['centre']} mm"
        dec = d["decode_attempts"][0]["text"] if d["decode_attempts"] else "-"
        extra += f" purity={d.get('row_purity', '-')}"
        print(f"  [{i}] {d['view']:6s} score={d['score']:.2f} {d['orientation']}{extra} decoded={dec}")
    print(f"Saved to {a.out_dir}/")


if __name__ == "__main__":
    main()