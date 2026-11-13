"""
barcode_region_detector.py -- Phase 5: automatic barcode localisation
=======================================================================

Finds barcode-like regions in a full camera photo of the flask + two mirrors,
without clicking, and reports each region in Phase 3 rectified coordinates.

Resolution handling:
- The calibration JSON stores the dimensions of the photo used during calibration.
- If the current input photo has a different resolution but the same aspect ratio,
  the saved homographies and photo-space corners are adapted automatically.
- The rectified mirror output remains width_mm * px_per_mm by height_mm * px_per_mm.

Usage:
  python barcode_region_detector.py photo.jpg --calib calibration_record.json
  python barcode_region_detector.py photo.jpg --calib calibration_record.json --out-dir det
  python barcode_region_detector.py photo.jpg
"""

import argparse
import json
import os

import cv2
import numpy as np


MIRROR_W_MM, MIRROR_H_MM, PX_PER_MM = 100.0, 150.0, 10.0
EXTENT_MIN_CONTRAST = 0.33
MAX_HINGE_MM, MIN_HINGE_MM = 93.5, 10.0
DEBUG = False


# ----------------------------------------------------------------- calibration
def load_calibration(path, image_shape=None):
    """
    Load a Phase 3 calibration and adapt it to the current input image size.

    A saved homography maps calibration-photo pixels to rectified pixels.
    If the current photo is resized relative to the calibration photo:

        p_calibration = S * p_current
        H_current = H_saved * S

    The rectified output dimensions and px/mm scale do not change.
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    cal = {
        "px_per_mm": float(d.get("px_per_mm", PX_PER_MM)),
        "width_mm": MIRROR_W_MM,
        "height_mm": MIRROR_H_MM,
    }

    if "mirrors" in d:
        cal["width_mm"] = float(
            d.get("mirror_visible_glass_mm", {}).get("width", MIRROR_W_MM)
        )
        cal["height_mm"] = float(
            d.get("mirror_visible_glass_mm", {}).get("height", MIRROR_H_MM)
        )
        for side in ("left", "right"):
            m = d["mirrors"][side]
            H_raw = m.get("homography_photo_to_rectified")
            C_raw = m.get("corners_photo_px_TL_TR_BR_BL")
            cal[side] = {
                "H": None if H_raw is None else np.asarray(H_raw, dtype=float),
                "corners": None if C_raw is None else np.asarray(C_raw, dtype=float),
            }
    else:
        cal["width_mm"] = float(d.get("mirror_width_mm", MIRROR_W_MM))
        cal["height_mm"] = float(d.get("mirror_height_mm", MIRROR_H_MM))
        for side in ("left", "right"):
            H_raw = d.get(f"{side}_photo_to_rectified")
            C_raw = d.get(f"{side}_corners_photo_px")
            cal[side] = {
                "H": None if H_raw is None else np.asarray(H_raw, dtype=float),
                "corners": None if C_raw is None else np.asarray(C_raw, dtype=float),
            }

    # Determine coordinate conversion from current photo pixels to calibration
    # photo pixels. This is the correction needed for resized input photographs.
    sx = sy = 1.0
    saved_size = d.get("image_size_px")
    if image_shape is not None and saved_size:
        current_h, current_w = image_shape[:2]
        saved_w = float(saved_size["width"])
        saved_h = float(saved_size["height"])

        if current_w <= 0 or current_h <= 0:
            raise ValueError("Invalid input image dimensions.")

        sx = saved_w / float(current_w)
        sy = saved_h / float(current_h)

        # A simple scale transform is valid only if aspect ratio is preserved.
        relative_difference = abs(sx - sy) / max(abs(sx), abs(sy), 1e-12)
        if relative_difference > 0.01:
            raise ValueError(
                "Input image aspect ratio differs from the calibration image. "
                f"Scale X={sx:.6f}, scale Y={sy:.6f}. "
                "Use the original calibration photo or recalibrate this image."
            )

        S = np.array(
            [[sx, 0.0, 0.0],
             [0.0, sy, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )

        for side in ("left", "right"):
            if cal[side]["H"] is not None and cal[side]["H"].size == 9:
                cal[side]["H"] = cal[side]["H"].reshape(3, 3) @ S

            if cal[side]["corners"] is not None:
                cal[side]["corners"] = cal[side]["corners"] / np.array(
                    [sx, sy], dtype=float
                )

        print(
            "Calibration resolution adaptation: "
            f"calibration={saved_w:.0f}x{saved_h:.0f}, "
            f"input={current_w}x{current_h}, "
            f"scale=({sx:.4f}, {sy:.4f})."
        )
    elif image_shape is not None and not saved_size:
        print(
            "WARNING: calibration JSON has no image_size_px field. "
            "Assuming the input photo has the calibration resolution."
        )

    out_w = int(round(cal["width_mm"] * cal["px_per_mm"]))
    out_h = int(round(cal["height_mm"] * cal["px_per_mm"]))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    for side in ("left", "right"):
        H = cal[side]["H"]
        if H is not None and H.size == 9 and np.isfinite(H).all():
            cal[side]["H"] = H.reshape(3, 3)
            continue

        C = cal[side]["corners"]
        if C is None or np.asarray(C).shape != (4, 2):
            raise ValueError(
                f"{side} mirror in {path} has neither a valid homography "
                "nor four saved photo-space corners."
            )

        cal[side]["H"] = cv2.getPerspectiveTransform(
            np.asarray(C, dtype=np.float32), dst
        ).astype(float)
        print(
            f"NOTE: {side} homography was rebuilt from the saved mirror corners."
        )

    cal["out_size"] = (out_w, out_h)
    return cal


def direct_roi_from_calibration(cal, img_shape, margin_px=40):
    """Estimate the photo-space strip between the two hinge-side mirror edges."""
    h, w = img_shape[:2]
    L = cal["left"]["corners"]
    R = cal["right"]["corners"]

    x0 = max(L[1][0], L[2][0]) - margin_px
    x1 = min(R[0][0], R[3][0]) + margin_px
    y0 = min(L[0][1], L[1][1], R[0][1], R[1][1])
    y1 = max(L[2][1], L[3][1], R[2][1], R[3][1])

    x0 = int(max(0, min(w, x0)))
    x1 = int(max(0, min(w, x1)))
    y0 = int(max(0, min(h, y0)))
    y1 = int(max(0, min(h, y1)))

    if x1 <= x0 or y1 <= y0:
        return (0, 0, w, h)
    return (x0, y0, x1 - x0, y1 - y0)


# ------------------------------------------------------------------- detection
def _transitions_per_row(gray_box):
    if gray_box.size == 0:
        return 0.0
    g = cv2.GaussianBlur(gray_box, (3, 3), 0)
    _, b = cv2.threshold(
        g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    transitions = np.abs(np.diff(b.astype(np.int8), axis=1)).sum(axis=1)
    return float(np.median(transitions))


def _gradient_candidates_vertical_bars(gray, min_area_frac=0.0004):
    """Find regions with strong horizontal gradients: likely vertical bars."""
    h, w = gray.shape
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    ax, ay = np.abs(gx), np.abs(gy)

    response = np.clip(ax - ay, 0, None)
    response = cv2.normalize(
        response, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    k = max(7, int(0.012 * min(h, w)))
    response = cv2.blur(response, (k, k))
    _, bw = cv2.threshold(
        response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    close_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(15, 3 * k), max(5, k // 2))
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_k)
    bw = cv2.erode(bw, None, iterations=3)
    bw = cv2.dilate(bw, None, iterations=4)

    contours, _ = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    out = []
    for contour in contours:
        x, y, bw_, bh_ = cv2.boundingRect(contour)
        if bw_ * bh_ < min_area_frac * h * w:
            continue

        aspect_ratio = bw_ / float(bh_)
        if not (0.25 < aspect_ratio < 12):
            continue

        box_gray = gray[y:y + bh_, x:x + bw_]
        sx = ax[y:y + bh_, x:x + bw_]
        sy = ay[y:y + bh_, x:x + bw_]

        coherence = float(
            np.clip(sx - sy, 0, None).sum()
            / (sx.sum() + sy.sum() + 1e-6)
        )
        transitions = _transitions_per_row(box_gray)
        if transitions < 8:
            continue

        score = float(
            np.clip(coherence * 1.6, 0, 1)
            * min(1.0, transitions / 30.0)
        )
        out.append({
            "box": (x, y, bw_, bh_),
            "score": score,
            "transitions_per_row": transitions,
            "method": "gradient",
        })
    return out


def _opencv_barcode_candidates(gray):
    """OpenCV barcode detector candidates, when available."""
    out = []
    if not hasattr(cv2, "barcode"):
        return out

    try:
        detector = cv2.barcode.BarcodeDetector()
        result = detector.detectMulti(gray)
        ok, points = result[0], result[1]
        if ok and points is not None:
            for quad in np.asarray(points).reshape(-1, 4, 2):
                x, y, w, h = cv2.boundingRect(quad.astype(np.float32))
                if w * h > 100:
                    out.append({
                        "box": (x, y, w, h),
                        "score": 0.85,
                        "transitions_per_row": None,
                        "method": "opencv_barcode",
                    })
    except Exception:
        pass
    return out


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = ix * iy
    union = aw * ah + bw * bh - intersection
    return intersection / float(union + 1e-9)


def _nms(candidates, threshold=0.3):
    candidates = sorted(candidates, key=lambda c: -c["score"])
    kept = []
    for candidate in candidates:
        if all(
            _iou(candidate["box"], existing["box"]) < threshold
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def detect_regions(img_bgr, border_px=0, max_regions=5):
    """Detect barcode-like regions with vertical or horizontal bars."""
    gray = (
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if img_bgr.ndim == 3 else img_bgr
    )
    h, w = gray.shape
    gray_eq = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

    candidates = []
    for candidate in _gradient_candidates_vertical_bars(gray_eq):
        candidate["orientation"] = "vertical_bars"
        candidates.append(candidate)

    rotated = cv2.rotate(gray_eq, cv2.ROTATE_90_CLOCKWISE)
    for candidate in _gradient_candidates_vertical_bars(rotated):
        x, y, bw_, bh_ = candidate["box"]
        # Rotated coordinates -> original coordinates.
        candidate["box"] = (y, h - (x + bw_), bh_, bw_)
        candidate["orientation"] = "horizontal_bars"
        candidates.append(candidate)

    for candidate in _opencv_barcode_candidates(gray):
        candidate["orientation"] = "opencv"
        candidates.append(candidate)

    if border_px > 0:
        candidates = [
            c for c in candidates
            if c["box"][0] > border_px
            and c["box"][1] > border_px
            and c["box"][0] + c["box"][2] < w - border_px
            and c["box"][1] + c["box"][3] < h - border_px
        ]

    return _nms(candidates)[:max_regions]


# ---------------------------------------------------------------------- decode
def try_decode(crop_bgr):
    """Best-effort decode; a failed decode does not mean detection failed."""
    results = []
    variants = {
        "as_is": crop_bgr,
        "flipped": cv2.flip(crop_bgr, 1),
    }

    for name, image in variants.items():
        image = cv2.copyMakeBorder(
            image, 30, 30, 60, 60, cv2.BORDER_REPLICATE
        )

        try:
            import zxingcpp
            for result in zxingcpp.read_barcodes(image):
                results.append({
                    "decoder": "zxing-cpp",
                    "view": name,
                    "format": str(result.format),
                    "text": result.text,
                })
        except Exception:
            pass

        try:
            from pyzbar import pyzbar
            for result in pyzbar.decode(image):
                results.append({
                    "decoder": "pyzbar",
                    "view": name,
                    "format": result.type,
                    "text": result.data.decode("utf-8", "replace"),
                })
        except Exception:
            pass

        if hasattr(cv2, "barcode"):
            try:
                detector = cv2.barcode.BarcodeDetector()
                result = detector.detectAndDecodeWithType(image)
                ok, texts = result[0], result[1]
                if ok and texts is not None:
                    for value in ([texts] if isinstance(texts, str) else texts):
                        if value:
                            results.append({
                                "decoder": "opencv",
                                "view": name,
                                "text": value,
                            })
            except Exception:
                pass

    return results


# ------------------------------------------------------------------- pipelines
def _box_to_photo_poly(box, H):
    """Map four rectified bounding-box corners back to photo pixels."""
    x, y, w, h = box
    quad = np.array(
        [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(
        quad, np.linalg.inv(H)
    ).reshape(-1, 2)


def refine_with_extent(rect, regions, ignore_below_px=None):
    """
    Expand detector seed boxes using barcode_extent.py when available.
    The main detector continues to work if barcode_extent.py is absent.
    """
    try:
        import barcode_extent as be
    except Exception:
        return regions

    candidates = be.find_barcodes(rect, ignore_below_px)

    gray = (
        cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        if rect.ndim == 3 else rect
    )
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    enhanced = cv2.addWeighted(
        enhanced,
        1.6,
        cv2.GaussianBlur(enhanced, (0, 0), 2),
        -0.6,
        0,
    )
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    candidates += be.find_barcodes(enhanced_bgr, ignore_below_px)
    candidates.sort(key=lambda c: -c["score"])

    unique = []
    for candidate in candidates:
        if all(
            _iou(candidate["box_xywh"], kept["box_xywh"]) < 0.3
            for kept in unique
        ):
            unique.append(candidate)
    candidates = unique

    if DEBUG:
        for candidate in candidates:
            features = candidate.get("features", {})
            print(
                "   extent candidate",
                candidate["box_xywh"],
                "score",
                candidate["score"],
                "features",
                features,
            )

    out = []
    for region in regions:
        sx, sy, sw, sh = region["box"]
        best = None

        for candidate in candidates:
            x, y, w, h = candidate["box_xywh"]
            ix = max(0, min(sx + sw, x + w) - max(sx, x))
            iy = max(0, min(sy + sh, y + h) - max(sy, y))
            if (
                ix * iy >= 0.5 * sw * sh
                and (best is None or candidate["score"] > best["score"])
            ):
                best = candidate

        if best is not None:
            region = dict(
                region,
                box=tuple(best["box_xywh"]),
                method=region["method"] + "+extent",
                score=max(region["score"], best["score"]),
                extent_features=best.get("features"),
                seed_box=region["box"],
            )
        out.append(region)

    for candidate in candidates:
        x, y, w, h = candidate["box_xywh"]
        roi = gray[max(0, y):y + h, max(0, x):x + w]
        candidate["orig_contrast"] = (
            float(np.clip(
                (np.percentile(roi, 90) - np.percentile(roi, 10)) / 110.0,
                0,
                1,
            ))
            if roi.size else 0.0
        )

    out = _nms(out)

    # Extent-only detections: useful for narrow or foreshortened barcode views.
    for candidate in candidates:
        features = candidate.get("features", {})
        x, y, w, h = candidate["box_xywh"]
        if (
            candidate["score"] >= 0.6
            and candidate["orig_contrast"] >= EXTENT_MIN_CONTRAST
            and h >= 80
            and candidate.get("n_bars", 0) >= 8
            and x > 15
            and x + w < rect.shape[1] - 15
            and all(_iou(candidate["box_xywh"], r["box"]) < 0.3 for r in out)
        ):
            out.append({
                "box": tuple(candidate["box_xywh"]),
                "score": candidate["score"],
                "method": "extent_only",
                "orientation": candidate.get("bar_axis", "unknown"),
                "transitions_per_row": None,
                "extent_features": features,
            })

    regular = [r for r in out if r["method"] != "extent_only"]
    extent_only = [r for r in out if r["method"] == "extent_only"]
    return regular + _merge_stacked(extent_only)


def _merge_stacked(regions):
    """Merge extent-only boxes that share an x-range and touch vertically."""
    regions = sorted(regions, key=lambda r: r["box"][1])
    merged = []

    for region in regions:
        x, y, w, h = region["box"]
        for existing in merged:
            mx, my, mw, mh = existing["box"]
            x_overlap = min(x + w, mx + mw) - max(x, mx)

            if x_overlap >= 0.8 * min(w, mw) and y <= my + mh + 20:
                nx, ny = min(x, mx), min(y, my)
                existing["box"] = (
                    nx,
                    ny,
                    max(x + w, mx + mw) - nx,
                    max(y + h, my + mh) - ny,
                )
                existing["score"] = max(existing["score"], region["score"])
                break
        else:
            merged.append(dict(region))

    return merged


MIN_PURITY = 0.0



def row_purity(gray_box, orientation="unknown"):
    """
    Estimate how consistently rows/columns are dark or light.
    This is a heuristic, not a probability.
    """
    # Safety check: prevent OpenCV from processing an empty crop.
    if gray_box is None or gray_box.size == 0:
        return 0.0

    if gray_box.ndim == 3:
        gray_box = cv2.cvtColor(gray_box, cv2.COLOR_BGR2GRAY)
    if gray_box.size < 50:
        return 0.0

    def measure(image):
        image = cv2.GaussianBlur(image, (3, 3), 0)
        threshold, _ = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        dark_fraction = (image < threshold).astype(float).mean(axis=1)
        return float(np.mean(np.abs(2 * dark_fraction - 1)))

    if "vertical" in orientation:
        return measure(gray_box.T.copy())
    if "horizontal" in orientation:
        return measure(gray_box)
    return max(measure(gray_box), measure(gray_box.T.copy()))


def detect_mirror(img, cal, side, border_mm=2.0, ignore_below_mm=None):
    H = cal[side]["H"]
    ppm = cal["px_per_mm"]
    width_mm = cal["width_mm"]

    rect = cv2.warpPerspective(img, H, cal["out_size"])
    regions = detect_regions(
        rect, border_px=int(border_mm * ppm)
    )

    if ignore_below_mm is not None:
        regions = [
            r for r in regions
            if (r["box"][1] + r["box"][3] / 2) / ppm < ignore_below_mm
        ]

    regions = refine_with_extent(
        rect,
        regions,
        None if ignore_below_mm is None else ignore_below_mm * ppm,
    )

    detections = []
    for region in regions:
        x, y, w, h = region["box"]
        x_mm0, x_mm1 = x / ppm, (x + w) / ppm

        if side == "left":
            # Hinge edge is the RIGHT edge of the rectified left mirror.
            near, far = width_mm - x_mm1, width_mm - x_mm0
        else:
            # Hinge edge is the LEFT edge of the rectified right mirror.
            near, far = x_mm0, x_mm1

        if far > MAX_HINGE_MM or near < MIN_HINGE_MM:
            continue

        purity = row_purity(
            rect[y:y + h, x:x + w], region["orientation"]
        )
        if purity < MIN_PURITY:
            continue

        detections.append({
            "view": side,
            "score": round(float(region["score"]), 3),
            "method": region["method"],
            "orientation": region["orientation"],
            "transitions_per_row": region["transitions_per_row"],
            "row_purity": round(purity, 2),
            "extent_features": region.get("extent_features"),
            "box_rectified_px": [int(x), int(y), int(w), int(h)],
            "box_rectified_mm": [
                round(x / ppm, 2),
                round(y / ppm, 2),
                round(w / ppm, 2),
                round(h / ppm, 2),
            ],
            "hinge_edge_distance_mm": {
                "near": round(near, 2),
                "far": round(far, 2),
                "centre": round((near + far) / 2, 2),
            },
            "height_range_mm_from_top": [
                round(y / ppm, 2),
                round((y + h) / ppm, 2),
            ],
            "photo_polygon_px": _box_to_photo_poly(
                region["box"], H
            ).round(1).tolist(),
            "_crop": rect[y:y + h, x:x + w],
        })

    return detections, rect


def detect_direct(img, roi):
    x0, y0, w0, h0 = roi
    sub = img[y0:y0 + h0, x0:x0 + w0]
    regions = detect_regions(sub, border_px=0)

    detections = []
    for region in regions:
        x, y, w, h = region["box"]
        purity = row_purity(
            sub[y:y + h, x:x + w], region["orientation"]
        )
        if purity < MIN_PURITY:
            continue

        detections.append({
            "row_purity": round(purity, 2),
            "view": "direct",
            "score": round(float(region["score"]), 3),
            "method": region["method"],
            "orientation": region["orientation"],
            "transitions_per_row": region["transitions_per_row"],
            "box_photo_px": [
                int(x + x0), int(y + y0), int(w), int(h)
            ],
            "_crop": sub[y:y + h, x:x + w],
        })

    return detections


def annotate(img, cal, all_detections, direct_roi):
    vis = img.copy()
    colours = {
        "left": (0, 200, 0),
        "right": (255, 120, 0),
        "direct": (0, 0, 255),
    }

    if direct_roi is not None:
        x, y, w, h = direct_roi
        cv2.rectangle(
            vis, (x, y), (x + w, y + h), (150, 150, 150), 2
        )

    if cal:
        for side in ("left", "right"):
            if side in cal:
                cv2.polylines(
                    vis,
                    [cal[side]["corners"].astype(np.int32)],
                    True,
                    (200, 200, 0),
                    2,
                )

    thickness = max(2, img.shape[1] // 600)
    for index, detection in enumerate(all_detections):
        colour = colours[detection["view"]]

        if "photo_polygon_px" in detection:
            polygon = np.array(
                detection["photo_polygon_px"], dtype=np.int32
            )
            cv2.polylines(vis, [polygon], True, colour, thickness + 1)
            x, y = map(int, polygon[0])
        else:
            x, y, w, h = detection["box_photo_px"]
            cv2.rectangle(
                vis, (x, y), (x + w, y + h), colour, thickness + 1
            )

        cv2.putText(
            vis,
            f"{detection['view']}#{index} s={detection['score']:.2f}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            thickness * 0.5,
            colour,
            thickness,
            cv2.LINE_AA,
        )

    return vis


def run(
    image_path,
    calib_path=None,
    out_dir="detections",
    direct_roi=None,
    decode=True,
    margin_px=40,
    min_score=0.5,
    ignore_below_mm=None,
    top_per_view=1,
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(
            f"Could not read input image: {image_path}"
        )

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    cal = None
    detections = []
    roi = direct_roi

    if calib_path:
        # IMPORTANT: pass img.shape so calibration coordinates are adapted
        # to the actual resolution of this input photo.
        cal = load_calibration(calib_path, img.shape)

        for side in ("left", "right"):
            side_detections, rect = detect_mirror(
                img,
                cal,
                side,
                ignore_below_mm=ignore_below_mm,
            )
            cv2.imwrite(
                os.path.join(out_dir, f"{stem}_{side}_rectified.jpg"),
                rect,
            )
            detections.extend(side_detections)

        if roi is None:
            roi = (0, 0, img.shape[1], img.shape[0])
    else:
        roi = roi or (0, 0, img.shape[1], img.shape[0])

    detections.extend(detect_direct(img, roi))

    # Save every crop before final score/top-per-view filtering, matching the
    # original detector's diagnostic behaviour.
    for index, detection in enumerate(detections):
        crop = detection.pop("_crop")
        crop_path = os.path.join(
            out_dir, f"{stem}_{detection['view']}_{index}.png"
        )
        cv2.imwrite(crop_path, crop)
        detection["crop_file"] = os.path.basename(crop_path)
        detection["decode_attempts"] = (
            try_decode(crop) if decode else []
        )

    detections = [
        d for d in detections if d["score"] >= min_score
    ]
    detections.sort(key=lambda d: -d["score"])

    if top_per_view:
        seen = {}
        kept = []
        for detection in detections:
            side = detection["view"]
            seen[side] = seen.get(side, 0) + 1
            if seen[side] <= top_per_view:
                kept.append(detection)
        detections = kept

    geometry = None
    if cal is not None:
        L = cal["left"]["corners"]
        R = cal["right"]["corners"]
        geometry = {
            "axis_x_px": round(float(
                (
                    (L[1][0] + L[2][0]) / 2
                    + (R[0][0] + R[3][0]) / 2
                ) / 2
            ), 1),
            "edge_height_px": round(float(
                (
                    (L[2][1] - L[1][1])
                    + (R[3][1] - R[0][1])
                ) / 2
            ), 1),
            "note": (
                "Approximate photo-space geometry from scaled mirror corners; "
                "not a tube-surface measurement."
            ),
        }

    result = {
        "direct_geometry": geometry,
        "image": image_path,
        "image_size_px": [img.shape[1], img.shape[0]],
        "calibration": calib_path,
        "direct_roi_px": list(roi),
        "score_note": (
            "Heuristic proposal score 0..1. NOT a calibrated probability."
        ),
        "detections": detections,
    }

    json_path = os.path.join(
        out_dir, f"{stem}_detections.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    annotated = annotate(img, cal, detections, roi)
    cv2.imwrite(
        os.path.join(out_dir, f"{stem}_annotated.jpg"),
        annotated,
    )

    return result


# ------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(
        description="Automatic barcode-region detection (Phase 5)"
    )
    parser.add_argument("image")
    parser.add_argument(
        "--calib",
        help="calibration_record_N.json or mirror_homographies.json",
    )
    parser.add_argument("--out-dir", default="detections")
    parser.add_argument(
        "--direct-roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
    )
    parser.add_argument(
        "--ignore-below-mm",
        type=float,
        default=124.0,
        help=(
            "Ignore mirror detections below this y coordinate in mm from top; "
            "use 0 to disable."
        ),
    )
    parser.add_argument(
        "--extent-contrast",
        type=float,
        default=0.33,
        help="Minimum contrast for extent-only hits.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print barcode_extent candidates per mirror.",
    )
    parser.add_argument(
        "--top-per-view",
        type=int,
        default=1,
        help="Maximum regions retained per view; 0 keeps all.",
    )
    parser.add_argument(
        "--max-hinge-mm",
        type=float,
        default=93.5,
        help="Ignore detections farther than this from hinge edge.",
    )
    parser.add_argument(
        "--min-purity",
        type=float,
        default=0.0,
        help="Minimum row-purity gate; 0 disables.",
    )
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Drop proposals below this heuristic score.",
    )

    args = parser.parse_args()

    global EXTENT_MIN_CONTRAST, DEBUG, MAX_HINGE_MM, MIN_PURITY
    MIN_PURITY = args.min_purity
    MAX_HINGE_MM = args.max_hinge_mm
    DEBUG = args.debug
    EXTENT_MIN_CONTRAST = args.extent_contrast

    result = run(
        args.image,
        args.calib,
        args.out_dir,
        tuple(args.direct_roi) if args.direct_roi else None,
        not args.no_decode,
        min_score=args.min_score,
        ignore_below_mm=args.ignore_below_mm or None,
        top_per_view=args.top_per_view,
    )

    print(f"{len(result['detections'])} candidate region(s)")
    for index, detection in enumerate(result["detections"]):
        extra = ""
        if "hinge_edge_distance_mm" in detection:
            hinge = detection["hinge_edge_distance_mm"]
            extra = (
                " hinge-edge near/far/centre="
                f"{hinge['near']}/{hinge['far']}/{hinge['centre']} mm"
            )

        decode_attempts = detection["decode_attempts"]
        decoded = decode_attempts[0]["text"] if decode_attempts else "-"
        extra += f" purity={detection.get('row_purity', '-')}"
        print(
            f"  [{index}] {detection['view']:6s} "
            f"score={detection['score']:.2f} "
            f"{detection['orientation']}{extra} decoded={decoded}"
        )

    print(f"Saved results to: {args.out_dir}/")


if __name__ == "__main__":
    main()
