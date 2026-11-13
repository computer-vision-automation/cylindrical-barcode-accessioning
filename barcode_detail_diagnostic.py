import cv2
import numpy as np
import importlib.util
import os

BASE = r"D:\tech\new"
PHOTO = os.path.join(BASE, "samples", "7cm+90.jpg")
CALIB = os.path.join(BASE, "phase(4)", "calibration_record7cm+90.json")
OUT = os.path.join(BASE, "barcode_detail_diagnostic")

os.makedirs(OUT, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "detector", os.path.join(BASE, "barcode_region_detector(1).py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

img = cv2.imread(PHOTO)
if img is None:
    raise FileNotFoundError(PHOTO)

cal = mod.load_calibration(CALIB, img.shape)
H = cal["right"]["H"]
ppm = cal["px_per_mm"]

rect = cv2.warpPerspective(img, H, cal["out_size"])
cv2.imwrite(os.path.join(OUT, "right_rectified.png"), rect)

regions = mod.detect_regions(rect, border_px=int(2 * ppm))
print("\nRIGHT MIRROR BARCODE DETAIL REPORT")
print("Original photo:", img.shape[1], "x", img.shape[0])
print("Rectified image:", rect.shape[1], "x", rect.shape[0])
print("Candidates found:", len(regions))

invH = np.linalg.inv(H)
gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)

for i, region in enumerate(regions[:15]):
    box = region.get("box") if isinstance(region, dict) else region
    x, y, w, h = map(int, box[:4])

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(gray.shape[1], x+w), min(gray.shape[0], y+h)
    crop = gray[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    # Map candidate corners back to original camera pixels
    corners = np.float32([[
        [x, y], [x+w, y], [x+w, y+h], [x, y+h]
    ]])
    original_corners = cv2.perspectiveTransform(corners, invH)[0]
    original_width = (
        np.linalg.norm(original_corners[1] - original_corners[0])
    )
    original_height = (
        np.linalg.norm(original_corners[3] - original_corners[0])
    )

    contrast = float(crop.std())
    sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())

    # Estimate narrow run widths across multiple scan lines.
    # This is a diagnostic estimate, not a confirmed barcode measurement.
    estimates = []
    for line in (crop if crop.shape[0] >= crop.shape[1] else crop.T):
        if len(line) < 20:
            continue
        smooth = cv2.GaussianBlur(
            line.reshape(1, -1), (1, 3), 0
        ).ravel()
        threshold = (float(smooth.min()) + float(smooth.max())) / 2
        bits = smooth < threshold
        changes = np.diff(np.r_[False, bits, False].astype(np.int8))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        runs = ends - starts
        runs = runs[runs >= 1]
        if len(runs) >= 5:
            estimates.append(float(np.percentile(runs, 25)))

    narrow_run = (
        float(np.median(estimates)) if estimates else float("nan")
    )

    crop_path = os.path.join(OUT, f"candidate_{i:02d}.png")
    cv2.imwrite(crop_path, crop)

    print(
        f"\nCandidate {i}: rectified={w}x{h}px"
        f" | original-photo estimate={original_width:.1f}x"
        f"{original_height:.1f}px"
        f" | contrast={contrast:.1f}"
        f" | sharpness={sharpness:.1f}"
        f" | estimated narrow run={narrow_run:.1f}px"
        f"\n  crop: {crop_path}"
    )

print("\nFull rectified image and candidate crops saved in:")
print(OUT)
