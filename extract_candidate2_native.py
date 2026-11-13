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
cal = mod.load_calibration(CALIB, img.shape)
H = cal["right"]["H"]

rect = cv2.warpPerspective(img, H, cal["out_size"])
regions = mod.detect_regions(rect, border_px=int(2 * cal["px_per_mm"]))

if len(regions) < 3:
    raise RuntimeError("Candidate 2 was not found.")

region = regions[2]
box = region.get("box") if isinstance(region, dict) else region
x, y, w, h = map(int, box[:4])

rect_corners = np.float32([[
    [x, y], [x+w, y], [x+w, y+h], [x, y+h]
]])

source_corners = cv2.perspectiveTransform(
    rect_corners, np.linalg.inv(H)
)[0]

# Save original photo with candidate boundary
marked = img.copy()
cv2.polylines(
    marked, [np.int32(source_corners)], True, (0, 0, 255), 5
)
cv2.imwrite(os.path.join(OUT, "candidate2_original_location.jpg"), marked)

# Native-source-resolution perspective extraction
top = np.linalg.norm(source_corners[1] - source_corners[0])
bottom = np.linalg.norm(source_corners[2] - source_corners[3])
left = np.linalg.norm(source_corners[3] - source_corners[0])
right = np.linalg.norm(source_corners[2] - source_corners[1])

out_w = max(1, round((top + bottom) / 2))
out_h = max(1, round((left + right) / 2))

destination = np.float32([
    [0, 0], [out_w-1, 0],
    [out_w-1, out_h-1], [0, out_h-1]
])

M = cv2.getPerspectiveTransform(
    source_corners.astype(np.float32), destination
)

native = cv2.warpPerspective(
    img, M, (out_w, out_h), flags=cv2.INTER_LINEAR
)

cv2.imwrite(os.path.join(OUT, "candidate2_native.png"), native)

print("CANDIDATE 2 SOURCE EXTRACTION")
print("Original photograph:", img.shape[1], "x", img.shape[0])
print("Native extraction:", out_w, "x", out_h)
print("Saved:")
print(os.path.join(OUT, "candidate2_original_location.jpg"))
print(os.path.join(OUT, "candidate2_native.png"))
