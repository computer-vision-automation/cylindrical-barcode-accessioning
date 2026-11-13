import cv2
import numpy as np
import importlib.util
import os

BASE = r"D:\tech\new"
PHOTO = os.path.join(BASE, "samples", "7cm+90.jpg")
CALIB = os.path.join(BASE, "phase(4)", "calibration_record7cm+90.json")
OUT = os.path.join(BASE, "barcode_detail_diagnostic")

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
    raise RuntimeError("Candidate 2 not found.")

box = regions[2].get("box") if isinstance(regions[2], dict) else regions[2]
x, y, w, h = map(int, box[:4])

corners = np.float32([[
    [x, y], [x+w, y], [x+w, y+h], [x, y+h]
]])
source = cv2.perspectiveTransform(corners, np.linalg.inv(H))[0]

top = np.linalg.norm(source[1] - source[0])
bottom = np.linalg.norm(source[2] - source[3])
left = np.linalg.norm(source[3] - source[0])
right = np.linalg.norm(source[2] - source[1])

ow = max(1, round((top + bottom) / 2))
oh = max(1, round((left + right) / 2))

dst = np.float32([
    [0, 0], [ow-1, 0], [ow-1, oh-1], [0, oh-1]
])
M = cv2.getPerspectiveTransform(source.astype(np.float32), dst)

methods = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}

for name, method in methods.items():
    result = cv2.warpPerspective(img, M, (ow, oh), flags=method)
    cv2.imwrite(os.path.join(OUT, f"candidate2_{name}.png"), result)

    # Nearest-neighbour enlargement is only for easier visual inspection
    enlarged = cv2.resize(
        result, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(
        os.path.join(OUT, f"candidate2_{name}_3x.png"), enlarged
    )

print("Interpolation comparison saved in:", OUT)
print("Native dimensions:", ow, "x", oh)
