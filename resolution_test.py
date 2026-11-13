import cv2
import numpy as np
import importlib.util
import os

BASE = r"D:\tech\new"
SOURCE = os.path.join(BASE, "samples", "7cm+90.jpg")
CALIB = os.path.join(BASE, "phase(4)", "calibration_record7cm+90.json")
OUT = os.path.join(BASE, "resolution_test")

os.makedirs(OUT, exist_ok=True)

# Load the existing detector module and its calibration function
module_path = os.path.join(BASE, "barcode_region_detector(1).py")
spec = importlib.util.spec_from_file_location("barcode_detector", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

img = cv2.imread(SOURCE)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {SOURCE}")

cal = module.load_calibration(CALIB, img.shape)

H10 = cal["right"]["H"]
if H10 is None:
    raise ValueError("Right mirror homography is missing.")

ppm = cal["px_per_mm"]
width_mm = cal["width_mm"]
height_mm = cal["height_mm"]

# Save the original 10 px/mm rectification
w10 = int(round(width_mm * ppm))
h10 = int(round(height_mm * ppm))

rect10 = cv2.warpPerspective(
    img, H10, (w10, h10), flags=cv2.INTER_LINEAR
)

# Scale the OUTPUT coordinates of the homography
target_ppm = 20.0
scale = target_ppm / ppm

S = np.array([
    [scale, 0, 0],
    [0, scale, 0],
    [0, 0, 1]
], dtype=np.float64)

H20 = S @ H10

w20 = int(round(width_mm * target_ppm))
h20 = int(round(height_mm * target_ppm))

rect20 = cv2.warpPerspective(
    img, H20, (w20, h20), flags=cv2.INTER_LINEAR
)

cv2.imwrite(os.path.join(OUT, "right_10ppm.png"), rect10)
cv2.imwrite(os.path.join(OUT, "right_20ppm.png"), rect20)

print("\nRESOLUTION COMPARISON COMPLETE")
print(f"Original sampling: {ppm:.1f} pixels/mm")
print(f"Test sampling:     {target_ppm:.1f} pixels/mm")
print(f"10 ppm dimensions: {w10} x {h10}")
print(f"20 ppm dimensions: {w20} x {h20}")
print(f"Saved results in:  {OUT}")
