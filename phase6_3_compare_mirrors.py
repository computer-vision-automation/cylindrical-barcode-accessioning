import cv2
import numpy as np
from pathlib import Path

base = Path(r"D:\tech\new\phase5_new\7cm+90")
names = [
    "7cm+90_left_rectified.jpg",
    "7cm+90_right_rectified.jpg"
]

panels = []

for name in names:
    img = cv2.imread(str(base / name))
    if img is None:
        print("Missing:", name)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    cv2.putText(enhanced, name, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    panels.append(enhanced)

if len(panels) == 2:
    comparison = np.hstack(panels)
    out = Path(r"D:\tech\new\phase6_new\left_right_mirror_comparison.jpg")
    cv2.imwrite(str(out), comparison)
    print("Saved:", out)
