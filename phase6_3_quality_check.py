import cv2
import numpy as np
from pathlib import Path

files = [
    Path(r"D:\tech\new\phase5_new\10cm0\10cm0_direct_0.png"),
    Path(r"D:\tech\new\phase5_new\7cm+90\7cm+90_right_1.png"),
    Path(r"D:\tech\new\phase5_new\7cm+90\7cm+90_right_2.png"),
    Path(r"D:\tech\new\phase5_new\7cm+90\7cm+90_right_3.png"),
    Path(r"D:\tech\new\phase5_new\7cm+90\7cm+90_right_4.png"),
    Path(r"D:\tech\new\phase5_new\7cm+90\7cm+90_right_5.png"),
]

tiles = []

for p in files:
    im = cv2.imread(str(p))
    if im is None:
        print(f"\nMISSING: {p.name}")
        continue

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()

    print(f"\n{p.name}")
    print(f"  Dimensions: {im.shape[1]} x {im.shape[0]}")
    print(f"  Sharpness (Laplacian): {sharpness:.2f}")
    print(f"  Contrast (std dev): {contrast:.2f}")

    scale = min(500 / im.shape[1], 220 / im.shape[0])
    resized = cv2.resize(im, None, fx=scale, fy=scale)
    canvas = np.full((260, 520, 3), 255, np.uint8)
    h, w = resized.shape[:2]
    canvas[30:30+h, :w] = resized
    cv2.putText(canvas, p.name, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
    tiles.append(canvas)

if tiles:
    sheet = np.vstack(tiles)
    out = Path(r"D:\tech\new\phase6_new\mirror_quality_comparison.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"\nCONTACT SHEET: {out}")
