import cv2, numpy as np, zxingcpp

g = cv2.imread("phase7_right.png", 0).astype(np.float32)
X0, X1, Y0, Y1 = 65, 135, 318, 600          # barcode bars only; adjust if clipped
cv2.imwrite("barcode_crop.png", g[Y0:Y1, X0:X1].astype(np.uint8))

p = g[Y0:Y1, X0:X1].mean(axis=1)
p = (p - p.min()) / (np.ptp(p) + 1e-6)
np.savetxt("profile.csv", p, fmt="%.3f")

bits = p < 0.5
edges = np.flatnonzero(np.diff(np.r_[0, bits, 0].astype(int)))
runs = np.diff(edges)
print("rows:", len(p), "| narrowest runs (px):", np.sort(runs)[:10])

img = (255 * np.repeat(p.reshape(-1, 1), 200, axis=1)).astype(np.uint8)
for name, rot in (("cw", cv2.ROTATE_90_CLOCKWISE), ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE)):
    r = cv2.copyMakeBorder(cv2.rotate(img, rot), 40, 40, 80, 80, cv2.BORDER_CONSTANT, value=255)
    print(name, [(x.format, x.text) for x in zxingcpp.read_barcodes(r)])