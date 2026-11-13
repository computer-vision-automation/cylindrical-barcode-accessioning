import cv2, numpy as np, zxingcpp

g = cv2.imread("crop180_left.png", 0).astype(np.float32)
roi = g[40:220, 194:220]                                   # bars only: no digits, no dark grazing edge
up = cv2.resize(roi, None, fx=1, fy=4, interpolation=cv2.INTER_CUBIC)   # 4x vertical for sub-pixel alignment
H, W = up.shape
M = 60
nz = lambda v: (v - v.mean()) / (v.std() + 1e-6)

ref = nz(up[:, W // 2])
core = ref[M:H - M]
cols, shifts = [], []
for x in range(W):
    c = nz(up[:, x])
    sc = [np.dot(core, c[M + s:H - M + s]) for s in range(-M, M + 1)]
    s = int(np.argmax(sc)) - M
    shifts.append(s)
    cols.append(c[M + s:H - M + s])
print("column shifts (4x px):", shifts)

p = np.mean(cols, axis=0)
p = (p - p.min()) / (np.ptp(p) + 1e-6)
np.savetxt("left180_profile_aligned.csv", p, fmt="%.3f")

bits = p < 0.5
runs = np.diff(np.flatnonzero(np.diff(np.r_[0, bits, 0].astype(int)))) / 4
print("rows(4x):", len(p), "| narrowest runs (orig px):", np.sort(runs)[:12])

img = (255 * np.repeat(p.reshape(-1, 1), 200, axis=1)).clip(0, 255).astype(np.uint8)
cv2.imwrite("left180_aligned.png", img)
for name, rot in (("cw", cv2.ROTATE_90_CLOCKWISE), ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE)):
    q = cv2.copyMakeBorder(cv2.rotate(img, rot), 40, 40, 100, 100, cv2.BORDER_CONSTANT, value=255)
    print(name, [(x.format, x.text) for x in zxingcpp.read_barcodes(q)])