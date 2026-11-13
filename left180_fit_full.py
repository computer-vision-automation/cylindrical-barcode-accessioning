import cv2, numpy as np, sys
from scipy.ndimage import gaussian_filter1d

L = ["0001101","0011001","0010011","0111101","0100011","0110001","0101111","0111011","0110111","0001011"]
G = [s[::-1].translate(str.maketrans("01","10")) for s in L]
R = [s.translate(str.maketrans("01","10")) for s in L]
PAR = ["LLLLLL","LLGLGG","LLGGLG","LLGGGL","LGLLGG","LGGLLG","LGGGLL","LGLGLG","LGLGGL","LGGLGL"]

def modules(d):
    s = "101"
    for i, ch in enumerate(PAR[d[0]]):
        s += (L if ch == "L" else G)[d[i + 1]]
    s += "01010" + "".join(R[x] for x in d[7:13]) + "101"
    return np.array([int(c) for c in s], float)

def check_ok(d):
    return (10 - (sum(d[i] * (3 if i % 2 else 1) for i in range(12)) % 10)) % 10 == d[12]

def render(mod, scale, off, sig, n):
    y = (np.arange(n) - off) / scale
    idx = np.clip(np.floor(y).astype(int), -1, 95)
    m = np.where((idx >= 0) & (idx < 95), mod[np.clip(idx, 0, 94)], 0.0)
    return gaussian_filter1d(1.0 - m, sig * scale)

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def best_fit(p, mod, scales, offs, sigs):
    best = (-2, None)
    for s in scales:
        for o in offs:
            for g in sigs:
                c = corr(p, render(mod, s, o, g, len(p)))
                if c > best[0]: best = (c, (s, o, g))
    return best

def run(p, ref_digits, seed=0):
    scales = np.arange(1.5, 2.6, 0.05); offs = np.arange(-5, 40, 0.5); sigs = (0.35, 0.5, 0.7)
    for rev in (False, True):
        q = p[::-1] if rev else p
        c0, geo = best_fit(q, modules(ref_digits), scales, offs, sigs)
        rng = np.random.default_rng(seed); null = []
        for _ in range(60):
            d = list(rng.integers(0, 10, 12)); d.append((10 - (sum(d[i] * (3 if i % 2 else 1) for i in range(12)) % 10)) % 10)
            null.append(best_fit(q, modules(d), scales[::2], offs[::2], sigs[1:2])[0])
        null = np.array(null)
        print(f"{'reversed' if rev else 'forward '}: corr with reference = {c0:.3f} | random valid codes: mean {null.mean():.3f}, max {null.max():.3f}, 95th pct {np.percentile(null,95):.3f}")

g = cv2.imread("crop180_left.png", 0).astype(np.float32)
Y0, Y1, X0, X1 = 25, 235, 194, 220          # extra rows top/bottom so the guard bars are not clipped
up = cv2.resize(g[Y0:Y1, X0:X1], None, fx=1, fy=4, interpolation=cv2.INTER_CUBIC)
H, W = up.shape; M = 60
pad = np.pad(up, ((M, M), (0, 0)), mode="edge")
nz = lambda v: (v - v.mean()) / (v.std() + 1e-6)
core = nz(pad[M:M + H, W // 2])
cols = []
for x in range(W):
    c = nz(pad[:, x])
    sc = [core[M:H - M] @ c[M + s + M:H - M + s + M] for s in range(-M, M + 1)]
    s = int(np.argmax(sc)) - M
    cols.append(pad[M + s:M + s + H, x])
p = np.mean(cols, axis=0)[:H // 4 * 4].reshape(-1, 4).mean(axis=1)
np.savetxt("left180_profile_full.csv", p, fmt="%.3f")
print("profile rows:", len(p))
run((p - p.mean()) / (p.std() + 1e-9), [int(c) for c in "8901425076811"])