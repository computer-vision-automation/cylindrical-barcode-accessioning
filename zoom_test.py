"""zoom_test.py  --  usage: python zoom_test.py IMAGE X0 X1 Y0 Y1
Crops the barcode box (original-image pixels), straightens the tilted bars column by column,
averages along the bar length, fits the EAN-13 pattern for 8901425076811 against random codes,
and tries ZXing on the straightened barcode. Needs: opencv-python numpy scipy zxing-cpp"""
import cv2, numpy as np, sys
from scipy.ndimage import gaussian_filter1d

L = ["0001101","0011001","0010011","0111101","0100011","0110001","0101111","0111011","0110111","0001011"]
G = [s[::-1].translate(str.maketrans("01","10")) for s in L]
R = [s.translate(str.maketrans("01","10")) for s in L]
PAR = ["LLLLLL","LLGLGG","LLGGLG","LLGGGL","LGLLGG","LGGLLG","LGGGLL","LGLGLG","LGLGGL","LGGLGL"]

def modules(d):                         # d = list of 13 digits -> 95 modules (1 = dark)
    s = "101"
    for i, ch in enumerate(PAR[d[0]]):
        s += (L if ch == "L" else G)[d[i + 1]]
    s += "01010" + "".join(R[x] for x in d[7:13]) + "101"
    return np.array([int(c) for c in s], float)

def check_ok(d):
    return (10 - (sum(d[i] * (3 if i % 2 else 1) for i in range(12)) % 10)) % 10 == d[12]

def render(mod, scale, off, sig, n):
    # profile is a brightness signal: dark bar -> low value. sample module sequence at row centres
    y = (np.arange(n) - off) / scale
    idx = np.clip(np.floor(y).astype(int), -1, 95)
    m = np.where((idx >= 0) & (idx < 95), mod[np.clip(idx, 0, 94)], 0.0)   # outside = quiet zone (light)
    return gaussian_filter1d(1.0 - m, sig * scale)

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def best_fit(p, mod, scales, offs, sigs):
    from scipy.optimize import minimize
    cand = []
    for sc in scales:
        for o in offs:
            for g in sigs:
                cand.append((corr(p, render(mod, sc, o, g, len(p))), sc, o, g))
    cand.sort(key=lambda t: -t[0])
    best = (-2, None)
    for c, sc, o, g in cand[:3]:                      # refine the 3 best coarse fits continuously
        f = lambda v: -corr(p, render(mod, v[0], v[1], max(0.15, v[2]), len(p)))
        r = minimize(f, [sc, o, g], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 300})
        if -r.fun > best[0]: best = (-r.fun, tuple(r.x))
    return best

def run(p, ref_digits, n_starts=40, seed=0):
    n = len(p)
    scales = np.arange(n / 130, n / 85, n / 95 * 0.004)      # rows per module; fine steps (errors add up over 95 modules)
    offs = np.arange(-0.1 * n, 0.3 * n, max(0.5, n / 400))
    sigs = (0.35, 0.5, 0.7)
    out = {}
    for rev in (False, True):
        q = p[::-1] if rev else p
        c0, geo = best_fit(q, modules(ref_digits), scales, offs, sigs)
        # null: random valid EAN-13s with the SAME search
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(40):
            d = list(rng.integers(0, 10, 12)); d.append((10 - (sum(d[i] * (3 if i % 2 else 1) for i in range(12)) % 10)) % 10)
            null.append(best_fit(q, modules(d), scales[::2], offs[::2], sigs[1:2])[0])
        null = np.array(null)
        out[rev] = (c0, geo, null)
        print(f"{'reversed' if rev else 'forward '}: corr with reference code = {c0:.3f} | random valid codes: mean {null.mean():.3f}, max {null.max():.3f}, 95th pct {np.percentile(null,95):.3f}")
    return out


import cv2, numpy as np, sys
try:
    import zxingcpp
except Exception:
    zxingcpp = None
sys.argv = sys.argv[:1] + sys.argv[1:]

def profile(img, X0, X1, Y0, Y1, M=60):
    roi = img[Y0:Y1, X0:X1].astype(np.float32)
    up = cv2.resize(roi, None, fx=1, fy=4, interpolation=cv2.INTER_CUBIC)
    H, W = up.shape
    pad = np.pad(up, ((M, M), (0, 0)), mode="edge")
    nz = lambda v: (v - v.mean()) / (v.std() + 1e-6)
    core = nz(pad[M:M + H, W // 2]); cols = []
    for x in range(W):
        c = nz(pad[:, x])
        sc = [core[M:H - M] @ c[2*M + s:H + s] for s in range(-M, M + 1)]
        s = int(np.argmax(sc)) - M
        cols.append(pad[M + s:M + s + H, x])
    p4 = np.mean(cols, axis=0)
    global ALIGNED
    ALIGNED = np.stack(cols, axis=1)
    return p4[:H // 4 * 4].reshape(-1, 4).mean(axis=1), roi

img = cv2.imread(sys.argv[1], 0)
X0, X1, Y0, Y1 = [int(v) for v in sys.argv[2:6]]
p, roi = profile(img, X0, X1, Y0, Y1)
cv2.imwrite('zoom_roi_6x.png', cv2.resize(roi.astype(np.uint8), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
pn = (p - p.mean()) / p.std()
print("rows", len(p))
run(pn, [int(c) for c in "8901425076811"])
v = (p - p.min()) / np.ptp(p)
bits = v < 0.5
runs = np.diff(np.flatnonzero(np.diff(np.r_[0, bits, 0].astype(int))))
print("narrowest runs:", np.sort(runs)[:12])
im = (255 * np.repeat(cv2.resize(v.reshape(-1,1),(1,len(v)*4),interpolation=cv2.INTER_CUBIC),200,axis=1)).clip(0,255).astype(np.uint8)
for nm, rot in (("cw", cv2.ROTATE_90_CLOCKWISE), ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE)):
    q = cv2.copyMakeBorder(cv2.rotate(im, rot), 40, 40, 100, 100, cv2.BORDER_CONSTANT, value=255)
    print(nm, [(x.format, x.text) for x in zxingcpp.read_barcodes(q)] if zxingcpp else "zxing n/a here")

al = np.clip(ALIGNED, 0, 255).astype(np.uint8)
cv2.imwrite("zoom_aligned_2d.png", al)
al = cv2.resize(al, None, fx=2, fy=0.5 * 1, interpolation=cv2.INTER_CUBIC)
for nm, rot in (("cw", cv2.ROTATE_90_CLOCKWISE), ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE)):
    q = cv2.copyMakeBorder(cv2.rotate(al, rot), 40, 40, 100, 100, cv2.BORDER_CONSTANT, value=int(al.max()))
    for scale in (1, 2, 3):
        qq = cv2.resize(q, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        r = [(x.format, x.text) for x in zxingcpp.read_barcodes(qq)] if zxingcpp else "zxing n/a here"
        print("aligned 2D", nm, f"x{scale}", r)


# ---------------- blind decode: digits chosen from the bars only, validated by the EAN-13 check digit ----------------
from scipy.optimize import minimize

def blind_all(p, geo, starts=30, seed=1):
    sc, o, g = geo; n = len(p); g = max(0.15, g); rng = np.random.default_rng(seed)
    score = lambda d: corr(p, render(modules(d), sc, o, g, n)); found = {}
    for _ in range(starts):
        d = [int(v) for v in rng.integers(0, 10, 13)]; cur = score(d); imp = True
        while imp:
            imp = False
            for i in range(13):
                bv, bc = d[i], cur
                for v in range(10):
                    if v != d[i]:
                        t = d.copy(); t[i] = v; c = score(t)
                        if c > bc + 1e-9: bv, bc = v, c
                if bv != d[i]: d[i] = bv; cur = bc; imp = True
        if check_ok(d): found["".join(map(str, d))] = cur
    return sorted(found.items(), key=lambda kv: -kv[1])

def refit(q, code, geo):
    d = [int(c) for c in code]
    f = lambda v: -corr(q, render(modules(d), v[0], v[1], max(0.15, v[2]), len(q)))
    r = minimize(f, list(geo), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 300})
    return tuple(r.x), -r.fun

def joint(q, geo, rounds=3):
    """alternate: pick digits for the current geometry, then re-fit the geometry to those digits"""
    for _ in range(rounds):
        ranked = blind_all(q, geo)
        if not ranked: return None, -1
        geo, c = refit(q, ranked[0][0], geo)
    return ranked[0][0], c

print("\nBLIND DECODE (digits chosen from the bars only; start geometry from a fit, then refined jointly)")
n = len(pn); refd = [int(c) for c in "8901425076811"]
scales = np.arange(n / 130, n / 85, n / 95 * 0.004); offs = np.arange(-0.1 * n, 0.3 * n, max(0.5, n / 400))
best = None
for rev in (False, True):
    q = pn[::-1] if rev else pn
    c0, geo = best_fit(q, modules(refd), scales, offs, (0.35, 0.5, 0.7))
    code, c = joint(q, geo)
    votes = 0
    for ds, do in ((-0.004, 0), (0.004, 0), (0, -0.5), (0, 0.5)):          # stability: nudge the start geometry
        code2, _ = joint(q, (geo[0] * (1 + ds), geo[1] + do, geo[2]))
        votes += (code2 == code)
    print(f"  {'reversed' if rev else 'forward '}: code = {code} | corr {c:.3f} | same code from {votes}/4 nudged starts")
    if code and (best is None or c > best[1]): best = (code, c, rev, votes)
if best:
    ok = best[1] >= 0.80 and best[3] >= 3
    print(f"CANDIDATE: {best[0]}  (corr {best[1]:.3f}, stable {best[3]}/4, read {'bottom-to-top' if best[2] else 'top-to-bottom'})")
    print("VERDICT: PASS - accepted (corr >= 0.80 and the same code from >= 3 of 4 nudged starts)" if ok else
          "VERDICT: FAIL - not accepted (needs corr >= 0.80 and >= 3/4 stable). Do not report this as a decode.")
else:
    print("VERDICT: FAIL - no check-valid code found")