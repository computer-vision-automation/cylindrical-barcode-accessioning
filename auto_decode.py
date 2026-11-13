"""
auto_decode.py  --  find and decode a 1D EAN-13 barcode from tube photos, no hand-picked boxes, no reference code.

  python auto_decode.py samples                      # every image in the folder
  python auto_decode.py samples\\zoomed.jpg           # one image
  python auto_decode.py samples --expect 8901425076811 --jobs 4

Pipeline per image:
  1. LOCATE  : find stacks of parallel, similar-length dark bars (both bar orientations, 3 image scales)
  2. STRAIGHTEN: crop each stack, align columns vertically (removes tilt/curvature), average along the bar length
  3. DECODE  : choose the 13 digits that best explain the bar profile (EAN-13 structure + check digit); the
               scale/offset start from the profile's own first/last bar -- the reference code is NOT used
  4. VERDICT : PASS only if corr >= 0.80 AND the same code comes back from >= 3 of 4 nudged starts
--expect is used only AFTER decoding, to flag whether the PASS code matches.
Needs: opencv-python numpy scipy   (optional: zxing-cpp)
"""
import argparse, csv, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

try:
    import zxingcpp
except Exception:
    zxingcpp = None

# ----------------------------------------------------------------------------- EAN-13 model
L = ["0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011"]
G = [s[::-1].translate(str.maketrans("01", "10")) for s in L]
R = [s.translate(str.maketrans("01", "10")) for s in L]
PAR = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]


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
    return gaussian_filter1d(1.0 - m, max(0.15, sig) * scale)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ----------------------------------------------------------------------------- 1. LOCATE
def _blobs(g, T=12):
    loc = cv2.GaussianBlur(g, (0, 0), 12)
    dark = (g.astype(np.int16) < loc.astype(np.int16) - T).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, 7), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w >= 10 and 1 <= h <= 22 and w / h >= 1.8 and a >= 0.35 * w * h:
            out.append((int(x), int(y), int(w), int(h), a / w))     # last = mean thickness
    if len(out) > 4000:
        out = sorted(out, key=lambda b: -b[2] * b[4])[:4000]
    return out


def _stacks(bl):
    if len(bl) < 12:
        return []
    B = np.array(bl, float)
    x, y, w, h, t = B.T
    cy = y + h / 2
    seen, out = set(), []
    for i in range(len(B)):
        ov = np.minimum(x + w, x[i] + w[i]) - np.maximum(x, x[i])
        ok = (ov >= 0.6 * np.minimum(w, w[i])) & (w / w[i] > 0.55) & (w / w[i] < 1.8)
        idx = np.flatnonzero(ok)
        if len(idx) < 12:
            continue
        idx = idx[np.argsort(cy[idx])]
        med_t = max(2.0, float(np.median(t[idx])))
        maxgap = 7 * med_t + 4
        chain = [idx[0]]
        chains = []
        for j in idx[1:]:
            if cy[j] - cy[chain[-1]] <= maxgap:
                chain.append(j)
            else:
                chains.append(chain); chain = [j]
        chains.append(chain)
        for ch in chains:
            if len(ch) >= 12:
                key = (ch[0], ch[-1], len(ch))
                if key not in seen:
                    seen.add(key); out.append(ch)
    return [[bl[k] for k in ch] for ch in out]


def _score(stack, g):
    x = np.array([b[0] for b in stack], float); y = np.array([b[1] for b in stack], float)
    w = np.array([b[2] for b in stack], float); h = np.array([b[3] for b in stack], float)
    cy = np.sort(y + h / 2)
    pitch = np.diff(cy)
    f = {}
    f["count"] = float(np.clip((len(stack) - 8) / 16.0, 0, 1))
    cv_p = float(np.std(pitch) / (np.mean(pitch) + 1e-6)) if len(pitch) > 2 else 0.0
    f["irregular"] = float(np.clip(cv_p / 0.4, 0, 1))
    mad = float(np.median(np.abs(w - np.median(w))) / (np.median(w) + 1e-6))
    f["uniform"] = float(np.clip(1 - mad / 0.25, 0, 1))
    x0, x1 = int(np.median(x)), int(np.median(x + w))
    y0, y1 = int(y.min()), int((y + h).max())
    roi = g[max(0, y0):y1, max(0, x0):x1]
    f["contrast"] = float(np.clip((np.percentile(roi, 90) - np.percentile(roi, 10)) / 80.0, 0, 1)) if roi.size else 0.0
    base = float(np.mean(list(f.values())))
    gate = 1.0 if (f["irregular"] > 0.6 and f["uniform"] > 0.4 and len(stack) <= 45) else 0.2   # rulers: periodic and many ticks
    return round(base * gate, 3), f


def locate(gray, top_k=3):
    """Return candidate strips [{score, orient, box(x0,y0,x1,y1 in bar-horizontal space), strip}] best first."""
    cands = []
    for orient in ("h", "v"):
        g_full = gray if orient == "h" else np.ascontiguousarray(gray.T)
        for s in (1, 2, 4):
            g = g_full if s == 1 else cv2.resize(g_full, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
            for st in _stacks(_blobs(g)):
                sc, feats = _score(st, g)
                if sc < 0.35:
                    continue
                x = np.array([b[0] for b in st]); w = np.array([b[2] for b in st])
                y0 = min(b[1] for b in st); y1 = max(b[1] + b[3] for b in st)
                x0, x1 = float(np.median(x)), float(np.median(x + w))
                inset = max(1.0, 0.08 * (x1 - x0))
                pad = 0.08 * (y1 - y0)
                box = [(x0 + inset) * s, (y0 - pad) * s, (x1 - inset) * s, (y1 + pad) * s]
                cands.append({"score": sc, "orient": orient, "scale": s, "n_bars": len(st), "features": feats, "box": box})
    cands.sort(key=lambda c: -c["score"])
    kept = []
    for c in cands:
        def iou(a, b):
            if a["orient"] != b["orient"]:
                return 0.0
            ax0, ay0, ax1, ay1 = a["box"]; bx0, by0, bx1, by1 = b["box"]
            iw = max(0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0, min(ay1, by1) - max(ay0, by0))
            i = iw * ih
            return i / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - i + 1e-9)
        if all(iou(c, k) < 0.3 for k in kept):
            kept.append(c)
    out = []
    for c in kept[:top_k]:
        g_full = gray if c["orient"] == "h" else np.ascontiguousarray(gray.T)
        H, W = g_full.shape
        x0, y0, x1, y1 = c["box"]
        x0, x1 = int(max(0, x0)), int(min(W, x1)); y0, y1 = int(max(0, y0)), int(min(H, y1))
        if x1 - x0 < 6 or y1 - y0 < 40:
            continue
        c["box"] = [x0, y0, x1, y1]
        c["strip"] = g_full[y0:y1, x0:x1].copy()
        out.append(c)
    return out


# ----------------------------------------------------------------------------- 2. STRAIGHTEN
def straighten(strip):
    roi = strip.astype(np.float32)
    up = cv2.resize(roi, None, fx=1, fy=4, interpolation=cv2.INTER_CUBIC)
    H, W = up.shape
    M = 4 * max(10, int(0.5 * W))
    pad = np.pad(up, ((M, M), (0, 0)), mode="edge")
    nz = lambda v: (v - v.mean()) / (v.std() + 1e-6)
    core = nz(pad[M:M + H, W // 2])
    cols = []
    for x in range(W):
        c = nz(pad[:, x])
        sc = [core[M:H - M] @ c[2 * M + s:H + s] for s in range(-M, M + 1)] if H > 2 * M + 8 else [0.0]
        s = int(np.argmax(sc)) - (M if H > 2 * M + 8 else 0)
        cols.append(pad[M + s:M + s + H, x])
    p = np.mean(cols, axis=0)[:H // 4 * 4].reshape(-1, 4).mean(axis=1)
    return p


# ----------------------------------------------------------------------------- 3. DECODE (reference-free)
def blind_all(p, geo, starts=16, seed=1):
    sc, o, g = geo; n = len(p); rng = np.random.default_rng(seed)
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
    f = lambda v: -corr(q, render(modules(d), v[0], v[1], v[2], len(q)))
    r = minimize(f, list(geo), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 250})
    return (r.x[0], r.x[1], max(0.15, r.x[2])), -r.fun


def joint(q, geo, rounds=2):
    for _ in range(rounds):
        ranked = blind_all(q, geo)
        if not ranked:
            return None, -1, geo
        geo, c = refit(q, ranked[0][0], geo)
    return ranked[0][0], c, geo


def initial_geometry(q):
    """scale/offset from the profile's own first and last dark bar (no code involved). Returns None if not bar-like."""
    v = (q - q.min()) / (np.ptp(q) + 1e-9)
    bits = v < 0.5
    edges = np.flatnonzero(np.diff(np.r_[0, bits, 0].astype(int)))
    if len(edges) < 24:                       # EAN-13 has ~30 bars -> ~60 edges even after blur merging
        return None
    start, end = float(edges[0]), float(edges[-1])
    if end - start < 40:
        return None
    return start, end


def decode_profile(p):
    p = np.asarray(p, float)
    # resample so one module is ~4 px (very close-up shots have huge modules -> slow and unnecessary)
    approx_mod = len(p) / 100.0
    if approx_mod > 6:
        f = int(round(approx_mod / 4.0))
        p = cv2.resize(p.reshape(-1, 1), (1, max(40, len(p) // f)), interpolation=cv2.INTER_AREA).ravel()
    n = len(p)
    q0 = (p - p.mean()) / (p.std() + 1e-9)
    info = {"rows": n, "px_per_module_est": round(n / 100.0, 2)}
    if n / 100.0 < 1.6:
        info["verdict"] = "SKIP - too small (< ~1.6 px/module)"; info["pass"] = False
        return info
    best = None
    for rev in (False, True):
        q = q0[::-1] if rev else q0
        ge = initial_geometry(q)
        if ge is None:
            continue
        start, end = ge
        centre = (start + end) / 2
        for ds in (-0.04, -0.02, 0.0, 0.02, 0.04):
            scale = (end - start) / 95.0 * (1 + ds)
            geo = (scale, centre - 47.5 * scale, 0.5)
            code, c, g2 = joint(q, geo)
            if code and (best is None or c > best["corr"]):
                best = {"code": code, "corr": c, "geo": g2, "rev": rev, "q": q}
    if best is None:
        info["verdict"] = "FAIL - no check-valid code found (profile not bar-like)"; info["pass"] = False
        return info
    sc, o, g = best["geo"]; votes = 0
    for ds, do in ((-0.004, 0), (0.004, 0), (0, -0.5), (0, 0.5)):
        code2, _, _ = joint(best["q"], (sc * (1 + ds), o + do, g))
        votes += (code2 == best["code"])
    ok = best["corr"] >= 0.80 and votes >= 3
    info.update({"code": best["code"], "corr": round(best["corr"], 3), "stable": f"{votes}/4",
                 "read": "reversed" if best["rev"] else "forward", "pass": bool(ok),
                 "module_px": round(sc, 2),
                 "verdict": "PASS" if ok else "FAIL - code not accepted (needs corr >= 0.80 and >= 3/4 stable)"})
    return info


def zxing_try(strip):
    if zxingcpp is None:
        return None
    for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        im = strip if rot is None else cv2.rotate(strip, rot)
        for f in (1, 2, 3):
            q = cv2.copyMakeBorder(cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC), 40, 40, 80, 80,
                                   cv2.BORDER_CONSTANT, value=int(im.max()))
            for r in zxingcpp.read_barcodes(q):
                if r.text:
                    return str(r.text)
    return None


# ----------------------------------------------------------------------------- batch
def process(path, out_dir, expect, top_k, save_crops):
    name = os.path.splitext(os.path.basename(path))[0]
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"image": name, "verdict": "unreadable"}
    cands = locate(gray, top_k)
    results = []
    for i, c in enumerate(cands):
        info = decode_profile(straighten(c["strip"]))
        info.update({"candidate": i, "locator_score": c["score"], "bars": c["n_bars"], "orient": c["orient"],
                     "box_x0y0x1y1": c["box"], "zxing": zxing_try(c["strip"])})
        if save_crops:
            cv2.imwrite(os.path.join(out_dir, f"{name}_cand{i}.png"), c["strip"])
        results.append(info)
    passed = [r for r in results if r.get("pass")]
    best = max(passed, key=lambda r: r["corr"]) if passed else (results[0] if results else None)
    # annotated overview
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, (c, r) in enumerate(zip(cands, results)):
        x0, y0, x1, y1 = c["box"]
        if c["orient"] == "v":
            x0, y0, x1, y1 = y0, x0, y1, x1
        col = (0, 200, 0) if r.get("pass") else (0, 0, 255)
        th = max(2, gray.shape[1] // 600)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, th)
        cv2.putText(vis, f"{i}:{r.get('code', '-')}", (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.6, gray.shape[1] / 2500), col, th)
    sc = 1600 / max(vis.shape[:2])
    cv2.imwrite(os.path.join(out_dir, f"{name}_annotated.jpg"),
                cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else vis)
    row = {"image": name, "candidates": len(results)}
    if best:
        row.update({k: best.get(k) for k in ("code", "corr", "stable", "read", "px_per_module_est", "verdict", "zxing")})
        row["box"] = best.get("box_x0y0x1y1")
        row["matches_expected"] = (best.get("code") == expect) if (expect and best.get("pass")) else ""
    else:
        row["verdict"] = "no barcode-like stack found"
    row["all_candidates"] = results
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="image file or folder")
    ap.add_argument("--out", default="auto_decode_out")
    ap.add_argument("--expect", default=None, help="optional known code, used only to flag matches after decoding")
    ap.add_argument("--top", type=int, default=3, help="candidates to try per image")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--save-crops", action="store_true")
    a = ap.parse_args()
    files = ([a.path] if os.path.isfile(a.path) else
             sorted(f for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG") for f in glob.glob(os.path.join(a.path, e))))
    files = sorted(set(files))
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(files)} image(s) -> {a.out}")
    if a.jobs > 1 and len(files) > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(process, files, [a.out] * len(files), [a.expect] * len(files),
                               [a.top] * len(files), [a.save_crops] * len(files)))
    else:
        rows = [process(f, a.out, a.expect, a.top, a.save_crops) for f in files]
    print("\n%-22s %-15s %-6s %-6s %-9s %s" % ("image", "code", "corr", "stab", "px/mod", "verdict"))
    for r in rows:
        print("%-22s %-15s %-6s %-6s %-9s %s %s" % (r["image"][:22], r.get("code") or "-", r.get("corr", "-"),
              r.get("stable", "-"), r.get("px_per_module_est", "-"), r.get("verdict", ""),
              ("[matches expected]" if r.get("matches_expected") else "[DIFFERS from expected]" if r.get("matches_expected") is False else "")))
    with open(os.path.join(a.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "code", "corr", "stable", "read", "px_per_module_est", "verdict", "zxing", "matches_expected", "box"])
        for r in rows:
            w.writerow([r["image"], r.get("code", ""), r.get("corr", ""), r.get("stable", ""), r.get("read", ""),
                        r.get("px_per_module_est", ""), r.get("verdict", ""), r.get("zxing", ""),
                        r.get("matches_expected", ""), r.get("box", "")])
    with open(os.path.join(a.out, "details.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, default=str)
    print(f"\nSaved: {os.path.join(a.out, 'summary.csv')}, details.json and *_annotated.jpg overviews")


if __name__ == "__main__":
    main()