"""Verify / rank EAN-13 candidates against blurred two-mirror barcode profiles.
Usage: python verify_barcode.py CODE [CODE ...]  (or --codes-file codes.txt)
Needs: crop180_left.png-style profile CSVs (left180_profile_full.csv) and crop180_right.png.
Reports each candidate's joint correlation and a z-score vs random valid codes."""
import sys, numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
L=["0001101","0011001","0010011","0111101","0100011","0110001","0101111","0111011","0110111","0001011"]
G=[s[::-1].translate(str.maketrans("01","10")) for s in L]
R=[s.translate(str.maketrans("01","10")) for s in L]
PAR=["LLLLLL","LLGLGG","LLGGLG","LLGGGL","LGLLGG","LGGLLG","LGGGLL","LGLGLG","LGLGGL","LGGLGL"]
chk=lambda d:(10-(sum(d[i]*(3 if i%2 else 1) for i in range(12))%10))%10
def modules(d):
    s="101"+"".join((L if c=="L" else G)[d[i+1]] for i,c in enumerate(PAR[d[0]]))+"01010"+"".join(R[x] for x in d[7:13])+"101"
    return np.array([int(c) for c in s],float)
def render(m,sc,off,sig,n):
    i=np.floor((np.arange(n)-off)/sc).astype(int)
    return gaussian_filter1d(1-np.where((i>=0)&(i<95),m[np.clip(i,0,94)],0.),sig*sc)
def corr(a,b):
    a=a-a.mean();b=b-b.mean();return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
def profile(path,Y0,Y1,X0,X1):  # column-aligned row profile (handles tilt/curvature)
    g=cv2.imread(path,0).astype(np.float32);up=cv2.resize(g[Y0:Y1,X0:X1],None,fx=1,fy=4,interpolation=cv2.INTER_CUBIC)
    H,W=up.shape;M=60;pad=np.pad(up,((M,M),(0,0)),mode="edge");nz=lambda v:(v-v.mean())/(v.std()+1e-6)
    core=nz(pad[M+60:M+H-60,W//2]);cols=[]
    for x in range(W):
        c=nz(pad[:,x]);s=int(np.argmax([core@c[M+60+k:M+H-60+k] for k in range(-M,M+1)]))-M;cols.append(pad[M+s:M+s+H,x])
    q=np.mean(cols,axis=0)[:H//4*4].reshape(-1,4).mean(axis=1);return (q-q.mean())/q.std()
def best(p,m):  # best fit over scale/offset/blur, tried in both orientations
    b=-2
    for q in (p,p[::-1]):
        for s in np.arange(1.5,2.3,0.05):
            for o in np.arange(-5,45,0.5):
                for g in (0.4,0.6):
                    b=max(b,corr(q,render(m,s,o,g,len(q))))
    return b
def joint(views,code):
    return float(np.mean([best(p,modules(code)) for p in views]))
if __name__=="__main__":
    views=[np.loadtxt("left180_profile_full.csv"),profile("crop180_right.png",40,240,64,80)]
    views[0]=(views[0]-views[0].mean())/views[0].std()
    codes=[[int(c) for c in a] for a in sys.argv[1:]]
    rng=np.random.default_rng(0);null=[]
    for _ in range(60):
        d=list(rng.integers(0,10,12));null.append(joint(views,d+[chk(d)]))
    mu,sd=np.mean(null),np.std(null)
    print(f"null (60 random valid codes): mean {mu:.3f} sd {sd:.3f} max {max(null):.3f}")
    for c in sorted(codes,key=lambda c:-joint(views,c)):
        if chk(c[:12])!=c[12]: print("".join(map(str,c)),"INVALID CHECK DIGIT");continue
        j=joint(views,c);print("".join(map(str,c)),f"joint={j:.3f} z={(j-mu)/sd:.1f}")