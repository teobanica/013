import sys, json
sys.path.insert(0,'.')
import numpy as np
from math import factorial
import matrices as m
import cases as cs
exec(open("basisglow.py").read().split("out={}")[0].split("import cases as cs")[1])
allc=dict(extra); allc.update(cs.C)
J=json.load(open('basisglow.json'))
def S(M,k):
    K=M**k; N=M.shape[0]; return np.linalg.norm(K@K.conj().T-N*np.eye(N))**2
rows=[]
for name,mk in allc.items():
    H=np.array(mk(),complex); N=H.shape[0]; d=J[name]["dev"]
    D=[d[p-1]*factorial(p)*N**(2*p) for p in range(1,len(d)+1)]
    if len(D)<4: continue
    rel=abs(D[3]-18*S(H,2))/max(1,abs(D[3]))
    rows.append((name,N,D,S(H,2),S(H,3)))
    print(f"{name:26s} N={N:2d} relerr(D4 vs 18 S2)={rel:.1e}  S2={S(H,2):10.3f}  S3={S(H,3):10.3f}")
# p=6 at N=6: fit D6 = a S2 + b S3
for N in [6]:
    R=[r for r in rows if r[1]==N and len(r[2])>=6]
    A=np.array([[r[3],r[4]] for r in R]); y=np.array([r[2][5] for r in R])
    c,res,_,_=np.linalg.lstsq(A,y,rcond=None)
    print("N=%d p=6: D6 = %.4f S2 + %.4f S3, max residual %.2e"%(N,*c,np.abs(A@c-y).max()))
    y=np.array([r[2][6] for r in R]); c,res,_,_=np.linalg.lstsq(A,y,rcond=None)
    print("N=%d p=7: D7 = %.4f S2 + %.4f S3, max residual %.2e"%(N,*c,np.abs(A@c-y).max()))
    for r in R: print("   ",r[0],round(r[2][5],3),round(r[3],3),round(r[4],3))
