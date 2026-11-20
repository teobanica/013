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
    P=M**k; Gm=P.T@P.conj()     # <C_x^k, C_y^k>
    return (np.abs(Gm)**2).sum()-np.trace(np.abs(Gm)**2).real
print(f"{'matrix':28s} {'Delta_4':>12s} {'18*S2(col)':>12s} {'18*S2(row)':>12s}  Delta_5")
for name,mk in allc.items():
    H=np.array(mk(),complex); N=H.shape[0]
    G=[None]+[None]*8
    d=J[name]["dev"]; D4=d[3]*factorial(4)*N**8 if len(d)>3 else float('nan')
    D5=d[4]*factorial(5)*N**10 if len(d)>4 else float('nan')
    print(f"{name:28s} {D4:12.2f} {18*S(H,2):12.2f} {18*S(H.T,2):12.2f}  {D5:14.2f}  ratio5/S2={D5/S(H,2) if S(H,2)>1e-9 else float('nan'):.3f}")
