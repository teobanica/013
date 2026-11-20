import numpy as np, itertools, sys, json
from math import factorial
def walsh(n):
    H=np.array([[1.]])
    for _ in range(n): H=np.block([[H,H],[H,-H]])
    return H
def moments_grid(H,pmax,chunk=200000):
    N=H.shape[0]; L=pmax+1
    roots=np.exp(2j*np.pi*np.arange(L)/L)
    f=np.array([1/factorial(m)**2 for m in range(pmax+1)])
    acc=np.zeros(pmax+1)
    total=L**(N-1); idx=np.arange(total)
    for s in range(0,total,chunk):
        ii=idx[s:s+chunk]; B=np.ones((len(ii),N),complex); r=ii.copy()
        for y in range(1,N): B[:,y]=roots[r%L]; r//=L
        W=B@H.T; A=np.abs(W)**2           # |w_x|^2
        P=np.zeros((len(ii),pmax+1)); P[:,0]=1
        for x in range(N):
            Q=A[:,x:x+1]**np.arange(pmax+1)*f   # f(t|w|^2) coefficients
            newP=np.zeros_like(P)
            for k in range(pmax+1): newP[:,k]=(P[:,:k+1]*Q[:,k::-1][:, :k+1]).sum(1)
            P=newP
        acc+=P.sum(0)
    acc/=total
    return [factorial(p)**2*acc[p] for p in range(pmax+1)]
if __name__=="__main__":
    pmax=int(sys.argv[1]); res={}
    for n,pm in [(1,pmax),(2,pmax),(3,min(pmax,6))]:
        m=moments_grid(walsh(n),pm); res[2**n]=m
        print(2**n,[f"{v:.6f}" for v in m[1:]],flush=True)
    json.dump(res,open('grid.json','w'))
