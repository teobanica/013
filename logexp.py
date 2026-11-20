import sys
from fractions import Fraction as F
from math import factorial, comb
def poly(f):
    P={}
    for l in open(f):
        p,rest=l.split(':'); p=int(p[2:])
        P[p]={int(t.split('^')[1]):int(t.split('*')[0]) for t in rest.replace(' + ',' ').split()}
    return P
def polymul(a,b):
    r={}
    for e1,v1 in a.items():
        for e2,v2 in b.items(): r[e1+e2]=r.get(e1+e2,0)+v1*v2
    return r
def logcoef(P):
    pm=max(P); G=[{0:F(1)}]+[{e:F(v,factorial(p)**2) for e,v in P[p].items()} for p in range(1,pm+1)]
    L=[{} for _ in range(pm+1)]
    for k in range(1,pm+1):
        s={e:k*v for e,v in G[k].items()}
        for j in range(1,k):
            for e,v in polymul({e:j*v for e,v in L[j].items()},G[k-j]).items(): s[e]=s.get(e,0)-v
        L[k]={e:v/k for e,v in s.items() if v!=0}
    return L
for f in sys.argv[1:]:
    L=logcoef(poly(f)); print("==",f)
    for g in range(0,4):
        print(" genus",g,[str(L[k].get(k+1-g,0)) for k in range(1,len(L))])
    print(" genus0 check Cat:",[L[k].get(k+1,0)==F((-1)**(k-1)*comb(2*k-2,k-1),k*k) for k in range(1,len(L))])
    print(" genus1 check (-1)^k C(2k-2,k-2)/4:",[L[k].get(k,0)==F((-1)**k*comb(2*k-2,k-2),4) for k in range(2,len(L))])
