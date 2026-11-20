from fractions import Fraction as F
from math import factorial, comb, isqrt
P={}
for l in open('walsh.out'):
    p,rest=l.split(':'); p=int(p[2:])
    P[p]={int(tt.split('^')[1]):int(tt.split('*')[0]) for tt in rest.replace(' + ',' ').split()}
pm=max(P)
print("P_p(1):",[sum(P[p].values()) for p in P])
print("sqrt [N^2]P_p:",[isqrt(P[p][2]) for p in P], [isqrt(P[p][2])**2==P[p][2] for p in P])
K={}
for p in P:
    K[p]=[F(P[p].get(2*p-k,0),factorial(p))*(-1)**k for k in range(0,2*p-1)]
    print(p,[str(v) for v in K[p][:7]])
# fit K_k(p) = sum_j c_j binom(p,j)
def fit(k,jmin,jmax):
    pts=[p for p in P if p>=jmin][: jmax-jmin+1]
    A=[[F(comb(p,j)) for j in range(jmin,jmax+1)] for p in pts]; b=[K[p][k] if k<len(K[p]) else F(0) for p in pts]
    n=len(A)
    for i in range(n):
        piv=next(r for r in range(i,n) if A[r][i]!=0); A[i],A[piv]=A[piv],A[i]; b[i],b[piv]=b[piv],b[i]
        for r in range(n):
            if r!=i and A[r][i]!=0:
                f=A[r][i]/A[i][i]; A[r]=[x-f*y for x,y in zip(A[r],A[i])]; b[r]-=f*b[i]
    c=[b[i]/A[i][i] for i in range(n)]
    chk=[(p,sum(cj*comb(p,j) for cj,j in zip(c,range(jmin,jmax+1)))==(K[p][k] if k<len(K[p]) else 0)) for p in P]
    return c,chk
for k,(jmin,jmax) in {1:(2,2),2:(2,4),3:(3,6),4:(3,8)}.items():
    if jmax-jmin+1<=len([p for p in P if p>=jmin]):
        c,chk=fit(k,jmin,jmax); print("K%d binom coeffs j=%d..%d:"%(k,jmin,jmax),[str(x) for x in c],chk)
# cumulant transform: log sum P_p t^p/p!^2, coefficients as polynomials in N
def polymul(a,b):
    r={}
    for e1,v1 in a.items():
        for e2,v2 in b.items(): r[e1+e2]=r.get(e1+e2,0)+v1*v2
    return r
G=[{0:F(1)}]+[{e:F(v,factorial(p)**2) for e,v in P[p].items()} for p in range(1,pm+1)]
Lc=[{}]*(pm+1)
for k in range(1,pm+1):
    s={e:k*v for e,v in G[k].items()}
    for j in range(1,k):
        for e,v in polymul({e:j*v for e,v in Lc[j].items()},G[k-j]).items(): s[e]=s.get(e,0)-v
    Lc[k]={e:v/k for e,v in s.items() if v!=0}
    print("C_%d:"%k," + ".join(f"({Lc[k][e]})N^{e}" for e in sorted(Lc[k],reverse=True)))
