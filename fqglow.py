# Glow moments of F_G, G = (Z_q)^n, q prime, as polynomials in N = q^n.
import sys, functools, json
from fractions import Fraction
from math import factorial
from collections import Counter
from walshglow import setparts, parts_list, mobius_weight, arrangements, padd, steinhaus_w
q=int(sys.argv[1])
def rankq(vecs):
    rows=[list(v) for v in vecs if any(v)]; r=0
    if not rows: return 0
    m=len(rows[0])
    for c in range(m):
        piv=next((i for i in range(r,len(rows)) if rows[i][c]%q),None)
        if piv is None: continue
        rows[r],rows[piv]=rows[piv],rows[r]; inv=pow(rows[r][c],q-2,q)
        rows[r]=[x*inv%q for x in rows[r]]
        for i in range(len(rows)):
            if i!=r and rows[i][c]%q:
                f=rows[i][c]; rows[i]=[(x-f*y)%q for x,y in zip(rows[i],rows[r])]
        r+=1
    return r
@functools.lru_cache(None)
def D(cols):
    k=len(cols); res={}
    for rho in parts_list(k):
        nb=max(rho)+1; merged=[[0]*len(cols[0]) for _ in range(nb)]
        for c,b in enumerate(rho): merged[b]=[(x+y)%q for x,y in zip(merged[b],cols[c])]
        e=nb-rankq(merged)
        res[e]=res.get(e,0)+mobius_weight(rho)
        if res[e]==0: del res[e]
    return res
@functools.lru_cache(None)
def I_blocks(sizes):
    if not sizes: return {0:1}
    K=sum(sizes); r=len(sizes); blk=[]
    for t,s in enumerate(sizes): blk+=[t]*s
    tot={}
    for f in setparts(K):
        k=max(f)+1
        for g in arrangements(f):
            cols=[[0]*r for _ in range(k)]
            for l in range(K):
                cols[f[l]][blk[l]]+=1; cols[g[l]][blk[l]]-=1
            tot=padd(tot,D(tuple(sorted(tuple(x%q for x in c) for c in cols))))
    return tot
def I_pi(bs):
    s=sorted(b for b in bs if b>=2); a=sum(1 for b in bs if b==1)
    return {e+a:v for e,v in I_blocks(tuple(s)).items()}
pmax=int(sys.argv[2]); w=steinhaus_w(pmax); out=sys.argv[3]
with open(out,'w') as fo:
    for p in range(1,pmax+1):
        tot={}
        for pi in setparts(p):
            sz=list(Counter(pi).values()); coef=Fraction(factorial(p))
            for s in sz: coef*=w[s]
            tot=padd(tot,{e+len(sz):coef*v for e,v in I_pi(sz).items()})
        assert all(v.denominator == 1 for v in tot.values())
        m={e:int(v) for e,v in tot.items()}
        line=f"p={p}: "+" + ".join(f"{m[e]}*N^{e}" for e in sorted(m,reverse=True))
        print(line,flush=True); fo.write(line+"\n"); fo.flush()
