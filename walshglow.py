# Exact complex glow moments of the Walsh matrix W_N, N=2^n, as polynomials in N.
# E|Omega|^{2p} = p! sum_{pi in P(p)} prod_beta w_{|beta|} N^{|pi|} I(pi),
# w_k = k! kappa_k, kappa = coefficients of log sum_m s^m/m!^2  (Steinhaus cumulants),
# I(pi) = #{i,j in G^p : [i]=[j], block sums equal}, computed symbolically in N=2^n.
import sys, itertools, functools, json
from fractions import Fraction
from math import factorial
from collections import Counter

def padd(a,b,c=1):
    r=dict(a)
    for e,v in b.items():
        r[e]=r.get(e,0)+c*v
        if r[e]==0: del r[e]
    return r
def pmul(a,b):
    r={}
    for e1,v1 in a.items():
        for e2,v2 in b.items():
            r[e1+e2]=r.get(e1+e2,0)+v1*v2
    return {e:v for e,v in r.items() if v!=0}

def setparts(n):
    # restricted growth strings
    def rec(i,cur,m):
        if i==n: yield tuple(cur); return
        for c in range(m+1):
            cur.append(c); yield from rec(i+1,cur,max(m,c+1) if c==m else m); cur.pop()
    if n==0: yield (); return
    yield from rec(0,[],0)

@functools.lru_cache(None)
def parts_list(k): return list(setparts(k))

def rank2(masks):
    basis=[]
    for m in masks:
        for b in basis: m=min(m,m^b)
        if m: basis.append(m)
    return len(basis)

@functools.lru_cache(None)
def mobius_weight(rgs):
    # mu(0hat, rho) = prod (-1)^{|b|-1}(|b|-1)!
    c=Counter(rgs); r=1
    for s in c.values(): r*=(-1)**(s-1)*factorial(s-1)
    return r

@functools.lru_cache(None)
def D(cols):
    # number of distinct v in (F_2^n)^k with sum_c cols[c]*v_c = 0 (rowwise), poly in N
    k=len(cols); res={}
    for rho in parts_list(k):
        nb=max(rho)+1; merged=[0]*nb
        for c,b in enumerate(rho): merged[b]^=cols[c]
        e=nb-rank2(merged)
        res[e]=res.get(e,0)+mobius_weight(rho)
        if res[e]==0: del res[e]
    return res

def arrangements(f):
    # all g with same multiset of values as f
    c=Counter(f); vals=sorted(c)
    K=len(f)
    def rec(i,cnt,cur):
        if i==K: yield tuple(cur); return
        for v in vals:
            if cnt[v]:
                cnt[v]-=1; cur.append(v); yield from rec(i+1,cnt,cur); cur.pop(); cnt[v]+=1
    yield from rec(0,dict(c),[])

@functools.lru_cache(None)
def I_blocks(sizes):
    # sizes: sorted tuple of block sizes >=2
    if not sizes: return {0:1}
    K=sum(sizes); blk=[]
    for t,s in enumerate(sizes): blk+= [t]*s
    tot={}
    for f in setparts(K):
        k=max(f)+1
        for g in arrangements(f):
            cols=[0]*k
            for l in range(K):
                cols[f[l]]^=1<<blk[l]; cols[g[l]]^=1<<blk[l]
            tot=padd(tot,D(tuple(sorted(cols))))
    return tot

def I_pi(blocksizes):
    s=sorted(b for b in blocksizes if b>=2); a=sum(1 for b in blocksizes if b==1)
    return {e+a:v for e,v in I_blocks(tuple(s)).items()}

def steinhaus_w(pmax):
    # f(s)=sum s^m/m!^2 ; log f = sum kappa_k s^k ; w_k = k! kappa_k
    f=[Fraction(1,factorial(m)**2) for m in range(pmax+1)]
    L=[Fraction(0)]*(pmax+1)
    # log via L' = f'/f : sum k L_k s^{k-1} * f = f'
    for k in range(1,pmax+1):
        # coefficient of s^{k-1}: sum_{j=1..k} j L_j f_{k-j} = k f_k
        s=k*f[k]-sum(j*L[j]*f[k-j] for j in range(1,k))
        L[k]=s/(k*f[0])
    return [factorial(k)*L[k] for k in range(pmax+1)]

def moment(p,w):
    tot={}
    for pi in setparts(p):
        sizes=list(Counter(pi).values()); nb=len(sizes)
        coef=Fraction(factorial(p))
        for s in sizes: coef*=w[s]
        Ip=I_pi(sizes)
        tot=padd(tot,{e+nb:coef*v for e,v in Ip.items()})
    return tot

if __name__=="__main__":
    pmax=int(sys.argv[1]); out=sys.argv[2]
    w=steinhaus_w(pmax)
    res={}
    with open(out,'w') as fo:
        for p in range(1,pmax+1):
            m=moment(p,w)
            assert all(v.denominator==1 for v in m.values())
            m={e:int(v) for e,v in m.items()}
            res[p]=m
            line=f"p={p}: "+" + ".join(f"{m[e]}*N^{e}" for e in sorted(m,reverse=True))
            print(line,flush=True); fo.write(line+"\n"); fo.flush()
    json.dump({p:{str(e):v for e,v in m.items()} for p,m in res.items()},open(out+'.json','w'))
