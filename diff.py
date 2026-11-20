from fractions import Fraction as F
from math import factorial, comb
def poly(f):
    P={}
    for l in open(f):
        p,rest=l.split(':'); p=int(p[2:])
        P[p]={int(t.split('^')[1]):int(t.split('*')[0]) for t in rest.replace(' + ',' ').split()}
    return P
def divlin(c,r):  # divide poly (list low->high) by (N-r)
    c=c[:]; out=[0]*(len(c)-1)
    for i in range(len(c)-1,0,-1):
        out[i-1]=c[i]; c[i-1]+=r*c[i]
    return out, c[0]
G=poly('q1000003.out')
for name,f in [('Walsh','walsh.out'),('q=3','q3.out'),('q=5','q5.out')]:
    P=poly(f); print('==',name)
    for p in [p for p in P if p in G]:
        d={e:P[p].get(e,0)-G[p].get(e,0) for e in set(P[p])|set(G[p])}; d={e:v for e,v in d.items() if v}
        if not d: continue
        lo=min(d); c=[d.get(e,0) for e in range(lo,max(d)+1)]
        facs=[]
        for r in [1,2,4,8,3,9]:
            while len(c)>1:
                q_,rem=divlin(c,r)
                if rem==0: c=q_; facs.append(r)
                else: break
        print(f"p={p}: N^{lo}"+"".join(f"(N-{r})" for r in facs)+" * "+str(c))
