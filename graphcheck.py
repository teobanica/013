from collections import defaultdict
from fractions import Fraction as Q
from itertools import combinations
from math import factorial, comb, prod
from functools import lru_cache

def bparts(pos,neg):
 if not pos:
  yield ();return
 for m in range(1,len(pos)+1):
  for p in combinations(pos[1:],m-1):
   plus=(pos[0],)+p
   for minus in combinations(neg,m):
    for rest in bparts(tuple(x for x in pos if x not in plus),tuple(x for x in neg if x not in minus)):
     yield (plus+minus,)+rest

def rank(entries,q):
 a=[[x%q for x in row] for row in entries]; r=0
 for j in range(len(a[0])):
  piv=next((i for i in range(r,len(a)) if a[i][j]),None)
  if piv is None:continue
  a[r],a[piv]=a[piv],a[r]; inv=pow(a[r][j],-1,q)
  for i in range(r+1,len(a)):
   c=a[i][j]*inv
   if c:a[i]=[(u-c*v)%q for u,v in zip(a[i],a[r])]
  r+=1
 return r
@lru_cache(None)
def ranks(m):return [rank(m,q) for q in (2,3,5)]
@lru_cache(None)
def cumul(m):return factorial(m)**2*co[m]
co=[Q(0)]
for m in range(1,11):
 co.append(Q(1,factorial(m)**2)-sum((s*co[s]/factorial(m-s)**2 for s in range(1,m)),Q(0))/m)
for k in range(1,5):
 P=[]
 for blocks in bparts(tuple(range(k)),tuple(range(k,2*k))):
  P.append(([next(i for i,S in enumerate(blocks) if h in S) for h in range(2*k)],len(blocks),prod(cumul(len(S)//2) for S in blocks)))
 polys=[defaultdict(Q) for _ in range(3)]
 for A,r,wa in P:
  for B,c,wb in P:
   components=list(range(r+c))
   def root(i):
    while components[i]!=i:i=components[i]
    return i
   m=[[0]*c for _ in range(r)]
   for h,(i,j) in enumerate(zip(A,B)):
    components[root(i)]=root(r+j);m[i][j]+=1 if h<k else -1
   if len({root(i) for i in range(r+c)})!=1:continue
   for d,p in zip(ranks(tuple(map(tuple,m))),polys):p[r+c-d]+=wa*wb/factorial(k)**2
 for p in polys:
  assert all(deg>=2 and deg<=k+1 for deg in p)
  assert p[k+1]==Q((-1)**(k-1)*comb(2*k-2,k-1),k*k)
  assert p[2]==(factorial(k)*co[k])**2
 assert polys[1][k]==polys[2][k]
 if k==4:
  delta={i:polys[0][i]-polys[1][i] for i in range(2,k+2)}
  assert {k:v for k,v in delta.items() if v}=={3:Q(-1,32),4:Q(1,32)}
 print('k',k,'leading',polys[1][k+1],'N^2',polys[1][2],'next odd and Walsh',polys[1][k],polys[0][k])
