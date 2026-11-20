import numpy as np, json, sys
from check import moments_grid
def fourier(ns):
    H=np.array([[1.+0j]])
    for n in ns:
        F=np.exp(2j*np.pi*np.outer(range(n),range(n))/n); H=np.kron(H,F)
    return H
def poly(f):
    P={}
    for l in open(f):
        p,rest=l.split(':'); p=int(p[2:])
        P[p]={int(t.split('^')[1]):int(t.split('*')[0]) for t in rest.replace(' + ',' ').split()}
    return P
cases=[("F_3",[3],7,'q3.out'),("Z_3^2",[3,3],4,'q3.out'),("F_5",[5],6,'q1000003.out'),("F_7",[7],5,'q1000003.out'),("F_5 vs q=5",[5],6,'q5.out')]
for name,ns,pm,f in cases:
    H=fourier(ns); N=H.shape[0]; P=poly(f)
    m=moments_grid(H,pm,chunk=50000)
    comparisons = [(p, round(m[p]), sum(c*N**e for e, c in P[p].items()))
                   for p in range(1, pm+1)]
    print(name, comparisons, flush=True)
    assert all(observed == expected for p, observed, expected in comparisons), name
