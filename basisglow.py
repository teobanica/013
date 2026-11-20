# Glow moments of the basis matrices, compared with the generic Fourier polynomials P^infty_p(N).
import sys, json, time
sys.path.insert(0,'.')
import numpy as np
from math import factorial
import matrices as m
from check import moments_grid
exec(open('cases.py').read().split('C = {}')[0])   # rng, R, tq as in the paper
import cases as cs
def poly(f):
    P={}
    for l in open(f):
        p,rest=l.split(':'); p=int(p[2:])
        P[p]={int(t.split('^')[1]):int(t.split('*')[0]) for t in rest.replace(' + ',' ').split()}
    return P
G=poly('gen.out'); W=poly('walsh.out')
pmax={6:7,7:6,8:6,9:5,10:4,11:4,12:3,16:2}
rng2=np.random.default_rng(7); E=lambda x: np.exp(2j*np.pi*x)
extra={
 "F_6":lambda: m.fourier(6), "F_7":lambda: m.fourier(7), "F_8":lambda: m.fourier(8), "W_8":lambda: m.kron(m.fourier(2),m.fourier(2),m.fourier(2)),
 "F_9":lambda: m.fourier(9), "F_10":lambda: m.fourier(10), "F_11":lambda: m.fourier(11),
 "F_2xQF_3, random":lambda: m.F6rs(*E(rng2.random(2))),
 "F_{8,5a}, random":lambda: m.F85a(*E(rng2.random(5))), "F_{8,5b}, random":lambda: m.F85b(*E(rng2.random(5))),
 "F_3xQF_3, random":lambda: m.dita(m.fourier(3),m.fourier(3),m.dephased_Q(3,3,E(rng2.random(4)))),
}
allc=dict(extra); allc.update(cs.C)
out={}; log=open('basisglow.log','w')
for name,mk in allc.items():
    H=np.array(mk(),complex); N=H.shape[0]
    assert np.allclose(H@H.conj().T,N*np.eye(N),atol=1e-8), name
    pm=pmax[N]; t0=time.time()
    mom=moments_grid(H,pm,chunk=100000)
    dev=[(mom[p]-sum(c*N**e for e,c in G[p].items()))/(factorial(p)*N**(2*p)) for p in range(1,pm+1)]
    out[name]={"N":N,"moments":mom[1:],"dev":dev}
    s=f"{name:28s} N={N:2d} " + " ".join(f"{d:+.3e}" for d in dev) + f"   ({time.time()-t0:.0f}s)"
    print(s,flush=True); log.write(s+"\n"); log.flush()
json.dump(out,open('basisglow.json','w'),indent=1)
