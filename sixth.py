"""Exact coefficient calculations for mixed row moments at order six."""
from fractions import Fraction as F
from math import factorial
from itertools import product
import sympy as sp

def cumulants(maximum):
    f=[F(1,factorial(k)**2) for k in range(maximum+1)]
    kappa=[F(0)]*(maximum+1)
    for k in range(1,maximum+1):
        kappa[k]=(k*f[k]-sum(j*kappa[j]*f[k-j] for j in range(1,k)))/k
    return kappa

K=cumulants(6)
def coefficient(caps,charge):
    """Coefficient in exp(log M), retaining Q and its conjugate once."""
    rows=len(caps);limit=sum(caps)
    bound=tuple(caps)*2+(limit,1,1)
    logarithm={}
    indices=list(product(*(range(c+1) for c in caps)))
    for r in indices:
        m=sum(r)
        if m==0:continue
        for s in indices:
            if sum(s)!=m:continue
            diff=tuple(a-b for a,b in zip(r,s))
            if not any(diff):tail=(1,0,0)
            elif diff==charge:tail=(0,1,0)
            elif diff==tuple(-x for x in charge):tail=(0,0,1)
            else:continue
            multi_r=factorial(m)
            multi_s=factorial(m)
            for v in r:multi_r//=factorial(v)
            for v in s:multi_s//=factorial(v)
            logarithm[r+s+tail]=K[m]*multi_r*multi_s
    power={(0,)*len(bound):F(1)}
    answer={}
    for k in range(1,limit+1):
        output={}
        for a,c in power.items():
            for b,d in logarithm.items():
                index=tuple(x+y for x,y in zip(a,b))
                if all(x<=maximum for x,maximum in zip(index,bound)):
                    output[index]=output.get(index,F(0))+c*d/k
        power={a:c for a,c in output.items() if c}
        for index,c in power.items():
            if index[:2*rows]==tuple(caps)*2:
                tail=index[2*rows:]
                answer[tail]=answer.get(tail,F(0))+c
    N,Q,Qb=sp.symbols('N Q Qb')
    scale=1
    for v in caps:scale*=factorial(v)**2
    return sp.expand(sum(sp.Rational(v.numerator,v.denominator)*N**i*Q**j*Qb**k for (i,j,k),v in answer.items())*scale)


from pathlib import Path
import json
import numpy as np
import matrices as matrices
from check import moments_grid, walsh

ROOT = Path(__file__).resolve().parent
N, Q, Qb = sp.symbols('N Q Qb')
A = 8100*N**4-137700*N**3+1062450*N**2-4436100*N+7663500
P6 = sp.sympify(next(line.split(':',1)[1] for line in
    (ROOT/'gen.out').read_text().splitlines() if line.startswith('p=6:')).replace('^','**'), locals={'N':N})

def check_coefficients():
    tests = [
        ((4,2),(2,-2),72*N**2-420*N+652),
        ((3,3),(2,-2),81*N**2-513*N+882),
        ((3,3),(3,-3),sp.Integer(16)),
        ((2,2,2),(2,-2,0),2*N**2-33*N+130),
        ((2,2,2),(2,-1,-1),16*N**2-144*N+392)]
    for caps,charge,expected in tests:
        actual = coefficient(caps,charge).coeff(Q).coeff(Qb)
        assert sp.expand(actual-expected)==0, (caps,charge,actual)
        print('Exact coefficient:',caps,charge,actual,flush=True)
    mixed = 3*tests[0][2]+3*(N-2)*tests[3][2]-3*tests[4][2]
    assert sp.expand(mixed-(6*N**3+57*N**2-240*N))==0
    a = factorial(6)**2*(N**4/sp.Integer(64)-N**2*(9*N-24)/36
        +11*tests[0][2]/768+tests[1][2]/162-mixed/384)
    assert sp.expand(a-A)==0
    assert 2*2**6*K[2]**3 == -2
    assert 2*2**6*(2*K[2])**3 == -16
    print('Exact assembly and triangle coefficients: passed',flush=True)

def check_symbolic_h6():
    q=sp.symbols('q',nonzero=True); i=sp.I
    H=sp.Matrix([[1,1,1,1,1,1],[1,-1,i,i,-i,-i],
      [1,i,-1,-i,q,-q],[1,i,-i,-1,-q,q],
      [1,-i,1/q,-1/q,i,-1],[1,-i,-1/q,1/q,-1,i]])
    def conj(z): return sp.conjugate(z).subs(sp.conjugate(q),1/q)
    assert (H*H.applyfunc(conj).T-6*sp.eye(6)).applyfunc(sp.expand)==sp.zeros(6)
    B=H.applyfunc(lambda z:z**2)
    B=(B*B.applyfunc(conj).T-6*sp.eye(6)).applyfunc(sp.expand)
    C=H.applyfunc(lambda z:z**3)
    C=(C*C.applyfunc(conj).T-6*sp.eye(6)).applyfunc(sp.expand)
    def split(a,b,c):
        return sp.expand(sum(H[a,j]**2*conj(H[b,j]*H[c,j]) for j in range(6)))
    t=sp.expand(sum(split(a,b,c)*split(b,c,a)*split(c,a,b)
                   for a,b,c in product(range(6),repeat=3) if len({a,b,c})==3))
    r=sp.expand(sp.trace(B**3))
    assert sp.expand(sp.trace(B**2))==120
    assert sp.expand(sp.trace(C**2))==0
    assert t==1920
    assert sp.expand(r-576-192*(q**4+q**-4))==0
    delta=sp.expand(A.subs(N,6)*120+2700*r+21600*t)
    assert sp.expand(delta-48967200-518400*(q**4+q**-4))==0
    print('Exact H6(q): S2=120, S3=0, T=1920, R=',r,flush=True)
    print('Exact H6(q) moment deviation:',delta,flush=True)

def invariants(H):
    n=len(H); J=np.eye(n)
    assert np.max(np.abs(H@H.conj().T-n*J))<1e-7
    B=H**2@(H**2).conj().T-n*J
    C=H**3@(H**3).conj().T-n*J
    s2=np.vdot(B,B).real; s3=np.vdot(C,C).real
    r=np.trace(B@B@B).real
    q=np.einsum('aj,bj,cj->abc',H**2,H.conj(),H.conj())
    t=sum(q[a,b,c]*q[b,c,a]*q[c,a,b]
          for a,b,c in product(range(n),repeat=3) if len({a,b,c})==3).real
    return np.array([s2,s3,r,t])

def predicted(H):
    n=len(H); s2,s3,r,t=invariants(H)
    return int(P6.subs(N,n))+float(A.subs(N,n))*s2+51200*s3+2700*r+21600*t

def check_stored():
    import cases
    stored=json.loads((ROOT/'basisglow.json').read_text())
    count=0; worst=0
    for name,make in cases.C.items():
        if name not in stored or len(stored[name]['moments'])<6:continue
        H=make(); target=stored[name]['moments'][5]
        rel=abs(predicted(H)-target)/max(1,abs(target))
        assert rel<2e-9,(name,rel)
        worst=max(worst,rel);count+=1
        # Transposition and equivalence must preserve every displayed invariant.
        n=len(H); phases=np.exp(1j*np.arange(n)*0.37)
        assert np.allclose(invariants(H),invariants(H.T),rtol=1e-8,atol=1e-7)
        assert np.allclose(invariants(H),invariants(phases[:,None]*H[::-1,::-1]*phases[None,:]),rtol=1e-8,atol=1e-7)
    print('Stored p=6 moments:',count,'passed; worst relative error',worst,flush=True)
    assert count>=17
    assert np.allclose(invariants(matrices.T6()),[0,1080,0,-3240],atol=1e-7)
    print('Tao invariants and equivalence/transposition checks: passed',flush=True)

def check_grids():
    def fourier(n):
        return np.exp(2j*np.pi*np.outer(np.arange(n),np.arange(n))/n)
    tests=[('F2',fourier(2)),('F3',fourier(3)),('F4',fourier(4)),
        ('Walsh4',walsh(2)),('F4 deformation',matrices.F4s(np.exp(0.41j))),
        ('Tao6',matrices.T6()),('H6(1)',matrices.H6(1+0j)),
        ('H6(exp(0.37i))',matrices.H6(np.exp(0.37j)))]
    for name,H in tests:
        # Seven roots in each variable integrate degree six exactly.
        # This is deterministic quadrature, evaluated in floating point.
        actual=moments_grid(H,6)[6]; expected=predicted(H)
        error=abs(actual-expected)/max(1,abs(actual))
        assert error<2e-10,(name,actual,expected,error)
        print('Independent grid:',name,'moment=',format(actual,'.12g'),
              'relative error=',format(error,'.3g'),flush=True)

if __name__=='__main__':
    check_coefficients()
    check_symbolic_h6()
    check_stored()
    check_grids()
    print('All sixth-moment checks passed.',flush=True)
