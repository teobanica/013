"""Section 3: the generic polynomials evaluated at N=2 are not moments.

Evaluates P^infty_p(2) from gen.out, and the 3 by 3 Hankel determinant of the
candidate moments 1, P_1, ..., P_4 of |Omega|^2. For comparison the same is
done with the Walsh polynomials (walsh.out), which at N=2 give the true glow
moments of F_2, and with the glow moments of F_2 computed directly.
"""
from fractions import Fraction
from math import comb

def load(path):
    P = {}
    for line in open(path):
        p, rest = line.split(':', 1)
        P[int(p[2:])] = {int(t.split('^')[1]): int(t.split('*')[0]) for t in rest.replace(' + ', ' ').split()}
    return P

def det(M):
    M = [[Fraction(x) for x in row] for row in M]; n = len(M); d = Fraction(1)
    for c in range(n):
        p = next((i for i in range(c, n) if M[i][c]), None)
        if p is None: return Fraction(0)
        if p != c: M[c], M[p] = M[p], M[c]; d = -d
        d *= M[c][c]
        for i in range(c+1, n):
            f = M[i][c]/M[c][c]; M[i] = [u - f*v for u, v in zip(M[i], M[c])]
    return d

def hankel(m):
    return [det([[m[i+j] for j in range(k)] for i in range(k)]) for k in (1, 2, 3)]

def f2_moment(p):
    # Omega for F_2: |(Hb)_1|^2=2+2cos t, |(Hb)_2|^2=2-2cos t, and
    # E|z_1 a_1+z_2 a_2|^{2p} = sum_m binom(p,m)^2 |z_1|^{2m}|z_2|^{2(p-m)}.
    # Then E[(2+2cos t)^m (2-2cos t)^{p-m}] = 4^p binom(2m,m)binom(2p-2m,p-m)/(4^p binom(p,m)).
    return sum(comb(p, m)*comb(2*m, m)*comb(2*p - 2*m, p - m) for m in range(p+1))

def main():
    G, W = load('gen.out'), load('walsh.out')
    gen = [1] + [sum(c*2**e for e, c in G[p].items()) for p in range(1, 6)]
    wal = [1] + [sum(c*2**e for e, c in W[p].items()) for p in range(1, 6)]
    true = [f2_moment(p) for p in range(6)]
    print('p          :', list(range(6)), flush=True)
    print('P^infty_p(2):', gen, flush=True)
    print('P^(2)_p(2)  :', wal, flush=True)
    print('F_2 direct  :', true, flush=True)
    assert gen == [1, 4, 20, 112, 532, -29296]
    assert wal == true
    hg, hw = hankel(gen), hankel(wal)
    print('Hankel minors, generic:', [int(x) for x in hg], flush=True)
    print('Hankel minors, Walsh  :', [int(x) for x in hw], flush=True)
    assert hg[2] == -496 and all(x >= 0 for x in hw)
    print('Hankel checks passed.', flush=True)

if __name__ == '__main__':
    main()
