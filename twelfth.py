"""Independent check of the 12th-moment formula (Section 4), without SymPy.

1. Exact invariants of the Tao matrix, over the integers of Q(w), w^3=1.
2. Exact invariants of H6(q) and of the order-4 family H(a), as Laurent
   polynomials with Gaussian integer coefficients.
3. The 12th moments of all saved matrices of order at most 8 (basisglow.json,
   exact phase-grid quadrature) are compared with the formula.
4. The coefficients are then recovered from the data alone: at N=6 and N=8 by
   least squares on S2,S3,R,T, and the polynomial A(N) by interpolation at
   N=2,4,6,7,8.
"""
import json
from fractions import Fraction
from itertools import product
import numpy as np
import matrices as m
import cases

def say(*a):
    print(*a, flush=True)

# ---------- universal polynomial and the claimed coefficients ----------
def load_poly(path, p):
    for line in open(path):
        if line.startswith(f'p={p}:'):
            terms = line.split(':', 1)[1].replace(' + ', ' ').split()
            return {int(t.split('^')[1]): int(t.split('*')[0]) for t in terms}
P6 = load_poly('gen.out', 6)
def p6(N):
    return sum(c * N**e for e, c in P6.items())
def A(N):
    return 8100*N**4 - 137700*N**3 + 1062450*N**2 - 4436100*N + 7663500

# ---------- floating invariants, written with explicit loops ----------
def invariants(H):
    n = len(H)
    H2, H3 = H**2, H**3
    B = np.array([[np.vdot(H2[b], H2[a]) for b in range(n)] for a in range(n)]) - n*np.eye(n)
    C = np.array([[np.vdot(H3[b], H3[a]) for b in range(n)] for a in range(n)]) - n*np.eye(n)
    s2 = float(np.sum(np.abs(B)**2)); s3 = float(np.sum(np.abs(C)**2))
    r = float(np.trace(B @ B @ B).real)
    Q = np.zeros((n, n, n), complex)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                Q[a, b, c] = np.sum(H2[a] * np.conj(H[b] * H[c]))
    t = 0.0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if a != b and b != c and a != c:
                    t += (Q[a, b, c] * Q[b, c, a] * Q[c, a, b]).real
    return s2, s3, r, t

# ---------- exact arithmetic in Z[w], w^2 = -1-w ----------
class Zw:
    def __init__(s, a, b=0): s.a, s.b = a, b
    def __add__(s, o): return Zw(s.a+o.a, s.b+o.b)
    def __sub__(s, o): return Zw(s.a-o.a, s.b-o.b)
    def __mul__(s, o): return Zw(s.a*o.a - s.b*o.b, s.a*o.b + s.b*o.a - s.b*o.b)
    def conj(s): return Zw(s.a - s.b, -s.b)
    def re2(s): return 2*s.a - s.b          # twice the real part
    def norm(s): return s.a*s.a - s.a*s.b + s.b*s.b
ONEw, Wroot = Zw(1), Zw(0, 1)
def wpow(k):
    out = ONEw
    for _ in range(k % 3): out = out * Wroot
    return out

def exact_tao():
    T = m.T6()
    E = np.rint(np.angle(T) / (2*np.pi/3)).astype(int) % 3
    assert np.allclose(np.exp(2j*np.pi*E/3), T)
    n = 6
    H = [[wpow(E[a][j]) for j in range(n)] for a in range(n)]
    def gram(k):
        G = [[Zw(0) for _ in range(n)] for _ in range(n)]
        for a, b in product(range(n), repeat=2):
            s = Zw(0)
            for j in range(n): s = s + wpow(k*E[a][j]) * wpow(k*E[b][j]).conj()
            G[a][b] = s - (Zw(n) if a == b else Zw(0))
        return G
    G1 = gram(1)
    assert all(G1[a][b].a == 0 and G1[a][b].b == 0 for a in range(n) for b in range(n))
    B, C = gram(2), gram(3)
    s2 = sum(B[a][b].norm() for a in range(n) for b in range(n))
    s3 = sum(C[a][b].norm() for a in range(n) for b in range(n))
    r2 = Zw(0)
    for a, b, c in product(range(n), repeat=3): r2 = r2 + B[a][b]*B[b][c]*B[c][a]
    Q = {}
    for a, b, c in product(range(n), repeat=3):
        s = Zw(0)
        for j in range(n): s = s + H[a][j]*H[a][j]*(H[b][j]*H[c][j]).conj()
        Q[a, b, c] = s
    t2 = 0
    for a, b, c in product(range(n), repeat=3):
        if len({a, b, c}) == 3: t2 += (Q[a, b, c]*Q[b, c, a]*Q[c, a, b]).re2()
    assert r2.re2() % 2 == 0 and t2 % 2 == 0
    return s2, s3, r2.re2()//2, t2//2

# ---------- exact Laurent polynomials in q with Gaussian integer coefficients ----------
class L:
    """Dict exponent -> (re, im), with q on the unit circle, so conj(q) = 1/q."""
    def __init__(s, d=None): s.d = {k: v for k, v in (d or {}).items() if v != (0, 0)}
    @staticmethod
    def c(re, im=0, k=0): return L({k: (re, im)})
    def __add__(s, o):
        d = dict(s.d)
        for k, (x, y) in o.d.items():
            u, v = d.get(k, (0, 0)); d[k] = (u+x, v+y)
        return L(d)
    def __neg__(s): return L({k: (-x, -y) for k, (x, y) in s.d.items()})
    def __sub__(s, o): return s + (-o)
    def __mul__(s, o):
        d = {}
        for k1, (a, b) in s.d.items():
            for k2, (c, e) in o.d.items():
                u, v = d.get(k1+k2, (0, 0)); d[k1+k2] = (u + a*c - b*e, v + a*e + b*c)
        return L(d)
    def conj(s): return L({-k: (x, -y) for k, (x, y) in s.d.items()})
    def __eq__(s, o): return s.d == o.d
    def __repr__(s): return ' + '.join(f'({x}{y:+d}i)q^{k}' for k, (x, y) in sorted(s.d.items())) or '0'
ZERO, ONE = L(), L.c(1)

def lmat_invariants(H):
    n = len(H)
    def gram(k):
        Hk = [[ONE for _ in range(n)] for _ in range(n)]
        for a, j in product(range(n), repeat=2):
            for _ in range(k): Hk[a][j] = Hk[a][j] * H[a][j]
        G = [[ZERO for _ in range(n)] for _ in range(n)]
        for a, b in product(range(n), repeat=2):
            s = ZERO
            for j in range(n): s = s + Hk[a][j] * Hk[b][j].conj()
            G[a][b] = s - (L.c(n) if a == b else ZERO)
        return G
    G1 = gram(1)
    assert all(G1[a][b] == ZERO for a in range(n) for b in range(n))
    B, C = gram(2), gram(3)
    s2, s3, r = ZERO, ZERO, ZERO
    for a, b in product(range(n), repeat=2):
        s2 = s2 + B[a][b]*B[a][b].conj(); s3 = s3 + C[a][b]*C[a][b].conj()
    for a, b, c in product(range(n), repeat=3): r = r + B[a][b]*B[b][c]*B[c][a]
    Q = {}
    for a, b, c in product(range(n), repeat=3):
        s = ZERO
        for j in range(n): s = s + H[a][j]*H[a][j]*(H[b][j]*H[c][j]).conj()
        Q[a, b, c] = s
    t = ZERO
    for a, b, c in product(range(n), repeat=3):
        if len({a, b, c}) == 3:
            x = Q[a, b, c]*Q[b, c, a]*Q[c, a, b]; t = t + x + x.conj()
    return s2, s3, r, t      # t is twice T

def exact_h6():
    i, q = L.c(0, 1), L.c(1, 0, 1)
    qb = q.conj()
    one = ONE
    H = [[one]*6,
         [one, -one, i, i, -i, -i],
         [one, i, -one, -i, q, -q],
         [one, i, -i, -one, -q, q],
         [one, -i, qb, -qb, i, -one],
         [one, -i, -qb, qb, -one, i]]
    return lmat_invariants(H)

def exact_ha():
    a = L.c(1, 0, 1)
    one = ONE
    H = [[one]*4, [one, a, -one, -a], [one, -one, one, -one], [one, -a, -one, a]]
    return lmat_invariants(H)[0]

def main():
    say('Tao matrix, exact over Z[w]:')
    tao = exact_tao()
    say('  (S2,S3,R,T) =', tao)
    assert tao == (0, 1080, 0, -3240)
    assert 51200*1080 + 21600*(-3240) == -14688000
    say('  12th moment minus P6(6) = 51200*1080 + 21600*(-3240) = -14688000')

    say('H6(q), exact Laurent polynomials:')
    s2, s3, r, t2 = exact_h6()
    say('  S2 =', s2, '  S3 =', s3); say('  R =', r); say('  2T =', t2)
    assert s2 == L.c(120) and s3 == ZERO and t2 == L.c(3840)
    assert r == L({-4: (192, 0), 0: (576, 0), 4: (192, 0)})
    a6 = A(6)
    assert a6*120 + 2700*576 + 21600*1920 == 48967200 and 2700*384 == 1036800
    say('  R = 576+384cos(4t), T = 1920, deviation = 48967200+1036800cos(4t): passed')

    say('H(a), order 4, exact:')
    sa = exact_ha()
    say('  S2 =', sa)
    assert sa == L({-2: (32, 0), 0: (128, 0), 2: (32, 0)})
    say('  S2 = 128+64cos(2t) = 64+128cos(t)^2: passed')

    say('Saved 12th moments, orders 2 to 8:')
    stored = json.load(open('basisglow.json'))
    rng2 = np.random.default_rng(7); E = lambda x: np.exp(2j*np.pi*x)
    extra = {"F_6": lambda: m.fourier(6), "F_7": lambda: m.fourier(7), "F_8": lambda: m.fourier(8),
             "W_8": lambda: m.kron(m.fourier(2), m.fourier(2), m.fourier(2)),
             "F_9": lambda: m.fourier(9), "F_10": lambda: m.fourier(10), "F_11": lambda: m.fourier(11),
             "F_2xQF_3, random": lambda: m.F6rs(*E(rng2.random(2))),
             "F_{8,5a}, random": lambda: m.F85a(*E(rng2.random(5))),
             "F_{8,5b}, random": lambda: m.F85b(*E(rng2.random(5))),
             "F_3xQF_3, random": lambda: m.dita(m.fourier(3), m.fourier(3), m.dephased_Q(3, 3, E(rng2.random(4))))}
    built = {name: np.array(make(), complex) for name, make in extra.items()}   # same order as basisglow.py
    built.update({name: np.array(make(), complex) for name, make in cases.C.items()})
    from check import walsh, moments_grid
    data = []
    for name, H in built.items():
        if name not in stored or len(stored[name]['moments']) < 6: continue
        assert np.allclose(np.abs(np.array(stored[name]['moments'][:5]) - moments_grid(H, 5)[1:]), 0, atol=1e-6*np.abs(stored[name]['moments'][4]))
        data.append((name, len(H), invariants(H), stored[name]['moments'][5]))
    for name, H in [('F_2', m.fourier(2)), ('F_3', m.fourier(3)), ('F_4', m.fourier(4)), ('W_4', walsh(2)),
                    ('F_5', m.fourier(5)), ('H(a), a=exp(0.7i)', m.F4s(np.exp(0.7j)))]:
        H = np.array(H, complex)
        data.append((name, len(H), invariants(H), moments_grid(H, 6)[6]))
    worst = 0
    for name, n, (s2, s3, r, t), mom in data:
        pred = p6(n) + A(n)*s2 + 51200*s3 + 2700*r + 21600*t
        rel = abs(pred - mom)/mom; worst = max(worst, rel)
        say(f'  {name:26s} N={n}  S2={s2:9.2f} S3={s3:8.2f} R={r:10.2f} T={t:9.2f}  rel.err={rel:.1e}')
        assert rel < 1e-10, name
    say(f'  {len(data)} matrices passed, worst relative error {worst:.1e}')

    say('Coefficients recovered from the data alone:')
    for n in (6, 8):
        rows = [(inv, mom - p6(n)) for _, k, inv, mom in data if k == n]
        X = np.array([inv for inv, _ in rows]); y = np.array([d for _, d in rows])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        say(f'  N={n}, {len(rows)} matrices: S2,S3,R,T coefficients',
            ' '.join(f'{c:.4f}' for c in coef), f'(A({n})={A(n)})')
        assert np.allclose(coef, [A(n), 51200, 2700, 21600], rtol=1e-6)
    pts = {}
    for _, n, (s2, s3, r, t), mom in data:
        if s2 > 1:
            pts.setdefault(n, (mom - p6(n) - 51200*s3 - 2700*r - 21600*t)/s2)
    xs = sorted(pts)
    assert xs == [2, 4, 6, 7, 8]
    V = np.array([[Fraction(x)**k for k in range(5)] for x in xs], dtype=object)
    vals = [Fraction(round(pts[x])) for x in xs]
    assert all(abs(pts[x] - round(pts[x])) < 1e-3 for x in xs)
    # Solve the 5 by 5 Vandermonde system exactly.
    M = [list(V[i]) + [vals[i]] for i in range(5)]
    for c in range(5):
        p = next(i for i in range(c, 5) if M[i][c] != 0); M[c], M[p] = M[p], M[c]
        for i in range(5):
            if i != c:
                f = M[i][c]/M[c][c]; M[i] = [u - f*v for u, v in zip(M[i], M[c])]
    poly = [M[k][5]/M[k][k] for k in range(5)]
    say('  interpolated S2 coefficient:', ' + '.join(f'{c}*N^{k}' for k, c in enumerate(poly)))
    assert poly == [7663500, -4436100, 1062450, -137700, 8100]
    say('All 12th-moment checks passed.')

if __name__ == '__main__':
    main()
