"""Numerical checks of the edge asymptotics of W_2 and W_4 (Section 6).

1. W_2: an exact finite sum for M_p(W_2) is evaluated numerically, and p*M_p tends to 2/pi.
2. W_4: with b_{s,t}=exp(i(D+sA+tB+stC)) and u,v,w=cos2A,cos2B,cos2C, the squared
   moduli r_j^2 of Ub are 1+(uv,uw,vw).s_j, with s_j running over the 4 sign
   vectors of product 1. Checked on random inputs.
3. The integral of exp(-Q/4) over R^3 is sqrt(2)Gamma(1/4)^3, computed here by
   a different route: integrate x first, then use polar coordinates in y,z.
4. E_b Y^{2p} is integrated numerically in the variables u,v,w, with arcsine
   densities, and compared with sqrt(2)Gamma(1/4)^3 p^{-3/4}/pi^3.
5. The alignment factor of Lemma 6.3 at the edge, N=4: the exact moments
   E|a_1+a_2+a_3+a_4|^{2p}/16^p, times p^{3/2}, tend to 2/pi^{3/2}.
Together, 4 and 5 give the constant 2sqrt(2)Gamma(1/4)^3/pi^{9/2} of Theorem 6.4.
"""
import numpy as np
from math import gamma, pi, sqrt, lgamma, exp, log
from scipy.integrate import quad
from scipy.special import i0e

def say(*a):
    print(*a, flush=True)

def logbinom(n, k):
    return lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)

def logsum(xs):
    m = max(xs); return m + log(sum(exp(x - m) for x in xs))

def walsh2(p):
    # E|Omega|^{2p} = sum_m binom(p,m) binom(2m,m) binom(2p-2m,p-m), divided by N^{3p} = 8^p.
    return exp(logsum([logbinom(p, m) + logbinom(2*m, m) + logbinom(2*p-2*m, p-m) for m in range(p+1)]) - p*log(8))

def edge4(p):
    # E|a_1+a_2+a_3+a_4|^{2p} = sum_k binom(p,k)^2 binom(2k,k) binom(2p-2k,p-k).
    return exp(logsum([2*logbinom(p, k) + logbinom(2*k, k) + logbinom(2*p-2*k, p-k) for k in range(p+1)]) - p*log(16))

SIGNS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)

def structure(trials=1000):
    rng = np.random.default_rng(3)
    W4 = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], float)
    worst = 0
    for _ in range(trials):
        D, A, B, C = 2*pi*rng.random(4)
        b = np.array([np.exp(1j*(D + s*A + t*B + s*t*C)) for s in (1, -1) for t in (1, -1)])
        r2 = np.sort(np.abs(W4 @ b)**2 / 4)
        u, v, w = np.cos(2*A), np.cos(2*B), np.cos(2*C)
        worst = max(worst, np.max(np.abs(r2 - np.sort(1 + SIGNS @ np.array([u*v, u*w, v*w])))))
    return worst

def gaussian_integral():
    # int dx exp(-x^2(y^2+z^2)/4) = 2 sqrt(pi)/rho, then polar coordinates; 8 equal sectors in phi.
    inner = lambda phi: quad(lambda r: exp(-r**4*np.sin(2*phi)**2/16), 0, np.inf, limit=200)[0]
    return 8*2*sqrt(pi)*quad(inner, 0, pi/4, limit=200)[0]

def edge_average(p, n=300, k=4):
    # u=sin(alpha) with alpha uniform on (-pi/2,pi/2) has the arcsine law; 8 sign sectors.
    x, wt = np.polynomial.legendre.leggauss(n)
    s = (x + 1)/2; ws = wt/2
    al = (pi/2)*s**k; wa = (pi/2)*k*s**(k-1)*ws
    u = np.sin(al)
    U, V = np.meshgrid(u, u, indexing='ij'); WW = np.outer(wa, wa)
    total = 0.0
    for i in range(n):
        c = np.stack([U*V, U*u[i], V*u[i]])
        Y = np.sqrt(np.clip(1 + np.tensordot(SIGNS, c, 1), 0, None)).mean(0)
        total += wa[i]*np.sum(WW*np.exp(2*p*np.log(np.clip(Y, 1e-300, None))))
    return 8*total/pi**3

def main():
    say('1. W_2, numerical evaluation of exact moments: p*M_p versus 2/pi =', f'{2/pi:.8f}')
    for p in (10, 100, 1000, 10000, 100000):
        say(f'   p={p:>6}: {p*walsh2(p):.8f}')
    assert abs(100000*walsh2(100000)/(2/pi) - 1) < 1e-4

    err = structure()
    say('2. W_4 squared moduli versus 1+(uv,uw,vw).s_j, 1000 random inputs: max error', f'{err:.1e}')
    assert err < 1e-12

    g, target = gaussian_integral(), sqrt(2)*gamma(0.25)**3
    say(f'3. Integral of exp(-Q/4): {g:.10f}, sqrt(2)Gamma(1/4)^3 = {target:.10f}')
    assert abs(g/target - 1) < 1e-8

    K = target/pi**3
    say('4. E_b Y^{2p} * p^{3/4} / (sqrt(2)Gamma(1/4)^3/pi^3):')
    ratios = []
    for p in (1e2, 1e4, 1e6, 1e8):
        a, b = edge_average(p), edge_average(p, 500, 6)
        assert abs(a/b - 1) < 1e-6
        ratios.append(a*p**0.75/K)
        say(f'   p=1e{round(log(p, 10))}: {ratios[-1]:.6f}')
    assert all(abs(1 - r2) < abs(1 - r1) for r1, r2 in zip(ratios, ratios[1:])) and abs(ratios[-1] - 1) < 1e-4

    say('5. Alignment at the edge, N=4: p^{3/2} E|a_1+...+a_4|^{2p}/16^p versus 2/pi^{3/2} =', f'{2/pi**1.5:.8f}')
    for p in (100, 1000, 10000, 100000):
        say(f'   p={p:>6}: {p**1.5*edge4(p):.8f}')
    assert abs(100000**1.5*edge4(100000)/(2/pi**1.5) - 1) < 1e-4
    say('   Product of the 2 constants:', f'{K*2/pi**1.5:.10f}',
        '= 2sqrt(2)Gamma(1/4)^3/pi^{9/2} =', f'{2*sqrt(2)*gamma(0.25)**3/pi**4.5:.10f}')
    # Integrate out the two row phases, then integrate the relative column angle.
    # The scaled I_0 and the variable x=sqrt(beta)*(pi/4-u) keep the peak resolved.
    expected = 4 / (pi * sqrt(2*pi))
    say('6. W_2, exp(-2 beta)*(2 beta)^(3/2)*Z(beta) versus', f'{expected:.8f}')
    for beta in (10, 100, 1000, 10000):
        def integrand(x):
            v = x / sqrt(beta)
            u = pi/4 - v
            r0 = sqrt(2) * np.cos(u)
            r1 = sqrt(2) * np.sin(u)
            return exp(-4*beta*np.sin(v/2)**2) * i0e(beta*r0) * i0e(beta*r1)
        integral = quad(integrand, 0, min(14.0, pi*sqrt(beta)/4),
                        epsabs=1e-14, epsrel=1e-11)[0]
        scaled = (2*beta)**1.5 * 4*integral / (pi*sqrt(beta))
        say(f'   beta={beta:5d}: {scaled:.8f}')
    assert abs(scaled / expected - 1) < 1e-4
    say('All edge checks passed.')

if __name__ == '__main__':
    main()
