"""The leading free energy and the energy curve (Section 5).

Phi_0(x) = sqrt(1+4x) - 1 - log((1+sqrt(1+4x))/2). The program checks that its
Taylor coefficients are (-1)^{k-1} Cat_{k-1}/k, by solving L' = 1 - x L'^2
exactly, and that the energy curve
    m(beta) = d/d beta Phi_0(beta^2/4) = beta/(1+sqrt(1+beta^2))
equals tanh(arcsinh(beta)/2), with m = 1 - 1/beta + O(1/beta^2). It prints the
points of the curve used in the figure.
"""
from fractions import Fraction
from math import comb, sqrt, log, tanh, asinh

def say(*a):
    print(*a, flush=True)

def phi0(x):
    r = sqrt(1 + 4*x); return r - 1 - log((1 + r)/2)

def m(beta):
    return beta/(1 + sqrt(1 + beta*beta))

def main():
    K = 30
    d = [Fraction(0)]*K           # coefficients of L'(x)
    d[0] = Fraction(1)
    for j in range(1, K):
        d[j] = -sum(d[i]*d[j-1-i] for i in range(j))
    L = [Fraction(0)] + [d[k-1]/k for k in range(1, K+1)]
    for k in range(1, K+1):
        assert L[k] == Fraction((-1)**(k-1)*comb(2*k-2, k-1), k*k)
    say('Taylor coefficients of Phi_0:', ', '.join(str(L[k]) for k in range(1, 9)), '...')
    say(f'Catalan formula checked through order {K}.')
    for x in (0.01, 0.1):
        series = sum(float(L[k])*x**k for k in range(1, K+1))
        assert abs(series - phi0(x)) < 1e-12
    worst = 0
    for i in range(1, 400):
        b = i/20; h = 1e-5
        num = (phi0((b+h)**2/4) - phi0((b-h)**2/4))/(2*h)
        worst = max(worst, abs(num - m(b)), abs(m(b) - tanh(asinh(b)/2)))
    assert worst < 1e-8
    say('m(beta) = derivative of Phi_0(beta^2/4) = tanh(arcsinh(beta)/2): checked')
    for b in (10, 100, 1000):
        say(f'  beta={b}: beta*(1-m) = {b*(1-m(b)):.6f}')
    assert abs(1000*(1 - m(1000)) - 1) < 2e-3
    say('Points of the curve (beta, m):')
    say(' '.join(f'({k/4:g},{m(k/4):.4f})' for k in range(0, 41)))

if __name__ == '__main__':
    main()
