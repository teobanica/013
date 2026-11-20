"""Ground states of the Fourier array and Chu sequences (Section 6).

For the cyclic Fourier matrix F_N, the input b attains the edge Y(b)=1 exactly
when all |(F_N b)_j| are equal to sqrt(N). The Chu sequences
    b_k = exp(-i pi k(k+1)/N)   (N odd),    b_k = exp(-i pi k^2/N)   (N even)
have this property: equivalently, their nonzero-shift periodic autocorrelations vanish (CAZAC).
The program checks this for N up to 64, computes |Omega| = N^{3/2} after aligning
the outputs, and prints the bar heights |(F_7 b)_j|/sqrt(7) for a random input
and for the Chu input, used in the figure.
"""
import numpy as np

def say(*a):
    print(*a, flush=True)

def chu(N):
    k = np.arange(N)
    return np.exp(-1j*np.pi*k*(k+1)/N) if N % 2 else np.exp(-1j*np.pi*k*k/N)

def main():
    for N in range(2, 65):
        F = np.exp(2j*np.pi*np.outer(np.arange(N), np.arange(N))/N)
        b = chu(N); out = F @ b
        assert np.allclose(np.abs(out), np.sqrt(N))
        auto = [np.vdot(np.roll(b, s), b) for s in range(1, N)]
        assert np.allclose(auto, 0)
        a = out/np.abs(out)                      # align the outputs
        assert abs(np.vdot(a, out) - N**1.5) < 1e-8*N**1.5
    say('Chu sequences: |F_N b| constant, zero off-peak autocorrelation, Re(Omega) = N^(3/2), for N = 2..64')
    N = 7
    F = np.exp(2j*np.pi*np.outer(np.arange(N), np.arange(N))/N)
    rng = np.random.default_rng(2026)
    rand = np.exp(2j*np.pi*rng.random(N))
    r1 = np.abs(F @ rand)/np.sqrt(N); r2 = np.abs(F @ chu(N))/np.sqrt(N)
    say('N=7, random input, |(Fb)_j|/sqrt(N):', ' '.join(f'{x:.3f}' for x in r1),
        f'  Y = {r1.mean():.4f}')
    say('N=7, Chu input,    |(Fb)_j|/sqrt(N):', ' '.join(f'{x:.3f}' for x in r2),
        f'  Y = {r2.mean():.4f}')
    say('All Chu checks passed.')

if __name__ == '__main__':
    main()
