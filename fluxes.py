"""Hadamard matrices as flux patterns on the crossed-wire array (Sections 1 and 4).

Row wire j and column wire k cross at a junction with phase factor H_jk. The
cell between row wires j,j+1 and column wires k,k+1 carries the flux
    Phi_jk = arg(H_jk conj(H_{j+1,k}) H_{j+1,k+1} conj(H_{j,k+1}))/(2 pi)
in units of the flux quantum, taken in [0,1). The program checks:
1. the (N-1)^2 cell fluxes determine H up to row and column phases;
2. orthogonality of rows a,b says that the rectangles with horizontal sides on
   a,b and j<l have first cosines summing to -N/2;
3. S_k(H) is the sum over these rectangles, with ordered a != b and all j,l, of cos(2 pi k Phi);
4. row and column phases leave all the fluxes unchanged;
and prints the flux patterns of F_8, W_8 and T_6 used for the figure.
"""
import numpy as np
import matrices as m
from check import walsh

def say(*a):
    print(*a, flush=True)

def cell_fluxes(H):
    c = H[:-1, :-1] * np.conj(H[1:, :-1]) * H[1:, 1:] * np.conj(H[:-1, 1:])
    return np.mod(np.round(np.angle(c)/(2*np.pi), 10), 1.0)

def rebuild(Phi):
    # Dephased matrix with first row and column 1, from the cell fluxes.
    n = Phi.shape[0] + 1
    G = np.ones((n, n), complex)
    for j in range(n-1):
        for k in range(n-1):
            # H_{j+1,k+1} = conj(H_jk) H_{j+1,k} H_{j,k+1} exp(2 pi i Phi_jk)
            G[j+1, k+1] = np.conj(G[j, k]) * G[j+1, k] * G[j, k+1] * np.exp(2j*np.pi*Phi[j, k])
    return G

def dephase(H):
    H = H / H[:, :1]
    return H / H[:1, :]

def rectangle_sums(H, k):
    # Sum over a != b and all j, l of exp(2 pi i k Phi(a,b;j,l)), where the
    # rectangle holonomy is H_aj conj(H_bj) H_bl conj(H_al).
    n = len(H); total = 0
    for a in range(n):
        for b in range(n):
            if a == b: continue
            r = (H[a] * np.conj(H[b]))**k
            total += np.sum(np.outer(r, np.conj(r)))
    return total

def s_k(H, k):
    Hk = H**k; G = Hk @ Hk.conj().T - len(H)*np.eye(len(H))
    return np.sum(np.abs(G)**2)

def main():
    rng = np.random.default_rng(11)
    tests = [('F_8', m.fourier(8)), ('W_8', walsh(3).astype(complex)), ('T_6', m.T6()),
             ('H_6^q', m.H6(np.exp(0.9j))), ('K_6', m.K6(*(np.pi*rng.random(3))))]
    for name, H in tests:
        H = np.array(H, complex); n = len(H)
        Phi = cell_fluxes(H)
        assert np.allclose(rebuild(Phi), dephase(H))
        phases = (np.exp(2j*np.pi*rng.random(n))[:, None]) * H * np.exp(2j*np.pi*rng.random(n))[None, :]
        assert np.allclose(np.exp(2j*np.pi*cell_fluxes(phases)), np.exp(2j*np.pi*Phi))
        j, l = np.triu_indices(n, k=1)
        for k in (1, 2, 3):
            cosine_sum = 0.0
            for a in range(n):
                for b in range(a + 1, n):
                    ratio = (H[a] * H[b].conj()) ** k
                    row_pair_sum = np.sum((ratio[j] * ratio[l].conj()).real)
                    if k == 1:
                        assert abs(row_pair_sum + n / 2) < 1e-9
                    cosine_sum += row_pair_sum
            harmonic = s_k(H, k)
            assert abs(n*n*(n-1) + 4*cosine_sum - harmonic) < 1e-7
            assert abs(rectangle_sums(H, k) - harmonic) < 1e-7
        say(f'{name}: fluxes rebuild H, gauge invariant, distinct-rectangle sum = -N/2 per row pair, '
            f'S_2={s_k(H, 2).real:.2f}, S_3={s_k(H, 3).real:.2f} as rectangle sums')
    say('Cell fluxes, in units of the flux quantum:')
    for name, H in tests[:3]:
        Phi = cell_fluxes(np.array(H, complex))
        say(name)
        for row in Phi:
            say('  ' + ' '.join(f'{x:.4f}' for x in row))
    say('All flux checks passed.')

if __name__ == '__main__':
    main()
