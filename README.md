# Planar rotators with Hadamard couplings

**Chat, Opus and Teo**, November 2026

A complex Hadamard matrix specifies the magnetic fluxes in an array of crossed superconducting wires coupled by Josephson junctions. With independent uniform phases on the wires, the energy distribution is a real projection of the *glow*, the law of the complex sum $\langle Hb,a\rangle$. With Villain bonds along neighboring wires and the junction coupling switched off, the same sum measures how the phases align.

- For the Fourier matrices of elementary abelian groups, every even glow moment is a polynomial in the matrix order. For arbitrary complex Hadamard matrices we compute the moments through order 12. Matrices equivalent to real Hadamard matrices are exactly those that attain the universal upper bounds at orders 8 and 10.
- At each fixed order of the high-temperature pressure expansion, the leading coefficient is universal and has a Catalan formula. At the ground-state edge, the Walsh matrices of orders 2 and 4 have different moment asymptotics. In order 4 the leading term comes from the crossings of three families of ground states.
- For the Fourier matrix of order $N$ and two independent periodic Villain chains, the normalized fourth moment crosses over when the correlation length is of order $\sqrt{N}$. At correlation length $c\sqrt{N}$ its limit $M(c)$ decreases strictly from 2 to 1 as $c$ increases. Above that scale the normalized Fourier sum converges in law to the uniform distribution on the unit circle.

## Contents

- [013.pdf](013.pdf) is the paper (48 pages, 21 figures).
- [013.tex](013.tex) is the self-contained LaTeX source, including the figures. Compile twice with `pdflatex` to resolve the references.
- [README.txt](README.txt) lists the programs and their recorded outputs by section. All paths below are relative to the repository root.

| Programs | Computation | Paper |
|---|---|---|
| [fluxes.py](fluxes.py), [walshglow.py](walshglow.py), [fqglow.py](fqglow.py) | Cell fluxes and exact moment polynomials for Walsh and elementary abelian Fourier matrices | Sections 1–3 |
| [sixth.py](sixth.py), [invariants6.py](invariants6.py), [twelfth.py](twelfth.py) | Twelfth-moment coefficients and invariants for the Tao matrix and the order-six family $H_6^q$ | Section 4 |
| [freeenergy.py](freeenergy.py), [graphcheck.py](graphcheck.py) | Catalan coefficients of the pressure and connected-graph enumeration | Section 5 |
| [walshedge.py](walshedge.py), [order16.py](order16.py) | Walsh edge asymptotics and counts of sign and fourth-root maximizers in order 16 | Section 6 |
| [crossover.py](crossover.py), [villaincheck.py](villaincheck.py), [hypercube.py](hypercube.py) | Fourier crossover curve, finite-size Villain checks and the hypercube coefficients of Theorem 10.3 | Sections 9–10 |

## Running the code

Use Python 3.10 or later with NumPy and SciPy. The full set of shorter checks also uses SymPy. From the repository root, reproduce the fourth-moment table and curve with

```sh
python3 crossover.py
```

The program prints 65 values. At $c=1$ it gives $M(1)=1.41935$ to the five decimal places used in the table. The full output is in [crossover.csv](crossover.csv).

For an independent exact enumeration, run

```sh
python3 order16.py
```

The five real Hadamard classes of order 16 give 200704, 36864, 4096, 0 and 0 dephased fourth-root maximizers, respectively. The sign counts are given alongside them in [order16.log](order16.log).

To rerun all 19 shorter checks and refresh their output files, use

```sh
sh runall.sh
```

The commands for the longer polynomial enumerations and their saved results are in [README.txt](README.txt).

