Planar rotators with Hadamard couplings
Code accompanying the paper

Python 3.10+ with NumPy and SciPy; sixth.py and invariants6.py also need SymPy.
Run the programs from inside this folder. The script runall.sh reruns the
faster ones and refreshes their outputs. Run it with sh runall.sh.

The longer enumerations can be regenerated separately:
  python3 walshglow.py 8 walsh.out
  python3 fqglow.py 1000003 8 gen.out
  python3 fqglow.py 3 7 q3.out
  python3 check.py 8
  python3 basisglow.py
For the other seven-order tables replace 3 in both arguments by 5, 7,
11, 13 or 1000003. Use 2 and order 6 for the separate q2.out check. The Walsh
program also writes walsh.out.json (included under the name walsh.json).
The bound on minors in Section 2 certifies the generic computation
at q = 1000003 through p = 8.

Section 1, planar rotators and Josephson arrays
  fluxes.py                       cell fluxes of F_8, W_8, T_6, gauge invariance,
                                  Hadamard condition and S_k as flux harmonics

Section 2, polynomiality
  walshglow.py -> walsh.out, walsh.json    polynomials P_p for Walsh matrices
  fqglow.py -> gen.out, q*.out             polynomials for q prime and q large
  check.py, check2.py, grid.json           quadrature checks at small N

Section 3, Walsh and generic moments
  diff.py, look.py                differences of polynomials, K_4 interpolation
  hankel.py                       P^infty_p(2), Hankel determinant -496

Section 4, Hadamard powers and plaquette fluxes
  matrices.py, cases.py           the matrices used in the paper
  basisglow.py, .json, .log       moments of 33 matrix examples of orders 6 to 12
  s2check.py, s2s3fit.py, s23.txt S_2, S_3 and the 8th and 10th moments
  sixth.py                        coefficient tables and H6(q), 12th moment
  invariants6.py                  exact invariants of the Tao matrix
  twelfth.py                      independent 12th-moment check, exact
                                  invariants, coefficients recovered from data

Section 5, free energy
  logexp.py                       coefficients of log G from the polynomials
  graphcheck.py                   connected graphs for k <= 4, q = 2, 3, 5
  freeenergy.py                   Catalan coefficients, energy curve m(beta)

Section 6, ground states
  walshedge.py                    edge asymptotics for W_2 and W_4, and the fixed-size
                                  partition function for W_2
  chu.py                          Chu sequences as ground states of F_N
  order16.py, order16-matrices.txt  sign and 4th-root maximizers in order 16,
                                  Sloane's had.16.0 to had.16.4

Section 9, the 4th-moment crossover
  crossover.py, crossover.csv     table of M(c) and the 65 points of the figure

Section 10, Villain glow of Hadamard matrices
  hypercube.py, .log              Theorem 10.3, last figure, independent pinned
                                  Laplacian and four-cycle checks
  villaincheck.py, .log           E|Z|^2 at N=16, formula against exact sampling
  had163.txt, bform.txt           had.16.3 and the phases of a bistochastic form

runall.sh captures the output from each check in the corresponding .log
file, except for crossover.py, whose output is crossover.csv.
