#!/bin/sh
# runall.sh -- reruns the faster checks of the paper and writes one output file next to each program.
# The long computations (walshglow.py, fqglow.py, basisglow.py) have their outputs in walsh.out, gen.out, q*.out,
# basisglow.json and basisglow.log. sixth.py and invariants6.py need SymPy.
set -ex
cd "$(dirname "$0")"
python3 check2.py > check2.log 2>&1
python3 diff.py > diff.log 2>&1
python3 look.py > look.log 2>&1
python3 logexp.py walsh.out q3.out q5.out q1000003.out > logexp.log 2>&1
python3 s2check.py > s2check.log 2>&1
python3 s2s3fit.py > s2s3fit.log 2>&1
python3 villaincheck.py > villaincheck.log 2>&1
python3 crossover.py > crossover.csv
python3 hankel.py > hankel.log
python3 twelfth.py > twelfth.log
python3 graphcheck.py > graphcheck.log
python3 order16.py > order16.log
python3 walshedge.py > walshedge.log
python3 fluxes.py > fluxes.log
python3 freeenergy.py > freeenergy.log
python3 chu.py > chu.log
python3 sixth.py > sixth.log 2>&1
python3 invariants6.py > invariants6.log
python3 hypercube.py > hypercube.log
