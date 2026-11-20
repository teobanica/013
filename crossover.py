"""Reproduce the Section 9 table and the 65 samples of the crossover curve."""
from math import pi
import csv
import sys

from scipy.integrate import quad
from scipy.special import k0, sici


def moment(c):
    """Integrate in z=sqrt(UV), for independent mean-one exponentials U,V."""
    if c == 0:
        return 2.0
    alpha = pi * c**2 / 2

    def integrand(z):
        sine_integral = sici(alpha * z**2)[0]
        return 4 * z * k0(2 * z) * (1 - 2 * sine_integral / pi)**2

    # The product-density tail beyond z=25 is negligible at printed precision.
    cuts = (0, 0.25, 1, 3, 6, 10, 15, 25)
    return 1 + sum(quad(integrand, a, b, epsabs=4e-11, limit=600)[0]
                   for a, b in zip(cuts, cuts[1:]))


def main():
    values = [moment(k / 16) for k in range(65)]
    assert all(a > b for a, b in zip(values, values[1:]))
    table = {4: '1.89106', 8: '1.69450', 16: '1.41935', 32: '1.19882', 64: '1.07900'}
    assert all(f'{values[k]:.5f}' == expected for k, expected in table.items())
    out = csv.writer(sys.stdout, lineterminator='\n')
    out.writerow(['c', 'M_figure_6dp', 'M_table_5dp'])
    for k, value in enumerate(values):
        out.writerow([f'{k/16:.4f}', f'{value:.6f}', table.get(k, '')])


if __name__ == '__main__':
    main()
