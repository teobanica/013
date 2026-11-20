"""Theorem 10.3. Exact cold-end coefficients for the Walsh glow on a cube.

The printed values of n^2 c_n give the data points in the last figure.
Each undirected edge carries one Villain bond.
"""
from fractions import Fraction
from math import comb

import numpy as np


def coefficient(n):
    """Group ordered pairs of supports by their sizes and intersection."""
    total = Fraction(0)
    for k in range(1, n + 1):
        for m in range(1, k + 1):
            first = max(1, m - (n - k))
            if first % 2 == 0:
                first += 1
            partners = sum(comb(k, h) * comb(n - k, m - h)
                           for h in range(first, m + 1, 2))
            multiplicity = 1 if k == m else 2
            total += Fraction(multiplicity * comb(n, k) * partners, k * m)
    return 16 * total / (4 ** n)


def pinned_coefficient(n):
    """Independent real-space calculation from the pinned cube Laplacian."""
    size = 2 ** n
    sites = range(size)
    walsh = np.array([[(-1) ** ((x & s).bit_count()) for s in sites]
                      for x in sites], dtype=float)
    laplacian = n * np.eye(size)
    for x in sites:
        for i in range(n):
            laplacian[x, x ^ (1 << i)] = -1
    green = np.zeros_like(laplacian)
    green[1:, 1:] = np.linalg.inv(laplacian[1:, 1:])
    # Linearizing both pinned phase fields gives a bilinear form with
    # variance proportional to Tr(W G W^t G).
    return 16 * np.trace(walsh @ green @ walsh.T @ green) / size**2


def square_moment(l):
    """Exact four-cycle currents summed to floating-point precision."""
    gray_order = (0, 1, 3, 2)
    modes = np.arange(-int(7*np.sqrt(l))-4, int(7*np.sqrt(l))+5)
    denominator = np.exp(-4*modes**2/l).sum()
    deficit = np.zeros((4, 4))
    for x in range(4):
        for y in range(4):
            charge = np.zeros(4, dtype=int)
            for site, sign in [(0, 1), (x ^ y, 1), (x, -1), (y, -1)]:
                charge[site] += sign
            offsets = np.cumsum(charge[list(gray_order)])
            action = ((modes[:, None] + offsets)**2).sum(axis=1)
            deficit[x, y] = 1 - np.exp(-action/l).sum()/denominator
    walsh = np.array([[(-1) ** ((x & y).bit_count()) for x in range(4)]
                      for y in range(4)])
    return 1 + (deficit * (walsh @ deficit @ walsh)).sum()/16


if __name__ == "__main__":
    assert [coefficient(n) for n in (1, 2, 3)] == [4, 4, Fraction(227, 72)]
    for dim in range(1, 7):
        assert abs(float(coefficient(dim)) - pinned_coefficient(dim)) < 1e-10
    l = 2000
    scaled = l*l*(square_moment(l)-1)
    assert abs(scaled - float(coefficient(2))) < .015
    print("Pinned Laplacian verified through n=6.")
    print(f"Independent four-cycle sum: l={l}, l^2*(E|Z|^4-1)={scaled:.6f}.")
    print("  n          c_n          n^2 c_n")
    for dim in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32,
                48, 64, 96, 128]:
        c = coefficient(dim)
        print(f"{dim:3d}  {float(c):12.8f}  {float(dim*dim*c):12.8f}")
