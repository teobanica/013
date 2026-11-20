"""Exact Section 4 examples. Run from the extracted code folder."""
from collections import Counter
from itertools import permutations, product

import numpy as np

from sixth import check_symbolic_h6
from matrices import T6


def tao_invariants():
    # Matrix entries are w**entry, where w**2+w+1=0.
    exponents = [[int(n) for n in row] for row in
                 ('000000', '001122', '010221', '012012', '022101', '021210')]
    assert np.allclose(np.exp(2j * np.pi * np.array(exponents) / 3), T6())
    # Orthogonality is equivalent to each ratio having two of each cube root.
    for a, b in permutations(exponents, 2):
        assert Counter((x-y) % 3 for x, y in zip(a, b)) == {0: 2, 1: 2, 2: 2}
    # Squaring conjugates the matrix, so the centered squared-row Gram matrix
    # vanishes (S2=R=0). Cubing gives the all-ones matrix, so S3=30*6**2.
    # For T, all three cyclic factors coincide since 2 == -1 modulo three.
    coefficients = [0, 0, 0]
    for a, b, c in permutations(exponents, 3):
        counts = Counter((-x-y-z) % 3 for x, y, z in zip(a, b, c))
        for i, j, k in product(range(3), repeat=3):
            coefficients[(i+j+k) % 3] += counts[i] * counts[j] * counts[k]
    c0, c1, c2 = coefficients
    assert c1 == c2
    return 0, 30*6**2, 0, c0-c1


if __name__ == '__main__':
    s2, s3, r, t = tao_invariants()
    assert (s2, s3, r, t) == (0, 1080, 0, -3240)
    delta = 51200*s3 + 21600*t
    assert delta == -14688000
    print(f'Tao: S2={s2}, S3={s3}, R={r}, T={t}, moment deviation={delta}')
    check_symbolic_h6()
