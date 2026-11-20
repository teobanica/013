"""Reproduce the dephased sign and fourth-root counts in Section 6.

order16-matrices.txt contains the 16 rows of each of Sloane's representatives
had.16.0 through had.16.4, in order, with blank lines between matrices.
"""
from pathlib import Path

DATA = Path(__file__).with_name('order16-matrices.txt')


def distance(x, y):
    return bin(x ^ y).count('1')


def masks(rows):
    if len(rows) != 16 or any(len(row) != 16 or set(row) - {'+', '-'} for row in rows):
        raise ValueError('Expected a 16 by 16 sign matrix')
    row_masks = [int(row.translate(str.maketrans('+-', '01')), 2) for row in rows]
    assert all(distance(x, y) == 8 for i, x in enumerate(row_masks) for y in row_masks[:i])
    return row_masks


def sign_count(rows):
    # The first entry is the most significant bit. Dephasing fixes that bit at 0.
    # If the Hamming distance is d, the corresponding row sum is 16 - 2*d.
    row_masks = masks(rows)
    return sum(all(distance(m, h) in (6, 10) for h in row_masks)
               for m in range(1 << 15))


def main():
    matrices = [block.splitlines() for block in DATA.read_text().strip().split('\n\n')]
    counts = [sign_count(rows) for rows in matrices]
    assert counts == [448, 192, 64, 0, 0]
    assert [''.join(col) for col in zip(*matrices[3])] == matrices[4]
    print('Sloane class   dephased signs   dephased fourth roots')
    for i, count in enumerate(counts):
        # b=(s+it)/(1+i). The only integer squares summing to 32 are 16 and 16.
        print(f'16.{i} {count:>22} {count**2:>23}')
    print('had.16.4 is the transpose of had.16.3.')


if __name__ == '__main__':
    main()
