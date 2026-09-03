"""Exact first-jet certificate for the tensor A^{theta}, sin(theta)=3/5.

All tensor entries below are integers after the indicated rescalings:

    A5 = 5 A,
    P3 = 3 P,
    A1 = (27/25) P = (9/25) P3.

The script checks, coefficient by coefficient (no random sampling), that P is
an algebraic covariant derivative curvature tensor, that the differentiated
1- and 2-Stein identities hold, and that

    tr((A1_X)^2) = (2592/25) a b^2,

where a=x1^2+x6^2 and b=x2^2+...+x5^2.  Consequently

    32 tr(A_X^3) - 9 tr((A1_X)^2) = -2176 |X|^6.

It also constructs the complete exact linear constraint matrix in the 720
coordinates V* tensor Sym^2(Lambda^2 V*) and computes its rank modulo a
prime.  The successive nullities are 630, 420, 300, and 30.
"""

from collections import defaultdict
import itertools
import math

import numpy as np


N = 6
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
PINDEX = {p: k for k, p in enumerate(PAIRS)}
SYM = [(a, b) for a in range(15) for b in range(a, 15)]
SINDEX = {p: k for k, p in enumerate(SYM)}


def pair_coord(i, j):
    if i == j:
        return None, 0
    if i < j:
        return PINDEX[(i, j)], 1
    return PINDEX[(j, i)], -1


def sym_pair_coord(i, j, k, l):
    a, sa = pair_coord(i, j)
    b, sb = pair_coord(k, l)
    if not sa or not sb:
        return None, 0
    if a > b:
        a, b = b, a
    return SINDEX[(a, b)], sa * sb


def put_pair_matrix(Q, i, j, k, l, value):
    a, sa = pair_coord(i - 1, j - 1)
    b, sb = pair_coord(k - 1, l - 1)
    value //= sa * sb
    if Q[a, b] not in (0, value):
        raise ValueError((i, j, k, l, value, Q[a, b]))
    Q[a, b] = Q[b, a] = value


def pair_matrix_to_tensor(Q):
    T = np.zeros((N, N, N, N), dtype=np.int64)
    for i, j, k, l in itertools.product(range(N), repeat=4):
        a, sa = pair_coord(i, j)
        b, sb = pair_coord(k, l)
        if sa and sb:
            T[i, j, k, l] = sa * sb * Q[a, b]
    return T


def build_A5():
    """Return the integer tensor 5*A at (s,c)=(3/5,4/5)."""
    Q = np.zeros((15, 15), dtype=np.int64)
    sections = {
        (1, 2): -5, (1, 3): -5, (1, 4): -5, (1, 5): -5,
        (1, 6): -20, (2, 3): -5, (2, 4): -5, (2, 6): -5,
        (2, 5): -20, (3, 5): -5, (3, 6): -5, (4, 5): -5,
        (4, 6): -5, (5, 6): -5, (3, 4): -20,
    }
    for (i, j), value in sections.items():
        put_pair_matrix(Q, i, j, i, j, value)
    mixed = [
        (1, 3, 2, 6, -3), (1, 4, 3, 6, 4),
        (1, 4, 5, 6, 3), (1, 5, 2, 6, 4),
        (1, 5, 4, 6, -3), (1, 2, 3, 6, 3),
        (1, 2, 5, 6, -4), (1, 3, 4, 6, -4),
        (2, 3, 4, 5, 5), (2, 4, 3, 5, -5),
        (1, 6, 2, 3, -6), (1, 6, 2, 5, 8),
        (1, 6, 3, 4, 8), (1, 6, 4, 5, -6),
        (2, 5, 3, 4, -10),
    ]
    for entry in mixed:
        put_pair_matrix(Q, *entry)
    return pair_matrix_to_tensor(Q)


# Nonzero entries of P3=3P, indexed as (derivative,i,j,k,l,value).
# Both curvature pairs are increasing.  Curvature pair symmetry and
# skew-symmetry generate all other components.
P3_DATA = [
    (1, 2, 3, 2, 3, 10), (1, 2, 3, 4, 5, 10),
    (1, 2, 4, 2, 4, -10), (1, 2, 4, 3, 5, 10),
    (1, 3, 5, 3, 5, -10), (1, 4, 5, 4, 5, 10),
    (2, 1, 3, 2, 3, 5), (2, 1, 3, 4, 5, 5),
    (2, 1, 4, 2, 4, -5), (2, 1, 4, 3, 5, 5),
    (2, 2, 3, 4, 6, 4), (2, 2, 4, 3, 6, 4),
    (2, 2, 4, 5, 6, 3), (2, 2, 5, 4, 6, 3),
    (2, 3, 4, 4, 6, 3), (2, 3, 5, 3, 6, -4),
    (2, 3, 5, 5, 6, -3), (2, 4, 5, 4, 6, 4),
    (3, 1, 2, 2, 3, -5), (3, 1, 2, 4, 5, -5),
    (3, 1, 5, 2, 4, 5), (3, 1, 5, 3, 5, -5),
    (3, 2, 3, 5, 6, -4), (3, 2, 4, 2, 6, -4),
    (3, 2, 4, 4, 6, 3), (3, 2, 5, 5, 6, -3),
    (3, 2, 6, 3, 5, 4), (3, 3, 4, 5, 6, -3),
    (3, 3, 5, 4, 6, -3), (3, 4, 5, 5, 6, -4),
    (4, 1, 2, 2, 4, 5), (4, 1, 2, 3, 5, -5),
    (4, 1, 5, 2, 3, 5), (4, 1, 5, 4, 5, 5),
    (4, 2, 3, 2, 6, -4), (4, 2, 4, 3, 6, -3),
    (4, 2, 4, 5, 6, 4), (4, 2, 5, 2, 6, -3),
    (4, 2, 6, 3, 4, -3), (4, 2, 6, 4, 5, -4),
    (4, 3, 5, 3, 6, 3), (4, 3, 5, 5, 6, -4),
    (5, 1, 3, 2, 4, -5), (5, 1, 3, 3, 5, 5),
    (5, 1, 4, 2, 3, -5), (5, 1, 4, 4, 5, -5),
    (5, 2, 3, 3, 6, 4), (5, 2, 4, 2, 6, -3),
    (5, 2, 4, 4, 6, -4), (5, 2, 5, 3, 6, 3),
    (5, 2, 6, 3, 5, 3), (5, 3, 4, 3, 6, 3),
    (5, 3, 5, 4, 6, 4), (5, 3, 6, 4, 5, 4),
    (6, 2, 3, 2, 4, -8), (6, 2, 3, 3, 5, 8),
    (6, 2, 4, 2, 5, -6), (6, 2, 4, 3, 4, -6),
    (6, 2, 4, 4, 5, -8), (6, 2, 5, 3, 5, 6),
    (6, 3, 4, 3, 5, 6), (6, 3, 5, 4, 5, 8),
]


def build_P3():
    result = np.zeros((N, N, N, N, N), dtype=np.int64)
    matrices = [np.zeros((15, 15), dtype=np.int64) for _ in range(N)]
    for m, i, j, k, l, value in P3_DATA:
        put_pair_matrix(matrices[m - 1], i, j, k, l, value)
    for m in range(N):
        result[m] = pair_matrix_to_tensor(matrices[m])
    return result


def add_polynomial_term(poly, indices, value):
    if value:
        poly[tuple(sorted(indices))] += int(value)


def audit_certificate(A5, P3):
    product5 = itertools.product(range(N), repeat=5)
    assert max(abs(P3[m, i, j, k, l] + P3[m, i, k, l, j]
                   + P3[m, i, l, j, k])
               for m, i, j, k, l in product5) == 0

    product5 = itertools.product(range(N), repeat=5)
    assert max(abs(P3[m, i, j, k, l] + P3[k, i, j, l, m]
                   + P3[l, i, j, m, k])
               for m, i, j, k, l in product5) == 0

    assert max(abs(sum(P3[m, i, j, k, j] for j in range(N)))
               for m, i, k in itertools.product(range(N), repeat=3)) == 0

    # Coefficients of tr(A_X P3_{m,X}) vanish for every derivative m.
    for m in range(N):
        poly = defaultdict(int)
        for i, j, k, l, p, q in itertools.product(range(N), repeat=6):
            add_polynomial_term(
                poly, (i, k, p, q), A5[i, j, k, l] * P3[m, p, j, q, l]
            )
        assert not {mon: value for mon, value in poly.items() if value}

    # ||P3_X||^2 = 800 (x1^2+x6^2)(x2^2+...+x5^2)^2.
    entries = [[defaultdict(int) for _ in range(N)] for _ in range(N)]
    for j, l in itertools.product(range(N), repeat=2):
        for m, i, k in itertools.product(range(N), repeat=3):
            add_polynomial_term(entries[j][l], (m, i, k), P3[m, i, j, k, l])

    norm_poly = defaultdict(int)
    for j, l in itertools.product(range(N), repeat=2):
        for mon1, value1 in entries[j][l].items():
            for mon2, value2 in entries[j][l].items():
                add_polynomial_term(norm_poly, mon1 + mon2, value1 * value2)

    target = defaultdict(int)
    for u in (0, 5):
        for v in range(1, 5):
            for w in range(1, 5):
                add_polynomial_term(target, (u, u, v, v, w, w), 800)
    norm_poly = {mon: value for mon, value in norm_poly.items() if value}
    target = {mon: value for mon, value in target.items() if value}
    assert norm_poly == target

    # Independently verify the advertised cubic Jacobi moment.  Since
    # A5=5A, this is
    # tr((A5_X)^3) = -8500 |X|^6 + 3645 a b^2.
    jacobi = [[defaultdict(int) for _ in range(N)] for _ in range(N)]
    for j, l in itertools.product(range(N), repeat=2):
        for i, k in itertools.product(range(N), repeat=2):
            add_polynomial_term(jacobi[j][l], (i, k), A5[i, j, k, l])
    cubic = defaultdict(int)
    for j, l, p in itertools.product(range(N), repeat=3):
        for mon1, value1 in jacobi[j][l].items():
            for mon2, value2 in jacobi[l][p].items():
                for mon3, value3 in jacobi[p][j].items():
                    add_polynomial_term(
                        cubic, mon1 + mon2 + mon3, value1 * value2 * value3
                    )
    cubic_target = defaultdict(int)
    for i, k, m in itertools.product(range(N), repeat=3):
        add_polynomial_term(cubic_target, (i, i, k, k, m, m), -8500)
    for u in (0, 5):
        for v, w in itertools.product(range(1, 5), repeat=2):
            add_polynomial_term(cubic_target, (u, u, v, v, w, w), 3645)
    cubic = {mon: value for mon, value in cubic.items() if value}
    cubic_target = {mon: value for mon, value in cubic_target.items() if value}
    assert cubic == cubic_target


def add_component_to_row(row, m, i, j, k, l, value=1):
    a, sign = sym_pair_coord(i, j, k, l)
    if sign:
        row[120 * m + a] += value * sign


def exact_constraint_matrix(A5):
    rows = []
    stages = []

    # Each derivative slice is an algebraic curvature tensor.
    for m in range(N):
        for i, j, k, l in itertools.combinations(range(N), 4):
            row = np.zeros(720, dtype=np.int64)
            add_component_to_row(row, m, i, j, k, l)
            add_component_to_row(row, m, i, k, l, j)
            add_component_to_row(row, m, i, l, j, k)
            rows.append(row)
            stages.append("algebraic Bianchi")

    # Differential Bianchi; the last three indices may be taken increasing.
    for i, j in PAIRS:
        for k, l, m in itertools.combinations(range(N), 3):
            row = np.zeros(720, dtype=np.int64)
            add_component_to_row(row, m, i, j, k, l)
            add_component_to_row(row, k, i, j, l, m)
            add_component_to_row(row, l, i, j, m, k)
            rows.append(row)
            stages.append("differential Bianchi")

    # Parallel Ricci.
    for m in range(N):
        for i in range(N):
            for k in range(i, N):
                row = np.zeros(720, dtype=np.int64)
                for j in range(N):
                    add_component_to_row(row, m, i, j, k, j)
                rows.append(row)
                stages.append("parallel Ricci")

    # Every coefficient of tr(A_X T_{m,X}) vanishes.
    monomials = list(itertools.combinations_with_replacement(range(N), 4))
    monomial_index = {mon: index for index, mon in enumerate(monomials)}
    moment_rows = np.zeros((N, len(monomials), 720), dtype=np.int64)
    for i, j, k, l in np.argwhere(A5 != 0):
        value = int(A5[i, j, k, l])
        for m, p, q in itertools.product(range(N), repeat=3):
            a, sign = sym_pair_coord(p, int(j), q, int(l))
            if sign:
                mon = tuple(sorted((int(i), int(k), p, q)))
                moment_rows[m, monomial_index[mon], 120 * m + a] += value * sign
    for m in range(N):
        for h in range(len(monomials)):
            rows.append(moment_rows[m, h])
            stages.append("second moment")

    return np.stack(rows), stages


def rank_mod_prime(matrix, prime=1_000_003):
    matrix = matrix.copy() % prime
    nrows, ncols = matrix.shape
    rank = 0
    for column in range(ncols):
        possible = np.flatnonzero(matrix[rank:, column])
        if not len(possible):
            continue
        pivot = rank + int(possible[0])
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        inverse = pow(int(matrix[rank, column]), prime - 2, prime)
        matrix[rank] = matrix[rank] * inverse % prime
        below = np.flatnonzero(matrix[rank + 1:, column]) + rank + 1
        if len(below):
            factors = matrix[below, column].copy()
            matrix[below] = (
                matrix[below] - factors[:, None] * matrix[rank][None, :]
            ) % prime
        rank += 1
    return rank


def rref_mod_prime(matrix, prime=1_000_003):
    """Return modular RREF data used to lift a rational kernel."""
    matrix = matrix.copy() % prime
    nrows, ncols = matrix.shape
    rank = 0
    pivots = []
    for column in range(ncols):
        possible = np.flatnonzero(matrix[rank:, column])
        if not len(possible):
            continue
        pivot = rank + int(possible[0])
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        inverse = pow(int(matrix[rank, column]), prime - 2, prime)
        matrix[rank] = matrix[rank] * inverse % prime
        other = np.flatnonzero(matrix[:, column])
        other = other[other != rank]
        if len(other):
            factors = matrix[other, column].copy()
            matrix[other] = (
                matrix[other] - factors[:, None] * matrix[rank][None, :]
            ) % prime
        pivots.append(column)
        rank += 1
    return matrix, pivots


def rational_reconstruct(residue, modulus):
    """Small numerator/denominator reconstruction, checked modulo modulus."""
    residue = int(residue) % modulus
    if residue == 0:
        return 0, 1
    bound = math.isqrt(modulus // 2)
    r0, r1 = modulus, residue
    t0, t1 = 0, 1
    while abs(r1) > bound:
        quotient = r0 // r1
        r0, r1 = r1, r0 - quotient * r1
        t0, t1 = t1, t0 - quotient * t1
    numerator, denominator = r1, t1
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    divisor = math.gcd(numerator, denominator)
    numerator //= divisor
    denominator //= divisor
    if (abs(numerator) > bound or denominator > bound
            or (residue * denominator - numerator) % modulus):
        raise ArithmeticError("rational reconstruction failed")
    return numerator, denominator


def exact_rational_kernel_dimension(matrix, prime=1_000_003):
    """Lift and directly verify a rational kernel basis.

    The modular rank supplies the lower bound on the rational rank.  Directly
    verified rational kernel vectors supply the matching upper bound.
    """
    rref, pivots = rref_mod_prime(matrix, prime)
    pivot_set = set(pivots)
    free = [column for column in range(matrix.shape[1])
            if column not in pivot_set]
    modular_basis = np.zeros((matrix.shape[1], len(free)), dtype=np.int64)
    for basis_index, free_column in enumerate(free):
        modular_basis[free_column, basis_index] = 1
        for row, pivot_column in enumerate(pivots):
            modular_basis[pivot_column, basis_index] = -rref[row, free_column] % prime

    integer_vectors = []
    for basis_index in range(len(free)):
        fractions = [rational_reconstruct(value, prime)
                     for value in modular_basis[:, basis_index]]
        denominator = 1
        for _, entry_denominator in fractions:
            denominator = math.lcm(denominator, entry_denominator)
        vector = np.array([
            numerator * (denominator // entry_denominator)
            for numerator, entry_denominator in fractions
        ], dtype=object)
        product = matrix.astype(object) @ vector
        assert all(value == 0 for value in product)
        integer_vectors.append(vector)

    # Their distinct unit entries in the free columns prove independence.
    return len(integer_vectors), len(pivots)


def main():
    A5 = build_A5()
    P3 = build_P3()
    audit_certificate(A5, P3)
    print("exact tensor and polynomial certificate: PASS")
    print("A1=(9/25)P3 and tr((A1_X)^2)=(2592/25)ab^2")
    print("third Ledger constant C=-2176")

    matrix, stages = exact_constraint_matrix(A5)
    for stage in (
        "algebraic Bianchi", "differential Bianchi",
        "parallel Ricci", "second moment",
    ):
        endpoint = max(i for i, name in enumerate(stages) if name == stage) + 1
        rank = rank_mod_prime(matrix[:endpoint])
        print(stage, "rank", rank, "nullity", 720 - rank)

    nullity, rank = exact_rational_kernel_dimension(matrix)
    print("verified over Q: rank", rank, "nullity", nullity)


if __name__ == "__main__":
    main()
