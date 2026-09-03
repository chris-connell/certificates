"""Uniform exact first-jet certificate for the tensors A^theta.

This verifies every identity over the quotient ring

    Z[s,c]/(s^2+c^2-1),

not by sampling numerical parameters or tangent vectors.  Tensor slots use
the convention in the paper's matrix computation:

    <A_X e_j,e_l> = A(X,e_j,X,e_l),
    <B_X e_j,e_l> = B(X,e_j,X,e_l;X).

Thus A_X(Y)=A(X,Y)X.  The isolated draft sentence defining A_X(Y) as
A(Y,X)X has the opposite sign and must be corrected.

The sparse component pattern is the table P3_DATA below.  Replace each
signed magnitude by

    10 -> 6s,   5 -> 3s,   4 -> 3sc,
     3 -> 3s^2, 8 -> 6sc,  6 -> 6s^2.

The resulting algebraic covariant derivative curvature tensor B satisfies

    B_X = 6s(-u Q(w)+v P^theta(w)),

    d Ric = 0,
    d tr(A_X^2) = 0,
    tr(B_X^2) = 288 s^2 a b^2,
    |B|^2 = 4608 s^2,

where a=x1^2+x6^2 and b=x2^2+x3^2+x4^2+x5^2.  It follows that

    32 tr(A_X^3)-9 tr(B_X^2) = -2176 |X|^6

for every parameter s^2+c^2=1.
"""

from collections import defaultdict
import itertools
import math


N = 6
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
PINDEX = {pair: index for index, pair in enumerate(PAIRS)}


def pair_coord(i, j):
    if i == j:
        return None, 0
    if i < j:
        return PINDEX[(i, j)], 1
    return PINDEX[(j, i)], -1


# Entries (derivative,i,j,k,l,signed magnitude), with both curvature pairs
# increasing.  Pair symmetry and skew-symmetry generate every other entry.
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


# A polynomial is a dict (s_degree,c_degree) -> integer coefficient.
ZERO = {}
ONE = {(0, 0): 1}
S = {(1, 0): 1}
C = {(0, 1): 1}
SC = {(1, 1): 1}
S2 = {(2, 0): 1}


def poly_add(left, right):
    result = defaultdict(int)
    result.update(left)
    for monomial, coefficient in right.items():
        result[monomial] += coefficient
    return {monomial: coefficient for monomial, coefficient in result.items()
            if coefficient}


def poly_scale(poly, scalar):
    return {monomial: scalar * coefficient
            for monomial, coefficient in poly.items() if scalar * coefficient}


def poly_multiply(left, right):
    result = defaultdict(int)
    for (s1, c1), coefficient1 in left.items():
        for (s2, c2), coefficient2 in right.items():
            result[s1 + s2, c1 + c2] += coefficient1 * coefficient2
    return {monomial: coefficient for monomial, coefficient in result.items()
            if coefficient}


def reduce_mod_circle(poly):
    """Use c^(2k+e)=c^e(1-s^2)^k, so the c-degree is at most one."""
    result = defaultdict(int)
    for (s_degree, c_degree), coefficient in poly.items():
        power, remainder = divmod(c_degree, 2)
        for index in range(power + 1):
            result[s_degree + 2 * index, remainder] += (
                coefficient * (-1) ** index * math.comb(power, index)
            )
    return {monomial: coefficient for monomial, coefficient in result.items()
            if coefficient}


def empty_pair_matrix():
    return [[{} for _ in range(15)] for _ in range(15)]


def put(Q, i, j, k, l, value):
    a, sign_a = pair_coord(i - 1, j - 1)
    b, sign_b = pair_coord(k - 1, l - 1)
    value = poly_scale(value, sign_a * sign_b)
    Q[a][b] = value
    Q[b][a] = value


def component(Q, i, j, k, l):
    a, sign_a = pair_coord(i, j)
    b, sign_b = pair_coord(k, l)
    if not sign_a or not sign_b:
        return ZERO
    return poly_scale(Q[a][b], sign_a * sign_b)


def build_A():
    Q = empty_pair_matrix()
    sections = {
        (1, 2): -1, (1, 3): -1, (1, 4): -1, (1, 5): -1,
        (1, 6): -4, (2, 3): -1, (2, 4): -1, (2, 6): -1,
        (2, 5): -4, (3, 5): -1, (3, 6): -1, (4, 5): -1,
        (4, 6): -1, (5, 6): -1, (3, 4): -4,
    }
    for (i, j), value in sections.items():
        put(Q, i, j, i, j, poly_scale(ONE, value))
    mixed = [
        (1, 3, 2, 6, poly_scale(S, -1)), (1, 4, 3, 6, C),
        (1, 4, 5, 6, S), (1, 5, 2, 6, C),
        (1, 5, 4, 6, poly_scale(S, -1)), (1, 2, 3, 6, S),
        (1, 2, 5, 6, poly_scale(C, -1)),
        (1, 3, 4, 6, poly_scale(C, -1)),
        (2, 3, 4, 5, ONE), (2, 4, 3, 5, poly_scale(ONE, -1)),
        (1, 6, 2, 3, poly_scale(S, -2)),
        (1, 6, 2, 5, poly_scale(C, 2)),
        (1, 6, 3, 4, poly_scale(C, 2)),
        (1, 6, 4, 5, poly_scale(S, -2)),
        (2, 5, 3, 4, poly_scale(ONE, -2)),
    ]
    for entry in mixed:
        put(Q, *entry)
    return Q


def build_B():
    matrices = [empty_pair_matrix() for _ in range(N)]
    magnitude = {
        10: poly_scale(S, 6),
        5: poly_scale(S, 3),
        4: poly_scale(SC, 3),
        3: poly_scale(S2, 3),
        8: poly_scale(SC, 6),
        6: poly_scale(S2, 6),
    }
    for derivative, i, j, k, l, signed_magnitude in P3_DATA:
        sign = 1 if signed_magnitude > 0 else -1
        value = poly_scale(magnitude[abs(signed_magnitude)], sign)
        put(matrices[derivative - 1], i, j, k, l, value)
    return matrices


def add_x_term(polynomial, x_indices, parameter_poly):
    x_monomial = tuple(sorted(x_indices))
    polynomial[x_monomial] = poly_add(polynomial[x_monomial], parameter_poly)


def audit():
    A_matrix = build_A()
    B_matrices = build_B()
    A = lambda i, j, k, l: component(A_matrix, i, j, k, l)
    B = lambda m, i, j, k, l: component(B_matrices[m], i, j, k, l)

    # Exhaustive curvature symmetries in all 6^5 slots.
    for m, i, j, k, l in itertools.product(range(N), repeat=5):
        value = B(m, i, j, k, l)
        assert not poly_add(value, B(m, j, i, k, l))
        assert not poly_add(value, B(m, i, j, l, k))
        assert value == B(m, k, l, i, j)

    # Algebraic and differential Bianchi identities, and parallel Ricci.
    for m, i, j, k, l in itertools.product(range(N), repeat=5):
        algebraic = poly_add(
            poly_add(B(m, i, j, k, l), B(m, i, k, l, j)),
            B(m, i, l, j, k),
        )
        differential = poly_add(
            poly_add(B(m, i, j, k, l), B(k, i, j, l, m)),
            B(l, i, j, m, k),
        )
        assert not algebraic
        assert not differential
    for m, i, k in itertools.product(range(N), repeat=3):
        contraction = ZERO
        for j in range(N):
            contraction = poly_add(contraction, B(m, i, j, k, j))
        assert not contraction

    # Full covariant-tensor norm.  This is distinct from the radial
    # derivative-Jacobi norm checked below.
    full_norm = ZERO
    for m, i, j, k, l in itertools.product(range(N), repeat=5):
        value = B(m, i, j, k, l)
        full_norm = poly_add(full_norm, poly_multiply(value, value))
    assert not reduce_mod_circle(poly_add(full_norm, poly_scale(S2, -4608)))

    # Every x-coefficient of tr(A_X B_{m,X}) vanishes modulo s^2+c^2=1.
    for m in range(N):
        moment = defaultdict(dict)
        for i, j, k, l, p, q in itertools.product(range(N), repeat=6):
            value = poly_multiply(A(i, j, k, l), B(m, p, j, q, l))
            if value:
                add_x_term(moment, (i, k, p, q), value)
        assert not {
            monomial: reduce_mod_circle(value)
            for monomial, value in moment.items() if reduce_mod_circle(value)
        }

    # Build B_X and verify ||B_X||^2=288s^2 a b^2.
    B_X = [[defaultdict(dict) for _ in range(N)] for _ in range(N)]
    for j, l, m, i, k in itertools.product(range(N), repeat=5):
        value = B(m, i, j, k, l)
        if value:
            add_x_term(B_X[j][l], (m, i, k), value)

    # Verify coefficientwise that this sparse tensor is the manuscript's
    # compact representative B_X=6s(-u Q(w)+v P^theta(w)).  Here
    # (u,a,b,d,e,v)=(x1,x2,x3,x4,x5,x6), and indices 1,...,4 are W.
    expected = [[defaultdict(dict) for _ in range(N)] for _ in range(N)]

    def expected_put(j, l, coefficient, *monomial):
        add_x_term(expected[j][l], monomial, coefficient)
        if j != l:
            add_x_term(expected[l][j], monomial, coefficient)

    u, a, b, d, e, v = 0, 1, 2, 3, 4, 5
    expected_put(1, 1, poly_scale(SC, -24), v, b, d)
    expected_put(1, 1, poly_scale(S2, -24), v, d, e)
    expected_put(1, 1, poly_scale(S, 12), u, b, b)
    expected_put(1, 1, poly_scale(S, -12), u, d, d)
    expected_put(1, 2, poly_scale(SC, 12), v, a, d)
    expected_put(1, 2, poly_scale(SC, 12), v, b, e)
    expected_put(1, 2, poly_scale(S2, -12), v, d, d)
    expected_put(1, 2, poly_scale(S2, 12), v, e, e)
    expected_put(1, 2, poly_scale(S, -12), u, a, b)
    expected_put(1, 2, poly_scale(S, 12), u, d, e)
    expected_put(1, 3, poly_scale(SC, 12), v, a, b)
    expected_put(1, 3, poly_scale(S2, 12), v, a, e)
    expected_put(1, 3, poly_scale(S2, 12), v, b, d)
    expected_put(1, 3, poly_scale(SC, -12), v, d, e)
    expected_put(1, 3, poly_scale(S, 12), u, a, d)
    expected_put(1, 3, poly_scale(S, 12), u, b, e)
    expected_put(1, 4, poly_scale(S2, 12), v, a, d)
    expected_put(1, 4, poly_scale(SC, -12), v, b, b)
    expected_put(1, 4, poly_scale(S2, -12), v, b, e)
    expected_put(1, 4, poly_scale(SC, 12), v, d, d)
    expected_put(1, 4, poly_scale(S, -24), u, b, d)
    expected_put(2, 2, poly_scale(SC, -24), v, a, e)
    expected_put(2, 2, poly_scale(S2, 24), v, d, e)
    expected_put(2, 2, poly_scale(S, 12), u, a, a)
    expected_put(2, 2, poly_scale(S, -12), u, e, e)
    expected_put(2, 3, poly_scale(SC, -12), v, a, a)
    expected_put(2, 3, poly_scale(S2, 12), v, a, d)
    expected_put(2, 3, poly_scale(S2, -12), v, b, e)
    expected_put(2, 3, poly_scale(SC, 12), v, e, e)
    expected_put(2, 3, poly_scale(S, -24), u, a, e)
    expected_put(2, 4, poly_scale(SC, 12), v, a, b)
    expected_put(2, 4, poly_scale(S2, -12), v, a, e)
    expected_put(2, 4, poly_scale(S2, -12), v, b, d)
    expected_put(2, 4, poly_scale(SC, -12), v, d, e)
    expected_put(2, 4, poly_scale(S, 12), u, a, d)
    expected_put(2, 4, poly_scale(S, 12), u, b, e)
    expected_put(3, 3, poly_scale(S2, -24), v, a, b)
    expected_put(3, 3, poly_scale(SC, 24), v, a, e)
    expected_put(3, 3, poly_scale(S, -12), u, a, a)
    expected_put(3, 3, poly_scale(S, 12), u, e, e)
    expected_put(3, 4, poly_scale(S2, -12), v, a, a)
    expected_put(3, 4, poly_scale(SC, -12), v, a, d)
    expected_put(3, 4, poly_scale(S2, 12), v, b, b)
    expected_put(3, 4, poly_scale(SC, -12), v, b, e)
    expected_put(3, 4, poly_scale(S, 12), u, a, b)
    expected_put(3, 4, poly_scale(S, -12), u, d, e)
    expected_put(4, 4, poly_scale(S2, 24), v, a, b)
    expected_put(4, 4, poly_scale(SC, 24), v, b, d)
    expected_put(4, 4, poly_scale(S, -12), u, b, b)
    expected_put(4, 4, poly_scale(S, 12), u, d, d)

    for j, l in itertools.product(range(N), repeat=2):
        monomials = set(B_X[j][l]) | set(expected[j][l])
        for monomial in monomials:
            difference = poly_add(
                B_X[j][l].get(monomial, ZERO),
                poly_scale(expected[j][l].get(monomial, ZERO), -1),
            )
            assert not reduce_mod_circle(difference)

    norm = defaultdict(dict)
    for j, l in itertools.product(range(N), repeat=2):
        for monomial1, value1 in B_X[j][l].items():
            for monomial2, value2 in B_X[j][l].items():
                add_x_term(
                    norm, monomial1 + monomial2,
                    poly_multiply(value1, value2),
                )
    for u in (0, 5):
        for v, w in itertools.product(range(1, 5), repeat=2):
            add_x_term(norm, (u, u, v, v, w, w), poly_scale(S2, -288))
    assert not {
        monomial: reduce_mod_circle(value)
        for monomial, value in norm.items() if reduce_mod_circle(value)
    }

    # Independently verify tr(A_X^3)=-68|X|^6+81s^2ab^2.
    A_X = [[defaultdict(dict) for _ in range(N)] for _ in range(N)]
    for j, l, i, k in itertools.product(range(N), repeat=4):
        value = A(i, j, k, l)
        if value:
            add_x_term(A_X[j][l], (i, k), value)
    cubic = defaultdict(dict)
    for j, l, p in itertools.product(range(N), repeat=3):
        for monomial1, value1 in A_X[j][l].items():
            for monomial2, value2 in A_X[l][p].items():
                for monomial3, value3 in A_X[p][j].items():
                    add_x_term(
                        cubic, monomial1 + monomial2 + monomial3,
                        poly_multiply(poly_multiply(value1, value2), value3),
                    )
    for i, k, m in itertools.product(range(N), repeat=3):
        add_x_term(cubic, (i, i, k, k, m, m), poly_scale(ONE, 68))
    for u in (0, 5):
        for v, w in itertools.product(range(1, 5), repeat=2):
            add_x_term(cubic, (u, u, v, v, w, w), poly_scale(S2, -81))
    assert not {
        monomial: reduce_mod_circle(value)
        for monomial, value in cubic.items() if reduce_mod_circle(value)
    }


if __name__ == "__main__":
    audit()
    print("uniform certificate over Z[s,c]/(s^2+c^2-1): PASS")
    print("|B|^2=4608s^2, tr(B_X^2)=288s^2ab^2, C=-2176")
