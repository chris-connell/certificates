"""Exact scalar audit for the algebraic harmonic-realizability theorem.

All curvature contractions are recomputed from the displayed components of
A^theta over Z[s,c]/(s^2+c^2-1).  No floating-point arithmetic is used.
"""

import itertools

import completion_family_exact_certificate as cert


def add(*terms):
    result = cert.ZERO
    for term in terms:
        result = cert.poly_add(result, term)
    return cert.reduce_mod_circle(result)


def scale(poly, scalar):
    return cert.reduce_mod_circle(cert.poly_scale(poly, scalar))


def multiply(*terms):
    result = cert.ONE
    for term in terms:
        result = cert.poly_multiply(result, term)
    return cert.reduce_mod_circle(result)


def audit():
    matrix = cert.build_A()
    A = lambda i, j, k, l: cert.component(matrix, i, j, k, l)

    ric = [[cert.ZERO for _ in range(cert.N)] for _ in range(cert.N)]
    for j, l, i in itertools.product(range(cert.N), repeat=3):
        ric[j][l] = add(ric[j][l], A(i, j, i, l))
    for j, l in itertools.product(range(cert.N), repeat=2):
        expected = cert.poly_scale(cert.ONE, -8) if j == l else cert.ZERO
        assert not add(ric[j][l], scale(expected, -1))

    norm = cert.ZERO
    hat = cert.ZERO
    circle = cert.ZERO
    for a, b, c, d in itertools.product(range(cert.N), repeat=4):
        rabcd = A(a, b, c, d)
        norm = add(norm, multiply(rabcd, rabcd))
        for u, v in itertools.product(range(cert.N), repeat=2):
            hat = add(hat, multiply(rabcd, A(a, b, u, v), A(c, d, u, v)))
            circle = add(
                circle,
                multiply(rabcd, A(a, u, c, v), A(b, u, d, v)),
            )

    expected_norm = scale(cert.ONE, 384)
    expected_hat = add(scale(cert.ONE, -4608), scale(cert.S2, 1152))
    expected_circle = add(scale(cert.ONE, -384), scale(cert.S2, 144))
    assert not add(norm, scale(expected_norm, -1))
    assert not add(hat, scale(expected_hat, -1))
    assert not add(circle, scale(expected_circle, -1))

    # Corrected Lichnerowicz:
    # N + 2 rho |R|^2 - hat - 4 circle = 0.
    rho = -8
    N_lich = add(hat, scale(circle, 4), scale(norm, -2 * rho))
    assert N_lich == scale(cert.S2, 1728)

    # In the contracted H3 identity, write
    # 315 f''' = 2176 + 9 alpha.  Exact simplification gives
    # N = 4608 s^2 + 160 alpha.  Comparing with Lichnerowicz forces
    # alpha = -18 s^2, whereas radial positivity forces alpha >= 0.
    n = 6
    # Every coefficient of hat is even, so (7/2)hat stays integral.
    half_hat = {monomial: coefficient // 2 for monomial, coefficient in hat.items()}
    bracket = add(
        scale(cert.ONE, n * rho ** 3),
        scale(norm, -36),
        scale(half_hat, 7),
        scale(circle, -1),
    )
    assert bracket == add(scale(cert.ONE, -32640), scale(cert.S2, 3888))

    # The general obstruction uses
    # F_A=(27 N_A-32 C_A)/(n(n+2)(n+4))=315 f'''(0).
    obstruction_numerator = add(scale(N_lich, 27), scale(bracket, -32))
    denominator = n * (n + 2) * (n + 4)
    assert all(coefficient % denominator == 0
               for coefficient in obstruction_numerator.values())
    F_A = {
        monomial: coefficient // denominator
        for monomial, coefficient in obstruction_numerator.items()
    }
    F_A = cert.reduce_mod_circle(F_A)
    assert F_A == add(scale(cert.ONE, 2176), scale(cert.S2, -162))

    # At X=e_1, tr(A_X^3)=-68, so the directional nonnegativity
    # condition 32 tr(A_X^3)+F_A>=0 fails by exactly -162s^2.
    directional_at_e1 = add(scale(cert.ONE, 32 * -68), F_A)
    assert directional_at_e1 == scale(cert.S2, -162)

    # Candidate first jet: alpha=0, so the contracted H3 norm is 4608s^2.
    candidate_norm = scale(cert.S2, 4608)
    assert add(candidate_norm, scale(N_lich, -1)) == scale(cert.S2, 2880)

    print("exact harmonic-obstruction audit: PASS")
    print("N_A=1728s^2; C_A=-32640+3888s^2; F_A=2176-162s^2")
    print("at X=e1, 32 tr(A_X^3)+F_A=-162s^2")
    print("equivalently, contracted H3 gives N=4608s^2+160alpha")
    print("therefore alpha=-18s^2, contradicting alpha>=0 for s!=0")


if __name__ == "__main__":
    audit()
