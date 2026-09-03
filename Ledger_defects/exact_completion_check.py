"""Exact rational certificate for the A^theta first-jet completion.

All scalar coefficients live in Q[s,c]/(s^2+c^2-1).  No floating-point
arithmetic or external CAS is used.
"""

from collections import defaultdict
from fractions import Fraction as F
import itertools
import math


class SC:
    def __init__(self, terms=None):
        self.d = {k: F(v) for k, v in (terms or {}).items() if v}

    @staticmethod
    def q(v):
        return SC({(0, 0): F(v)})

    def __add__(self, other):
        other = as_sc(other)
        out = defaultdict(F); out.update(self.d)
        for k, v in other.d.items(): out[k] += v
        return SC(out)

    __radd__ = __add__

    def __neg__(self):
        return SC({k: -v for k, v in self.d.items()})

    def __sub__(self, other):
        return self + (-as_sc(other))

    def __rsub__(self, other):
        return as_sc(other) - self

    def __mul__(self, other):
        other = as_sc(other)
        out = defaultdict(F)
        for (a, b), x in self.d.items():
            for (c, d), y in other.d.items(): out[(a+c, b+d)] += x*y
        return SC(out)

    __rmul__ = __mul__

    def __truediv__(self, value):
        return SC({k: v/F(value) for k, v in self.d.items()})

    def reduced(self):
        # Repeatedly replace s^a c^b by s^(a-2)c^b - s^(a-2)c^(b+2).
        work = defaultdict(F); work.update(self.d)
        while any(a >= 2 and v for (a, b), v in work.items()):
            key = next(k for k, v in work.items() if k[0] >= 2 and v)
            a, b = key; v = work.pop(key)
            work[(a-2, b)] += v
            work[(a-2, b+2)] -= v
        return SC(work)

    def is_zero(self):
        return not self.reduced().d

    def __repr__(self):
        return repr(self.reduced().d)


def as_sc(x):
    return x if isinstance(x, SC) else SC.q(x)


S_PAR = SC({(1, 0): 1})
C_PAR = SC({(0, 1): 1})
N = 6


def add_xpoly(dst, mon, val):
    mon = tuple(sorted(mon))
    dst[mon] = dst.get(mon, SC()) + val
    if dst[mon].is_zero(): dst.pop(mon)


def mul_xpoly(p, q):
    out = {}
    for a, x in p.items():
        for b, y in q.items(): add_xpoly(out, a+b, x*y)
    return out


def candidate_polynomial():
    """Return matrix entries as dictionaries: cubic x-monomial -> SC coefficient."""
    M = [[{} for _ in range(N)] for _ in range(N)]

    def put(j, l, coeff, *mon):
        add_xpoly(M[j][l], mon, coeff)
        if j != l: add_xpoly(M[l][j], mon, coeff)

    # u=x_1 is index 0, v=x_6 is index 5; a,b,d,e are indices 1,2,3,4.
    u, a, b, d, e, v = 0, 1, 2, 3, 4, 5
    put(1,1,-4*C_PAR,u,b,d); put(1,1,-4*S_PAR,u,d,e); put(1,1,-2,v,b,b); put(1,1,2,v,d,d)
    put(1,2,2*C_PAR,u,a,d); put(1,2,2*C_PAR,u,b,e); put(1,2,-2*S_PAR,u,d,d); put(1,2,2*S_PAR,u,e,e); put(1,2,2,v,a,b); put(1,2,-2,v,d,e)
    put(1,3,2*C_PAR,u,a,b); put(1,3,2*S_PAR,u,a,e); put(1,3,2*S_PAR,u,b,d); put(1,3,-2*C_PAR,u,d,e); put(1,3,-2,v,a,d); put(1,3,-2,v,b,e)
    put(1,4,2*S_PAR,u,a,d); put(1,4,-2*C_PAR,u,b,b); put(1,4,-2*S_PAR,u,b,e); put(1,4,2*C_PAR,u,d,d); put(1,4,4,v,b,d)
    put(2,2,-4*C_PAR,u,a,e); put(2,2,4*S_PAR,u,d,e); put(2,2,-2,v,a,a); put(2,2,2,v,e,e)
    put(2,3,-2*C_PAR,u,a,a); put(2,3,2*S_PAR,u,a,d); put(2,3,-2*S_PAR,u,b,e); put(2,3,2*C_PAR,u,e,e); put(2,3,4,v,a,e)
    put(2,4,2*C_PAR,u,a,b); put(2,4,-2*S_PAR,u,a,e); put(2,4,-2*S_PAR,u,b,d); put(2,4,-2*C_PAR,u,d,e); put(2,4,-2,v,a,d); put(2,4,-2,v,b,e)
    put(3,3,-4*S_PAR,u,a,b); put(3,3,4*C_PAR,u,a,e); put(3,3,2,v,a,a); put(3,3,-2,v,e,e)
    put(3,4,-2*S_PAR,u,a,a); put(3,4,-2*C_PAR,u,a,d); put(3,4,2*S_PAR,u,b,b); put(3,4,-2*C_PAR,u,b,e); put(3,4,-2,v,a,b); put(3,4,2,v,d,e)
    put(4,4,4*S_PAR,u,a,b); put(4,4,4*C_PAR,u,b,d); put(4,4,2,v,b,b); put(4,4,-2,v,d,d)
    return M


M = candidate_polynomial()


def polarized_S(a, b, c, d, e):
    mon = tuple(sorted((a, b, c)))
    coeff = M[d][e].get(mon, SC())
    counts = [mon.count(k) for k in set(mon)]
    multiplicity = math.factorial(3)
    for z in counts: multiplicity //= math.factorial(z)
    return coeff / multiplicity


def T(a, b, c, d, e):
    # Inverse of the radial Jacobi polarization map.
    return (polarized_S(a,b,c,d,e)/2
            - polarized_S(a,b,d,c,e)/2
            + polarized_S(a,c,e,b,d)
            - polarized_S(a,d,e,b,c))


PAIRS = [(i, j) for i in range(N) for j in range(i+1, N)]
PI = {p: k for k, p in enumerate(PAIRS)}


def pair(i, j):
    if i == j: return None, 0
    return (PI[(i,j)], 1) if i < j else (PI[(j,i)], -1)


def curvature_A():
    Q = [[SC() for _ in range(15)] for _ in range(15)]
    def put(i,j,k,l,val):
        a,sa=pair(i-1,j-1); b,sb=pair(k-1,l-1); val=val/(sa*sb)
        Q[a][b]=Q[b][a]=val
    sections={(1,2):-1,(1,3):-1,(1,4):-1,(1,5):-1,(1,6):-4,
              (2,3):-1,(2,4):-1,(2,6):-1,(2,5):-4,(3,5):-1,
              (3,6):-1,(4,5):-1,(4,6):-1,(5,6):-1,(3,4):-4}
    for (i,j),v in sections.items(): put(i,j,i,j,SC.q(v))
    mixed=[(1,3,2,6,-S_PAR),(1,4,3,6,C_PAR),(1,4,5,6,S_PAR),
           (1,5,2,6,C_PAR),(1,5,4,6,-S_PAR),(1,2,3,6,S_PAR),
           (1,2,5,6,-C_PAR),(1,3,4,6,-C_PAR),(2,3,4,5,SC.q(1)),
           (2,4,3,5,SC.q(-1)),(1,6,2,3,-2*S_PAR),(1,6,2,5,2*C_PAR),
           (1,6,3,4,2*C_PAR),(1,6,4,5,-2*S_PAR),(2,5,3,4,SC.q(-2))]
    for row in mixed: put(*row)
    def A(i,j,k,l):
        a,sa=pair(i,j); b,sb=pair(k,l)
        return SC() if not (sa and sb) else sa*sb*Q[a][b]
    return A


def check():
    # M_X X=0.
    mx = [dict() for _ in range(N)]
    for j,l in itertools.product(range(N), repeat=2):
        for mon,coef in M[j][l].items(): add_xpoly(mx[j], mon+(l,), coef)
    assert all(not p for p in mx), "candidate does not annihilate X"

    # The inverse formula really returns the prescribed radial Jacobi polynomial.
    radial = [[{} for _ in range(N)] for _ in range(N)]
    for j,l,a,c,e in itertools.product(range(N), repeat=5):
        add_xpoly(radial[j][l], (a,c,e), T(a,j,c,l,e))
    for j,l in itertools.product(range(N), repeat=2):
        mons=set(radial[j][l])|set(M[j][l])
        assert all((radial[j][l].get(mon,SC())-M[j][l].get(mon,SC())).is_zero()
                   for mon in mons), "inverse polarization mismatch"

    # Full differentiated Einstein identity, not merely its diagonal restriction.
    ric_bad=[]
    for e,a,c in itertools.product(range(N), repeat=3):
        val=sum((T(a,b,c,b,e) for b in range(N)), SC())
        if not val.is_zero(): ric_bad.append((e,a,c,val))
    assert not ric_bad, ric_bad[:3]

    # Full differentiated second Jacobi moment for arbitrary base direction e.
    A=curvature_A()
    moment_by_e=[{} for _ in range(N)]
    for e in range(N):
        for a,c,p,q,j,l in itertools.product(range(N), repeat=6):
            val=A(a,j,c,l)*T(p,l,q,j,e)
            if val.d: add_xpoly(moment_by_e[e],(a,c,p,q),val)
    bad=[(e,mon,val) for e,P in enumerate(moment_by_e) for mon,val in P.items() if not val.is_zero()]
    assert not bad, bad[:3]

    # Exact norm identity ||M_X||^2=8 (x_1^2+x_6^2)(x_2^2+...+x_5^2)^2.
    norm={}
    for j,l in itertools.product(range(N), repeat=2):
        product=mul_xpoly(M[j][l],M[j][l])
        for mon,val in product.items(): add_xpoly(norm,mon,val)
    target={}
    for u in (0,5):
        for w1 in range(1,5):
            for w2 in range(1,5): add_xpoly(target,(u,u,w1,w1,w2,w2),SC.q(8))
    allmons=set(norm)|set(target)
    badnorm=[(mon,norm.get(mon,SC())-target.get(mon,SC())) for mon in allmons
             if not (norm.get(mon,SC())-target.get(mon,SC())).is_zero()]
    assert not badnorm,badnorm[:3]
    print("EXACT CERTIFICATE PASSED")
    print("M_X X = 0; radial polarization matches; dRic = 0;")
    print("d tr(A_X^2) = 0; ||M_X||^2 = 8 a b^2 in Q[s,c]/(s^2+c^2-1).")


if __name__ == "__main__":
    check()
