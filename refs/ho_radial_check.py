import sys, math, time

sys.path.insert(0, "/home/soham/Projects/nuclear-physics/pre-quantum-projects/csm")
sys.path.insert(0, "/home/soham/Projects/nuclear-physics/nuhamil-alt/refs")
import numpy as np, mpmath as mp
from csm.basis.harmonic_oscillator import BasisParameters, basis
import oracle_numpy as o

mp.mp.dps = 60


def exact(n, l, x):  # normalized (no sqrt(b)): g_nl(x)
    x = mp.mpf(x)
    a = l + mp.mpf(1) / 2
    return (
        mp.sqrt(2 * mp.factorial(n) / mp.gamma(n + a + 1))
        * x ** (l + 1)
        * mp.exp(-x * x / 2)
        * mp.laguerre(n, a, x * x)
    )


# closed-form recurrence (no sympy), vectorized, as would go into lax.scan
def rec(x, l, nmax):
    a = l + 0.5
    x2 = x * x
    g = np.zeros((nmax + 1, len(x)))
    g[0] = np.sqrt(2 / math.gamma(a + 1)) * x ** (l + 1) * np.exp(-x2 / 2)
    if nmax > 0:
        g[1] = g[0] * (a + 1 - x2) / math.sqrt(a + 1)
    for n in range(2, nmax + 1):
        m = n - 1
        A = -1 / math.sqrt((m + 1) * (m + a + 1))
        B = (2 * m + a + 1) / math.sqrt((m + 1) * (m + a + 1))
        C = math.sqrt(m * (m + a) / ((m + 1) * (m + a + 1)))
        g[n] = (A * x2 + B) * g[n - 1] - C * g[n - 2]
    return g


hbw = 16.0
tests = [
    (l, nmax, xs)
    for l, nmax in [(0, 50), (1, 50), (8, 46), (20, 40)]
    for xs in [np.linspace(0.05, 13, 40)]
]
print(
    "max abs err vs mpmath (dimensionless g_nl(x)); csm basis | closed-form rec | oracle scipy form"
)
for l, nmax, xs in tests:
    P = BasisParameters(max_n=nmax, L=l, hbarw=hbw, mu=1.0)  # b = hbarc/sqrt(hbw)
    b = P.b
    ref = np.array([[float(exact(n, l, x)) for x in xs] for n in range(nmax + 1)])
    c1 = basis(xs * b, P) * math.sqrt(b)
    c2 = rec(xs, l, nmax)
    c3 = np.array(
        [
            [o.ho_radial(n, l, 1.0, x) / math.sqrt(2 * 1.0) * math.sqrt(2) for x in [x]][0]
            for n in range(nmax + 1)
            for x in xs
        ]
    ).reshape(nmax + 1, -1)
    # oracle ho_radial(n,l,anu,r) = sqrt(2nu)*... with nu=sqrt(anu)=1 -> equals g(x)
    print(
        f" l={l:2d} nmax={nmax}: {np.abs(c1 - ref).max():.1e} | {np.abs(c2 - ref).max():.1e} | {np.abs(c3 - ref).max():.1e}"
    )
# extreme: large x and large n, relative error where function non-negligible
print("large x (underflow behaviour), l=2, n=60:")
xs = np.array([15.0, 20.0, 30.0, 36.0, 40.0, 45.0])
ref = np.array([[float(exact(n, 2, x)) for x in xs] for n in (0, 30, 60)])
c2 = rec(xs, 2, 60)[[0, 30, 60]]
c3 = np.array([[o.ho_radial(n, 2, 1.0, x) for x in xs] for n in (0, 30, 60)])
for i, n in enumerate((0, 30, 60)):
    print(f" n={n}: ref {ref[i]}\n      rec {c2[i]}\n      sci {c3[i]}")
# orthonormality on NuHamil-like mesh with kmax=8
k, w = np.polynomial.legendre.leggauss(100)
k = 4 * (k + 1)
w = 4 * w
for hw in (10, 16, 40):
    b = math.sqrt(o.hc**2 / (o.mred * hw))
    for l in (0, 5):
        R = rec(b * k, l, 50) * math.sqrt(b)
        G = (R * w) @ R.T
        print(f"hw={hw} l={l}: ortho dev (n<=50, 100pt mesh) {np.abs(G - np.eye(51)).max():.1e}")
