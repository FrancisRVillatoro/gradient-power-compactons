"""Reproducible numerics for the gradient-power compacton paper.

Generates
  fig_profiles.pdf      : compacton profiles for p=2, r=1,3,5 (mu=c=1) and edge scaling
  table_spectrum.txt    : FD eigenvalues of S for p=2, r=3 at three resolutions vs exact
  widths.txt            : numerical vs exact half-widths

All formulas are those of Theorems 1 and 3 of the manuscript.
"""
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.special import beta as Beta, betainc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def compacton(p, r, c=1.0, mu=1.0):
    """Return dict with A, Wstar, ell, and callables U(z), b(z), a(z) on |z|<ell."""
    A = (2*p*c/(p+1))**(1/(p-1))
    Wstar = A**p
    D = (r+1)*(p-1)**2/(2*p**2)
    fa = (p*r-1)/((p-1)*(r+1)); fb = r/(r+1)
    ell = (2*mu*r/(r+1))**(1/(r+1))*p/(p-1)*A**(p*(r-1)/(r+1))*Beta(fa, fb)

    def Wp_abs(y):
        W = Wstar*y**(p/(p-1))
        val = (r+1)/(mu*r)*(c*p/(p+1)*W**((p+1)/p) - 0.5*W**2)
        return np.maximum(val, 0.0)**(1.0/(r+1))
    def b_of_y(y): return r*Wp_abs(y)**(r-1)
    # exact z(y) from the regularized incomplete beta: s = ell*I_y(fa,fb), z = ell - s
    def z_of_y(y): return ell*(1.0 - betainc(fa, fb, y))
    ys = np.linspace(0, 1, 200001)
    zs = z_of_y(ys)
    y_of_z = PchipInterpolator(zs[::-1], ys[::-1])
    def U(z):
        y = np.clip(y_of_z(np.clip(np.abs(z), 0, ell)), 0, 1)
        return np.where(np.abs(z) < ell, A*y**(1/(p-1)), 0.0)
    def b(z):
        y = np.clip(y_of_z(np.clip(np.abs(z), 0, ell)), 1e-300, 1)
        return b_of_y(y)
    def a(z): return p*U(z)**(p-1)
    return dict(A=A, Wstar=Wstar, ell=ell, D=D, fa=fa, fb=fb, U=U, b=b, a=a, y_of_z=y_of_z)

# ---------------- Figure: profiles and edge scaling ----------------
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
out = []
for r, ls in [(1, "-"), (3, "--"), (5, ":")]:
    cp = compacton(2.0, float(r))
    z = np.linspace(-cp["ell"], cp["ell"], 2001)
    ax[0].plot(z, cp["U"](z), ls, color="k", lw=1.4, label=rf"$r={r}$")
    # edge scaling: distance d to the right edge
    d = np.logspace(-6, -0.5, 200)
    Ud = cp["U"](cp["ell"] - d)
    alpha = (r+1)/(2*r-1)   # p=2
    ax[1].loglog(d, Ud, ls, color="k", lw=1.4, label=rf"$r={r}$, $\alpha={alpha:.3g}$")
    slope = np.polyfit(np.log(d[:80]), np.log(Ud[:80]), 1)[0]
    out.append((r, cp["ell"], slope, alpha))
ax[0].set_xlabel(r"$z$"); ax[0].set_ylabel(r"$U(z)$"); ax[0].legend(frameon=False)
ax[0].set_title(r"(a) $p=2$, $c=\mu=1$")
ax[1].set_xlabel(r"$d=\ell-|z|$"); ax[1].set_ylabel(r"$U$"); ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title(r"(b) edge behaviour $U\sim C_e d^{\alpha}$")
fig.tight_layout(); fig.savefig("fig_profiles.pdf")
with open("widths.txt", "w") as f:
    for r, ell, slope, alpha in out:
        f.write(f"p=2 r={r}: ell={ell:.6f}  L={2*ell:.6f}  fitted edge exponent={slope:.4f}  exact alpha={alpha:.4f}\n")
print(open("widths.txt").read())

# ---------------- Spectrum of S by conservative finite differences ----------------
def fd_spectrum(p, r, N, k=8, c=1.0, mu=1.0):
    """Conservative second-order FD with exact homogeneous Dirichlet endpoint values.

    N is the total number of grid points including the two prescribed endpoints.
    The matrix is assembled only on the N-2 interior unknowns.
    """
    cp = compacton(p, r, c, mu)
    ell = cp["ell"]
    zg = np.linspace(-ell, ell, N); h = zg[1]-zg[0]
    zi = zg[1:-1]
    U = cp["U"](zi)
    V = c/(p*np.maximum(U, 1e-300)**(p-1)) - 1.0
    zm = 0.5*(zg[1:]+zg[:-1]); bm = cp["b"](zm)

    # Interior row i uses midpoint coefficients bm[i] and bm[i+1] in full-grid indexing.
    bleft = bm[:-1]          # length N-2
    bright = bm[1:]          # length N-2
    main = mu*(bleft+bright)/h**2 + V
    off = -mu*bm[1:-1]/h**2 # between adjacent interior unknowns
    S = diags([off, main, off], [-1, 0, 1], format="csc")

    vals, vecs = eigsh(S, k=k, sigma=-1.0, which="LM")
    idx = np.argsort(vals); vals = vals[idx]; vecs = vecs[:, idx]
    par = []
    for j in range(k):
        v = vecs[:, j]
        par.append("even" if np.dot(v, v[::-1]) > 0 else "odd")
    return vals, par

p, r = 2.0, 3.0
exact_even = [-(p-1)*(p*r+2*p+1)/(2*p**2*(r+1)) + (p-1)*(p*(r+1)+2)/(2*p**2)*n + (r+1)*(p-1)**2/(2*p**2)*n**2 for n in range(4)]
exact_odd  = [(p-1)*(r+3)/(2*p)*n + (r+1)*(p-1)**2/(2*p**2)*n**2 for n in range(4)]
lines = ["p=2, r=3, c=mu=1. Conservative second-order FD on uniform grid, Dirichlet at the edges.",
         "exact even: " + ", ".join(f"{v:.6f}" for v in exact_even),
         "exact odd : " + ", ".join(f"{v:.6f}" for v in exact_odd)]
for N in (6001, 12001, 24001, 48001):
    vals, par = fd_spectrum(p, r, N)
    lines.append(f"N={N:6d}: " + "  ".join(f"{v:+.5f}({q[0]})" for v, q in zip(vals, par)))
open("table_spectrum.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
