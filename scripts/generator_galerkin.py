#!/usr/bin/env python3
"""Jacobi-Galerkin diagnostic for the linearised generator a d_z S.

Reproduces data/generator_convergence.csv and figures/fig_generator_realparts.pdf
for p=2, c=mu=1, r=3,5.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import beta as Bfun, eval_jacobi
from scipy.linalg import eig
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)


def pars(p,r,c=1.0,mu=1.0):
    A=(2*p*c/(p+1))**(1/(p-1))
    aa=(p*r-1)/((p-1)*(r+1)); bb=r/(r+1)
    ell=(2*mu*r/(r+1))**(1/(r+1))*p/(p-1)*A**(p*(r-1)/(r+1))*Bfun(aa,bb)
    Cq=ell/Bfun(aa,bb)
    gamma=(p+1)/((p-1)*(r+1))
    Cpar=(p*r+2*p+1)/((p-1)*(r+1))
    delta=1/(r+1)
    A0=2*p*p*c/(p+1)
    D=(r+1)*(p-1)**2/(2*p**2)
    return A,aa,bb,ell,Cq,gamma,Cpar,delta,A0,D


def lam_even(p,r,n):
    *_,D=pars(p,r)
    lam0=-(p-1)*(p*r+2*p+1)/(2*p**2*(r+1))
    S=(p*(r+1)+2)/((p-1)*(r+1))
    return lam0+D*n*(S+n)


def lam_odd(p,r,n):
    *_,D=pars(p,r)
    return D*n*(n+p*(r+3)/((p-1)*(r+1)))


def basis_array(p,r,n,y,odd=False,deriv=False):
    _,aa,bb,ell,Cq,gamma,Cpar,delta,A0,D=pars(p,r)
    alpha=Cpar-1
    betaJ=delta if odd else -delta
    x=1-2*y
    P=eval_jacobi(n,alpha,betaJ,x)
    pref=y**gamma*((1-y)**delta if odd else 1.0)
    if not deriv:
        return pref*P
    if n==0:
        dPdy=np.zeros_like(y)
    else:
        dPdx=.5*(n+alpha+betaJ+1)*eval_jacobi(n-1,alpha+1,betaJ+1,x)
        dPdy=-2*dPdx
    dpref=pref*(gamma/y-(delta/(1-y) if odd else 0.0))
    return dpref*P+pref*dPdy


def generator_matrix(p,r,Nm,Q=1400):
    _,aa,bb,ell,Cq,gamma,Cpar,delta,A0,D=pars(p,r)
    xq,wq=leggauss(Q); y=(xq+1)/2; w=wq/2
    E=np.stack([basis_array(p,r,n,y,False,False) for n in range(Nm)],axis=1)
    O=np.stack([basis_array(p,r,n,y,True,False) for n in range(Nm)],axis=1)
    dE=np.stack([basis_array(p,r,n,y,False,True) for n in range(Nm)],axis=1)
    dO=np.stack([basis_array(p,r,n,y,True,True) for n in range(Nm)],axis=1)
    wdz=2*Cq*y**(aa-1)*(1-y)**(bb-1)
    ME=E.T@((w*wdz)[:,None]*E)
    MO=O.T@((w*wdz)[:,None]*O)
    a=A0*y
    KOE=O.T@((w*2*a)[:,None]*dE)
    KEO=E.T@((w*2*a)[:,None]*dO)
    le=np.array([lam_even(p,r,n) for n in range(Nm)])
    lo=np.array([lam_odd(p,r,n) for n in range(Nm)])
    BOE=-np.linalg.solve(MO,KOE@np.diag(le))
    BEO=-np.linalg.solve(ME,KEO@np.diag(lo))
    Z=np.zeros((Nm,Nm))
    return np.block([[Z,BEO],[BOE,Z]])


def main():
    rows=[]
    for r in (3,5):
        for Nm in (6,8,10,12,16,20,24):
            G=generator_matrix(2.0,r,Nm)
            ev=eig(G,right=False)
            pos=np.sort(ev.imag[ev.imag>1e-7])
            rows.append(dict(r=r,Nm=Nm,dim=2*Nm,
                max_abs_Re=float(np.max(np.abs(ev.real))),
                omega1=float(pos[0]),omega2=float(pos[1]),omega3=float(pos[2])))
    df=pd.DataFrame(rows)
    df.to_csv(DATA/"generator_convergence.csv",index=False)

    plt.figure(figsize=(6.0,4.1))
    for r in (3,5):
        d=df[df.r==r]
        plt.semilogy(d["dim"],d["max_abs_Re"],marker="o",label=fr"$r={r}$")
    plt.xlabel("Galerkin matrix dimension")
    plt.ylabel(r"$\max |\Re\lambda|$")
    plt.title("Jacobi--Galerkin generator diagnostic")
    plt.grid(True,which="both",alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(FIG/"fig_generator_realparts.pdf",bbox_inches="tight")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
