#!/usr/bin/env python3
"""Regenerate the manuscript figures from exact formulas and archived CSV data."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import beta, betaincinv

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"; FIG=ROOT/"figures"
FIG.mkdir(exist_ok=True)
L=60.0; p=2.0; c=mu=1.0

def pars(r):
    A=(2*p*c/(p+1))**(1/(p-1))
    aa=(p*r-1)/((p-1)*(r+1)); bb=r/(r+1)
    ell=(2*mu*r/(r+1))**(1/(r+1))*p/(p-1)*A**(p*(r-1)/(r+1))*beta(aa,bb)
    return A,aa,bb,ell

def exact(x,r,x0=0.0):
    A,aa,bb,ell=pars(r)
    z=(x-x0+L/2)%L-L/2
    u=np.zeros_like(x); m=np.abs(z)<=ell
    q=1-np.abs(z[m])/ell
    if r==1:
        u[m]=A*np.cos(z[m]/4.0)**2
    else:
        u[m]=A*betaincinv(aa,bb,np.clip(q,0,1))
    return u

def fig_profiles():
    fig,ax=plt.subplots(1,2,figsize=(9.2,3.4))
    styles={1:"-",3:"--",5:":"}
    for r in (1,3,5):
        A,aa,bb,ell=pars(r)
        z=np.linspace(-ell,ell,2501); U=exact(z,r)
        ax[0].plot(z,U,styles[r],lw=1.5,label=fr"$r={r}$")
        d=np.logspace(-6,-.5,220); Ud=exact(ell-d,r)
        alpha=(r+1)/(2*r-1)
        ax[1].loglog(d,Ud,styles[r],lw=1.5,label=fr"$r={r}$, $\alpha={alpha:.3g}$")
    ax[0].set(xlabel=r"$z$",ylabel=r"$U(z)$",title=r"(a) $p=2$, $c=\mu=1$"); ax[0].legend(frameon=False)
    ax[1].set(xlabel=r"$d=\ell-|z|$",ylabel=r"$U$",title=r"(b) edge behaviour $U\sim C_e d^\alpha$"); ax[1].legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(FIG/"fig_profiles.pdf",bbox_inches="tight"); plt.close(fig)

def fig_time_phase():
    d3=pd.read_csv(DATA/"history_padded_r3_N384.csv"); d5=pd.read_csv(DATA/"history_padded_r5_N384.csv")
    plt.figure(figsize=(6.0,4.0))
    for r,d in [(3,d3),(5,d5)]: plt.plot(d.t,d.relL2_dynamic,marker="o",ms=3,label=fr"$r={r}$")
    plt.xlabel("time"); plt.ylabel("phase-aligned relative $L^2$ error"); plt.title("Padded Fourier--Galerkin propagation")
    plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"fig_time_error.pdf",bbox_inches="tight"); plt.close()
    plt.figure(figsize=(6.0,4.0))
    for r,d in [(3,d3),(5,d5)]: plt.plot(d.t,d.phase_error,marker="o",ms=3,label=fr"$r={r}$")
    plt.axhline(0,lw=.8); plt.xlabel("time"); plt.ylabel(r"phase correction $\delta$"); plt.title("Phase drift")
    plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"fig_phase_error.pdf",bbox_inches="tight"); plt.close()

def fig_fourier():
    N=2**18; x=np.linspace(-L/2,L/2,N,endpoint=False)
    for r in (3,5):
        u=exact(x,r); uh=np.fft.rfft(u)/N; k=np.arange(len(uh)); amp=np.abs(uh)
        sel=(k>=500)&(k<=5000)&(amp>0); slope,_=np.polyfit(np.log(k[sel]),np.log(amp[sel]),1)
        alpha=(r+1)/(2*r-1); theory=-(1+alpha)
        kk=k[1:]; aa=amp[1:]; sel2=(kk>=500)&(kk<=5000)&(aa>0)
        logC=np.mean(np.log(aa[sel2])-theory*np.log(kk[sel2])); ref=np.exp(logC)*kk**theory
        plt.figure(figsize=(6.0,4.2)); plt.loglog(kk,aa,lw=.9,label=fr"$r={r}$ exact compacton")
        plt.loglog(kk,ref,"--",lw=1.1,label=fr"edge law $k^{{{theory:.3g}}}$")
        plt.xlim(20,20000); plt.ylim(1e-12,2e-2); plt.xlabel(r"Fourier mode $k$"); plt.ylabel(r"$|\widehat U_k|$")
        plt.title(fr"$r={r}$: fitted slope {slope:.5f}"); plt.grid(True,which="both",alpha=.22); plt.legend(); plt.tight_layout()
        plt.savefig(FIG/f"fig_fourier_decay_r{r}.pdf",bbox_inches="tight"); plt.close()

def fig_generator():
    d=pd.read_csv(DATA/"generator_convergence.csv")
    plt.figure(figsize=(6.0,4.1))
    for r in (3,5):
        q=d[d.r==r]; plt.semilogy(q.dim,q.max_abs_Re,marker="o",label=fr"$r={r}$")
    plt.xlabel("Galerkin matrix dimension"); plt.ylabel(r"$\max |\Re\lambda|$"); plt.title("Jacobi--Galerkin generator diagnostic")
    plt.grid(True,which="both",alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"fig_generator_realparts.pdf",bbox_inches="tight"); plt.close()

if __name__=="__main__":
    fig_profiles(); fig_time_phase(); fig_fourier(); fig_generator()
    print("Figures written to",FIG)
