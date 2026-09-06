import numpy as np
from scipy.special import beta as Bfun, betaincinv
from scipy.optimize import minimize_scalar
from scipy.fft import fft, ifft
import time, argparse, csv, os

L=60.0; P=2.0; C=1.0; MU=1.0

def pars(r):
    A=(2*P*C/(P+1))**(1/(P-1))
    aa=(P*r-1)/((P-1)*(r+1)); bb=r/(r+1)
    ell=(2*MU*r/(r+1))**(1/(r+1))*P/(P-1)*A**(P*(r-1)/(r+1))*Bfun(aa,bb)
    return A,aa,bb,ell

def exact(x,r,x0=0.0):
    A,aa,bb,ell=pars(r)
    z=(x-x0+L/2)%L-L/2
    u=np.zeros_like(x)
    m=np.abs(z)<=ell
    q=1-np.abs(z[m])/ell
    if r==1:
        u[m]=A*np.cos(z[m]/4.0)**2
    else:
        u[m]=A*betaincinv(aa,bb,np.clip(q,0,1))
    return u

class PaddedGalerkin:
    def __init__(self,N,pad=3.0):
        assert N%2==0
        self.N=N; self.dx=L/N
        self.K=N//3  # retained modes |j|<=K
        M=int(np.ceil(pad*N))
        if M%2: M+=1
        self.M=M; self.pad=M/N
        self.k=2*np.pi*np.fft.fftfreq(N,d=self.dx)
        self.keep=np.zeros(N,dtype=bool)
        self.keep[:self.K+1]=True; self.keep[-self.K:]=True
    def project_hat(self,fh):
        out=np.zeros_like(fh)
        out[self.keep]=fh[self.keep]
        return out
    def project(self,f):
        return ifft(self.project_hat(fft(f))).real
    def to_pad_from_hat(self,fh):
        # fh uses numpy FFT normalization on N samples; only retained modes matter.
        M,N,K=self.M,self.N,self.K
        ph=np.zeros(M,dtype=complex)
        ph[:K+1]=(M/N)*fh[:K+1]
        ph[-K:]=(M/N)*fh[-K:]
        return ifft(ph).real
    def project_pad_to_hat(self,fM):
        M,N,K=self.M,self.N,self.K
        hM=fft(fM)
        out=np.zeros(N,dtype=complex)
        out[:K+1]=(N/M)*hM[:K+1]
        out[-K:]=(N/M)*hM[-K:]
        return out
    def rhs(self,u,r):
        # State is in retained Fourier space. Oversample it, evaluate F=|u|u,
        # project F back to retained modes, form q=F_x, oversample q, evaluate
        # Psi_r(q), and project back again. This prevents aliases from q^r
        # contaminating the retained band when padding is sufficient.
        uh=self.project_hat(fft(u))
        uM=self.to_pad_from_hat(uh)
        FM=np.abs(uM)*uM
        Fh=self.project_pad_to_hat(FM)
        qh=1j*self.k*Fh
        qM=self.to_pad_from_hat(qh)
        GM=np.abs(qM)**(r-1)*qM
        Gh=self.project_pad_to_hat(GM)
        rh=-1j*self.k*Fh + (self.k**2)*Gh
        rh[~self.keep]=0
        return ifft(rh).real

def rk4(u,dt,op,r):
    k1=op.rhs(u,r)
    k2=op.rhs(u+.5*dt*k1,r)
    k3=op.rhs(u+.5*dt*k2,r)
    k4=op.rhs(u+dt*k3,r)
    v=u+dt*(k1+2*k2+2*k3+k4)/6
    # roundoff-only projection; mathematically all RK stages are already retained.
    return op.project(v)

def invs(u,dx):
    return dx*np.sum(u), dx*np.sum(np.abs(u)**3)/3.0

def projected_exact(x,r,x0,op):
    return op.project(exact(x,r,x0))

def phase_dyn(u,x,r,t,op,bound=.08):
    # compare only within the same retained Galerkin space
    den0=np.linalg.norm(u)
    def obj(d):
        v=projected_exact(x,r,t+d,op)
        return np.vdot(u-v,u-v).real
    res=minimize_scalar(obj,bounds=(-bound,bound),method='bounded',options={'xatol':2e-11})
    return float(res.x)

def diagnostics(u,x,r,t,op,u0,M0,Q0):
    d=phase_dyn(u,x,r,t,op)
    up=projected_exact(x,r,t+d,op)
    ue=exact(x,r,t+d)
    M,Q=invs(u,op.dx)
    dynL2=np.linalg.norm(u-up)/np.linalg.norm(up)
    totalL2=np.linalg.norm(u-ue)/np.linalg.norm(ue)
    reprL2=np.linalg.norm(up-ue)/np.linalg.norm(ue)
    return dict(t=t,phase_error=d,
                relL2_dynamic=dynL2, relL2_total=totalL2, relL2_representation=reprL2,
                relLinf_dynamic=np.max(np.abs(u-up))/np.max(np.abs(up)),
                dM=(M-M0)/M0,dQ=(Q-Q0)/Q0,
                min_u=float(u.min()),max_u=float(u.max()))

def run(r,N,T,dt0,pad=3.0,times=None):
    x=np.linspace(-L/2,L/2,N,endpoint=False)
    op=PaddedGalerkin(N,pad)
    ue0=exact(x,r)
    u0=op.project(ue0)
    u=u0.copy(); M0,Q0=invs(u0,op.dx)
    n=int(np.ceil(T/dt0)); dt=T/n
    if times is None: times=[T]
    times=sorted(times); out=[]; ir=0; tic=time.time(); stable=True
    for j in range(n+1):
        t=j*dt
        while ir<len(times) and t+.5*dt>=times[ir]:
            out.append(diagnostics(u,x,r,t,op,u0,M0,Q0)); ir+=1
        if j==n: break
        u=rk4(u,dt,op,r)
        if not np.all(np.isfinite(u)) or np.max(np.abs(u))>100:
            stable=False; break
    meta=dict(r=r,N=N,K=op.K,Mpad=op.M,pad_actual=op.pad,dx=op.dx,dt=dt,T=t,nsteps=j,
              runtime=time.time()-tic,stable=stable)
    return out,meta

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--r',type=int,required=True);ap.add_argument('--N',type=int,required=True)
    ap.add_argument('--T',type=float,default=.1);ap.add_argument('--pad',type=float,default=3.0)
    ap.add_argument('--dtfactor',type=float,default=1.0);ap.add_argument('--nout',type=int,default=1)
    ap.add_argument('--out',type=str,default='')
    a=ap.parse_args()
    dt0=5e-5*(256/a.N)**3*a.dtfactor
    times=np.linspace(a.T/a.nout,a.T,a.nout)
    rows,meta=run(a.r,a.N,a.T,dt0,a.pad,times.tolist())
    print(meta)
    for row in rows: print(row)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
        with open(a.out,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(meta.keys())+list(rows[0].keys()))
            w.writeheader()
            for row in rows:w.writerow({**meta,**row})
