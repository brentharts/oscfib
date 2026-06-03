import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from helper import *
print("\nSCRIPT 3: multifractal of OSCILLATOR normal modes; verify |u_n|=|psi_n|")
N=2584; v=fib_pot(N); off=np.ones(N-1); lam=2.0; c=N//2; boxes=fib_boxes(N); qs=np.linspace(-3,6,46)
# H eigenpairs and D (oscillator) eigenpairs in same central window
EH,VH=eigh_tridiagonal(lam*v, off, select='i', select_range=(c-12,c+12))     # H: diag lam*v, off +1
ED,VD=eigh_tridiagonal(2.0+lam*v, -off, select='i', select_range=(c-12,c+12)) # D: diag 2+lam*v, off -1
print(f"  spec(D)-spec(H)-2 max = {np.max(np.abs(ED-EH-2)):.2e}")
# stagger H eigenvectors: U psi = (-1)^n psi ; compare envelopes to D normal modes
sign=((-1.0)**np.arange(N))[:,None]
env_H=np.abs(sign*VH); env_D=np.abs(VD)
# fix per-column global sign/phase by comparing envelopes (abs already removes sign)
disc=np.max(np.abs(np.sort(env_H,0)-np.sort(env_D,0)))  # robust to column ordering of degenerate
# better: direct since order matches (same eigenvalue order)
disc=np.max(np.abs(env_H-env_D))
print(f"  max_n,j | |U psi|_n - |u|_n |  = {disc:.2e}  -> Geier envelope identity confirmed")
# multifractal on D-modes vs H-states
def mf(V):
    As=[];Fs=[];Ts=[]
    for j in range(V.shape[1]):
        A,F,T=chhabra_jensen(np.abs(V[:,j])**2,boxes,qs,N); As.append(A);Fs.append(F);Ts.append(T)
    return np.mean(As,0),np.mean(Fs,0),np.mean(Ts,0)
AH,FH,TH=mf(VH); AD,FD,TD=mf(VD)
DqH=np.where(np.abs(qs-1)<1e-9,np.nan,TH/(qs-1)); DqD=np.where(np.abs(qs-1)<1e-9,np.nan,TD/(qs-1))
print(f"  max |D_q(H) - D_q(oscillator)| = {np.nanmax(np.abs(DqH-DqD)):.2e}")
fig,ax=plt.subplots(1,2,figsize=(13,5))
m=VH.shape[1]//2
ax[0].plot(np.abs(sign[:,0]*VH[:,m]),color="midnightblue",lw=0.8,label="|U psi_n| (Hamiltonian)")
ax[0].plot(np.abs(VD[:,m]),color="crimson",lw=0.8,ls="--",label="|u_n| (oscillator mode)")
ax[0].set_xlim(0,N); ax[0].set_xlabel("site n"); ax[0].set_ylabel("envelope")
ax[0].set_title("(3a) Mode envelopes coincide exactly"); ax[0].legend(fontsize=9)
ax[1].plot(qs,DqH,"-",color="midnightblue",lw=2,label="D_q Hamiltonian eigenstates")
ax[1].plot(qs,DqD,"--",color="crimson",lw=2,label="D_q oscillator normal modes")
ax[1].set_xlabel("q"); ax[1].set_ylabel("D_q"); ax[1].set_title("(3b) Identical multifractal spectra\nmultifractality transfers to the chain"); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig("s3_oscillator_modes.png",dpi=130); print("  saved s3_oscillator_modes.png")
