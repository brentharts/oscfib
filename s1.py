import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from helper import *
print("SCRIPT 1: f(alpha) and D_q as a function of lambda")
N=2584; v=fib_pot(N); off=np.ones(N-1); boxes=fib_boxes(N)
qs=np.linspace(-3,6,46); c=N//2
lams=[0.5,1.0,2.0,4.0,8.0]; cols=plt.cm.viridis(np.linspace(0,0.9,len(lams)))
fig,ax=plt.subplots(1,3,figsize=(17,5))
widths=[]; D2s=[]
for lam,col in zip(lams,cols):
    Ev,Vec=eigh_tridiagonal(lam*v,off,select='i',select_range=(c-12,c+12))
    As=[]; Fs=[]; Ts=[]
    for j in range(Vec.shape[1]):
        A,F,T=chhabra_jensen(np.abs(Vec[:,j])**2,boxes,qs,N); As.append(A);Fs.append(F);Ts.append(T)
    A=np.mean(As,0); F=np.mean(Fs,0); TAU=np.mean(Ts,0)
    Dq=np.where(np.abs(qs-1)<1e-9,np.nan,TAU/(qs-1))
    ax[0].plot(A,F,"-",color=col,lw=1.8,label=f"lambda={lam}")
    ax[1].plot(qs,Dq,"-",color=col,lw=1.8,label=f"lambda={lam}")
    w=A.max()-A.min(); widths.append(w); D2s.append(TAU[np.argmin(np.abs(qs-2))]/1.0)
    print(f"  lambda={lam}: f(a) width={w:.3f}  alpha in [{A.min():.3f},{A.max():.3f}]  D_2={TAU[np.argmin(np.abs(qs-2))]:.3f}")
ax[0].plot([0.7,1.3],[0.7,1.3],"k:",lw=0.6)
ax[0].set_xlabel("alpha"); ax[0].set_ylabel("f(alpha)"); ax[0].set_title("(1a) f(alpha) broadens with lambda\n(narrows toward point (1,1) as lambda->0)"); ax[0].legend(fontsize=8)
ax[1].axhline(1,color="gray",ls="--",lw=0.8); ax[1].set_xlabel("q"); ax[1].set_ylabel("D_q")
ax[1].set_title("(1b) D_q spreads from 1 (weak) to multifractal (strong)"); ax[1].legend(fontsize=8)
ax[2].plot(lams,widths,"o-",color="crimson",label="f(alpha) width")
ax[2].set_xlabel("lambda"); ax[2].set_ylabel("alpha_max - alpha_min")
ax[2].set_title("(1c) Multifractal width grows with lambda\n-> approaches ballistic/extended limit as lambda->0"); ax[2].legend(fontsize=9)
plt.tight_layout(); plt.savefig("s1_falpha_lambda.png",dpi=130); print("  saved s1_falpha_lambda.png")
