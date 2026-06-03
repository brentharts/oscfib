import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from helper import *
print("SCRIPT 2: spectrum dimension & optimal IDS Holder exponent vs DEGT")
N=4181; v=fib_pot(N); off=np.ones(N-1)
CONST=np.log(1+np.sqrt(2))   # DEGT: dim(sigma)*ln(lambda) -> 0.88137

def box_dim(E):
    E=np.sort(E); W=E[-1]-E[0]
    eps=W*0.6**np.arange(1,16)
    nb=[len(np.unique(np.floor((E-E[0])/e))) for e in eps]
    le,ln=np.log(eps),np.log(nb)
    sl=slice(3,12)                       # intermediate scaling window
    d=-np.polyfit(le[sl],ln[sl],1)[0]
    return d,le,ln,sl

def dos_alpha_min(E,qs=np.linspace(0.5,8,40)):
    E=np.sort(E); W=E[-1]-E[0]
    eps=W*0.55**np.arange(1,14); lnd=[]; an=[]
    for e in eps:
        idx=np.floor((E-E[0])/e).astype(int)
        _,cnt=np.unique(idx,return_counts=True); p=cnt/cnt.sum()
        lnd.append(np.log(e/W)); 
        Z=np.array([np.sum(p**q) for q in qs])
        an.append([np.sum((p**q/Z[k])*np.log(p)) for k,q in enumerate(qs)])
    lnd=np.array(lnd); an=np.array(an); sl=slice(2,11)
    A=np.array([np.polyfit(lnd[sl],an[sl,k],1)[0] for k in range(len(qs))])
    return A.min()                       # optimal Holder exponent of IDS

lams=np.array([2,3,4,6,8,12,16,24,32,48])
dims=[]; amins=[]
fig,ax=plt.subplots(1,3,figsize=(17,5))
for lam in lams:
    E=eigh_tridiagonal(lam*v,off,eigvals_only=True)
    d,le,ln,sl=box_dim(E); dims.append(d)
    amins.append(dos_alpha_min(E))
    if lam in (2,8,32): ax[0].plot(le,ln,"o-",ms=3,label=f"lambda={lam} (d={d:.3f})")
dims=np.array(dims); amins=np.array(amins)
for lam,d in zip(lams,dims): print(f"  lambda={lam:4.0f}: dim(spectrum)={d:.3f}   dim*ln(lambda)={d*np.log(lam):.3f}   IDS Holder a_min={amins[list(lams).index(lam)]:.3f}")
print(f"  DEGT strong-coupling constant ln(1+sqrt2) = {CONST:.5f}")
ax[0].set_xlabel("log eps"); ax[0].set_ylabel("log N_boxes"); ax[0].set_title("(2a) Box-counting of the spectrum"); ax[0].legend(fontsize=8)
ax[1].plot(lams,dims,"o-",color="crimson",label="dim(spectrum)")
ax[1].plot(lams,amins,"s-",color="teal",label="IDS Holder exponent a_min")
ax[1].axhline(1,color="gray",ls="--",lw=0.8)
ax[1].set_xlabel("lambda"); ax[1].set_ylabel("dimension / exponent")
ax[1].set_title("(2b) dim ->1 as lambda->0; both shrink as lambda grows"); ax[1].legend(fontsize=9); ax[1].set_xscale("log")
ax[2].plot(lams,dims*np.log(lams),"o-",color="purple",label="dim(spectrum) * ln(lambda)")
ax[2].axhline(CONST,color="black",ls="--",lw=1.2,label=f"DEGT limit ln(1+sqrt2)={CONST:.4f}")
ax[2].set_xlabel("lambda"); ax[2].set_ylabel("dim * ln(lambda)")
ax[2].set_title("(2c) Strong-coupling: numerics approach the DEGT constant"); ax[2].legend(fontsize=9); ax[2].set_xscale("log")
plt.tight_layout(); plt.savefig("s2_dimension_holder.png",dpi=130); print("  saved s2_dimension_holder.png")
