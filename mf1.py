import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
phi=(1+np.sqrt(5))/2; alpha=1/phi
def fib_word(nmin):
    s="A"
    while len(s)<nmin: s="".join("AB" if c=="A" else "A" for c in s)
    return s[:nmin]
def fib_pot(N): 
    return np.array([1.0 if c=="A" else 0.0 for c in fib_word(N)])

# ---------- PART 1: gap-labeling fan across lambda ----------
print("PART 1: gap-labeling spectrum across lambda")
Nfan=610; v=fib_pot(Nfan); off=np.ones(Nfan-1)
lams=np.linspace(0.2,8,90)
fig,ax=plt.subplots(1,2,figsize=(13,5))
for lam in lams:
    E=eigh_tridiagonal(lam*v,off,eigvals_only=True)
    ax[0].plot(np.full_like(E,lam),E,",",color="black",alpha=0.25,ms=0.5)
ax[0].set_xlabel("pinning / coupling  lambda"); ax[0].set_ylabel("E")
ax[0].set_title("(1a) Spectrum vs lambda: Cantor gaps widen")

# IDS at two lambdas -> gap-label HEIGHTS are lambda-independent (topological), in Z+phi Z
for lam,col in [(2.0,"midnightblue"),(5.0,"crimson")]:
    E=np.sort(eigh_tridiagonal(lam*v,off,eigvals_only=True))
    nE=np.arange(1,Nfan+1)/Nfan
    ax[1].plot(E,nE,lw=0.8,color=col,label=f"lambda={lam}")
for m in range(-8,9):
    h=(m*alpha)%1.0
    ax[1].axhline(h,color="gray",ls=":",lw=0.5)
ax[1].set_xlabel("E"); ax[1].set_ylabel("N(E)")
ax[1].set_title("(1b) IDS plateaus pinned to frac(m/phi)\nsame heights for different lambda = topological gap labels")
ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig("mf_gaplabel.png",dpi=130); print("  saved mf_gaplabel.png")

# ---------- PART 2: multifractal f(alpha), D_q via Chhabra-Jensen ----------
print("\nPART 2: multifractal analysis of critical eigenstates (lambda=2)")
N=2584; v=fib_pot(N); off=np.ones(N-1); lam=2.0
c=N//2
# central window of critical states
Evar, Vec = eigh_tridiagonal(lam*v, off, select='i', select_range=(c-15,c+15))
boxes=np.array([b for b in [2,3,5,8,13,21,34,55,89,144,233,377] if b<=N//8])
qs=np.linspace(-3,6,46)

def chhabra_jensen(psi):
    mu=np.abs(psi)**2; mu/=mu.sum()
    lnd=[]; an=[]; fn=[]; lz=[]
    for l in boxes:
        nb=N//l
        p=np.array([mu[i*l:(i+1)*l].sum() for i in range(nb)])
        p=p[p>1e-300]
        lnd.append(np.log(l/N))
        Z=np.array([np.sum(p**q) for q in qs])
        an_q=[]; fn_q=[]
        for k,q in enumerate(qs):
            w=p**q/Z[k]
            an_q.append(np.sum(w*np.log(p)))
            fn_q.append(np.sum(w*np.log(w)))
        an.append(an_q); fn.append(fn_q); lz.append(np.log(Z))
    lnd=np.array(lnd); an=np.array(an); fn=np.array(fn); lz=np.array(lz)
    A=np.array([np.polyfit(lnd,an[:,k],1)[0] for k in range(len(qs))])   # alpha(q)
    F=np.array([np.polyfit(lnd,fn[:,k],1)[0] for k in range(len(qs))])   # f(q)
    TAU=np.array([np.polyfit(lnd,lz[:,k],1)[0] for k in range(len(qs))]) # tau(q)
    return A,F,TAU

# average over the central critical states
As=[]; Fs=[]; Ts=[]
for j in range(Vec.shape[1]):
    A,F,T=chhabra_jensen(Vec[:,j]); As.append(A); Fs.append(F); Ts.append(T)
A=np.mean(As,0); F=np.mean(Fs,0); TAU=np.mean(Ts,0)
Dq=np.where(np.abs(qs-1)<1e-9, np.nan, TAU/(qs-1))
def Dat(q): return TAU[np.argmin(np.abs(qs-q))]/(q-1) if abs(q-1)>1e-6 else np.nan
print(f"  D_0 (support) = {Dat(0):.3f}   (extended->1, localized->0)")
print(f"  D_2 (correlation/IPR) = {Dat(2):.3f}")
print(f"  alpha range of f(alpha): [{A.min():.3f}, {A.max():.3f}]  width={A.max()-A.min():.3f}")
print(f"  f(alpha) max = {F.max():.3f} (should ~ D_0)")

fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].plot(qs,Dq,"o-",ms=3,color="crimson",label="Fibonacci critical states")
ax[0].axhline(1,color="green",ls="--",lw=1,label="extended (D_q=1)")
ax[0].axhline(0,color="gray",ls="--",lw=1,label="localized (D_q=0, q>0)")
ax[0].set_xlabel("q"); ax[0].set_ylabel("D_q"); ax[0].set_title("(2a) Generalized dimensions D_q\nvaries with q => genuinely multifractal")
ax[0].legend(fontsize=9)
ax[1].plot(A,F,"o-",ms=3,color="midnightblue")
ax[1].plot([A.min(),A.max()],[A.min(),A.max()],"k:",lw=0.8)
ax[1].set_xlabel("alpha"); ax[1].set_ylabel("f(alpha)")
ax[1].set_title("(2b) Singularity spectrum f(alpha)\nbroad curve = multifractal critical states")
plt.tight_layout(); plt.savefig("mf_spectrum.png",dpi=130); print("  saved mf_spectrum.png")
