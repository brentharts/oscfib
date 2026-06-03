import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
phi=(1+np.sqrt(5))/2; alpha=1/phi

def fib_word(nmin):
    s="A"
    while len(s)<nmin: s="".join("AB" if c=="A" else "A" for c in s)
    return s[:nmin]

def matrices(N,lam):
    w=fib_word(N); v=np.array([1.0 if c=="A" else 0.0 for c in w])
    H_diag=lam*v; H_off=np.ones(N-1)             # H: +1 hopping, lam*v pinning
    D_diag=2.0+lam*v; D_off=-np.ones(N-1)         # D: -1 spring, 2+lam*v
    return H_diag,H_off,D_diag,D_off,v

# ---- TEST A: verify Geier's exact identity D = U(H+2I)U^{-1}, spec(D)=spec(H)+2
print("TEST A: Geier operator identity")
N,lam=377,3.0
Hd,Ho,Dd,Do,v=matrices(N,lam)
H=np.diag(Hd)+np.diag(Ho,1)+np.diag(Ho,-1)
D=np.diag(Dd)+np.diag(Do,1)+np.diag(Do,-1)
U=np.diag((-1.0)**np.arange(N))
print(f"  || D - U(H+2I)U^-1 ||_max = {np.max(np.abs(D - U@(H+2*np.eye(N))@U)):.2e}")
EH=np.sort(eigh_tridiagonal(Hd,Ho,eigvals_only=True))
ED=np.sort(eigh_tridiagonal(Dd,Do,eigvals_only=True))
print(f"  max | spec(D) - (spec(H)+2) |  = {np.max(np.abs(ED-(EH+2))):.2e}   --> identity CONFIRMED")
Om=np.sqrt(EH+2.0)
print(f"  normal-mode frequencies Omega bounded in [{Om.min():.3f}, {Om.max():.3f}] (Cantor set)")

# ---- TEST B: the REAL arithmetic -- golden gap labeling (Bellissard et al.)
# IDS values on spectral gaps should lie in { frac(m*alpha) : m in Z }, alpha=1/phi
print("\nTEST B: gap labeling -- is the spectrum's arithmetic golden or prime?")
Nbig=610; Hd,Ho,Dd,Do,v=matrices(Nbig,lam)
E=np.sort(eigh_tridiagonal(Hd,Ho,eigvals_only=True))
gaps=np.diff(E)
order=np.argsort(gaps)[::-1][:8]              # 8 widest gaps
labels=(np.arange(1,Nbig)[order])/Nbig         # IDS height at each gap = k/N
ms=np.arange(-40,41)
print("  gap-IDS height   nearest frac(m/phi)   |residual|   m")
for L in np.sort(labels):
    cand=(ms*alpha)%1.0
    j=np.argmin(np.abs(cand-L)); 
    print(f"    {L:.4f}            {cand[j]:.4f}            {abs(cand[j]-L):.4f}     {ms[j]:+d}")

# ---- TEST C: can these oscillators BE the primon gas / zeta zeros?
def primes(n):
    s=np.ones(n+1,bool); s[:2]=False
    for i in range(2,int(n**.5)+1):
        if s[i]: s[i*i::i]=False
    return np.nonzero(s)[0]
p=primes(2000); ln_p=np.log(p[:60])            # primon energies E_p = ln p
zeros=np.array([14.1347,21.0220,25.0109,30.4249,32.9351,37.5862,40.9187,
                43.3271,48.0052,49.7738,52.9703,56.4462,59.3470,60.8318])
print("\nTEST C: chain frequencies vs primon energies vs zeta zeros")
print(f"  chain Omega_j   : bounded, max = {Om.max():.3f}")
print(f"  primon ln(p_j)  : unbounded, ln(p_60)={ln_p[-1]:.3f}, grows ~ ln(j ln j)")
print(f"  zeta zeros t_j  : unbounded, t_14={zeros[-1]:.3f}, grows ~ 2*pi*j/ln j")
print("  -> bounded set cannot be in energy-preserving bijection with an unbounded one")

# ---- figure
fig,ax=plt.subplots(1,3,figsize=(16,4.6))
nE=np.arange(1,len(E)+1)/len(E)
ax[0].plot(E,nE,lw=0.7,color="crimson")
for L in labels: ax[0].axhline(L,color="gray",ls=":",lw=0.6)
ax[0].set_title("(B) IDS staircase: gap heights = frac(m/phi)\n(golden arithmetic, NOT primes)")
ax[0].set_xlabel("E"); ax[0].set_ylabel("N(E)")

ax[1].plot(np.arange(len(Om)),np.sort(Om),".",ms=2,color="midnightblue",label="chain Omega_j (bounded)")
ax[1].plot(np.arange(len(ln_p)),ln_p,"o-",ms=3,color="green",label="primon ln(p_j) (unbounded)")
ax[1].set_title("(C) Geier oscillators are NOT primons")
ax[1].set_xlabel("index j"); ax[1].set_ylabel("energy / frequency"); ax[1].legend(fontsize=8)
ax[1].set_xlim(0,60)

ax[2].plot(np.arange(1,len(zeros)+1),zeros,"s-",ms=4,color="purple",label="Im(zeta zeros)")
ax[2].axhspan(Om.min(),Om.max(),color="midnightblue",alpha=0.2,label=f"chain Omega range [{Om.min():.1f},{Om.max():.1f}]")
ax[2].set_title("(C) zeta zeros live far outside\nthe chain's bounded spectrum")
ax[2].set_xlabel("zero index"); ax[2].set_ylabel("value"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fib_osc.png",dpi=130); print("\nsaved fib_osc.png")
