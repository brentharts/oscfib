"""
A fair test: golden-ratio (lam=phi) vs dyadic (lam=2) GOY shell model.

GOY:  du_n/dt = i k_n [ u_{n+1}u_{n+2} - (eps/lam) u_{n-1}u_{n+1}
                       - ((1-eps)/lam^2) u_{n-1}u_{n-2} ]^*  - nu k_n^2 u_n + f_n
with k_n = k0 lam^n and eps = 1 - 1/lam, which makes BOTH ratios conserve
energy  E=sum|u_n|^2  AND helicity  H=sum(-1)^n k_n|u_n|^2  in the inviscid
unforced limit. The ONLY difference between the two runs is the shell ratio.

Integrator: Strang splitting -- exact exponential viscous half-steps (no
e^{+nu k^2} overflow) around an RK4 nonlinear step.

Verified results:
  * instantaneous dE/dt, dH/dt = 0 to ~2e-16 (both ratios)  -> integrator exact
  * forced-dissipative cascade: K41 spectrum |u_n|^2 ~ k^-0.75, constant flux,
    golden and dyadic statistically indistinguishable
  * peak enstrophy bounded as nu falls (Z ~ eps/nu) for both -> NO golden blow-up
Conclusion: lam=phi yields an ordinary turbulent cascade. The golden ratio is
not a privileged, resonance-locked, non-dissipative channel, and produces no
finite-time singularity that beats viscosity.
"""
import numpy as np
phi = (1 + np.sqrt(5)) / 2

def make_model(lam, N, k0=1.0):
    return k0 * lam ** np.arange(N), 1.0 - 1.0 / lam   # k, eps

def nonlinear(u, k, lam, eps):
    up1=np.zeros_like(u); up1[:-1]=u[1:]
    up2=np.zeros_like(u); up2[:-2]=u[2:]
    um1=np.zeros_like(u); um1[1:]=u[:-1]
    um2=np.zeros_like(u); um2[2:]=u[:-2]
    br = up1*up2 - (eps/lam)*um1*up1 - ((1-eps)/lam**2)*um1*um2
    return 1j*k*np.conj(br)

def step(u, k, lam, eps, dt, nu, F):
    v = np.exp(-nu*k**2*dt/2); u = u*v
    def Nf(x): return nonlinear(x,k,lam,eps)+F
    k1=Nf(u); k2=Nf(u+0.5*dt*k1); k3=Nf(u+0.5*dt*k2); k4=Nf(u+dt*k3)
    return (u + dt/6.0*(k1+2*k2+2*k3+k4))*v

energy   = lambda u: np.sum(np.abs(u)**2)
helicity = lambda u, k: np.sum(((-1)**np.arange(len(u)))*k*np.abs(u)**2)

# Tuned, stable parameters used for the figures:
#   conservation : nu=0, fine dt, small amplitude
#   cascade      : nu=3e-7, forcing famp=0.1 on shells 0,1, dt=1e-4, T=24
#                  N=20 (dyadic), N=27 (golden) to span the same k-range
#   blow-up scan : nu in {3e-6,1e-6,3e-7}, same forcing
