import numpy as np
phi=(1+np.sqrt(5))/2; alpha=1/phi
def fib_word(nmin):
    s="A"
    while len(s)<nmin: s="".join("AB" if c=="A" else "A" for c in s)
    return s[:nmin]
def fib_pot(N): return np.array([1.0 if c=="A" else 0.0 for c in fib_word(N)])
def fib_boxes(N): return np.array([b for b in [2,3,5,8,13,21,34,55,89,144,233,377,610] if b<=N//8])
def chhabra_jensen(weights, boxes, qs, N):
    """Direct CJ multifractal of a probability vector (eigenstate |psi|^2 or DOS)."""
    mu=np.asarray(weights,float); mu=mu/mu.sum()
    lnd=[]; an=[]; fn=[]; lz=[]
    for l in boxes:
        nb=N//l
        p=np.array([mu[i*l:(i+1)*l].sum() for i in range(nb)])
        p=p[p>1e-300]
        lnd.append(np.log(l/N))
        Z=np.array([np.sum(p**q) for q in qs])
        an.append([np.sum((p**q/Z[k])*np.log(p)) for k,q in enumerate(qs)])
        fn.append([np.sum((p**q/Z[k])*np.log(p**q/Z[k])) for k,q in enumerate(qs)])
        lz.append(np.log(Z))
    lnd=np.array(lnd); an=np.array(an); fn=np.array(fn); lz=np.array(lz)
    A=np.array([np.polyfit(lnd,an[:,k],1)[0] for k in range(len(qs))])
    F=np.array([np.polyfit(lnd,fn[:,k],1)[0] for k in range(len(qs))])
    TAU=np.array([np.polyfit(lnd,lz[:,k],1)[0] for k in range(len(qs))])
    return A,F,TAU
