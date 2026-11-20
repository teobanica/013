# villaincheck.py -- Villain glow of a general Hadamard matrix H: Z = a^* H b / N, with a, b independent periodic
# Villain chains of parameter l on the cycle Z_N.  Checks E|Z|^2 = Tr(C H C H^*)/N^2, C_{jk} = C_N(j-k), against
# exact sampling of the chain (winding n with weight exp(-pi^2 n^2 l/N), then a Gaussian bridge of variance 2/l per bond
# with total 2 pi n), and prints the second moment curve for a few matrices.
import numpy as np, sys
rng=np.random.default_rng(1)
def CN(N,l,r):
    n=np.arange(-30,31)
    return np.sum(np.exp(-((N-r)*n**2+r*(n+1)**2)/l))/np.sum(np.exp(-N*n**2/l))
def Cmat(N,l):
    c=np.array([CN(N,l,r) for r in range(N)]); idx=(np.arange(N)[:,None]-np.arange(N)[None,:])%N
    return c[idx]
def sample(N,l,M):
    n=np.arange(-30,31); w=np.exp(-np.pi**2*n**2*l/N); w/=w.sum()
    wind=rng.choice(n,size=M,p=w)
    t=rng.normal(0,np.sqrt(2/l),(M,N)); t=t-t.mean(1,keepdims=True)+2*np.pi*wind[:,None]/N
    th=np.cumsum(t,1); th=th-th[:,-1:]+2*np.pi*rng.random((M,1))   # theta_N = theta_0, uniform global rotation
    return np.exp(1j*th)
def second(H,l):
    N=len(H); C=Cmat(N,l); return np.real(np.trace(C@H@C@H.conj().T))/N**2
if __name__=='__main__':
    pass
    N=16
    F=np.exp(2j*np.pi*np.outer(range(N),range(N))/N)
    W=np.array([[(-1)**bin(i&j).count('1') for j in range(N)] for i in range(N)])
    H3=np.loadtxt('had163.txt')
    b8=np.exp(2j*np.pi*np.loadtxt('bform.txt')); s=H3@b8; Bf=(4/s)[:,None]*H3*b8[None,:]   # a bistochastic form of had16.3, row sums 4
    for name,H in [('F_16',F),('W_16',W),('had16.3',H3),('bistochastic form of had16.3',Bf)]:
        print(name)
        for l in [0.5,2,8,32,128,1024]:
            a=sample(N,l,100000); b=sample(N,l,100000)
            Z=np.einsum('mj,jk,mk->m',a.conj(),H,b)/N
            print(f"   l={l:6}: formula {second(H,l):9.5f}   simulation {np.mean(abs(Z)**2):9.5f}   E|Z|^4 sim {np.mean(abs(Z)**4):8.4f}")
