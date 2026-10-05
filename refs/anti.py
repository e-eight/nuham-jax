import sys, math, numpy as np
sys.path.insert(0,'/home/soham/Projects/nuclear-physics/nuhamil-alt/refs')
sys.path.insert(0,'/home/soham/Projects/nuclear-physics/nuhamil-alt/src')
from moshinsky_reference import bracket
from nuham.angmom.racah import sixj, ninej

def triag(a,b,c): return c<abs(a-b) or c>a+b or (a+b+c)%2

def states(J2,P,T2,N):
    out=[]
    for n12 in range(N//2+1):
        for n3 in range((N-2*n12)//2+1):
            for l12 in range(N-2*n12-2*n3+1):
                l3=N-2*n12-2*n3-l12
                if (-1)**(l12+l3)!=P: continue
                for s12 in (0,1):
                    for j12 in range(abs(l12-s12),l12+s12+1):
                        for t12 in (0,1):
                            if triag(2*t12,1,T2): continue
                            if (-1)**(l12+s12+t12)!=-1: continue
                            for j3 in range(abs(2*l3-1),2*l3+2,2):
                                if triag(2*j12,j3,J2): continue
                                out.append((n12,l12,s12,j12,t12,n3,l3,j3))
    return out

def anti(a,b,J2,T2):
    n12,l12,s12,j12,t12,n3,l3,j3=a; n45,l45,s45,j45,t45,n6,l6,j6=b
    if 2*n12+l12+2*n3+l3!=2*n45+l45+2*n6+l6: return 0.0
    ex=0.0
    for S in range(max(abs(2*s12-1),abs(2*s45-1)),min(2*s12+1,2*s45+1)+1,2):
        for lam in range(max(abs(l12-l3),abs(l45-l6),abs(J2-S)//2),min(l12+l3,l45+l6,(J2+S)//2)+1):
            ex+=(math.sqrt((2*j12+1)*(j3+1)*(2*lam+1)*(S+1))*math.sqrt((2*j45+1)*(j6+1)*(2*lam+1)*(S+1))
                 *ninej(1,j3,2*l3,S,J2,2*lam,2*s12,2*j12,2*l12)*ninej(1,j6,2*l6,S,J2,2*lam,2*s45,2*j45,2*l45)
                 *math.sqrt((2*s12+1)*(2*s45+1))*sixj(1,1,2*s12,1,S,2*s45)
                 *bracket(n12,l12,n3,l3,n45,l45,n6,l6,lam,1/3))
    ex*=(-1)**(s12+t12+s45+t45)*math.sqrt((2*t12+1)*(2*t45+1))*sixj(1,1,2*t12,1,T2,2*t45)
    if a==b: return (1-2*ex)/3
    return -2*ex/3

def run(J2,P,T2,Nmax):
    tot_nas=tot_as=0; per=[]
    for N in range(Nmax+1):
        if (-1)**N!=P: continue
        st=states(J2,P,T2,N)
        if not st: continue
        A=np.array([[anti(a,b,J2,T2) for b in st] for a in st])
        w=np.linalg.eigvalsh(A)
        nas=len(st); nas1=int(np.sum(abs(w-1)<1e-2)); nz=int(np.sum(abs(w)<1e-2))
        sym=abs(A-A.T).max(); idem=abs(A@A-A).max()
        per.append((N,nas,nas1,nz,round(A.trace(),6),sym,idem,nas-nas1-nz))
        tot_nas+=nas; tot_as+=nas1
    return tot_nas,tot_as,per
if __name__=="__main__":
    J2,P,T2,Nmax=map(int,sys.argv[1:5])
    t,a,per=run(J2,P,T2,Nmax)
    print(J2,P,T2,Nmax,"NAS",t,"AS",a)
    for p in per: print(p)
