import sys, numpy as np
from collections import defaultdict
from count3 import lab_counts, orbits
def AS_full(Nmax):
    """F[(N,J2,T2)] = number of antisymmetric 3-nucleon multiplets (all parities, HO total quanta N), N<=Nmax"""
    # sp states
    sps=[]
    for e in range(Nmax+1):
        for l in range(e%2,e+1,2):
            for tj in sorted({abs(2*l-1),2*l+1}):
                for m2 in range(-tj,tj+1,2):
                    for tz in (-1,1): sps.append((e,m2,tz))
    MM=3*(2*Nmax+1)+4  # |2M| bound ~ 3*(2*Nmax+1)
    # dp[k][N][m2+MM][tz+3]
    shape=(4,Nmax+1,2*MM+1,7)
    dp=np.zeros(shape,dtype=np.int64); dp[0,0,MM,3]=1
    for (e,m2,tz) in sps:
        new=dp.copy()
        for k in range(3):
            # add one particle
            src=dp[k,:Nmax+1-e] if e>0 else dp[k]
            tgt=new[k+1,e:]
            sh=np.roll(np.roll(src,m2,axis=1),tz,axis=2)
            tgt+=sh
        dp=new
    c=dp[3]
    F={}
    for N in range(Nmax+1):
        for T2 in (1,3):
            for J2 in range(1,2*Nmax+10,2):
                if J2+1>2*MM-0 : continue
                def g(m2,tz2):
                    if abs(m2)>MM or abs(tz2)>3: return 0
                    return c[N,m2+MM,tz2+3]
                # multiplicity = c(M=J,Tz=T) - c(M=J+1,Tz=T) - c(M=J,Tz=T+1) + c(M=J+1,Tz=T+1)   (2M=J2, 2Tz=T2)
                F[(N,J2,T2)]=g(J2,T2)-g(J2+2,T2)-g(J2,T2+2)+g(J2+2,T2+2)
    return F
def jacobi_AS(F,Nmax):
    G={}
    for E in range(Nmax+1):
        for T2 in (1,3):
            for jr in range(1,2*Nmax+10,2):
                val=F.get((E,jr,T2),0)
                for Ncm in range(0,E//2+1):
                    for Lcm in range(0,E-2*Ncm+1):
                        Ee=E-2*Ncm-Lcm
                        if Ncm==0 and Lcm==0: continue
                        # sum over jrel' with triangle(jrel', Lcm, jr)
                        for jp in range(1,2*Nmax+10,2):
                            if abs(jp-2*Lcm)<=jr<=jp+2*Lcm:
                                val-=G.get((Ee,jp,T2),0)
                G[(E,jr,T2)]=val
    return G
if __name__=="__main__":
    emax,e2max,e3max=map(int,sys.argv[1:4])
    F=AS_full(e3max)
    G=jacobi_AS(F,e3max)
    # sanity: known from log: J=5/2,P=+,T=1/2 Nmax=8: AS=113 (sum over E<=8 even), J=5/2 P- : 79
    for P,ref in ((1,113),(-1,79)):
        s=sum(G[(E,5,1)] for E in range(0,min(e3max,8)+1) if (-1)**E==P) if e3max>=8 else None
        print("check J=5/2 P",P,"AS dim E<=8:",s,"log:",ref)
    o,chan,chanN=lab_counts(emax,e2max,e3max)
    tot_n2=sum(n*n for n in chan.values())
    print("sum n^2 =",tot_n2,"-> double GiB",tot_n2*8/2**30,"single",tot_n2*4/2**30,"half",tot_n2*2/2**30)
    # T coef sizes: per channel, per N: nAS_full(N,J,T,P)*nlab(N,..)
    tc=0; flops=0
    for (J2,P,T2),n in sorted(chan.items()):
        for N in range(e3max+1):
            if (-1)**N!=P: continue
            nl=chanN.get((J2,P,T2,N),0); na=F.get((N,J2,T2),0)
            tc+=na*nl
    print("T-coef total elements (AS x lab, all N blocks):",tc,"= %.2f GiB double"%(tc*8/2**30))
    # GEMM flops per lab channel: sum over (jrel,Ncm,Lcm) blocks of 2 m^2 k + 2 m k^2, m = AS Jacobi dim E<=Emax_jac, k = nlab columns with N>=...
    # use m=sum_{E<=e3max-2Ncm-Lcm} G(E,jrel), k=sum over lab states with N>=? (all N<= e3max connected through E<=N-... ): approximate k = number of lab states with N = E+2Ncm+Lcm for E in range
    fl=0
    for (J2,P,T2),n in chan.items():
        for Lcm in range(0,e3max+1):
            for Ncm in range(0,e3max//2+1):
                if 2*Ncm+Lcm>e3max: continue
                prel=P*(-1)**Lcm
                for jr in range(abs(J2-2*Lcm),J2+2*Lcm+1,2):
                    Es=[E for E in range(0,e3max-2*Ncm-Lcm+1) if (-1)**E==prel]
                    m=sum(G.get((E,jr,T2),0) for E in Es)
                    k=sum(chanN.get((J2,P,T2,E+2*Ncm+Lcm),0) for E in Es)
                    fl+=2*m*m*k+2*m*k*k
    print("dense GEMM flops (upper estimate, T^T V T, one pass over blocks) = %.3e"%fl)
    # biggest channel
    big=max(chan.items(),key=lambda kv:kv[1]); print("largest channel",big)
    # Jacobi dims
    for jr in (1,3,5,9,21):
        for T2 in (1,3):
            print("Jac AS dim jrel2=%d T2=%d"%(jr,T2),[G.get((E,jr,T2),0) for E in range(e3max+1)])
