"""Independent count of the 3N lab orbits, channels and me3j values (spec 5d): usage `python refs/count3.py EMAX E2MAX E3MAX [file]`.

With any fourth argument it also prints `file values`, the number of values of the me3j file (456 320 for 6 12 6).
Throwaway oracle script, not part of the package.
"""
import sys, itertools
from collections import defaultdict
def orbits(emax,lmax=None):
    lmax = emax if lmax is None else lmax
    o=[]
    for e in range(emax+1):
        for l in range(e%2, min(e,lmax)+1, 2):
            n=(e-l)//2
            for tj in sorted({abs(2*l-1),2*l+1}):
                o.append((e,n,l,tj))
    return o
def lab_counts(emax,e2max,e3max):
    o=orbits(emax); no=len(o)
    chan=defaultdict(int)       # (J2,P,T2) -> nstates
    chanN=defaultdict(int)      # (J2,P,T2,N) -> nstates
    tuples=0
    for i1 in range(no):
      for i2 in range(i1+1):
        for i3 in range(i2+1):
          e1,e2,e3=o[i1][0],o[i2][0],o[i3][0]
          if e1+e2>e2max or e1+e3>e2max or e2+e3>e2max or e1+e2+e3>e3max: continue
          P=(-1)**(o[i1][2]+o[i2][2]+o[i3][2])
          j1,j2,j3=o[i1][3],o[i2][3],o[i3][3]
          for J12 in range(abs(j1-j2)//2,(j1+j2)//2+1):
            for J2 in range(abs(2*J12-j3),2*J12+j3+1,2):
              for T12 in (0,1):
                if i1==i2 and (J12+T12)%2==0: continue
                for T2 in (1,3):
                  if T2>2*T12+1: continue
                  chan[(J2,P,T2)]+=1; chanN[(J2,P,T2,e1+e2+e3)]+=1
    return o,chan,chanN
def file_tuples(emax,e2max,e3max):
    o=orbits(emax); no=len(o); tot=0
    # per bra-triple: list of (j1,j2,j3,P, e) ; count pairs bra>=ket
    trip=[]
    for i1 in range(no):
      for i2 in range(i1+1):
        for i3 in range(i2+1):
          e1,e2,e3=o[i1][0],o[i2][0],o[i3][0]
          if e1+e2>e2max or e1+e3>e2max or e2+e3>e2max or e1+e2+e3>e3max: continue
          trip.append((i1,i2,i3))
    # ket constraints: i4<=i1; i5<=i4 (if i1==i4: i5<=i2); i6<=i5 (if i1==i4&i2==i5: i6<=i3)  => lexicographic ket<=bra
    def nJ(t,u):
        a,b,c=[o[x][3] for x in t]; d,e,f=[o[x][3] for x in u]
        n=0
        for J12 in range(abs(a-b)//2,(a+b)//2+1):
          for J45 in range(abs(d-e)//2,(d+e)//2+1):
            lo=max(abs(2*J12-c),abs(2*J45-f)); hi=min(2*J12+c,2*J45+f)
            if hi>=lo: n+=(hi-lo)//2+1
        return n
    idx={t:k for k,t in enumerate(trip)}
    Ptrip=[(-1)**sum(o[x][2] for x in t) for t in trip]
    for k,t in enumerate(trip):
        for k2 in range(k+1):
            u=trip[k2]
            if Ptrip[k]!=Ptrip[k2]: continue
            tot+=5*nJ(t,u)
    return tot
if __name__=="__main__":
    emax,e2max,e3max=map(int,sys.argv[1:4])
    o,chan,chanN=lab_counts(emax,e2max,e3max)
    mem=sum(n*(n+1)/2 for n in chan.values())
    print("norb",len(o),"nchan",len(chan),"sum n(n+1)/2",mem,"GiB double",mem*8/2**30, "maxdim",max(chan.values()))
    if len(sys.argv)>4: print("file values", file_tuples(emax,e2max,e3max))

def file_tuples_fast(emax,e2max,e3max):
    o=orbits(emax); no=len(o)
    cnt=defaultdict(int)
    for i1 in range(no):
      for i2 in range(i1+1):
        for i3 in range(i2+1):
          e1,e2,e3=o[i1][0],o[i2][0],o[i3][0]
          if e1+e2>e2max or e1+e3>e2max or e2+e3>e2max or e1+e2+e3>e3max: continue
          P=(-1)**(o[i1][2]+o[i2][2]+o[i3][2])
          cnt[(o[i1][3],o[i2][3],o[i3][3],P)]+=1
    def nJ(k,u):
        a,b,c=k[:3]; d,e,f=u[:3]; n=0
        for J12 in range(abs(a-b)//2,(a+b)//2+1):
          for J45 in range(abs(d-e)//2,(d+e)//2+1):
            lo=max(abs(2*J12-c),abs(2*J45-f)); hi=min(2*J12+c,2*J45+f)
            if hi>=lo: n+=(hi-lo)//2+1
        return n
    keys=list(cnt); allsum=0; diag=0
    for k in keys:
        diag+=cnt[k]*nJ(k,k)
        for u in keys:
            if u[3]!=k[3]: continue
            allsum+=cnt[k]*cnt[u]*nJ(k,u)
    return 5*(allsum+diag)//2
