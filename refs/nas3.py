import sys
from collections import defaultdict
from count3 import orbits
def nas_counts(e3max, jmaxrel=10**9):
    # count[(J2,P,T2,N,t12)] of NAS+CM states
    cnt=defaultdict(int)
    # first count Jacobi NAS (no CM) per (E, jrel2, P, t12): requires l12+s12+t12 odd
    jac=defaultdict(int)
    for E in range(e3max+1):
      for n12 in range(E//2+1):
        for n3 in range((E-2*n12)//2+1):
          for l12 in range(E-2*n12-2*n3+1):
            l3=E-2*n12-2*n3-l12
            for s12 in (0,1):
              for j12 in range(abs(l12-s12),l12+s12+1):
                for t12 in (0,1):
                  if (l12+s12+t12)%2==0: continue
                  for j3 in {abs(2*l3-1),2*l3+1}:
                    for jr in range(abs(2*j12-j3),2*j12+j3+1,2):
                      jac[(E,jr,(-1)**(l12+l3),t12)]+=1
    for (E,jr,Pr,t12),c in jac.items():
        for Ncm in range(0,(e3max-E)//2+1):
            for Lcm in range(0,e3max-E-2*Ncm+1):
                N=E+2*Ncm+Lcm
                P=Pr*(-1)**Lcm
                for J2 in range(abs(jr-2*Lcm),jr+2*Lcm+1,2):
                    for T2 in (1,3):
                        if T2>2*t12+1: continue
                        cnt[(J2,P,T2,N,t12)]+=c
    return cnt
