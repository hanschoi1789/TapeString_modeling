import sympy as sp
import numpy as np
import pickle, time
from scipy.integrate import solve_bvp

t0=time.time()
d = pickle.load(open('coupled_sym.pkl','rb'))
b0s,b1s,b2s,b3s,be,be1,be2,M = d['syms']
f_c4  = sp.lambdify((b0s,b1s,b2s,b3s,M), d['coeff4'], modules='numpy', cse=True)
f_rem = sp.lambdify((b0s,b1s,b2s,b3s,M), d['rem'],   modules='numpy', cse=True)
f_TH  = sp.lambdify((be,be1,be2,M),      d['TH'],    modules='numpy', cse=True)
print("lambdified", time.time()-t0, flush=True)

beta0=0.95; L=200.0
bcL=(np.pi/2,'tight'); bcR=(beta0,'tight')   # LEFT end flattened to 90deg

def rhs(s,y,p):
    Mv=p[0]
    y4=-f_rem(y[0],y[1],y[2],y[3],Mv)/f_c4(y[0],y[1],y[2],y[3],Mv)
    return np.vstack([y[1],y[2],y[3],y4,f_TH(y[0],y[1],y[2],Mv)])
def make_bc(thL):
    def bc(ya,yb,p):
        return np.array([ya[0]-bcL[0], ya[1], yb[0]-bcR[0], yb[1], ya[4], yb[4]-thL])
    return bc

x=np.linspace(0,L,161)
y=np.zeros((5,x.size)); y[0]=beta0+(np.pi/2-beta0)*np.exp(-x/12.)
p=np.array([0.0]); results=[]
th_list=np.concatenate([np.linspace(0.005,0.12,8), np.linspace(0.135,0.5,20)])
t_deadline=time.time()+240
for thL in th_list:
    if time.time()>t_deadline: print("budget reached"); break
    sol=solve_bvp(rhs, make_bc(thL), x, y, p=p, max_nodes=60000, tol=1e-5)
    if not sol.success:
        print(f"thL={thL:.4f} FAILED -> fold limit region"); break
    x,y,p=sol.x,sol.y,sol.p
    ss=np.linspace(0,L,2000); prof=sol.sol(ss)
    imin=int(np.argmin(prof[0]))
    results.append((thL,p[0],ss[imin],float(prof[0][imin]),prof[0].copy(),ss))
    print(f"thL={thL:7.4f}  M={p[0]:9.2f}  min be={np.degrees(prof[0][imin]):6.2f} deg @ {ss[imin]:6.1f} mm", flush=True)
pickle.dump(results, open('case2_results.pkl','wb'))
print("done", time.time()-t0)
