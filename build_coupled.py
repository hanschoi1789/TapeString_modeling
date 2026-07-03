"""Build symbolic Routhian R(be,be1,be2;M) for the coupled tape problem.
Energy = full Guinot 2012 planar strains with CENTERED coordinates:
  e11 = er + z_c*TH + es,  er eliminated by N=0  ->
  membrane = A/2 [ TH^2*Int(z_c^2) + 2 TH Int(z_c es) + Int(es^2) - (Int es)^2/(2a) ]
  k11 = TH*cos(beta) + ks11,  ks11 = z_c,11 cos - y,11 sin
  k22 = (be-beta0)/a,  k12 = s2*be1/a
Convention: s2 in [-a,a], a = half width = 15.96, beta = be*s2/a."""
import sympy as sp
import numpy as np
import pickle, time

t0=time.time()
# numeric params inline (keeps trees small)
a, beta0, E, nu, h = 16, sp.Rational(95,100), 210000, sp.Rational(3,10), sp.Rational(15,100)
A = E*h
D = E*h**3/(12*(1-nu**2))

be, be1, be2, M = sp.symbols('be be1 be2 M', real=True)

def pieces(s2v):
    """exact centered kinematics at quadrature node s2v (numeric)"""
    phi = be*s2v/a
    cp, sp_ = sp.cos(phi), sp.sin(phi)
    # centered z_c and y (y odd -> already centered)
    zc = (a/be)*(sp.sin(be)/be - cp)
    y  = (a/be)*sp_
    dz  = sp.diff(zc, be);  d2z = sp.diff(zc, be, 2)
    dy  = sp.diff(y,  be);  d2y = sp.diff(y,  be, 2)
    y1  = dy*be1;            z1  = dz*be1
    y11 = dy*be2 + d2y*be1**2
    z11 = dz*be2 + d2z*be1**2
    es   = sp.Rational(1,2)*(y1**2 + z1**2)
    ks11 = z11*cp - y11*sp_
    k12  = s2v*be1/a
    return zc, cp, es, ks11, k12

nq = 10
nodes, weights = np.polynomial.legendre.leggauss(nq)
Izc2=0; Izces=0; Ies=0; Ies2=0; Icos2=0; Icosks=0; Icos=0; Iks2=0; Iks=0; Ik12_2=0
for xi, wi in zip(nodes, weights):
    s2v = sp.nsimplify(round(float(xi),12))*a
    w   = sp.nsimplify(round(float(wi),12))*a
    zc, cp, es, ks11, k12 = pieces(s2v)
    Izc2   += w*zc**2
    Izces  += w*zc*es
    Ies    += w*es
    Ies2   += w*es**2
    Icos2  += w*cp**2
    Icosks += w*cp*ks11
    Icos   += w*cp
    Iks2   += w*ks11**2
    Iks    += w*ks11
    Ik12_2 += w*k12**2
print("quadrature sums built", time.time()-t0)

k22 = (be-beta0)/a
Ctt = A*Izc2 + D*Icos2                              # coeff of TH^2 (x1/2)
Cts = A*Izces + D*(Icosks + nu*k22*Icos)            # coeff of TH
u0  = sp.Rational(1,2)*A*(Ies2 - Ies**2/(2*a)) \
    + sp.Rational(1,2)*D*(Iks2 + 2*a*k22**2 + 2*nu*k22*Iks + 2*(1-nu)*Ik12_2)

R = u0 - (M - Cts)**2/(2*Ctt)                        # Routhian (do NOT expand)
TH_expr = (M - Cts)/Ctt                              # theta,1 recovery
print("R assembled", time.time()-t0)

# Euler-Lagrange of R via chain rule
s1 = sp.symbols('s1')
f  = sp.Function('f')(s1)
traj = {be: f, be1: sp.diff(f,s1), be2: sp.diff(f,s1,2)}
tB  = sp.diff(R,be).subs(traj)
tB1 = sp.diff(R,be1).subs(traj)
tB2 = sp.diff(R,be2).subs(traj)
EL = tB - sp.diff(tB1,s1) + sp.diff(tB2,s1,2)
print("EL derived", time.time()-t0)

b0s,b1s,b2s,b3s,b4s = sp.symbols('b0s b1s b2s b3s b4s')
sm = {sp.Derivative(f,(s1,4)):b4s, sp.Derivative(f,(s1,3)):b3s,
      sp.Derivative(f,(s1,2)):b2s, sp.Derivative(f,(s1,1)):b1s, f:b0s}
ELp = EL.subs(sm)
coeff4 = sp.diff(ELp, b4s)
rem    = ELp.subs(b4s, 0)
dR_db2 = sp.diff(R, be2)      # natural BC (loose end): dR/dbe2 = 0
print("split done", time.time()-t0)

with open('coupled_sym.pkl','wb') as fpk:
    pickle.dump(dict(coeff4=coeff4, rem=rem, TH=TH_expr, dR_db2=dR_db2, Ctt=Ctt,
                     syms=(b0s,b1s,b2s,b3s,be,be1,be2,M)), fpk)
print("saved", time.time()-t0)
