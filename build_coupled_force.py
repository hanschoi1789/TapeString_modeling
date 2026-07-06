"""Build symbolic Routhian EL system for a TIP-FORCE loaded cantilever:
prescribed internal moment  M(s1) = P*(L - s1),  dM/ds1 = -P  (undeformed
moment-arm approximation, valid for moderate rotations up to the fold).

Differences vs build_coupled.py (constant-M version):
  * M is now a FUNCTION of s1. The Euler-Lagrange chain rule therefore
    generates extra terms proportional to M' = dM/ds1 (and M'' = 0 since
    M is linear). These terms were identically zero in the constant-M code.
  * We also export the third-order natural BC
        Q3 = dR/dbe1 - d/ds1 (dR/dbe2)
    needed for a genuinely FREE tip (no cross-section fixture), which
    depends on (be, be1, be2, be3, M, M1).

Energy model, kinematics, quadrature: identical to build_coupled.py
(full Guinot 2012 planar strains, centered coordinates, N=0 elimination,
a = 16 mm half-width, beta0 = 0.95 rad, Gauss-Legendre nq=10).
"""
import sympy as sp
import numpy as np
import pickle, time

t0 = time.time()
# numeric params inline (keeps trees small) -- identical to build_coupled.py
a, beta0, E, nu, h = 16, sp.Rational(95, 100), 210000, sp.Rational(3, 10), sp.Rational(15, 100)
A = E * h
D = E * h**3 / (12 * (1 - nu**2))

be, be1, be2, M = sp.symbols('be be1 be2 M', real=True)

def pieces(s2v):
    """exact centered kinematics at quadrature node s2v (numeric)"""
    phi = be * s2v / a
    cp, sp_ = sp.cos(phi), sp.sin(phi)
    zc = (a / be) * (sp.sin(be) / be - cp)
    y = (a / be) * sp_
    dz = sp.diff(zc, be); d2z = sp.diff(zc, be, 2)
    dy = sp.diff(y, be);  d2y = sp.diff(y, be, 2)
    y1 = dy * be1;  z1 = dz * be1
    y11 = dy * be2 + d2y * be1**2
    z11 = dz * be2 + d2z * be1**2
    es = sp.Rational(1, 2) * (y1**2 + z1**2)
    ks11 = z11 * cp - y11 * sp_
    k12 = s2v * be1 / a
    return zc, cp, es, ks11, k12

nq = 10
nodes, weights = np.polynomial.legendre.leggauss(nq)
Izc2 = Izces = Ies = Ies2 = Icos2 = Icosks = Icos = Iks2 = Iks = Ik12_2 = 0
for xi, wi in zip(nodes, weights):
    s2v = sp.nsimplify(round(float(xi), 12)) * a
    w   = sp.nsimplify(round(float(wi), 12)) * a
    zc, cp, es, ks11, k12 = pieces(s2v)
    Izc2   += w * zc**2
    Izces  += w * zc * es
    Ies    += w * es
    Ies2   += w * es**2
    Icos2  += w * cp**2
    Icosks += w * cp * ks11
    Icos   += w * cp
    Iks2   += w * ks11**2
    Iks    += w * ks11
    Ik12_2 += w * k12**2
print("quadrature sums built", time.time() - t0, flush=True)

k22 = (be - beta0) / a
Ctt = A * Izc2 + D * Icos2
Cts = A * Izces + D * (Icosks + nu * k22 * Icos)
u0 = sp.Rational(1, 2) * A * (Ies2 - Ies**2 / (2 * a)) \
   + sp.Rational(1, 2) * D * (Iks2 + 2 * a * k22**2 + 2 * nu * k22 * Iks + 2 * (1 - nu) * Ik12_2)

R = u0 - (M - Cts)**2 / (2 * Ctt)      # Routhian (do NOT expand)
TH_expr = (M - Cts) / Ctt              # theta,1 recovery (pointwise, unchanged)
print("R assembled", time.time() - t0, flush=True)

# ---- Euler-Lagrange with M = M(s1) ----------------------------------------
s1 = sp.symbols('s1')
f = sp.Function('f')(s1)     # be(s1)
g = sp.Function('g')(s1)     # M(s1)  <-- key difference vs constant-M build
traj = {be: f, be1: sp.diff(f, s1), be2: sp.diff(f, s1, 2), M: g}

tB  = sp.diff(R, be).subs(traj)
tB1 = sp.diff(R, be1).subs(traj)
tB2 = sp.diff(R, be2).subs(traj)
EL = tB - sp.diff(tB1, s1) + sp.diff(tB2, s1, 2)
print("EL derived", time.time() - t0, flush=True)

b0s, b1s, b2s, b3s, b4s = sp.symbols('b0s b1s b2s b3s b4s')
Ms, M1s = sp.symbols('Ms M1s')   # M(s1) value and its first derivative (-P)
sm = {sp.Derivative(f, (s1, 4)): b4s, sp.Derivative(f, (s1, 3)): b3s,
      sp.Derivative(f, (s1, 2)): b2s, sp.Derivative(f, (s1, 1)): b1s, f: b0s,
      sp.Derivative(g, (s1, 2)): 0,   # M linear in s1 -> M'' = 0
      sp.Derivative(g, (s1, 1)): M1s, g: Ms}
ELp = EL.subs(sm)
coeff4 = sp.diff(ELp, b4s)
rem    = ELp.subs(b4s, 0)

# natural BCs
dR_db2 = sp.diff(R, be2)                       # (be,be1,be2,M): moment-like BC
# third-order natural BC for a FREE end: dR/dbe1 - d/ds1(dR/dbe2) = 0
Q3 = (sp.diff(R, be1).subs(traj) - sp.diff(sp.diff(R, be2).subs(traj), s1)).subs(sm)
print("split done", time.time() - t0, flush=True)

with open('coupled_force_sym.pkl', 'wb') as fpk:
    pickle.dump(dict(coeff4=coeff4, rem=rem, TH=TH_expr, dR_db2=dR_db2, Q3=Q3,
                     Ctt=Ctt,
                     syms=(b0s, b1s, b2s, b3s, be, be1, be2, M, Ms, M1s)), fpk)
print("saved coupled_force_sym.pkl", time.time() - t0, flush=True)
