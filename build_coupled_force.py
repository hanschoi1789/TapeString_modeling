"""Build symbolic Routhian EL system for a TIP-FORCE loaded cantilever,
now also supporting a UNIFORM DISTRIBUTED SELF-WEIGHT along the tape
(in addition to the tip force P and any point loads from clamp plates,
which are handled at the segment level in run_case_*.py):

    M(s1) = P*(L - s1) + (w/2)*(L - s1)^2 + [point-load terms handled outside]
    M'(s1)  = -P - w*(L - s1) - [sum of point loads distal to s1]
    M''(s1) = w                                   <-- constant, no longer 0

Differences vs the original build_coupled_force.py:
  * M is now a FUNCTION of s1 with a NONZERO, CONSTANT second derivative.
    A new symbol M2s represents M''(s1) = w (the self-weight per unit
    length). The old version hardcoded M''=0 (valid only for a point tip
    force with no distributed load); that is now a special case M2s=0.
  * coeff4 and rem (from the full Euler-Lagrange, which involves d^2/ds1^2
    of dR/dbe2) now depend on M2s and are lambdified with it as an extra
    argument.
  * Q3 (third-order free-tip BC) and dR_db2 (moment-type BC) only ever
    need M and M' (never M''), so their signatures are UNCHANGED.

Energy model, kinematics, quadrature: identical to the original
build_coupled_force.py (full Guinot 2012 planar strains, centered
coordinates, N=0 elimination, a = 16 mm half-width, beta0 = 0.95 rad,
Gauss-Legendre nq=10).
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
Ms, M1s, M2s = sp.symbols('Ms M1s M2s')   # M(s1), M'(s1), M''(s1) = w (self-weight/length)
sm = {sp.Derivative(f, (s1, 4)): b4s, sp.Derivative(f, (s1, 3)): b3s,
      sp.Derivative(f, (s1, 2)): b2s, sp.Derivative(f, (s1, 1)): b1s, f: b0s,
      sp.Derivative(g, (s1, 3)): 0,      # M''' = 0 (still exact: M'' = w is constant)
      sp.Derivative(g, (s1, 2)): M2s,    # M'' = w  (nonzero -> distributed self-weight)
      sp.Derivative(g, (s1, 1)): M1s, g: Ms}
ELp = EL.subs(sm)
coeff4 = sp.diff(ELp, b4s)
rem    = ELp.subs(b4s, 0)

# natural BCs -- these only ever touch M and M' (never M''), so unchanged
dR_db2 = sp.diff(R, be2)                       # (be,be1,be2,M): moment-like BC
# third-order natural BC for a FREE end: dR/dbe1 - d/ds1(dR/dbe2) = 0
# (needs only Ms, M1s: the s1-derivative here is a single order, same as before)
sm_q3 = {sp.Derivative(f, (s1, 4)): b4s, sp.Derivative(f, (s1, 3)): b3s,
         sp.Derivative(f, (s1, 2)): b2s, sp.Derivative(f, (s1, 1)): b1s, f: b0s,
         sp.Derivative(g, (s1, 2)): M2s, sp.Derivative(g, (s1, 1)): M1s, g: Ms}
Q3 = (sp.diff(R, be1).subs(traj) - sp.diff(sp.diff(R, be2).subs(traj), s1)).subs(sm_q3)
# Q3 as derived contains no M2s term algebraically (single s1-derivative only),
# but we substitute with the same dict for safety/consistency; verify below.
print("split done", time.time() - t0, flush=True)

has_m2_in_q3 = Q3.has(M2s)
has_m2_in_coeff4 = coeff4.has(M2s)
has_m2_in_rem = rem.has(M2s)
print(f"Q3 depends on M2s: {has_m2_in_q3}  (expected False)", flush=True)
print(f"coeff4 depends on M2s: {has_m2_in_coeff4}", flush=True)
print(f"rem depends on M2s: {has_m2_in_rem}  (expected True)", flush=True)

with open('coupled_force_sym.pkl', 'wb') as fpk:
    pickle.dump(dict(coeff4=coeff4, rem=rem, TH=TH_expr, dR_db2=dR_db2, Q3=Q3,
                     Ctt=Ctt,
                     syms=(b0s, b1s, b2s, b3s, be, be1, be2, M, Ms, M1s, M2s)), fpk)
print("saved coupled_force_sym.pkl", time.time() - t0, flush=True)
