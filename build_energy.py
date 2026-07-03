"""
풀 모델 에너지 밀도 (단위 s1 길이당) 를 sympy로 조립.
식 (36) Ue = Uer + Ues + Uers, 변형률 (28)(43), 단면적분 (A.2), 강성 (34).
종속변수: u1,u3,theta,be 와 s1-미분.
"""
import sympy as sp
from char_funcs import DEFS, b as bsym

# 재료/형상
E,nu,h,a,be0 = sp.symbols('E nu h a be0', positive=True)
A   = E*h
D11 = E*h**3/(12*(1-nu**2))
D12 = nu*D11
D33 = E*h**3/(24*(1+nu))

# 종속변수(점값)와 미분
u1p,u3p = sp.symbols('u1p u3p')       # u1_,1 ; u3_,1
th,thp  = sp.symbols('th thp')        # theta ; theta_,1
be,bep,bepp = sp.symbols('be bep bepp')  # beta_e ; ,1 ; ,11

def I(name): return DEFS[name].subs(bsym, be)

# 막대(전체) 변형률 (식 28)
er = u1p + sp.Rational(1,2)*(u1p**2+u3p**2)
kr = thp

# 단면적분 (A.2)
ov_z2   = a**3*I('I_z2')
ov_c2   = a   *I('I_c2')
ov_z    = a**2*I('I_z')
ov_es2  = a**5*I('I_d4')*bep**4
ov_k11s2= a**3*( I('I_k4')*bep**4 + I('I_k2kp')*bep**2*bepp + I('I_k2')*bepp**2 )
ov_k22s2= sp.Rational(4,1)/a*(be-be0)**2
ov_k11k22= a*( I('I_k2')*bep**2 + I('I_kp')*bepp )*(be-be0)
ov_k12s2= a/sp.Integer(3)*bep**2
ov_es   = a**3*I('I_d2')*bep**2
ov_zes  = a**4*I('I_ze')*bep**2
ov_ck11 = a**2*( I('I_ck2')*bep**2 + I('I_ckp')*bepp )
ov_ck22 = I('I_c')*(be-be0)

# --- 에너지 밀도 (식 35/36). 쉘 변형에너지를 N,M 형태로:
# u = 1/2[ A e11^2 + D11 k11^2 + 2 D12 k11 k22 + D11 k22^2 + 4 D33 k12^2 ] 를 s2 적분
# e11 = er + z kr + e^s,  k11 = -kr cos(beta) + k11^s,  k22=k22^s, k12=k12^s
# 적분하면 오버라인 양으로:
# A 항: A[ a er^2 + 2 ov_z er kr + ov_z2 kr^2 + 2 ov_es er + 2 ov_zes kr + ov_es2 ]   (e11^2)
# D11 항(k11^2): D11[ ov_c2 kr^2 - 2 ov_ck11 kr + ov_k11s2 ]
# 2 D12 k11 k22: 2 D12[ -kr ov_ck22 + ov_k11k22 ]     (cos beta * k22 -> ov_ck22; k11^s k22 -> ov_k11k22)
# D11 k22^2: D11 ov_k22s2
# 4 D33 k12^2: 4 D33 ov_k12s2

U_A   = A*( a*er**2 + 2*ov_z*er*kr + ov_z2*kr**2 + 2*ov_es*er + 2*ov_zes*kr + ov_es2 )
U_D11 = D11*( ov_c2*kr**2 - 2*ov_ck11*kr + ov_k11s2 )
U_D12 = 2*D12*( -kr*ov_ck22 + ov_k11k22 )
U_k22 = D11*ov_k22s2
U_k12 = 4*D33*ov_k12s2

U_density = sp.Rational(1,2)*( U_A + U_D11 + U_D12 + U_k22 + U_k12 )

if __name__=="__main__":
    # 수치 대입 테스트 (Table 1,2)
    subs0 = {E:210e9, nu:0.3, h:0.15e-3, a:0.06, be0:0.6}
    # 변형 없음: er=kr=0, be=be0, bep=bepp=0  -> 에너지 0 이어야
    test = U_density.subs(subs0).subs({u1p:0,u3p:0,th:0,thp:0,be:0.6,bep:0,bepp:0})
    print('변형없음 에너지 밀도 =', float(test), '(0 이어야)')
    # 약간 굽힘: kr=0.5
    test2 = U_density.subs(subs0).subs({u1p:0,u3p:0,th:0,thp:0.5,be:0.6,bep:0,bepp:0})
    print('kr=0.5 에너지 밀도 =', float(test2))
    # 단면 평탄화: be=0.3
    test3 = U_density.subs(subs0).subs({u1p:0,u3p:0,th:0,thp:0.5,be:0.3,bep:0,bepp:0})
    print('kr=0.5,be=0.3 에너지 밀도 =', float(test3))
    print('OK - 에너지 조립 완료')
