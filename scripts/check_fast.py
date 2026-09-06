"""Verificacion exacta (aritmetica racional) de las identidades clave del hilo,
evaluando en varios (p,r) racionales genericos y varios y racionales.
Si una identidad polinomica/algebraica en (p,r,y) se cumple exactamente en puntos
genericos multiples, queda verificada con confianza practica total."""
import sympy as sp

lam, n = sp.symbols('lambda n')

def run(pv, rv, cv=sp.Integer(1), muv=sp.Integer(1)):
    p, r, c, mu = map(sp.nsimplify, (pv, rv, cv, muv))
    m = (p-1)/p
    Wstar = (2*p*c/(p+1))**(p/(p-1))
    y = sp.symbols('y', positive=True)
    W = Wstar*y**(p/(p-1))            # 1/m = p/(p-1)
    RHS = c*p/(p+1)*W**((p+1)/p) - W**2/2
    Wp = (sp.simplify((r+1)/(mu*r)*RHS))**(sp.Rational(1,1)/(r+1))
    y_z = m*(W/Wstar)**(m-1)*Wp/Wstar
    b = r*Wp**(r-1)
    D = (r+1)*(p-1)**2/(2*p**2)
    fa = (p*r-1)/((p-1)*(r+1)); fb = r/(r+1)

    id1 = mu*b*y_z**2 - D*y*(1-y)
    id2 = mu*y_z*sp.diff(b*y_z, y) - D*(fa-(fa+fb)*y)
    ok1 = all(sp.simplify(id1.subs(y, sp.Rational(k,7))) == 0 for k in (1,2,3,5))
    ok2 = all(sp.simplify(id2.subs(y, sp.Rational(k,7))) == 0 for k in (1,2,3,5))

    # ecuacion espectral y autovalores
    K = (p+1)/((r+1)*(p-1)**2)
    g = (p+1)/((p-1)*(r+1))
    ok3 = sp.simplify(g*(g-1) + fa*g - K) == 0
    ok4 = sp.simplify((-1/(p-1))*(-1/(p-1)-1) + fa*(-1/(p-1)) - K) == 0

    f = sp.Function('f')
    eta = y**g*f(y)
    ode = sp.expand(sp.simplify(sp.expand(
        (y*(1-y)*sp.diff(eta,y,2) + (fa-(fa+fb)*y)*sp.diff(eta,y)
         + ((lam+1)/D - K/y)*eta)/y**g)))
    c2 = sp.simplify(ode.coeff(sp.diff(f(y),y,2)))
    c1 = sp.expand(sp.simplify(ode.coeff(sp.diff(f(y),y))))
    c0 = sp.expand(sp.simplify(ode.coeff(f(y))))
    Cpar = sp.simplify(c1.subs(y,0))
    ApB = sp.simplify(-(c1-Cpar)/y - 1)
    C_t = (p*r+2*p+1)/((p-1)*(r+1)); S_t = (p*r+p+2)/((p-1)*(r+1))
    ok5 = sp.simplify(c2 - y*(1-y)) == 0
    ok6 = sp.simplify(Cpar - C_t) == 0 and sp.simplify(ApB - S_t) == 0
    AB_expr = sp.simplify(-c0)   # AB(lambda)
    lam_even = sp.solve(sp.Eq(AB_expr, -n*(S_t+n)), lam)[0]
    lam_odd  = sp.solve(sp.Eq(AB_expr, (C_t+n)*(S_t-C_t-n)), lam)[0]
    t_even = -(p-1)*(p*r+2*p+1)/(2*p**2*(r+1)) + (p-1)*(p*(r+1)+2)/(2*p**2)*n + (r+1)*(p-1)**2/(2*p**2)*n**2
    t_odd  = (p-1)*(r+3)/(2*p)*n + (r+1)*(p-1)**2/(2*p**2)*n**2
    ok7 = sp.simplify(sp.expand(lam_even - t_even)) == 0
    ok8 = sp.simplify(sp.expand(lam_odd - t_odd)) == 0
    print(f"p={pv}, r={rv}: primera-integral->y(1-y):{ok1}  coef eta_y:{ok2}  indicial s+:{ok3} s-:{ok4}  hipergeom C,S:{ok5 and ok6}  lambda+ :{ok7}  lambda- :{ok8}")
    return all((ok1,ok2,ok3,ok4,ok5,ok6,ok7,ok8))

allok = True
for pv, rv in [(2,3),(2,5),(3,2),(sp.Rational(5,2),sp.Rational(7,3)),(2,1)]:
    allok &= run(pv, rv)
# independencia de c y mu (p=2, r=3, c=5/4, mu=3/7)
allok &= run(2,3,sp.Rational(5,4),sp.Rational(3,7))
print("TODO OK" if allok else "HAY DISCREPANCIAS")
