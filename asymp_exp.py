#%% Program 05d: The Lindstedt-Poincare Method
# Deriving the order epsilon equations.
# See Example 9.

#from sympy import collect, expand, Function, Symbol
import sympy as sp
import pprint

x0 = sp.Function('x0')
x1 = sp.Function('x1')
x2 = sp.Function('x2')
x = sp.Function('x')
t = sp.Symbol('t')
eps = sp.Symbol('eps')
w1 = sp.Symbol('w1')
w2 = sp.Symbol('w2')

x = x0(t) + eps * x1(t) + eps ** 2 * x2(t)
exp1 = (1 + eps * w1 + eps ** 2 * w2) ** 2 * x.diff(t, t) + x - eps * x **3
exp1= sp.expand(exp1)

sp.pprint(exp1)

#%% Zeroth order
eq0 = exp1.coeff(eps,0)
print(eq0)
sol0=sp.solvers.ode.dsolve(eq0,ics={x0(0): 1, x0(t).diff(t).subs(t, 0): 0})
sp.pprint(sol0)


# %% First order
eq1 = exp1.coeff(eps,1)
#sp.pprint(eq1)
prob1=eq1.subs(x0(t),sol0.rhs)
prob1=prob1.simplify().rewrite(sp.exp).expand()
#%%
secular= prob1.simplify().rewrite(sp.exp).expand().coeff(sp.exp(sp.I*t))
#sp.pprint(secular)
w1=sp.solve(secular,w1)

#sol1=sp.solvers.ode.dsolve(prob1,ics={x1(0): 1, x1(t).diff(t).subs(t, 0): 0})
# %%
prob1=prob1.subs('w1',w1[0])
#%%
sp.solvers.ode.dsolve(prob1,ics={x1(0):0,x1(t).diff(t).subs(t, 0): 0})
# %%

