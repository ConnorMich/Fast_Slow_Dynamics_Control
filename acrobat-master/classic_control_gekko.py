from gekko import GEKKO
import numpy as np
import matplotlib
from math import *
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  






# initialize gekko
m = GEKKO()
nt = 101
m.time = np.linspace(0,2,nt)

# Variables
q1 = m.Var(value=1)
q1dot = m.Var(value=0)
v1 = m.Var(value=0)
q2 = m.Var(value=0)
q2dot = m.Var(value=0)
v2 = m.Var(value=0)
tau = m.Var(value=0,lb=-1,ub=1)

#Defining system dynamics
l1 = 1.  # [m]
l2 = 1.  # [m]
m1 = 1.  #: [kg] mass of link 1
m2 = 1.  #: [kg] mass of link 2
lc1 = 0.5  #: [m] position of the center of mass of link 1
lc2 = 0.5  #: [m] position of the center of mass of link 2
I1 = 1.  #: moments of inertia for both links
I2 = 1.  #: moments of inertia for both links



# c1 = cos(q1)
# c2 = cos(q2)

# s1 = sin(q1)
# s2 = sin(q2)

# c12 = cos(q1 + q2)
# s12 = sin(q1 + q2)


# A = I1 + I2 + m2*l1*l1 + 2*m2*l1*lc2*c2
# B = I2 + m2*l1*lc2*c2
# C = -2*m2*l1*lc2*s2
# D = -m2*l1*lc2*s2
# E = m1*g*lc1*s1 + m2*g*(l1*s1 + lc2*s12)
# F = I2 + m2*l1*lc2*c2
# G = I2
# H = m2*l1*lc2*s2
# I = m2*g*lc2*s12



p = np.zeros(nt) # mark final time point
p[-1] = 1.0
final = m.Param(value=p)

# Equations
m.Equation(v1 == q1.dt())
m.Equation(v2 == q2.dt())
m.Equation(v1.dt() == m.cos(q1))
# m.Equation(v2.dt() == )


# m.Equation(x1.dt()==u)
# m.Equation(x2.dt()==0.5*x1**2)


m.Obj(thetadot + theta) # Objective function

m.options.IMODE = 6 # optimal control mode
m.solve(disp=False) # solve
plt.figure(1) # plot results
plt.plot(m.time,x1.value,'k-',label=r'$x_1$')
plt.plot(m.time,x2.value,'b-',label=r'$x_2$')
plt.plot(m.time,u.value,'r--',label=r'$u$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()