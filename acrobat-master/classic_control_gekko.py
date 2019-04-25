from scores.score_logger import FS_score

from gekko import GEKKO
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  
from numpy import pi
import gym




# # initialize gekko
# m = GEKKO()
# start = 0
# stop = 4
# dt = 0.02
# nt = int((stop-start)/dt)
# # m.time = np.linspace(0,5,nt)
# m.time = np.linspace(start,stop,nt)



# # Variables
# q1 = m.Var(value=0)
# q1dot = m.Var(value=0)
# q2 = m.Var(value=0)
# q2dot = m.Var(value=0)
# tau = m.MV(value=0, lb=-1, ub=1, integer=True)
# tau.STATUS = 1
# min_func = m.Var(value=0)
# p = np.ones(nt) # mark final time point
# # p[-1] = 1.0
# final = m.Param(value=p)


# #Defining system dynamics
# l1 = 1.  # [m]
# l2 = 1.  # [m]
# m1 = 1.  #: [kg] mass of link 1
# m2 = 1.  #: [kg] mass of link 2
# lc1 = 0.5  #: [m] position of the center of mass of link 1
# lc2 = 0.5  #: [m] position of the center of mass of link 2
# I1 = 1.  #: moments of inertia for both links
# I2 = 1.  #: moments of inertia for both links
# g = 9.8

# c1 = m.cos(q1)
# c2 = m.cos(q2)
# s1 = m.sin(q1)
# s2 = m.sin(q2)
# c12 = m.cos(q1 + q2)
# s12 = m.sin(q1 + q2)

# A = I1 + I2 + m2*l1*l1 + 2*m2*l1*lc2*c2
# B = I2 + m2*l1*lc2*c2
# C = -2*m2*l1*lc2*s2
# D = -m2*l1*lc2*s2
# E = m1*g*lc1*s1 + m2*g*(l1*s1 + lc2*s12)
# F = I2 + m2*l1*lc2*c2
# G = I2
# H = m2*l1*lc2*s2
# I = m2*g*lc2*s12

# # Equations
# m.Equation(q1dot == q1.dt())
# m.Equation(q2dot == q2.dt())
# m.Equation(q2dot.dt() == (A*(H*q1dot + I + tau ) + F*(-C*q1dot*q2dot - D*q2dot - E))/(F*B - A*G))
# m.Equation(q1dot.dt() == (1/F)*(tau - F-q2dot.dt() - H*q1dot**2 - I))
# m.Obj((m.cos(q1)*final)**2) # Objective function
# # m.Obj(min_func + 15*(q1*final - pi)**2 )

# m.options.IMODE = 6 # optimal control mode


# m.solve(disp=False) # solve

# # plot results
# plt.figure(1) 

# plt.subplot(3,1,1)
# plt.plot(m.time,q1.value,'k-',label=r'$q1$')
# plt.plot(m.time,q2.value,'b-',label=r'$q2$')
# plt.ylabel('Angle')
# plt.legend(loc='best')

# plt.subplot(3,1,2)
# plt.plot(m.time,q1dot.value,'k-',label=r'$q1dot$')
# plt.plot(m.time,q2dot.value,'b-',label=r'$q2dot$')
# plt.ylabel('Joint Velocity')
# plt.legend(loc='best')

# plt.subplot(3,1,3)
# plt.plot(m.time,tau.value,'r--',label=r'$tau$')
# plt.legend(loc='best')
# plt.xlabel('Time')
# plt.ylabel('Input Force')
# # plt.show()

# plt.close()

# print(len(tau.value))
# inputs = tau.value

# forceInput = np.array([])
# for i in range(0,len(tau.value)):
# 	forceInput = np.append(forceInput, int(round(tau.value[i])))

forceInput = np.ones(2000)*2
print(forceInput)

def test_acrobot(model_name, num_tests):
    # generate the environment
    env = gym.make("Acrobot-v1")

    # define the observation and action spaces
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Create the performance analyzer
    test_score_manager = FS_score(-1,0, model_name)
    test_score_manager.clear_fs_states()

    # Prep the environemnt
    state = env.reset()
    state = np.reshape(state, [1, observation_space])

    # Test the Model num_tests of times
    i=0 
    steps = 0
    while(i<num_tests):  
        
        # Render the environment
        env.render()

        # Determine and perform the action
        action = int(forceInput[steps])
       	print(action)
        state_next, reward, terminal, info = env.step(action)

        # save the state of the system
        test_score_manager.add_state(state[0],  action)

        # Set the next action of the state
        state_next = np.reshape(state_next, [1, observation_space])
        state = state_next
        steps += 1;


        # When the run is finished:
        if steps > 199 or state[0][0] < -0.99:
            steps = 0
            # Save the CSV
            test_score_manager.save_csv()

            # Add the run to the PNG
            test_score_manager.save_run(i, num_tests)
            test_score_manager.clear_run_data()

            # Reset the environment
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            i = i + 1
    env.close()
test_acrobot('gekko_acrobot', 1)