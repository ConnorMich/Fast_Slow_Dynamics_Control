import gym
import sys
import argparse
from scores.score_logger import ScoreLogger
from scores.score_logger import FS_score
from scores.score_logger import Test_Score
import datetime


from gekko import GEKKO
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  
from numpy import pi
import math


ENV_NAME = "CartPole-v1"
TRAINING_EPISODE_TIME = 225
# REQ_MED_TRAIN_REWARD = 250
REQ_MED_TRAIN_REWARD = 350





# desired states
fo = 0
so = 2


# initialize gekko
m = GEKKO()
start = 0
stop = 4
dt = 0.02
nt = int((stop-start)/dt)
# m.time = np.linspace(0,5,nt)
m.time = np.linspace(start,stop,nt)



# Variables
theta = m.Var(value=0, lb= -12 * 2 * math.pi / 360, ub = 12 * 2 * math.pi / 360)
thetadot = m.Var(value=0)
x = m.Var(value=0)
xdot = m.Var(value=0)
force = m.MV(value=0, lb=-10, ub=10, integer=True)
force.STATUS = 1
min_func = m.Var(value=0)
p = np.ones(nt) # mark final time point
# p[-51:-1] = 1.0
final = m.Param(value=p)


#Defining system dynamics
g = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 0.5 # actually half the pole's length
polemass_length = (masspole * length)
force_mag = 10.0

costheta = m.cos(theta)
sintheta = m.sin(theta)


temp = (force + polemass_length * thetadot * thetadot * sintheta) / total_mass

#these are how the accelerations are calculated.  This (and temp above) are what I need to adjust to change where the force of the pole acts on the
# thetaacc = (g * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
# xacc  = temp - polemass_length * thetadot.dt() * costheta / total_mass


# Equations
m.Equation(thetadot == theta.dt())
m.Equation(xdot == x.dt())
m.Equation(thetadot.dt() == (g * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass)))
m.Equation(xdot.dt() == temp - polemass_length * thetadot.dt() * costheta / total_mass)

m.Obj((xdot - so)**2 + (theta - fo)**2) # Objective function
# m.Obj(min_func + 15*(q1*final - pi)**2 )

m.options.IMODE = 6 # optimal control mode


m.solve(disp=False) # solve

# plot results
plt.figure(1) 

plt.subplot(3,1,1)
plt.plot(m.time,theta.value,'k-',label=r'$theta$')
# plt.plot(m.time,x.value,'b-',label=r'$x$')
plt.ylabel('Angle')
plt.legend(loc='best')

plt.subplot(3,1,2)
plt.plot(m.time,xdot.value,'b-',label=r'$xdot$')
plt.ylabel(' Velocity')
plt.legend(loc='best')

plt.subplot(3,1,3)
plt.plot(m.time,force.value, 'r--o',label=r'$tau$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Input Force')
plt.show()

xlist = np.array([])
tlist = np.array([])
for i in range(0,len(xdot.value)):
	xlist = np.append(xlist, xdot.value[i])
	tlist = np.append(tlist, theta.value[i])


tlist[:] = [x - fo for x in tlist]
xlist[:] = [x - so for x in xlist]

xlist = np.absolute(xlist)
tlist = np.absolute(tlist)

average_xdot_err = round(tlist.mean(),2)
average_theta_err = round(xlist.mean(),2)
print(len(xdot.value))
print(average_xdot_err)



forceInput = np.array([])
for i in range(0,len(force.value)):
	end = len(force.value)
	if abs(force.value[i]) < 1:
		forceInput = np.append(forceInput,int(0))
	else:
		forceInput = np.append(forceInput,int(10*np.sign(force.value[i])))



def test_cartpole(model_name, num_tests, slow_d):
    # generate the environment
    env = gym.make(ENV_NAME)

    # Create the performance analyzer
    test_score_manager = FS_score(0,slow_d, model_name)
    test_score_manager.clear_fs_tests()
    

    observation_space = env.observation_space.shape[0]

    # Prep the environemnt
    state = env.reset()
    state = np.reshape(state, [1, observation_space])

    steps = 0
    run = 0 
    episode_reward = 0
    switch = True
    while(run<num_tests):  
         # save the state of the system
        test_score_manager.add_state(state[0])

        # Render the environment
        env.render()

        # # Determine and perform the action
        action = forceInput[steps]
        # if action > 5:
        # 	action = 2
        # elif action < -5:
        # 	action = 0
        # else:
        # 	action = 1


        #if less than zero, then switch back and forth
        if abs(action) < 1:
	        if steps % 2 == 0:
	        	action = 0
	        else:
	        	action = 1
        else:
	        action = int(np.sign(action))
	        if action < 0:
	        	action = 0



        state_next, reward, terminal, info = env.step(action)

        # Set the next action of the state
        state_next = np.reshape(state_next, [1, observation_space])
        state = state_next

        # increment the number of steps and add the episode reward
        steps +=1
        # sum_reward += dqn_solver.reward(state[0], reward_func)

        # When the run is finished:
        if terminal or steps>TRAINING_EPISODE_TIME:
            # Save the CSV
            test_score_manager.save_csv()

            # Add the run to the PNG
            test_score_manager.save_run(run, num_tests)
            test_score_manager.clear_run_data()

            # Reset the environment
            state = env.reset()
            state = np.reshape(state, [1, observation_space])


            print("steps: " + str(steps))
            print("reward: " + str(episode_reward))
            steps = 0
            run += 1
    test_score_manager.close_graphs()
slow_d = 3
num_tests = 1
model_name = 'gekko_' + str(slow_d)
test_cartpole(model_name, num_tests, slow_d)

