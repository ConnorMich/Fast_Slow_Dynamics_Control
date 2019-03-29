import gym
import control
import numpy as np
import ipdb
from scores.score_logger import Test_Score
from sympy import *
from simple_pid import PID


class classicController:
    def __init__(self):
        self.fast_d = 0
        self.slow_d = 2
        #define our matrices for the linearized model, which we have
        M = 0.5  #cart mass
        m = 0.2  #pedulum mass
        b = 0.1  #coefficient of friction
        I = 0.006 #inertia
        g = 9.8 #gravity
        l = 0.3 #length of the pendulum
        k=I*(M+m)+M*m*l**2.0;
        U2=(I+m*l**2.0)/k;
        U4=-m*l/k;
        TF_2_3 = -(g*m**2.0*l**2.0)/k;
        TF_4_3 = (g*(m+M))/k;
        # TF_4_3 = (m*l*(M+m))*g/k;


        self.B=[[0], [U2], [0],[U4]];
        self.A=[[0, 1, 0, 0],
                [0, 0, TF_2_3, 0],
                [0, 0, 0, 1],
                [0, 0, TF_4_3, 0]]
        self.C=[[1, 0, 0, 0],[0, 0, 1, 0]];
        self.D = [[0],[0]];
        self.sys = control.StateSpace(self.A, self.B, self.C, self.D)
        # R = 1
        # Q = [[20, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0,]]

        #develop our fast dynamic controller (Using pole placement)
        self.K = control.place(self.A, self.B, [-5.1,-5.2,-5.3,-5.4])
        self.K = np.matrix(self.K.tolist()[0])

        #deevelop our slow dynamic controller (PID)
        self.pid = PID(5, 0.1, 0.05)

    def fast_act(self, state):
        state = np.transpose(np.matrix(state[0]))
        u = 0 + np.dot(self.K,state)[0,0]
        u = -np.sign(u)


        return np.int64(0) if u == -1 else np.int64(u)

    def slow_act(self,state):
        # getting the velocity variable
        state = state[0][1]

        # finding the error signal
        e = self.slow_d - state

        #determine the pid output
        u = self.pid(e)

        #send it
        u = np.sign(u)
        return np.int64(0) if u == 1 else np.int64(1)





def cartpole():
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    classCont = classicController()

    # while True:
    state = env.reset()
    state = np.reshape(state, [1, observation_space])

    i = 0

    while True:

        env.render()

        #for fast classic control
        action =  np.int64(1)


        # #for iterative  control
        # if i < 100:
        #     action = np.int64(1)

        # i = i+1
        print(action)

        state_next, reward, terminal, info = env.step(action)
        state_next = np.reshape(state_next, [1, observation_space])
        state = state_next
        if terminal:
            break
    env.close()

    # print(classCont.err)
cartpole()
