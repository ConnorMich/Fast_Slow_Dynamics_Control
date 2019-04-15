import gym
import control
import numpy as np
import ipdb
from scores.score_logger import Test_Score
from sympy import *
from simple_pid import PID


class classicController:
    def __init__(self):
        self.input = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True) = 0

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

    test_score_manager = Test_Score(classCont.fast_d,classCont.slow_d, 'FastClassicController') # the desired fast slow dynamics are 0,5
    test_score_manager.clear_test_scores()
    # while True:
    state = env.reset()
    state = np.reshape(state, [1, observation_space])

    i = 0

    while True:
        test_score_manager.add_state(state[0])

        env.render()


        #for fast classic control
        action = classCont.fast_act(state)

        #for iterative classic control
        # if i % 5 == 0:
        #     action = classCont.fast_act(state)
        # else:
        #     action = classCont.slow_act(state)

        state_next, reward, terminal, info = env.step(action)
        state_next = np.reshape(state_next, [1, observation_space])
        state = state_next
        if terminal:
            test_score_manager.save_last_run()
            break
    env.close()

    # print(classCont.err)
# cc = classicController()
cartpole()
