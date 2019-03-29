import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import plot_model


import math
import random




GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
# EXPLORATION_MIN = 0.01
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.999995


class DQNSolver:
    
    def __init__(self, observation_space, action_space):
        self.cart_vel_d = np.float64(2)
        self.pole_ang_d = np.float64(0)

        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))

        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        #tweaked to encourage movement in the +x direction
        
        self.memory.append((state, action, reward, next_state, done))

    def train_act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def test_act(self,state):
        # import ipdb
        # ipdb.set_trace()
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    # class reward_function:
    #     def __init__(self, dynamics, function_type):
    #         self.dynamics = dynamics
    #         self.function_type = function_type
    #     def get_reward(self, state):
    #         if self.function_type == 'linear':
    #             if  self.dynamics == "fast-slow":
    #                 return dqn_solver.linear_reward_function(state[0])
    #             elif self.dynamics == 'fast':
    #                 return = dqn_solver.linear_reward_function(state[0])
    #             elif self.dynamics == 'slow':
    #                 return = dqn_solver.linear_reward_function(state[0])
    #             else:
    #                 raise Exception("The system must be trained on fast dynamic, slow dynamic, or some combination")
    #         elif self.function_type == 'exponential':
    #             if self.dynamics == "fast-slow":
    #                 return = dqn_solver.exponential_reward_function(state[0])
    #             elif self.dynamics == 'fast':
    #                 return = dqn_solver.exponential_reward_function(state[0])
    #             elif self.dynamics == 'slow':
    #                 return = dqn_solver.exponential_reward_function(state[0])
    #             else:
    #                 raise Exception("The system must be trained on fast dynamic, slow dynamic, or some combination")
    #         else:
    #             raise Exception(reward_func + " not defined as a valid reward function type")

    def reward(self,state,dynamics, reward_func):
            if reward_func == 'linear':
                if  dynamics == "fast-slow":
                    return self.linear_reward_slow_function(state) + self.linear_reward_fast_function(state[0]) 
                elif dynamics == 'fast':
                    return self.linear_reward_fast_function(state)
                elif dynamics == 'slow':
                    return self.linear_reward_slow_function(state)
                else:
                    raise Exception("The system must be trained on fast dynamic, slow dynamic, or some combination")
            elif reward_func == 'exponential':
                if dynamics == "fast-slow":
                    return self.exponential_reward_fast_function(state) + self.exponential_reward_slow_function(state[0])
                elif dynamics == 'fast':
                    return self.exponential_reward_fast_function(state)
                elif dynamics == 'slow':
                    return self.exponential_reward_slow_function(state)
                else:
                    raise Exception("The system must be trained on fast dynamic, slow dynamic, or some combination")
            else:
                raise Exception(reward_func + " not defined as a valid reward function type")

    def linear_reward_slow_function(self, state):
        cart_pos = state[0]
        cart_vel = state[1]
        pole_ang = state[2]
        pole_vel = state[3]

        #Linearly reward the slow dynamic
        if cart_vel > self.cart_vel_d:
            return -(1/self.cart_vel_d)*(cart_vel) + 2
        else:
            return (1/self.cart_vel_d)*(cart_vel)

    def linear_reward_fast_function(self, state):
        cart_pos = state[0]
        cart_vel = state[1]
        pole_ang = state[2]
        pole_vel = state[3]

        #Linearly reward the fast dynamic
        lim = 12 * 2 * math.pi / 360 #the 12 degree angle limit in radians
        x1,y1 = 0,1
        x2,y2 = lim,0
        m = (y2-y1)/(x2-x1)
        b = y1
        if pole_ang > 0:
            return m*pole_ang +b
        else:
            return -m*pole_ang +b

        #Step Function Reward for the fast dynamic
        # if abs(pole_ang) > 12 * 2 * math.pi / 360:
        #     reward = 0
        # else:
        #     reward+=1

    def exponential_reward_slow_function(self, state):
        cart_pos = state[0]
        cart_vel = state[1]
        pole_ang = state[2]
        pole_vel = state[3]

        #Reward the slow dynamic
        if cart_vel > self.cart_vel_d:
            return np.power((-cart_vel + self.cart_vel_d),2)
        else:
            return np.power((cart_vel - self.cart_vel_d),2)

    def exponential_reward_fast_function(self, state):
        cart_pos = state[0]
        cart_vel = state[1]
        pole_ang = state[2]
        pole_vel = state[3]

        # Reward for the fast dynamic
        if pole_ang > self.pole_ang_d:
            return np.power(-pole_ang,2)
        else:
            return np.power(pole_ang,2)

    def exponential_reward_function(self, state):
        cart_pos = state[0]
        cart_vel = state[1]
        pole_ang = state[2]
        pole_vel = state[3]
        reward = 0;
        #Exponential reward functions

        #Reward the slow dynamic
        if cart_vel > self.cart_vel_d:
            reward+= np.power((-cart_vel + self.cart_vel_d),2)
        else:
            reward+= np.power((cart_vel - self.cart_vel_d),2)

        # Reward for the fast dynamic
        if pole_ang > self.pole_ang_d:
            reward+= np.power(-pole_ang,2)
        else:
            reward+= np.power(pole_ang,2)
        

        return reward

    def save_model(self, name):
        self.model.save('./models/'+name+'.h5')
        

    def load_model(self, name):
        del self.model
        self.model = load_model('./models/' + name +'.h5')

