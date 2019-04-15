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
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.9985


class DQNSolver:
    def __init__(self, observation_space, action_space, slow_d = 2, nn_bredth = 2, nn_depth = 2):
        nn_bredth = nn_bredth if nn_bredth is not None else 2
        nn_depth = nn_depth if nn_depth is not None else 2



        self.cart_vel_d = np.float64(slow_d)
        self.pole_ang_d = np.float64(0)

        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(nn_bredth, input_shape=(observation_space,), activation="relu"))
        
        for i in range(0,nn_depth-1):
            self.model.add(Dense(nn_bredth, activation="relu"))
            # print("there are: +1 layers")

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

    def reward(self,state, reward_func):
            if reward_func == 'linear':
                return self.linear(state[0])
            elif reward_func == 'exponential':
                return self.exponential(state[0])
            else:
                raise Exception(reward_func + " not defined as a valid reward function type")

    def linear(self, state):
        cart_pos = state[0]
        cart_vel = state[1]
        pole_ang = state[2]
        pole_vel = state[3]
        reward = 0;

        #Linear reward functions

        #Reward the slow dynamic
        # reward += -0.5*abs(cart_vel-self.cart_vel_d) + 1

        max_slow_reward = 2
        x2,y2 = self.cart_vel_d,max_slow_reward
        m = (y2)/(x2)
        reward += m*(-1)*abs(cart_vel-self.cart_vel_d) + max_slow_reward


        # for slow = 1
        # reward += -abs(cart_vel-self.cart_vel_d) + 2


        # #Reward the fast dynamic
        # lim = 12 * 2 * math.pi / 360 #the 12 degree angle limit in radians
        # x1,y1 = 0,1
        # x2,y2 = lim,0
        # m = (y2-y1)/(x2-x1)
        # b = y1
        # if pole_ang > 0:
        #     reward+= m*pole_ang +b
        # else:
        #     reward+= -m*pole_ang +b

        #Step Function Reward for the fast dynamic
        if abs(pole_ang) < 12 * 2 * math.pi / 360:
            reward += 1

        return reward

    def exponential(self, state):
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

