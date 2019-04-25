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
EXPLORATION_DECAY = 0.995

class DQNSolver:
    
    def __init__(self, observation_space, action_space, nn_bredth = 2, nn_depth = 2):
        nn_bredth = nn_bredth if nn_bredth is not None else 2
        nn_depth = nn_depth if nn_depth is not None else 2


        self.costheta1d = np.float64(-1)
        self.costheta1dotd = np.float64(0)
        # self.pole_ang_d = np.float64(0)

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

    def reward(self,state, reward_func):
            if reward_func == 'linear':
                return self.linear(state[0])
            elif reward_func == 'exponential':
                return self.exponential(state[0])
            else:
                raise Exception(reward_func + " not defined as a valid reward function type")

    def linear(self, state):
        costheta1 = state[0]
        sintheta1 = state[1]
        costheta2 = state[2]
        sintheta2 = state[3]
        thetaDot1 = state[4]
        thetaDot2 = state[5]

        #m is maximum reward
        m = 10
        reward = -m*costheta1

        # if costheta1 < -0.8:
        #     reward +=  -m*abs(thetaDot1)+m

        # if costheta1 < -0.01:
        #     reward += abs(costheta1)* (-m*abs(thetaDot1)+m)
        #     emphasise this!
        #     reward += 10*np.exp(-5*abs(thetaDot1))
        return reward

    def exponential(self, state):
        costheta1 = state[0]
        sintheta1 = state[1]
        costheta2 = state[2]
        sintheta2 = state[3]
        thetaDot1 = state[4]
        thetaDot2 = state[5]

        #m is maximum reward
        return 10/(costheta1+1.5)-6

    def save_model(self, name):
        self.model.save('./models/'+name+'.h5')
        

    def load_model(self, name):
        del self.model
        self.model = load_model('./models/' + name +'.h5')

