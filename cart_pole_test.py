import gym
import numpy as np

#1. Load environment and Q-Table structure
env = gym.make('CartPole-v0')
q_table = np.zeros([env.observation_space.n,env.action_space.n])


# 2. Parameters of Q-leanring
eta = .628
epsilon = 0.1
alpha = 0.1
gamma = 0.6
episodes = 5000
rev_list = [] # rewards per episode calculate
all_epochs = []
all_penalties = []

# 3. Q-learning Algorithm
for i_episode in range(episodes):
    state = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = false
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        env.render()
        next_state, reward, done, info = env.step(action) 

        #update the q table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

print("Training finished");
