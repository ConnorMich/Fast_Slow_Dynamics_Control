import gym
import sys
import numpy as np
import argparse
from scores.score_logger import ScoreLogger
from scores.score_logger import FS_score
# from scores.score_logger import Test_Score

from scores.score_logger import video
from dqn import DQNSolver
import datetime
import os

ENV_NAME = "Acrobot-v1"
DURATION = 150


def test_acrobot(model_name, num_tests):
    # generate the environment
    env = gym.make(ENV_NAME)

    # define the observation and action spaces
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Create and Load the DQN Controller Model
    dqn_solver = DQNSolver(observation_space, action_space)
    dqn_solver.load_model(model_name)

    # Create the performance analyzer
    test_score_manager = FS_score(dqn_solver.costheta1d,dqn_solver.costheta1dotd, model_name)
    test_score_manager.clear_fs_states()

    # Prep the environemnt
    state = env.reset()
    state = np.reshape(state, [1, observation_space])

    # Test the Model num_tests of times
    i=0 
    steps = 0
    while(i<num_tests):  
        
        # Render the environment
        # env.render()

        # Determine and perform the action
        action = dqn_solver.test_act(state)
        state_next, reward, terminal, info = env.step(action)

        # save the state of the system
        test_score_manager.add_state(state[0],  action)

        # Set the next action of the state
        state_next = np.reshape(state_next, [1, observation_space])
        state = state_next
        steps += 1;


        # When the run is finished:
        if steps > 300 or state[0][0] < -0.99:
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

if __name__ == "__main__":
    directory = './test_scores/BreadthDQN/models 550/'
    names = np.array([])
    slow_d = np.array([])
    for filename in os.listdir(directory):
        # check1 = filename.rfind('X')
        # if (filename[check1 - 3:check1 - 1] == filename[check1 + 1:check1+3]) or (filename[check1 - 2:check1 - 1] == filename[check1 + 1:check1+2]):
            #finding slow dynamic
        if filename.endswith('.h5'):
            names = np.append(names, filename[0:len(filename)-3])

            sd_index = filename.rfind('_s') + 2
            print(filename)
            print(filename[sd_index])
            slow_d = np.append(slow_d, int(filename[sd_index]))


    for i in range(0,len(names)):
        test_acrobot(names[i],100)

