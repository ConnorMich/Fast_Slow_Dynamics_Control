import gym
import sys
import numpy as np
import argparse
from scores.score_logger import ScoreLogger
from scores.score_logger import FS_score
from scores.score_logger import Test_Score

from scores.score_logger import video
from dqn import DQNSolver
import datetime
import os

ENV_NAME = "CartPole-v1"
TRAINING_EPISODE_TIME = 225
# REQ_MED_TRAIN_REWARD = 250
REQ_MED_TRAIN_REWARD = 360


def test_cartpole(model_name, num_tests, slow_d):
    # generate the environment
    env = gym.make(ENV_NAME)

    # define the observation and action spaces
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Create and Load the DQN Controller Model
    dqn_solver = DQNSolver(observation_space, action_space, slow_d)
    dqn_solver.load_model(model_name)

    # Create the performance analyzer
    test_score_manager = FS_score(dqn_solver.pole_ang_d,dqn_solver.cart_vel_d, model_name)
    test_score_manager.clear_fs_tests()
    
    # Prep the environemnt
    state = env.reset()
    state = np.reshape(state, [1, observation_space])

    steps = 0
    run = 0 
    episode_reward = 0
    while(run<num_tests):  
         # save the state of the system
        test_score_manager.add_state(state[0])

        # Render the environment
        # env.render()

        # Determine and perform the action
        action = dqn_solver.test_act(state)
        state_next, reward, terminal, info = env.step(action)
        episode_reward += dqn_solver.reward(state, 'linear')

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
            episode_reward = 0
            run += 1
    test_score_manager.close_graphs()


if __name__ == "__main__":
    directory = './models/'
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

    print(slow_d)
    print((names))

    for i in range(0,len(names)):
        print(str(names[i]) + '___' + str(slow_d[i]))
        test_cartpole(names[i],10, slow_d[i])

