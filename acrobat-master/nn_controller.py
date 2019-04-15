import gym
import sys
import numpy as np
import argparse
from scores.score_logger import ScoreLogger
from scores.score_logger import FS_score
import random
import ipdb

from scores.score_logger import video
from dqn import DQNSolver


ENV_NAME = "Acrobot-v1"
DURATION = 100
REQ_MED_TRAIN_REWARD = 10


def train_acrobot(reward_func, model_name):
    # Define environment
    env = gym.make(ENV_NAME)

    # Identify observation/action space
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Initialize DQN controller
    dqn_solver = DQNSolver(observation_space, action_space)


    # initialize the test score manager
    test_score_manager = FS_score(dqn_solver.costheta1d,dqn_solver.costheta1dotd, model_name)

    #initialize array to keep track of reward
    rewards_list = np.array([])

    # Initialize the run counter
    run = 0

    while run < 1000 or (len(rewards_list) > 50 and np.median(rewards_list[-51:-1]) > REQ_MED_TRAIN_REWARD):
        
        # Increment the run counter
        run += 1

        # Reset the environment for the next episode
        state = env.reset()
        import ipdb
        # print(type(env))
        # ipdb.set_trace()
        # state = env.env.reset_train_2()
        state = np.reshape(state, [1, observation_space])

        # Initialize steps and episodic reward
        step = 0
        total_rew = 0


        while True:
            # increment the step counter
            step += 1

            # Render the environment
            # env.render()

            # Perform action
            action = dqn_solver.train_act(state)
            state_next, reward, terminal, info = env.step(action)

            # Get reward for performing that action
            reward = dqn_solver.reward(state, reward_func)
            total_rew += reward

            # Create transition to next state
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next

            # If duration exceeded
            if step > DURATION or state[0][0] < -0.95:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(total_rew))
                step = 0
                if state[0][0] < -0.95:
                    test_score_manager.add_reward(total_rew + 1000)
                    rewards_list = np.append(rewards_list,total_rew + 1000)

                else:
                    test_score_manager.add_reward(total_rew)
                    rewards_list = np.append(rewards_list,total_rew)


                total_rew = 0

                if len(rewards_list) > 52:
                    print("median reward of latest 50 = " + str(np.median(rewards_list[-51:-1])))
                else:
                    print("median reward = " + str(np.median(rewards_list)))

                # if reward has reached desired benchmark, quit the trainingfunction
                if len(rewards_list) > 50:
                    if np.median(rewards_list[-51:-1]) > 300:
                        # initialize the graph builder
                        test_score_manager = FS_score(dqn_solver.costheta1d,dqn_solver.costheta1dotd, model_name)

                        test_score_manager.save_reward(rewards_list)

                        return
                break

            dqn_solver.experience_replay()
        dqn_solver.save_model(model_name)
    

    test_score_manager.graph_reward()

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
        env.render()

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
        if steps > DURATION:
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

if __name__ == "__main__":
    args = sys.argv 
    # del args[0]

    #Parameters
    # reward_func = args[1];
    # train_acrobot(trained dynamic, reward function, model name)

    train_acrobot('linear','acrobot_get_to_top')
    # test_dual_DQN('fast_3_3_19', 'slow_3_3_19', 10)

    test_acrobot('acrobot_v3',10)
