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


ENV_NAME = "CartPole-v1"
TRAINING_EPISODE_TIME = 225
# REQ_MED_TRAIN_REWARD = 250
REQ_MED_TRAIN_REWARD = 360
MAX_TRAIN_RUNS = 400

def train_cartpole(trained_dynamic, reward_func, model_name, nn_bredth, nn_depth, slow_d):
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space, nn_bredth, nn_depth, slow_d)

    #initialize run counter
    run = 0

    #initialize array to keep track of reward
    rewards_list = np.array([])

    while run < MAX_TRAIN_RUNS:
        # increment the run counter
        run += 1

        # initialize the environment and state variables
        state = env.reset()
        state = np.reshape(state, [1, observation_space])

        # initialize the steps & episode reward
        step = 0
        episode_reward = 0

        while True:
            # inrement the step counter
            step += 1

            # render the environment
            # env.render()

            # perform the training action
            action = dqn_solver.train_act(state)
            state_next, reward, terminal, info = env.step(action)

            # reward function
            reward = dqn_solver.reward(state, reward_func)
            episode_reward += reward

            # create transition to next state
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next

            # test if the simulation should terminate
            if terminal or step>TRAINING_EPISODE_TIME:
                # Print message to the terminal
                print("Run: " + str(run) + ", run length: " + str(step) +", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(episode_reward))
                
                # keep track of the reward
                rewards_list = np.append(rewards_list,episode_reward)

                # reset the episode reward
                episode_reward = 0



                if len(rewards_list) > 100:
                    print("median reward of latest 50 = " + str(np.median(rewards_list[-51:-1])))
                else:
                    print("median reward = " + str(np.median(rewards_list)))

                # if reward has reached desired benchmark, quit the trainingfunction
                if len(rewards_list) > 50:
                    if np.median(rewards_list[-51:-1]) > REQ_MED_TRAIN_REWARD or run == MAX_TRAIN_RUNS - 1:
                        # initialize the graph builder
                        test_score_manager = FS_score(dqn_solver.pole_ang_d,dqn_solver.cart_vel_d,model_name)

                        test_score_manager.save_reward(rewards_list)

                        return
                break
            # train the model
            dqn_solver.experience_replay()
        #save the model.  Note, this will overwrite the model every time, in case training is interupted
        dqn_solver.save_model(model_name)

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
    args = sys.argv 
    now = datetime.datetime.now()

    # del args[0]

    #Parameters
    # reward_func = args[1];
    # train_cartpole(trained dynamic, reward function, model name)

    # nn_bredth = 2
    # nn_depth = 50
    # slow_d = 2
    # name = 'fast_slow_4_8_19_' + str(nn_bredth) + 'X' + str(nn_depth) + '_s'+str(slow_d)
    # train_cartpole('fast-slow','linear',name, slow_d, nn_bredth, nn_depth)
    # test_cartpole(name,10)

    # old number of tests
    # tests = [70, 50, 30, 10, 6, 3]

    # what I think we should try
    # tests = [150, 110, 90, 70, 50, 30]



    # # tests = [2^9, 2^8, 2^7, 2^6, 2^5]
    # tests = [50]

    # timestamp = str(now.month)+ "_" + str(now.day) + "_" + str(now.year) + "_"
    # slow_d = 4
    # name = 'nfast_slow_' + timestamp + str(70) + 'X' + str(2) + '_s'+str(slow_d) + '_min_med_rew_' + str(REQ_MED_TRAIN_REWARD)
    # test_cartpole(name,10, slow_d)

    # slow_dynamics = [2, 3, 4]
    tests = [110, 90, 70, 50, 30]
    slow_dynamics = [1,2]

    # testing with varying depth and slow dynamic seperations
    timestamp = str(now.month)+ "_" + str(now.day) + "_" + str(now.year) + "_"

    for slow_d in slow_dynamics:
        for t in tests:
            print("test: " + str(t))
            print(str(slow_d) + ', slow dynamic.  depth: ' + str(t))
            nn_bredth = 3
            nn_depth = t
            name = 'fast_slow_' + timestamp + str(nn_bredth) + 'X' + str(nn_depth) + '_s'+str(slow_d) + '_min_med_rew_' + str(REQ_MED_TRAIN_REWARD)
            train_cartpole('fast-slow','linear',name, slow_d, nn_bredth, nn_depth)
            test_cartpole(name,10, slow_d)

