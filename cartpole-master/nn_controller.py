import gym
import sys
import numpy as np
import argparse
from scores.score_logger import ScoreLogger
from scores.score_logger import FS_score
from scores.score_logger import Test_Score
import random
import ipdb

from scores.score_logger import video
from dqn import DQNSolver


ENV_NAME = "CartPole-v1"


def train_cartpole(trained_dynamic, reward_func, model_name):

    # Create Environment
    env = gym.make(ENV_NAME)

    # Identify observation/action space
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Create score_logger to keep track of when to terminate training
    score_logger = ScoreLogger(ENV_NAME)

    # Initialize DQN controller
    dqn_solver = DQNSolver(observation_space, action_space)

    # Initialize video for recording training sequences
    # vid_manager = video(fs_score_logger.FS_PNG_SINGLE)

    run = 0
    while run < 150:
        run += 1

        # Prep the environment
        state = env.reset()
        state = np.reshape(state, [1, observation_space])

        step = 0
        while True:
            step += 1
            # env.render()

            # Decide and perform action
            action = dqn_solver.train_act(state)
            state_next, reward, terminal, info = env.step(action)
            state_next = np.reshape(state_next, [1, observation_space])


            # Determine reward for action & remember the experience
            reward = dqn_solver.reward(state, trained_dynamic, reward_func)
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next

            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                score_logger.add_score(step, run)

                break

            #Learn from previous iterations
            dqn_solver.experience_replay()
        #save the model; note this saves every iteration, so it can be quit halfaway and still tested
        dqn_solver.save_model(model_name)

def test_cartpole(model_name, num_tests):
    # generate the environment
    env = gym.make(ENV_NAME)

    # define the observation and action spaces
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Create and Load the DQN Controller Model
    dqn_solver = DQNSolver(observation_space, action_space)
    dqn_solver.load_model(model_name)

    # Create the performance analyzer
    test_score_manager = FS_score(dqn_solver.pole_ang_d,dqn_solver.cart_vel_d, model_name)
    test_score_manager.clear_fs_states()

    # Prep the environemnt
    state = env.reset()
    state = np.reshape(state, [1, observation_space])

    # Test the Model num_tests of times
    i=0 
    while(i<num_tests):  
        # save the state of the system
        test_score_manager.add_state(state[0])
        
        # Render the environment
        # env.render()

        # Determine and perform the action
        action = dqn_solver.test_act(state)
        state_next, reward, terminal, info = env.step(action)

        # Set the next action of the state
        state_next = np.reshape(state_next, [1, observation_space])
        state = state_next


        # When the run is finished:
        if terminal:

            # Save the CSV
            test_score_manager.save_csv()

            # Add the run to the PNG
            test_score_manager.save_run(i, num_tests)
            test_score_manager.clear_run_data()

            # Reset the environment
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            i = i + 1

def test_dual_DQN(fast_model_name, slow_model_name, num_tests):
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver_slow = DQNSolver(observation_space, action_space)
    dqn_solver_fast = DQNSolver(observation_space, action_space)

    dqn_solver_slow.load_model(slow_model_name)
    dqn_solver_fast.load_model(fast_model_name)

    # score_logger = ScoreLogger(ENV_NAME)
    test_score_manager = Test_Score(dqn_solver_fast.pole_ang_d,dqn_solver_fast.cart_vel_d, 'iterative_dual_control') # the desired fast slow dynamics are 0,5
    # vid_manager = video(fs_score_logger.FS_PNG_SINGLE)
    test_score_manager.clear_test_scores()




    state = env.reset()
    state = np.reshape(state, [1, observation_space])

    actions = []
    fasts = 0
    slows = 0
    i=0 
    j = 0
    while(i<num_tests):  
        test_score_manager.add_state(state[0])
        env.render()

        # if len(actions) < 3:
        #     actions.append(dqn_solver_fast.test_act(state))
        #     actions.append(dqn_solver_slow.test_act(state))
        #     actions.append(dqn_solver_fast.test_act(state))
        # else:
        #     actions.pop()

        #     if j % 2 == 0:
        #         slows = slows + 1
        #         actions.insert(dqn_solver_slow.test_act(state),0)

        #     else:
        #         fasts = fasts + 1
        #         actions.insert(dqn_solver_fast.test_act(state),0)
        # action = np.int64(np.median(actions))



        if j % 2 == 0:
            slows = slows + 1
            action = dqn_solver_slow.test_act(state)
        else:
            fasts = fasts + 1
            action = dqn_solver_fast.test_act(state)

        state_next, reward, terminal, info = env.step(action)
        state_next = np.reshape(state_next, [1, observation_space])
        state = state_next
        j = j+1

        if terminal:
            state = env.reset()
            test_score_manager.save_last_run()
            state = np.reshape(state, [1, observation_space])
            # test_score_manager.calculate_episodic_error()
            i = i + 1
            j = 0
            print('slows' , slows)
            print('fasts' ,fasts)


if __name__ == "__main__":
    args = sys.argv 
    # del args[0]

    #Parameters
    # reward_func = args[1];

    # train_cartpole(trained dynamic, reward function, model name)
    # train_cartpole('fast-slow','linear','fast_slow_3_29_19')
    train_cartpole('slow', 'linear', 'slow_3_29_19')



    # test_cartpole('slow_3_29_19',10)

    # test_cartpole('fast_slow_3_3_19',10)
