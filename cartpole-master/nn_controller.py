import gym
import sys
import numpy as np
import argparse
from scores.score_logger import ScoreLogger
from scores.score_logger import FS_score
from scores.score_logger import Test_Score

from scores.score_logger import video
from dqn import DQNSolver


ENV_NAME = "CartPole-v1"


def train_cartpole(trained_dynamic, reward_func, model_name):
    env = gym.make(ENV_NAME)
    # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    # fs_score_logger = FS_score(dqn_solver.pole_ang_d,dqn_solver.cart_vel_d) # the desired fast slow dynamics are 0,5
    # vid_manager = video(fs_score_logger.FS_PNG_SINGLE)
    # fs_score_logger.clear_fs_scores()

    run = 0
    i = 0;
    rewards_list = np.array([])

    while i < 200:
        i = i + 1
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        # fs_score_logger.add_state(state[0])

        step = 0
        episode_reward = 0
        av_ep_reward = 0
        while True:
            step += 1
            # env.render()
            action = dqn_solver.train_act(state)
            state_next, reward, terminal, info = env.step(action)

            #reward function
            reward = dqn_solver.linear_reward_function(state[0])
            episode_reward += reward


            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next

            #store the next state
            # fs_score_logger.add_state(state[0])

            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(episode_reward))
                rewards_list = np.append(rewards_list,episode_reward)
                episode_reward = 0
                print("average reward = " + str(np.sum(rewards_list)/len(rewards_list)))
                if sum(rewards_list)/len(rewards_list) > 350:
                    exit()
                # score_logger.add_score(step, run)
                # fs_score_logger.save_last_run()
                # vid_manager.add_frame()
                # fs_score_logger.calculate_episodic_error()
                # fs_score_logger.add_episodic_error()
                break
            dqn_solver.experience_replay()
        dqn_solver.save_model(model_name)

    # fs_score_logger.save_error_png()
    # vid_manager.stop_video()

def test_cartpole(model_name, num_tests):
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    dqn_solver.load_model(model_name)

    # score_logger = ScoreLogger(ENV_NAME)
    test_score_manager = Test_Score(dqn_solver.pole_ang_d,dqn_solver.cart_vel_d) # the desired fast slow dynamics are 0,5
    # vid_manager = video(fs_score_logger.FS_PNG_SINGLE)
    test_score_manager.clear_test_scores()

    state = env.reset()
    state = np.reshape(state, [1, observation_space])

    steps = 0
    i=0 
    sum_reward = 0
    while(i<num_tests):  
        test_score_manager.add_state(state[0])
        env.render()

        action = dqn_solver.test_act(state)
        state_next, reward, terminal, info = env.step(action)
        state_next = np.reshape(state_next, [1, observation_space])

        steps +=1
        sum_reward += dqn_solver.linear_reward_function(state[0])
        
        state = state_next

        if terminal:
            state = env.reset()
            test_score_manager.save_last_run()
            state = np.reshape(state, [1, observation_space])
            # test_score_manager.calculate_episodic_error()
            i = i + 1
            print("steps: " + str(steps))
            print("reward: " + str(sum_reward))
            steps = 0
            sum_reward = 0


if __name__ == "__main__":
    args = sys.argv 
    # del args[0]

    #Parameters
    # reward_func = args[1];
    # train_cartpole(trained dynamic, reward function, model name)

    # train_cartpole('fast-slow','linear','fast_slow_4_1_19_reverted')
    test_cartpole('fast_slow_4_1_19_reverted',10)
