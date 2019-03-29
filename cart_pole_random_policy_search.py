# code taken from https://medium.com/@m.alzantot/deep-reinforcement-learning-demystified-episode-0-2198c05a6124

import gym
import numpy as np

# a function that returns a random set of policy parameters.
def gen_random_policy():
	return (np.random.uniform(-1,1, size=4), np.random.uniform(-1,1))


# a function that selects an action for the agent who follows the given 
# policy will chose when it observes the state observation given by obs
def policy_to_action(env, policy, obs):
    if np.dot(policy[0], obs) + policy[1] > 0:
    	return 1
    else:
    	return 0


# a function that starts an episode of maximum length t_max timesteps 
# and returns the total reward that an agent following the given policy has recieved.
def run_episode(env, policy, t_max=1000, render=False):
    obs = env.reset()
    total_reward = 0
    for i in range(t_max):
        if render:
            env.render()
        selected_action = policy_to_action(env, policy, obs)
        obs, reward, done, _ = env.step(selected_action)
        total_reward += reward
        if done:
            break
    return total_reward

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	
	## Generate a pool or random policies
	n_policy = 500
	policy_list = [gen_random_policy() for _ in range(n_policy)]
	
	# Evaluate the score of each policy.
	scores_list = [run_episode(env, p) for p in policy_list]
	
	# Select the best policy.
	print('Best policy score = %f' %max(scores_list))
	
	best_policy= policy_list[np.argmax(scores_list)]
	print('Running with best policy:\n')
	run_episode(env, best_policy, render=True)