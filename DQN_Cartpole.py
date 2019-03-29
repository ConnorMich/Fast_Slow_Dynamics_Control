import gym
import numpy as np
import random

#define nn weights, one for each state and action.  Also define the bias
w = np.zeros(4)
b = 0


class DQN:
    def __init__(self, name):
      env = gym.make(name)
      state_space = env.observation_space.shape[0]
      action_space = env.action_space.n

      #define the weights and biases
      weights = [0] * (state_space)
      biases = [0]

    def update_mini_batch(self, mini_batch, eta):
      """Update the network's weights and biases by applying
      gradient descent using backpropagation to a single mini batch.
      The "mini_batch" is a list of tuples "(x, y)", and "eta"
      is the learning rate."""
      nabla_b = [np.zeros(b.shape) for b in self.biases]
      nabla_w = [np.zeros(w.shape) for w in self.weights]
      for x, y in mini_batch:
          delta_nabla_b, delta_nabla_w = self.backprop(x, y)
          nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
          nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
      self.weights = [w-(eta/len(mini_batch))*nw 
                      for w, nw in zip(self.weights, nabla_w)]
      self.biases = [b-(eta/len(mini_batch))*nb 
                     for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
       """Return a tuple "(nabla_b, nabla_w)" representing the
       gradient for the cost function C_x.  "nabla_b" and
       "nabla_w" are layer-by-layer lists of numpy arrays, similar
       to "self.biases" and "self.weights"."""
       nabla_b = [np.zeros(b.shape) for b in self.biases]
       nabla_w = [np.zeros(w.shape) for w in self.weights]
       # feedforward
       activation = x
       activations = [x] # list to store all the activations, layer by layer
       zs = [] # list to store all the z vectors, layer by layer
       for b, w in zip(self.biases, self.weights):
           z = np.dot(w, activation)+b
           zs.append(z)
           activation = sigmoid(z)
           activations.append(activation)
        # backward pass
       delta = self.cost_derivative(activations[-1], y) * \
           sigmoid_prime(zs[-1])
       nabla_b[-1] = delta
       nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       # Note that the variable l in the loop below is used a little
       # differently to the notation in Chapter 2 of the book.  Here,
       # l = 1 means the last layer of neurons, l = 2 is the
       # second-last layer, and so on.  It's a renumbering of the
       # scheme in the book, used here to take advantage of the fact
       # that Python can use negative indices in lists.
       for l in xrange(2, self.num_layers):
           z = zs[-l]
           sp = sigmoid_prime(z)
           delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
           nabla_b[-l] = delta
           nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
       return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) 

    def sigmoid(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z)*(1-sigmoid(z))
        

def Q_NN(state):
	#takes state, returns the best action based on Q-Value
	if (np.sign(np.dot(state,w) + b)) == -1:
		return 0
	else:
		return 1

def run_episode(M):
	state = env.reset()
	done = False;
	while not done:
	#create the memory files:
		if random.randint(1,101)/100 < 0.7:
			#with .7 probability, choose the best decision
			action = Q_NN(state)
		else:
			#with 0.3 probability, choose a random action
			action = random.randint(0,1)

		next_state, reward, done, nad = env.step(action)
		if done:
			break
		M.append((state,action,reward,next_state))
		# M_for_b.append((state, reward))
		state = next_state



if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  DQN('CartPole-v0')

  M = [] #generate the empty list of tuples
  num_episodes = 50 #run 50 episodes

  for j in range(0,num_episodes):
    print(j)
    # run_episode(M)
    print("length of M:", end = " ")
  print(len(M))
  
  # print(M)
  # Successfully created the training tables!  Now, to pull the bins

  #resampling a selection:
  #convert to the M list to a numpy array
  M = np.array(M)
  resample_size = 20
  M_bin = []
  for k in range(0,resample_size):
    M_bin.append(M[random.randint(1,len(M))])

  #so, we can train the NN on the M_bin data points.
  # print(M)
  # update_mini_batch(M)