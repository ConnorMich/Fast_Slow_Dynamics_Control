import gym
import sys
import control
import numpy as np
import ipdb
from scores.score_logger import FS_score
from sympy import *
from simple_pid import PID
import datetime


class classicController:
    def __init__(self, slow_d):
        self.fast_d = 0

        #stay still
        # self.slow_d = -5 + 5*0.2

        #move
        self.slow_d = slow_d


        self.desired_states = np.matrix([[0], [slow_d], [0], [self.fast_d]])
        #define our matrices for the linearized model, which we have
        M = 0.5  #cart mass
        m = 0.2  #pedulum mass
        b = 0.1  #coefficient of friction
        I = 0.006 #inertia
        g = 9.8 #gravity
        l = 0.3 #length of the pendulum
        k=I*(M+m)+M*m*l**2.0;
        U2=(I+m*l**2.0)/k;
        U4=-m*l/k;
        TF_2_3 = -(g*m**2.0*l**2.0)/k;
        TF_4_3 = (g*(m+M))/k;
        # TF_4_3 = (m*l*(M+m))*g/k;


        self.B=[[0], [U2], [0],[U4]];
        self.A=[[0, 1, 0, 0],
                [0, 0, TF_2_3, 0],
                [0, 0, 0, 1],
                [0, 0, TF_4_3, 0]]
        self.C=[[1, 0, 0, 0],[0, 0, 1, 0]];
        self.D = [[0],[0]];
        self.sys = control.StateSpace(self.A, self.B, self.C, self.D)
        # R = 1
        # Q = [[20, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0,]]

        #develop our fast dynamic controller (Using pole placement)
        self.K = control.place(self.A, self.B, [-5.1,-5.2,-5.3,-5.4])
        self.K = np.matrix(self.K.tolist()[0])
        print("K = ", np.matrix(self.K))
        print("A = ", np.matrix(self.A))
        # print("B = ",np.matrix(self.B))
        # print(np.linalg.eig(np.subtract(self.A, np.matmul(self.B,self.K))))
        # print(np.subtract(self.A, np.matmul(self.B,self.K)))
        exit()

        self.closed_loop = np.subtract(self.A, np.matmul(self.B,self.K))
        self.K2 = control.place(self.closed_loop, self.B, [-5.1,-5.2,-5.3,-5.4])



        #deevelop our slow dynamic controller (PID)
        self.pid = PID(2, 1, 0.0)

    def fast_act(self, state):
        state = np.transpose(np.matrix(state[0]))
        print(state)
        u = 0 + np.dot(self.K,state)[0,0]
        u = -np.sign(u)
        return np.int64(0) if u == -1 else np.int64(u)

    def fast_slow_act(self, state):
        state = np.transpose(np.matrix(state[0]))
        e = np.subtract(self.desired_states, state)

        #ensures the velocity of the cart doesn't affect control 
        e[0,0] = 0

        u = np.dot(self.K,e)[0,0]
        u = np.sign(u)


        return np.int64(0) if u == -1 else np.int64(u)


    def fast_slow_act_1(self, state):
        #trying full state feedback using pole placement
        velocity_setpoint = self.slow_d
        state = np.transpose(np.matrix(state[0]))
        t1 = np.dot(self.K,state)[0,0]
        t2 = np.dot(self.K2,state)[0,0]
        u = velocity_setpoint - t1 - t2
        u = np.sign(u)
        return np.int64(0) if u == -1 else np.int64(u)

    def fast_slow_act_2(self, state):
        #try using a PID to set position
        state = np.transpose(np.matrix(state[0]))

        #trying
        velocity_setpoint = self.slow_d
        current_velocity =state[1,0]
        e = velocity_setpoint - current_velocity

        pidout = -self.pid(e)

        self.desired_states = np.matrix([[0], [pidout], [0], [self.fast_d]])
        e = np.subtract(self.desired_states, state)

        #ensures the velocity of the cart doesn't affect control 
        e[0,0] = 0
        u = np.dot(self.K,e)[0,0]
        u = np.sign(u)


        return np.int64(0) if u == -1 else np.int64(u)

    def slow_act(self,state):
        # getting the velocity variable
        state = state[0][1]

        # finding the error signal
        e = self.slow_d - state

        #determine the pid output
        u = self.pid(e)

        #send it
        u = np.sign(u)
        return np.int64(0) if u == 1 else np.int64(1)

def test_cartpole(modelname, num_tests, slow_d):
    # generate the environment
    env = gym.make("CartPole-v1")

    # define the observation and action spaces
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Create the DQN Controller Model
    classCont = classicController(slow_d)

    # Create the performance analyzer
    test_score_manager = FS_score(classCont.fast_d,classCont.slow_d, modelname) # the desired fast slow dynamics are 0,5
    test_score_manager.clear_fs_tests()

    # Prep the environemnt
    state = env.reset()
    state = np.reshape(state, [1, observation_space])


    # Test the Model num_tests of times
    i = 0
    while(i<num_tests):
        # save the state of the system
        test_score_manager.add_state(state[0])

        # Render the environment
        env.render()

        #for fast classic control
        action = classCont.fast_slow_act_2(state)

        # for iterative classic control
        # if i % 3 == 0:
        #     action = classCont.fast_act(state)
        # else:
        #     action = classCont.slow_act(state)

        # # for iterative classic control
        # if i % 4 == 0:
        #     action = classCont.slow_act(state)
        # else:
        #     action = classCont.fast_act(state)
       

        # Set the next action of the state
        state_next, reward, terminal, info = env.step(action)
        state_next = np.reshape(state_next, [1, observation_space])
        state = state_next

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
    test_score_manager.close_graphs()
    env.close()



if __name__ == "__main__":
    args = sys.argv 
    # del args[0]

    #Parameters
    # reward_func = args[1];
    # train_cartpole(trained dynamic, reward function, model name)

    # train_cartpole('slow','linear','slow_3_3_19')
    # test_dual_DQN('fast_3_3_19', 'slow_3_3_19', 10)
    now = datetime.datetime.now()
    timestamp = str(now.month)+ "_" + str(now.day) + "_" + str(now.year) + "_"
    slow_dynamics = [ 1, 2, 3, 4]
    # slow_dynamics = [1]
    for slow_d in slow_dynamics:
        print("test: " + str(slow_d))
        name = 'classic_controller_' + timestamp + '_slow_dynamic_' +str(slow_d)
        test_cartpole(name,1, slow_d)

