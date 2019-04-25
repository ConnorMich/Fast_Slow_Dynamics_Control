import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
filepath = './test_scores/DepthDQN/test_results/rewards_csv_for_depth_DQN.txt'

reward = []
# i = 0
# with open(filepath) as fp:  
#    line = fp.readline()
#    while i < 20:
#     i = i+1
#     print((line.strip()))
#     line = fp.readline()



with open(filepath) as fp:  
   line = fp.readline()
   while line:
     while line.strip().rfind('slow dynamic') < 0:
      line = fp.readline()

     print('found:   ' + str(line.strip()))
     slow_d = line.strip()[line.strip().rfind('slow dynamic')-3]
     depth = line.strip()[line.strip().rfind('depth: ')+7:len(line.strip())-1]   
     print('slow_d: ' + str(slow_d) + ' size: 3x' + str(depth))
     line = fp.readline()
     while(line.strip().rfind('slow dynamic') < 0):
         ind = line.strip().rfind('score: ')
         if ind > 0:
          reward.append(float((line.strip()[ind+7:len(line.strip())-1])))
         line = fp.readline()
     ind = np.arange(0,len(reward))
     print(len(reward))
     plt.plot(ind,reward)
     plt.xlabel('Iterations')
     plt.ylabel('Reward')
     plt.savefig('./test_scores/DepthDQN/test_results/fast_slow_4_19_2019_3X'+str(depth)+'_s'+str(slow_d)+'_min_med_rew_360training_reward.png')
     reward = []
     plt.close()