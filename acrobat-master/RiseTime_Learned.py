#Script to determine the rise time and steady state error
import pandas as pd
import numpy as np
import os
directory = './test_scores/BreadthDQN/test results 550/'
directory = './test_scores/square DQN/test_scores/'
directory = './test_scores/Depth DQN/'


for filename in os.listdir(directory):
	if filename.endswith(".csv"):
		#finding slow dynamic
		print(filename)
		index = filename.rfind('_s') + 2
		slow_d = int(filename[index])

		riseTimes = np.array([])
		best_state = np.array([])

		did_learn = 0

		df = pd.read_csv(directory +filename)
		selected_columns = [col for col in df.columns if "costheta1" in col]
		df_filtered = df[selected_columns]


		#loop through all the columns
		for col in df_filtered.columns:
			xvals = df[col]
			#loop through the particular run
			bs = 1;
			for i in range(0, len(xvals)-1):
				if xvals[i] < bs:
					bs = xvals[i]
				if xvals[i] <= -0.8:
					riseTimes = np.append(riseTimes, i*0.02)
					did_learn = did_learn + 1
					break
			best_state = np.append(best_state, bs)				


		print(did_learn/float(len(df_filtered.columns)))
		print(np.mean(riseTimes))

		file = open(directory + str(filename) +'.txt','w')  
		file.write("Probability of Learning: " + str(float(did_learn)/float(len(df_filtered.columns))))
		file.write("\nMean riseTime: " + str(np.mean(riseTimes))) 
		file.write("\nMean Peak: " + str(np.median(best_state))) 
		file.close() 

	else:
		continue





# name = 'fast_slow_4_15_2019_110X3_s4_min_med_rew_360.csv'
# slow_d = 2
# ext = "./test_scores/" + str(name)


# riseTimes = np.array([])
# ssError = np.array([])

# df = pd.read_csv(ext)
# selected_columns = [col for col in df.columns if "xdot" in col]
# df_filtered = df[selected_columns]


# #loop through all the columns
# for col in df_filtered.columns:
# 	xdotvals = df[col]
# 	#loop through the particular run
# 	for i in range(0, len(xdotvals)-1):
# 		if xdotvals[i] >= slow_d:
# 			riseTimes = np.append(riseTimes, i*0.02)
# 			break
# 	run_ss_err = np.array([])
# 	for j in range(i, len(xdotvals)-1):
# 		if not np.isnan(xdotvals[j]):
# 			run_ss_err = np.append(run_ss_err, abs(xdotvals[j] - slow_d))
# 	ssError = np.append(ssError, np.mean(run_ss_err))


# print(np.mean(ssError))
# print(np.mean(riseTimes))

# print((ssError))
# print((riseTimes))
