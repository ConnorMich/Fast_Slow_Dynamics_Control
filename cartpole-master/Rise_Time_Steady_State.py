#Script to determine the rise time and steady state error
import pandas as pd
import numpy as np
import os
directory = './test_scores/DepthDQN/test results/'

for filename in os.listdir(directory):
	if filename.endswith(".csv"):
		#finding slow dynamic
		print(filename)
		index = filename.rfind('_s') + 2
		slow_d = int(filename[index])

		riseTimes = np.array([])
		ssError = np.array([])

		df = pd.read_csv(directory +filename)
		selected_columns = [col for col in df.columns if "xdot" in col]
		df_filtered = df[selected_columns]


		#loop through all the columns
		for col in df_filtered.columns:
			xdotvals = df[col]
			#loop through the particular run
			for i in range(0, len(xdotvals)-1):
				if xdotvals[i] >= slow_d*0.9:
					riseTimes = np.append(riseTimes, i*0.02)
					break
			run_ss_err = np.array([])
			for j in range(i, len(xdotvals)-1):
				if not np.isnan(xdotvals[j]):
					run_ss_err = np.append(run_ss_err, abs(xdotvals[j] - slow_d))
			ssError = np.append(ssError, np.mean(run_ss_err))


		print(np.mean(ssError))
		print(np.mean(riseTimes))

		file = open(directory + str(filename) +'.txt','w')  
		file.write("mean ssError: " + str(np.mean(ssError))) 
		file.write("mean riseTime: " + str(np.mean(riseTimes))) 
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
