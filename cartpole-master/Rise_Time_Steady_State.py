#Script to determine the rise time and steady state error
import pandas as pd
import numpy as np

name = 'classic_controller_4_18_2019__slow_dynamic_2.csv'
slow_d = 2
ext = "./test_scores/" + str(name)


riseTimes = np.array([])
ssError = np.array([])

df = pd.read_csv(ext)
selected_columns = [col for col in df.columns if "xdot" in col]
df_filtered = df[selected_columns]


#loop through all the columns
for col in df_filtered.columns:
	xdotvals = df[col]
	#loop through the particular run
	for i in range(0, len(xdotvals)-1):
		if xdotvals[i] >= slow_d:
			riseTimes = np.append(riseTimes, i*0.02)
			break
	for j in range(i, len(xdotvals)-1):
		ssError = np.append(ssError, abs(xdotvals(j) - slow_d))


print(np.mean(ssError))
print(riseTimes)