from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np
import pandas as pd

from cv2 import VideoWriter, VideoWriter_fourcc
import cv2



FS_CSV_PATH = "./scores/fs_scores.csv"
FS_PNG_SINGLE = "./scores/fs_scores.png"
FS_PNG_ITERATIVE = "./scores/fs_iterative_scores.png"

AVERAGE_SCORE_TO_SOLVE = 195
CONSECUTIVE_RUNS_TO_SOLVE = 100

class video:
  def __init__(self, path):
    self.path = path
    FPS = 1
    frame = cv2.imread(self.path)
    
    #get frame shape
    self.height, self.width, self.channels = frame.shape
    
    #create the video
    self.fourcc = VideoWriter_fourcc(*'MP42')
    self.video = VideoWriter('./noise.mp4', self.fourcc, float(FPS), (self.width, self.height))

  def add_frame(self):
    frame = cv2.imread(self.path)
    self.video.write(frame)
  def stop_video(self):
    self.video.release()


class Test_Score:
  def __init__(self, fast_objective, slow_objective):
    self.CSV_PATH = "./test_scores/test_scores.csv"
    self.PNG_PATH = "./test_scores/test_scores.png"

    #initialize the fast-slow objective dynamic
    self.fo = fast_objective
    self.so = slow_objective

    #initialize error metrics per run.
    self.xdot_average_error = 0
    self.xdot_total_error = 0
    self.theta_average_error = 0
    self.theta_total_error = 0

    #initialize data frames to keep track of all dynamics for a single episode
    self.dynamics =  pd.DataFrame(columns=['x','xdot','theta','thetadot'])
    self.dynamics = self.dynamics.fillna(0)

  def clear_test_scores(self):
    # #initialize the error metrics
    if os.path.exists(self.CSV_PATH):
      os.remove(self.CSV_PATH)
    if os.path.exists(self.PNG_PATH):
      os.remove(self.PNG_PATH)


  def add_state(self, observations):
    #dynamics is a vector recording the observations of the system
    self.dynamics.loc[len(self.dynamics)] = observations

  def print_states(self):
    print(self.dynamics)

  def calculate_episodic_error(self):
    xdot_value_sum = self.dynamics['xdot'].sum()
    xdot_desired_sum = self.dynamics.shape[1]*self.so
    self.xdot_total_error = abs(xdot_value_sum - xdot_desired_sum)
    self.xdot_average_error = self.xdot_total_error/self.dynamics.shape[1]

    theta_value_sum = self.dynamics['theta'].sum()
    theta_desired_sum = self.dynamics.shape[1]*self.so
    self.theta_total_error = abs(theta_value_sum - theta_desired_sum)
    self.theta_average_error = self.theta_total_error/self.dynamics.shape[1]

    #empty out the dynamics matrix
    self.dynamics = self.dynamics[0:0]

    #add the episodic error metric to the CSV file, creating one if need be
    if not os.path.exists(self.CSV_PATH):
      print(self.CSV_PATH, ' file not found.  Creating file')
      with open(self.CSV_PATH, 'w') as csvfile:
        scores_file = open(self.CSV_PATH, "a")
        with scores_file:
          writer = csv.writer(scores_file)
          writer.writerow(['xdot_average_error', 'theta_average_error'])
    else:
      with open(self.CSV_PATH, 'a') as scores_file:
        writer = csv.writer(scores_file)
        writer.writerow([self.xdot_average_error, self.theta_average_error])

  def save_last_run(self):
    #the environment updates every 0.02 seconds
    #this will save a graph of the run
    time_scale = [x*0.02 for x in [i for i in range(0,len(self.dynamics['xdot'])) ]]
    plt.subplot(2, 1, 1)
    plt.plot(time_scale, self.dynamics['xdot'], label='xdot measurment')
    plt.axhline(y=self.so, color='r', linestyle='-', label='xdot desired')
    # naming the axises and title
    plt.xlabel('Time') 
    plt.ylabel('Velocity') 


    plt.subplot(2,1,2)
    plt.plot(time_scale,self.dynamics['theta'], label='theta measurment')
    plt.axhline(y=self.fo, color='r', linestyle='-', label='theta desired')
    plt.xlabel('Time') 
    plt.ylabel('Angle') 

    plt.savefig(self.PNG_PATH, bbox_inches="tight")

    self.dynamics = None
    self.dynamics =  pd.DataFrame(columns=['x','xdot','theta','thetadot'])
    self.dynamics = self.dynamics.fillna(0)


  def save_error_png(self):
    if not os.path.exists(self.CSV_PATH):
      raise Exception(self.CSV_PATH, ' file not found')
    data = pd.read_csv(self.CSV_PATH)


    ind = [number for number in range(len(data['xdot_average_error']))]

    #plotting xdot error
    plt.subplot(2, 1, 1)
    plt.plot(ind,data['xdot_average_error'], label='xdot error measurment')
    plt.axhline(y=self.so, color='r', linestyle='-', label='xdot desired')
    plt.xlabel('Time') 
    plt.legend()
    plt.ylabel('Velocity Episodic Error, xdot') 
    plt.title('Velocity Error vs. Time')
    
    #plotting theta error
    plt.subplot(2,1,2)
    plt.plot(ind,data['theta_average_error'], label='theta error measurment')
    plt.axhline(y=self.fo, color='r', linestyle='-', label='theta desired')
    plt.xlabel('Time') 
    plt.ylabel('Angle Episodic Error, theta') 
    plt.title('Andle Error vs. Time')
  
    plt.legend()
    plt.savefig(self.PNG_PATH, bbox_inches="tight")


#Create an additional score logger class to maintain fast and slow dynamic performance values
class FS_score:
    def __init__(self, fast_objective, slow_objective, model_name):
        self.FS_CSV_PATH = "./test_scores/" + model_name + ".csv"
        self.FS_PNG_PATH = "./test_scores/" + model_name + ".png"
        self.TRAINING_REWARD_PATH = "./test_scores/" + model_name + "training_reward.png"


        #initialize the fast-slow objective dynamic
        self.fo = fast_objective
        self.so = slow_objective

        #initialize error metrics per run.
        self.xdot_average_error = 0
        self.xdot_total_error = 0
        self.theta_average_error = 0
        self.theta_total_error = 0


        #initialize data frames to keep track of all dynamics for a single episode
        self.dynamics =  pd.DataFrame(columns=['x','xdot','theta','thetadot'])
        self.dynamics = self.dynamics.fillna(0)

    def clear_fs_states(self):
        # #initialize the error metrics
        if os.path.exists(self.FS_CSV_PATH):
            os.remove(self.FS_CSV_PATH)
        if os.path.exists(self.FS_PNG_PATH):
            os.remove(self.FS_PNG_PATH)
        if os.path.exists(self.TRAINING_REWARD_PATH):
            os.remove(self.TRAINING_REWARD_PATH)

    def add_state(self, observations):
        #dynamics is a vector recording the observations of the system
        self.dynamics.loc[len(self.dynamics)] = observations

    def save_csv(self):
        if os.path.exists(self.FS_CSV_PATH):
            data = pd.read_csv(self.FS_CSV_PATH)
            result = pd.concat([self.dynamics, data], axis=1, sort=False)
            export_csv = result.to_csv(self.FS_CSV_PATH, index = None, header=True) 
        else:
            export_csv = self.dynamics.to_csv(self.FS_CSV_PATH, index = None, header=True) 
    
    def clear_run_data(self):
        self.dynamics = None
        self.dynamics =  pd.DataFrame(columns=['x','xdot','theta','thetadot'])
        self.dynamics = self.dynamics.fillna(0)
    
    def save_run(self, i, num_tests):
        #getting averaged integrated error
        if i == num_tests -1: #then this is the last run
            df = pd.read_csv(self.FS_CSV_PATH)
            df = df.set_index(['xdot', 'theta'])
            df = df.groupby(by=df.columns, axis=1).mean()
            df = df.reset_index()
            df['xdot'] = df['xdot'] - self.so
            df['theta'] = df['theta'] - self.fo
            df = df.abs()
            average_xdot_err = round(df["xdot"].mean(),2)
            average_theta_err = round(df["theta"].mean(),2)
            total_xdot_err = round(df["xdot"].sum(),2)
            total_theta_err = round(df["theta"].sum(),2)
            plt.figtext(.3, .85, "Mean Slow Dynamic Error = " + str(average_xdot_err))
            # plt.figtext(.3, .81, "Summed Slow Dynamic Error = " + str(total_xdot_err))

            plt.figtext(.3, 0.43, "Mean Fast Dynamic Error = " + str(average_theta_err))
            # plt.figtext(.3, 0.39, "Summed Fast Dynamic Error = " + str(total_theta_err))

        #the environment updates every 0.02 seconds
        #this will save a graph of the run
        time_scale = [x*0.02 for x in [i for i in range(0,len(self.dynamics['xdot'])) ]]
        plt.subplot(2, 1, 1)
        plt.plot(time_scale, self.dynamics['xdot'], label='xdot measurment')
        plt.axhline(y=self.so, color='r', linestyle='-', label='xdot desired')
        # naming the axises and title
        # plt.xlabel('Time') 
        plt.ylabel('Velocity') 


        plt.subplot(2,1,2)
        plt.plot(time_scale,self.dynamics['theta'], label='theta measurment')
        plt.axhline(y=self.fo, color='r', linestyle='-', label='theta desired')
        plt.xlabel('Time') 
        plt.ylabel('Angle') 

        plt.savefig(self.FS_PNG_PATH, bbox_inches="tight")

    def save_reward(self,reward_list):
        x = np.arange(len(reward_list))
        plt.plot(x, reward_list)
        plt.xlabel("Iteration")
        plt.ylabel("Reward")

        plt.savefig(self.TRAINING_REWARD_PATH, bbox_inches="tight")


    def save_error_png(self):
        if not os.path.exists(FS_CSV_PATH):
          raise Exception(FS_CSV_PATH, ' file not found')
        data = pd.read_csv(FS_CSV_PATH)


        ind = [number for number in range(len(data['xdot_average_error']))]

        #plotting xdot error
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(ind,data['xdot_average_error'], label='xdot error measurment')
        plt.axhline(y=self.so, color='r', linestyle='-', label='xdot desired')
        plt.xlabel('Time') 
        plt.legend()
        plt.ylabel('Velocity Episodic Error, xdot') 
        plt.title('Velocity Error vs. Time')
        
        #plotting theta error
        plt.subplot(2,1,2)
        plt.plot(ind,data['theta_average_error'], label='theta error measurment')
        plt.axhline(y=self.fo, color='r', linestyle='-', label='theta desired')
        plt.xlabel('Time') 
        plt.ylabel('Angle Episodic Error, theta') 
        plt.title('Andle Error vs. Time')
      
        plt.legend()
        plt.savefig(FS_PNG_ITERATIVE, bbox_inches="tight")
        plt.close()

SCORES_CSV_PATH = "./scores/scores.csv"
SCORES_PNG_PATH = "./scores/scores.png"
SOLVED_CSV_PATH = "./scores/solved.csv"
SOLVED_PNG_PATH = "./scores/solved.png"
AVERAGE_SCORE_TO_SOLVE = 195
CONSECUTIVE_RUNS_TO_SOLVE = 100

class ScoreLogger:

    def __init__(self, env_name):
        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.env_name = env_name

        if os.path.exists(SCORES_PNG_PATH):
            os.remove(SCORES_PNG_PATH)
        if os.path.exists(SCORES_CSV_PATH):
            os.remove(SCORES_CSV_PATH)

    def add_score(self, score, run):
        self._save_csv(SCORES_CSV_PATH, score)
        self._save_png(input_path=SCORES_CSV_PATH,
                       output_path=SCORES_PNG_PATH,
                       x_label="runs",
                       y_label="scores",
                       average_of_n_last=CONSECUTIVE_RUNS_TO_SOLVE,
                       show_goal=True,
                       show_trend=True,
                       show_legend=True)
        self.scores.append(score)
        mean_score = mean(self.scores)
        print("Scores: (min: " + str(min(self.scores)) + ", avg: " + str(mean_score) + ", max: " + str(max(self.scores)) + ")\n")
        if mean_score >= AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= CONSECUTIVE_RUNS_TO_SOLVE:
            solve_score = run-CONSECUTIVE_RUNS_TO_SOLVE
            print("Solved in " + str(solve_score) + " runs, " + str(run) + " total runs.")
            self._save_csv(SOLVED_CSV_PATH, solve_score)
            self._save_png(input_path=SOLVED_CSV_PATH,
                           output_path=SOLVED_PNG_PATH,
                           x_label="trials",
                           y_label="steps before solve",
                           average_of_n_last=None,
                           show_goal=False,
                           show_trend=False,
                           show_legend=False)
            exit()

    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))

        plt.subplots()
        plt.plot(x, y, label="score per run")

        average_range = average_of_n_last if average_of_n_last is not None else len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--", label="last " + str(average_range) + " runs average")

        if show_goal:
            plt.plot(x, [AVERAGE_SCORE_TO_SOLVE] * len(x), linestyle=":", label=str(AVERAGE_SCORE_TO_SOLVE) + " score average goal")

        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend")

        plt.title(self.env_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])
