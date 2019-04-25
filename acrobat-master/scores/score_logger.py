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


#Create an additional score logger class to maintain fast and slow dynamic performance values
class FS_score:
    def __init__(self, angle_objective, velocity_objective, name):
        self.FS_CSV_PATH = "./test_scores/" + name + ".csv"
        self.FS_PNG_PATH = "./test_scores/" + name + ".png"
        self.TRAINING_REWARD_PATH = "./test_scores/" + name + "training.png"


        #initialize the fast-slow objective dynamic
        self.ao = angle_objective
        self.vo = velocity_objective

        #initialize error metrics per run.
        self.xdot_average_error = 0
        self.xdot_total_error = 0
        self.theta_average_error = 0
        self.theta_total_error = 0


        #initialize data frames to keep track of all dynamics for a single episode
        self.dynamics =  pd.DataFrame(columns=['costheta1','sintheta1','costheta2','sintheta2','thetaDot1','thetaDot2', 'input'])
        self.dynamics = self.dynamics.fillna(0)

        #initialize array to keep track of reward
        self.rewards = []

    def add_reward(self, reward):
        self.rewards.append(reward)

    def graph_reward(self):
        x = np.arange(len(self.rewards))

        plt.scatter(x,self.rewards, s = 2)
        # plt.plot(np.polyfit(x,self.rewards,1))
        # naming the axises and title
        plt.xlabel('Iteration')
        plt.ylabel('Reward') 
        plt.savefig(self.TRAINING_REWARD_PATH, bbox_inches="tight")


    def close_graphs(self):
        plt.close();
    def clear_fs_states(self):
        # #initialize the error metrics
        if os.path.exists(self.FS_CSV_PATH):
            os.remove(self.FS_CSV_PATH)
        if os.path.exists(self.FS_PNG_PATH):
            os.remove(self.FS_PNG_PATH)
            
    def add_state(self, observations,input_force):
        #dynamics is a vector recording the observations of the system
        input_force = input_force - 1
        self.dynamics.loc[len(self.dynamics)] = np.append(observations,(input_force))

    def save_csv(self):
        if os.path.exists(self.FS_CSV_PATH):
            data = pd.read_csv(self.FS_CSV_PATH)
            result = pd.concat([self.dynamics, data], axis=1, sort=False)
            export_csv = result.to_csv(self.FS_CSV_PATH, index = None, header=True) 
        else:
            export_csv = self.dynamics.to_csv(self.FS_CSV_PATH, index = None, header=True) 

    def clear_run_data(self):
        self.dynamics = None
        self.dynamics =  pd.DataFrame(columns=['costheta1','sintheta1','costheta2','sintheta2','thetaDot1','thetaDot2', 'input'])
        self.dynamics = self.dynamics.fillna(0)

    def save_run(self, i, num_tests):
        #getting averaged integrated error
        if i == num_tests -1: #then this is the last run
            df = pd.read_csv(self.FS_CSV_PATH)
            df = df.set_index(['costheta1', 'thetaDot1'])
            df = df.groupby(by=df.columns, axis=1).mean()
            df = df.reset_index()
            df['costheta1'] = df['costheta1'] - self.ao
            df['thetaDot1'] = df['thetaDot1'] - self.vo
            df = df.abs()
            average_cosTheta1_err = round(df["costheta1"].mean(),2)
            average_thetaDot1_err = round(df["thetaDot1"].mean(),2)
            # total_cosTheta1_err = round(df["costheta1"].sum(),2)
            # total_thetaDot1_err = round(df["thetaDot1"].sum(),2)
            plt.figtext(.3, .85, "Mean Slow Dynamic Error = " + str(average_thetaDot1_err))
            # plt.figtext(.3, .81, "Summed Slow Dynamic Error = " + str(total_xdot_err))

            plt.figtext(.3, 0.43, "Mean Fast Dynamic Error = " + str(average_cosTheta1_err))
            # plt.figtext(.3, 0.39, "Summed Fast Dynamic Error = " + str(total_theta_err))

        #the environment updates every 0.02 seconds
        #this will save a graph of the run
        time_scale = [x*0.02 for x in [i for i in range(0,len(self.dynamics['thetaDot1'])) ]]
        plt.subplot(3, 1, 1)
        plt.plot(time_scale, self.dynamics['thetaDot1'], label='thetaDot1 measurment')
        plt.axhline(y=self.vo, color='r', linestyle='-', label='thetaDot1 desired')
        # naming the axises and title
        # plt.xlabel('Time') 
        plt.ylabel('thetaDot1') 


        plt.subplot(3,1,2)
        plt.plot(time_scale,self.dynamics['costheta1'], label='costheta1 measurment')
        plt.axhline(y=self.ao, color='r', linestyle='-', label='costheta1 desired')
        plt.xlabel('Time') 
        plt.ylabel('costheta1') 

        plt.subplot(3,1,3)
        plt.scatter(time_scale,self.dynamics['input'], label='input force', alpha = 0.7)
        plt.xlabel('Time') 
        plt.ylabel('Input Force') 

        plt.savefig(self.FS_PNG_PATH, bbox_inches="tight")
    
    def save_reward(self,reward_list):
        x = np.arange(len(reward_list))
        plt.plot(x, reward_list)
        plt.xlabel("Iteration")
        plt.ylabel("Reward")

        plt.savefig(self.TRAINING_REWARD_PATH, bbox_inches="tight")

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
