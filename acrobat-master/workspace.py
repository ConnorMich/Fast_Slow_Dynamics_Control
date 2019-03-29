from scores.score_logger import ScoreLogger
from scores.score_logger import FS_score


import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
from scores.score_logger import video

import random

fs_score_logger = FS_score(0,5) # the desired fast slow dynamics are 0,5
# fs_score_logger.clear_fs_scores()
vid_manager = video('./scores/fs_iterative_scores.png')
print(vid_manager.height)
print(vid_manager.width)


for j in range(0,10):
	for i in range(0,51):
		fs_score_logger.add_state([random.randint(1,10),random.randint(1,10),random.randint(1,10),random.randint(1,10)])
		fs_score_logger.add_state([random.randint(1,10),random.randint(1,10),random.randint(1,10),random.randint(1,10)])
		fs_score_logger.add_state([random.randint(1,10),random.randint(1,10),random.randint(1,10),random.randint(1,10)])
		fs_score_logger.add_state([random.randint(1,10),random.randint(1,10),random.randint(1,10),random.randint(1,10)])
		fs_score_logger.calculate_episodic_error()
		fs_score_logger.add_episodic_error()
	fs_score_logger.save_error_png()
	vid_manager.add_frame()
vid_manager.stop_video()



# FPS = 24
# seconds = 10

# frame = cv2.imread('./scores/fs_iterative_scores.png')
# #get frame shape
# height, width, channels = frame.shape


# fourcc = VideoWriter_fourcc(*'MP42')
# video = VideoWriter('./noise.mp4', fourcc, float(FPS), (width, height))
# for _ in range(FPS*seconds):
# 	# frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
# 	frame = cv2.imread('./scores/fs_iterative_scores.png')


# 	video.write(frame)
# video.release()

