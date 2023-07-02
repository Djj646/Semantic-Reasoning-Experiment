import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import cv2

from utils.io import read_png
from utils.distortion import apply_distortion

nav = read_png(path='E:/CARLADataset/Town03/1/nav',file_name=1685429099.4925046)
nav_inter = apply_distortion(nav, error_rate=1)

# 显示干扰后的图像
cv2.imshow('nav', nav)
cv2.imshow('nav_inter', nav_inter)
cv2.waitKey(3000)
cv2.destroyAllWindows()