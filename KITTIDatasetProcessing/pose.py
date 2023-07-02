import sys
import numpy as np
import cv2
from tqdm import tqdm

from utils.draw import get_map, draw_route, get_nav, get_global_nav_map, get_global_nav, draw_angles
from utils.transformations import euler_from_matrix

def parse_poses_file(file_path, pose_num):
    poses = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(pose_num):
            line = lines[i]
            elements = line.strip().split()
            elements = [float(element) for element in elements]
            transformation_matrix = np.array(elements).reshape(3, 4)
            poses.append(transformation_matrix)
            
    return poses

def reconstruct_trajectory(poses):
    trans = []
    euler = []
    for current_pose in poses:
        translation = current_pose[:3, 3] # 位置，列3
        rotation = current_pose[:3, :3] # 旋转矩阵，行0-3，列0-3
        # euler_angles = rotation_to_euler_angles(rotation, True)
        euler_angles = euler_from_matrix(rotation)
        trans.append(translation)
        euler.append(euler_angles)
    
    trans = np.array(trans)
    euler = np.array(euler)
    euler = np.degrees(euler)
    
    return trans, euler, np.concatenate((trans, euler), axis=1)

file_path = 'E:/DATASET/KITTI/data_odometry_poses/dataset/poses/00.txt'

poses = parse_poses_file(file_path, pose_num=2000)

# 重建轨迹
trans, euler_angles, traj = reconstruct_trajectory(poses)
np.savetxt("data.csv", traj, delimiter=',') # 保存数组

draw_angles(euler_angles)

xy_poses = trans[:, [0,2]]
# roll, yaw, pitch
yaw_poses = euler_angles[:, 1]



# origin_map = get_map(xy_poses)
# plan_map = draw_route(xy_poses, origin_map)
# global_nav_map = get_global_nav_map(plan_map)

# start = 0
# end = 925

# for i in tqdm(range(start, end)):
#     global_nav = get_global_nav(xy_poses[i], global_nav_map)
#     nav = get_nav(plan_map, xy_poses[i], yaw_poses[i])

#     cv2.imshow('Nav', nav)
#     cv2.imwrite(f'./kitti_odometry_result/nav/{i}.png', nav)
    
#     cv2.imshow('Global_nav', global_nav)
#     cv2.waitKey(10)
