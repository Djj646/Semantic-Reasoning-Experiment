import numpy as np
import math

turning_offset = 0 # 转弯开始对yaw角变化的提前量
yaw_threshold = 1.5 # 转弯微分量判定阈值 0.5

def set_args(offset, threshold):
    global turning_offset, yaw_threshold
    turning_offset = offset
    yaw_threshold = threshold

def read_pos(file_path):
    yaw_list = []
    with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                time = float(sp_line[0])
                yaw = float(sp_line[5])
                yaw_list.append((time,yaw))

    return np.array(yaw_list)

def find_turning_index(yaw_list):
    global yaw_threshold
    # 取第二列判断
    yaw_array = yaw_list[:, 1]

    # 使用NumPy的差分函数计算偏航角变化率
    diff_yaw = np.abs(np.diff(yaw_array))

    # 根据阈值，将偏航角变化率超过阈值的位置视为转弯点，下标值存入列表
    turning_indexes = np.where((diff_yaw < 10) & (diff_yaw >= yaw_threshold))[0] - turning_offset # np数组

    # 将转弯点 time 时间戳存入列表
    turning_list = [yaw_list[i][0] for i in turning_indexes]

    # 直行点下标
    straight_indexes = np.array([i for i in range(len(yaw_list)) if i not in turning_indexes]) # np数组
    
    # 将直行点 time 时间戳存入列表
    straight_list = [yaw_list[i][0] for i in straight_indexes]
    
    return turning_indexes, straight_indexes, turning_list, straight_list