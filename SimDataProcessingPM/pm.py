# ----------------
# pos 到 pm 图生成
# ----------------
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import cv2
import carla
import argparse
import time
from tqdm import tqdm
from ff.collect_pm import CollectPerspectiveImage
from ff.carla_sensor import Sensor, CarlaSensorMaster

camera_config = {
    'img_length': 640,
    'img_width': 360,
    'fov': 120,
    'fps': 30,
}

sensor_dict = {
    'camera':{
        'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
    },
    'lidar':{
        'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
    },
}

MAX_SPEED = 40
TRAJ_LENGTH = 25 # 25
vehicle_width = 2.0
longitudinal_sample_number_near = 8
longitudinal_sample_number_far = 0.5
lateral_sample_number = 20
lateral_step_factor = 1.0

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data-index', type=int, default=1, help='data index')
args = parser.parse_args()

save_path = 'E:/CARLADataset/Town01/'+str(args.data_index)+'/'

def mkdir(save_path, path):
    if not os.path.exists(save_path + path):
        os.makedirs(save_path + path)
        
def read_img(time_stamp):
    img_path = save_path + 'img/'
    file_name = str(time_stamp) + '.png'
    img = cv2.imread(img_path + file_name)
    return img

def read_state():
    state_path = save_path + 'state/'

    # read pose
    pose_file = state_path + 'pos.txt'
    time_stamp_list = []
    time_stamp_pose_dict = dict()
    file = open(pose_file, 'r') 
    while 1:
        line = file.readline()
        if not line:
            break
        if line == '\n':
            continue

        line_list = line.split()

        index = eval(line_list[0])

        transform = carla.Transform()
        transform.location.x = eval(line_list[1])
        transform.location.y = eval(line_list[2])
        transform.location.z = eval(line_list[3])
        transform.rotation.pitch = eval(line_list[4])
        transform.rotation.yaw = eval(line_list[5])
        transform.rotation.roll = eval(line_list[6])

        time_stamp_list.append(index)
        time_stamp_pose_dict[index] = transform

    file.close()

    return time_stamp_list, time_stamp_pose_dict
    

def distance(pose1, pose2):
    return pose1.location.distance(pose2.location)

def find_traj_with_fix_length(start_index, time_stamp_list, time_stamp_pose_dict):
    length = 0.0
    for i in range(start_index, len(time_stamp_list)-1):
        length += distance(time_stamp_pose_dict[time_stamp_list[i]], time_stamp_pose_dict[time_stamp_list[i+1]])
        if length >= TRAJ_LENGTH:
            return i
    return -1


class Param(object):
    def __init__(self):
        self.traj_length = float(TRAJ_LENGTH)
        self.target_speed = float(MAX_SPEED)
        self.vehicle_width = float(vehicle_width)
        self.longitudinal_sample_number_near = longitudinal_sample_number_near
        self.longitudinal_sample_number_far = longitudinal_sample_number_far
        self.lateral_sample_number = lateral_sample_number
        self.lateral_step_factor = lateral_step_factor

def main():
    mkdir(save_path, 'pm/')
    time_stamp_list, time_stamp_pose_dict = read_state()
    time_stamp_list.sort()

    param = Param()
    sensor = Sensor(sensor_dict['camera']['transform'], camera_config)
    sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)
    collect_perspective = CollectPerspectiveImage(param, sensor_master)

    for index in tqdm(range(len(time_stamp_list))):
        time_stamp = time_stamp_list[index]
        end_index = find_traj_with_fix_length(index, time_stamp_list, time_stamp_pose_dict)
        if end_index < 0:
            print('no enough traj: ', str(index), index/len(time_stamp_list))
            break

        vehicle_transform = time_stamp_pose_dict[time_stamp]  # in world coordinate
        traj_pose_list = []
        for i in range(index, end_index):
            time_stamp_i = time_stamp_list[i]
            time_stamp_pose = time_stamp_pose_dict[time_stamp_i]
            traj_pose_list.append((time_stamp_i, time_stamp_pose))

        img = read_img(time_stamp)
        empty_image = collect_perspective.getPM(traj_pose_list, vehicle_transform, img)

        cv2.imwrite(save_path+'pm/'+str(time_stamp)+'.png', empty_image)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Exit by user !")
        pass
