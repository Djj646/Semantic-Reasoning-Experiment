import numpy as np

def parse_pose(line):
    # 解析每行的姿态信息
    pose = np.fromstring(line, sep=' ')
    pose = pose.reshape((3, 4))
    return pose

def extract_position(pose):
    # 提取位置信息（平移部分）
    position = pose[:, 3]
    return position

def extract_angles(pose):
    # 提取角度信息（旋转部分）
    rotation = pose[:, :3]
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    pitch = np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1]**2 + rotation[2, 2]**2))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    angles = np.degrees([pitch, yaw, roll])
    return angles

# 读取00.txt文件
with open('00.txt', 'r') as f:
    lines = f.readlines()

# 处理每一行的姿态信息
for i, line in enumerate(lines):
    pose = parse_pose(line)
    position = extract_position(pose)
    angles = extract_angles(pose)

    # 格式化输出位置和角度信息
    print(f"帧 {i+1}: 位置 {position}, 角度 pitch:{angles[0]:.2f}, yaw:{angles[1]:.2f}, roll:{angles[2]:.2f}")
