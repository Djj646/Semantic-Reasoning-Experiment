import numpy as np
from scipy.spatial.transform import Rotation

# 读取位姿文件中的每一行，并存储在一个 NumPy 数组中
pose_file = "./00.txt"
poses = np.genfromtxt(pose_file)

# 取出旋转矩阵
rotations = poses[:, :9].reshape(-1, 3, 3)

# 通过旋转矩阵分解得到转换欧拉角，并取出yaw角
yaw_angles = np.array([Rotation.from_rotvec(r).as_euler('xyz', degrees=True)[2] for r in rotations])

np.savetxt('./yaw_angles.txt', yaw_angles, fmt='%.6f')
