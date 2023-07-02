import random
import numpy as np
import cv2

fake_nav_path = 'D:/VSCodeProjects/AutoDrive/SimDataProcessingPM/nav_test/fake_nav/normal'

def read_png(file_path:str):
    item = cv2.imread(file_path)
    item = cv2.resize(item,(256,128))
    return item

# 施加干扰
# 4次方等级: 0.03, 0.16, 0.44
def apply_distortion(img, error_rate):
    # 保持形状不变
    rows, cols = img.shape[:2]
    
    # 是否随机抽取
    if random.random() < error_rate:
        fake_nav_list = ['st', 'wh', 'r3', 'r2', 'l3', 'l2']
        fake_nav_kind = random.choice(fake_nav_list)
        file_path = f'{fake_nav_path}/{fake_nav_kind}.png'
        img = read_png(file_path)
    
    # 旋转干扰
    angle = random.randint(-15, 15)  # 随机选择一个角度
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    
    # 判断是否应用缩放干扰
    if random.random() < error_rate:
        scale = random.uniform(0.6, 1.4)  # 随机选择一个缩放比例
        img = scale_distortion(img, rows, cols, scale)
    
    # 判断是否应用平移干扰
    if random.random() < error_rate:
        shift_x = random.randint(-cols//10, cols//10)  # 随机选择一个x轴平移距离
        shift_y = random.randint(-rows//10, rows//10)  # 随机选择一个y轴平移距离
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (cols, rows))
    
    # 判断是否应用对称干扰
    if random.random() < error_rate:
        img = cv2.flip(img, 1)  # 水平翻转
    
    return img

def scale_distortion(img, rows, cols, scale):
    # 缩放图片
    resized_img = cv2.resize(img, None, fx=scale, fy=scale)

    # 计算缩放后的目标尺寸
    new_height, new_width = resized_img.shape[:2]

    if scale<1:
        # 计算填充尺寸
        padding_top = (rows - new_height) // 2
        padding_bottom = padding_top
        padding_left = (cols - new_width) // 2
        padding_right = padding_left

        # 白色填充
        padded_img = np.pad(resized_img, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), constant_values=255)

        return padded_img
    else:
        # 计算裁剪区域
        x_start = (new_width - cols) // 2
        y_start = (new_height - rows) // 2
        x_end = x_start + cols
        y_end = y_start + rows
        
        # 裁剪图像
        cropped_img = resized_img[y_start:y_end, x_start:x_end]

        return cropped_img