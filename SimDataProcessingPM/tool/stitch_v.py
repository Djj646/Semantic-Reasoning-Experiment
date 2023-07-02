import cv2
import numpy as np
import os

def vertical_concatenate_images(folder_path):
    images = []
    
    # 遍历文件夹中的所有PNG图片
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            images.append(image)
    
    # 按字母顺序排列，拼接顺序：MA2, UNet, CNN, SD
    img_merged = np.vstack((images[1], images[3], images[0], images[2]))
    
    cv2.imwrite(os.path.join(folder_path, 'total.png'), img_merged)
    
vertical_concatenate_images('D:/VSCodeProjects/AutoDrive/SimDataProcessingPM/result/lab5')