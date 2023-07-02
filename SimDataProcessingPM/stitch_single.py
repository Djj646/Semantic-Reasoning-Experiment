import cv2
import numpy as np

# 设置文件夹路径和图片尺寸
# 1-3: rain, 4-6: night, 7-9: foggy
folder1 = 'E:/CARLADataset/Town01/1/model_L1_result_blur_dis000/pm_insert'
folder2 = 'E:/CARLADataset/Town01/1/model_L2_500_result_blur_dis000/pm_insert'
folder3 = 'E:/CARLADataset/Town01/1/model_MA2_result_blur_dis000/pm_insert'
folder4 = 'E:/CARLADataset/Town04/1/model_L1_result_blur_dis000/pm_insert'
folder5 = 'E:/CARLADataset/Town04/1/model_L2_500_result_blur_dis000/pm_insert'
folder6 = 'E:/CARLADataset/Town04/1/model_MA2_result_blur_dis000/pm_insert'
folder7 = 'E:/CARLADataset/Town04/2/model_L1_result_blur_dis000/pm_insert'
folder8 = 'E:/CARLADataset/Town04/2/model_L2_500_result_blur_dis000/pm_insert'
folder9 = 'E:/CARLADataset/Town04/2/model_MA2_result_blur_dis000/pm_insert'

index0 = '1686469655.7230213'
index1 = '1686473544.3907394'
index2 = '1686476096.9399996'

img_width = 640  # 图片的尺寸
img_height = 360

img_list = []

for i in range(9):
    k = i//3
    index = globals()['index'+str(k)]
    folder = globals()['folder'+str(i+1)]
    img_path = f'{folder}/{index}.png'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_width, img_height))
    img_list.append(img)
    
img_row1 = np.vstack((img_list[0], img_list[1], img_list[2]))
img_row2 = np.vstack((img_list[3], img_list[4], img_list[5]))
img_row3 = np.vstack((img_list[6], img_list[7], img_list[8]))
img_merged = np.hstack((img_row1, img_row2, img_row3))

cv2.imwrite('./merged.png', img_merged)