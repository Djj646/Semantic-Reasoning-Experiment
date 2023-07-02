import cv2
import numpy as np
import glob
from tqdm import tqdm

def get_filelist(path):
    files = glob.glob(path+'/*.png')
    file_names = []
    for file in files:
        file = file.split('/')[-1]
        file_name = file.split('\\')[-1][:-4] # win文件路径 '\\' 修改
        file_names.append(file_name)
    file_names.sort()
    return file_names

# 设置文件夹路径和图片尺寸
folder1 = 'E:/CARLADataset/Town01/1/model_MA2_result_pure/pm_insert'  # 文件夹路径
folder2 = 'E:/CARLADataset/Town01/1/model_UNet_result_pure/pm_insert'
folder3 = 'E:/CARLADataset/Town01/1/model_CNN_result_pure/pm_insert'
folder4 = 'E:/CARLADataset/Town01/1/model_SD_result_pure/pm_insert'
folder5 = 'E:/CARLADataset/Town04/1/model_MA2_result_pure/pm_insert'
folder6 = 'E:/CARLADataset/Town04/1/model_UNet_result_pure/pm_insert'
folder7 = 'E:/CARLADataset/Town04/1/model_CNN_result_pure/pm_insert'
folder8 = 'E:/CARLADataset/Town04/1/model_SD_result_pure/pm_insert'
folder9 = 'E:/CARLADataset/Town04/2/model_MA2_result_pure/pm_insert'
folder10 = 'E:/CARLADataset/Town04/2/model_UNet_result_pure/pm_insert'
folder11 = 'E:/CARLADataset/Town04/2/model_CNN_result_pure/pm_insert'
folder12 = 'E:/CARLADataset/Town04/2/model_SD_result_pure/pm_insert'
folders_path = []

for i in range(1, 13):
    folder_var_name = "folder" + str(i)
    folder_path = globals()[folder_var_name]
    folders_path.append(folder_path)

img_width = 640  # 图片的尺寸
img_height = 360
file_list_town01 = get_filelist(folders_path[0]) # 一个文件夹下的索引
file_list_town04_1 = get_filelist(folders_path[4])
file_list_town04_2 = get_filelist(folders_path[8])

# 设置视频编码器和输出视频对象，帧率默认30.0
output_video = f'./video/lab5.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, 30.0, (img_width*3, img_height*4))

try: 
    # 获取所有文件夹中的同名图片并按顺序拼接
    for j in tqdm(range(10000)):
        index0 = file_list_town01[j]
        index1 = file_list_town04_1[j]
        index2 = file_list_town04_2[j]
        
        img_list = [] # 12张图片列表
        for i in range(12):
            k = i//4
            index = globals()['index'+str(k)]
            img_path = f'{folders_path[i]}/{index}.png'
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_width, img_height))
            img_list.append(img)
        
        img_col1 = np.vstack((img_list[0], img_list[1], img_list[2], img_list[3]))
        cv2.putText(img_col1, index0.split('.')[0], (img_width-220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_col1, 'MA', (10, img_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_col1, 'UNet', (10, img_height+img_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_col1, 'CNN', (10, img_height*2+img_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_col1, 'SD', (10, img_height*3+img_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        img_col2 = np.vstack((img_list[4], img_list[5], img_list[6], img_list[7]))
        cv2.putText(img_col2, index1.split('.')[0], (img_width-220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        img_col3 = np.vstack((img_list[8], img_list[9], img_list[10], img_list[11]))
        cv2.putText(img_col3, index2.split('.')[0], (img_width-220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        img_merged = np.hstack((img_col1, img_col2, img_col3))

        # 将帧写入视频
        video.write(img_merged)

except KeyboardInterrupt:
    print('Exit by usr !')
finally:
    # 释放视频对象
    video.release()