import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

fps = 30.0
path = 'F:/CARLADataset/Town10HD/1'

raw_folder = f'{path}/raw_result_heat'
folders = []
folders.append(raw_folder)

for model in ['CNN', 'UNet', 'MA2', 'SD']:
    folders.append(f'{path}/model_{model}_result_heat')

def read_png(path, file):
    item_path = f'{path}/{file}.png'
    item = cv2.imread(item_path)

    return item

def get_filelist(path):
    files = glob.glob(path+'/*.png')
    file_names = []
    for file in files:
        file = file.split('/')[-1]
        file_name = file.split('\\')[-1][:-4] # win文件路径 '\\' 修改
        file_names.append(file_name)
    file_names.sort()
    return file_names

filelist = get_filelist(raw_folder)

video_frame_width = 256
video_frame_height = 256
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(f'../video/heat_10HD.mp4', fourcc, fps, (video_frame_width*5, video_frame_height))

try:
    for file in tqdm(filelist):
        heatmaps = []
        heatmaps.append(read_png(folders[0], file))
        heatmaps.append(read_png(folders[3], file))
        heatmaps.append(read_png(folders[1], file))
        heatmaps.append(read_png(folders[2], file))
        heatmaps.append(read_png(folders[4], file))
        
        total = np.hstack((heatmaps[0], heatmaps[1], heatmaps[2], heatmaps[3], heatmaps[4]))
        cv2.putText(total, file, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(total, 'MA2', (video_frame_width+2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(total, 'CNN', (video_frame_width*2+2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(total, 'UNet', (video_frame_width*3+2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(total, 'SD', (video_frame_width*4+2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        video.write(total)
except KeyboardInterrupt:
    print('Exit by usr !')
finally:
    print('>> End writing the video and release...')
    video.release()
    print('>> Done')