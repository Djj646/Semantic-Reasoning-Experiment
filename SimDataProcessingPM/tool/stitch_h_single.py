import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

fps = 30.0
path = 'F:/CARLADataset/Town10HD/1'
result_save_path = f'../result/heat'
os.makedirs(f'{result_save_path}', exist_ok=True)

raw_folder = f'{path}/raw_result_heat'
folders = []
folders.append(raw_folder)

for model in ['MA2', 'UNet', 'CNN', 'SD']:
    folders.append(f'{path}/model_{model}_result_heat')

def read_png(path, file):
    item_path = f'{path}/{file}.png'
    item = cv2.imread(item_path)

    return item

try:
    file = '1686062892.508554'
    heatmaps = []
    heatmaps.append(read_png(folders[0], file))
    heatmaps.append(read_png(folders[1], file))
    heatmaps.append(read_png(folders[2], file))
    heatmaps.append(read_png(folders[3], file))
    heatmaps.append(read_png(folders[4], file))
    total = np.hstack((heatmaps[0], heatmaps[1], heatmaps[2], heatmaps[3], heatmaps[4]))
    
    # total = np.vstack((heat_row[0], heat_row[1]))
    cv2.imwrite(f'{result_save_path}/heat3.png', total)

except KeyboardInterrupt:
    print('Exit by usr !')
finally:
    print('>> Done')