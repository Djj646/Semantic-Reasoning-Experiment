import os

os.environ["CUDA_VISIBLE_DEVICES"]='0'

import sys
# 禁止 Python 解释器在运行代码时自动生成 .pyc 文件，从而减少程序在磁盘空间和 I/O 上的消耗
sys.dont_write_bytecode=True

import glob
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import torch
torch.backends.cudnn.benchmark = True # 根据硬件情况和自身的运行状态自动选择最适合当前运行的算法

import warnings

# 忽略 Python 产生的警告信息
warnings.filterwarnings('ignore')

path = 'F:/CARLADataset/Town10HD'

argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    '--data',
    type=int,
    default=1,
    help='data index'
)
argparser.add_argument(
    '--num',
    type=int,
    default=5000,
    help='Generate total nums'
)
args = argparser.parse_args()

def get_filelist(index:int):
    global path

    files = glob.glob(path+'/'+str(index)+'/img/*.png')
    file_names = []
    for file in files:
        file = file.split('/')[-1]
        file_name = file.split('\\')[-1][:-4] # win文件路径 '\\' 修改
        file_names.append(file_name)
    file_names.sort()
    return file_names

def read_png(file_name:str,kind:str,index:int):
    global path
    item_path = path+'/'+str(index)+'/'+kind+'/'+file_name+'.png'
    item = cv2.imread(item_path)
    item = cv2.resize(item,(256,128))
    return item

def main():
    img_width = 1024
    img_height = 512
    
    result_save_path = path+'/'+str(args.data)+f'/raw_result_heat/'
    os.makedirs(result_save_path , exist_ok=True)
    
    print('>> Loading the png...')
    file_names = get_filelist(args.data)
    print('>> Generating raw_heat...')
    try:
        for file in tqdm(file_names[:args.num]):
            # Reading the img and nav
            img = read_png(file_name=file,kind='img',index=args.data)
            nav = read_png(file_name=file,kind='nav',index=args.data)
            
            raw_heat = np.vstack((img, nav))
            
            cv2.imwrite(f'{result_save_path}{file}.png', raw_heat)
    except KeyboardInterrupt:
        print('Exit by usr !')
    finally:
        print('>> End writing the video and release...')
        print('>> Done')
        
        
if __name__ == '__main__':
    main()