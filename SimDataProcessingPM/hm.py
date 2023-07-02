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
import torchvision.transforms as transforms
torch.backends.cudnn.benchmark = True # 根据硬件情况和自身的运行状态自动选择最适合当前运行的算法

from PIL import Image
from model_train_costmap import GeneratorUNet, GeneratorCNN # GAN模型
import warnings

# 热力图
from utils.heat_map import heatmap

# 忽略 Python 产生的警告信息
warnings.filterwarnings('ignore')

path = 'F:\CARLADataset\Town10HD'

argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    '--data',
    type=int,
    default=2,
    help='data index'
)
argparser.add_argument(
    '--fps',
    type=int,
    default=60,
    help='FPS of video'
)
argparser.add_argument(
    '--num',
    type=int,
    default=2000,
    help='Generate total nums'
)
argparser.add_argument(
    '--model',
    type=str,
    default='L2',
    help='Model'
)
args = argparser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = args.model
date = '0622' if not model=='MA2' else '0608'
print('>> loading model...')
G_model = GeneratorUNet() if not model=='CNN' else GeneratorCNN()
G_model.load_state_dict(torch.load(f'./model/{date}/{model}.pth'))
G_model.to(device)
G_model.eval()

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
    result_save_path = path+'/'+str(args.data)+f'/model_{model}_result_heat/'
    os.makedirs(result_save_path , exist_ok=True)
    
    print('>> Loading the png...')
    file_names = get_filelist(args.data)
    print('>> Generating pm and insert into imgs...')
    try:
        for file in tqdm(file_names[:args.num]):
            # Reading the img and nav
            img = read_png(file_name=file,kind='img',index=args.data)
            nav = read_png(file_name=file,kind='nav',index=args.data)
            
            heat = heatmap(img, nav, G_model, alpha=0.6)
            
            # heat = cv2.resize(heat,(img_width,img_height))
            
            cv2.imwrite(f'{result_save_path}{file}.png', heat)
    except KeyboardInterrupt:
        print('Exit by usr !')
    finally:
        print('>> End writing the video and release...')
        print('>> Done')
        
        
if __name__ == '__main__':
    main()