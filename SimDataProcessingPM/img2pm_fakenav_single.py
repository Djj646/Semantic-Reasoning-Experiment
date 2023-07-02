# ------------------------------
# img 和 nav 经过 GAN 到 pm 图生成
# ------------------------------
import cProfile # 代码执行性能分析
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

# 忽略 Python 产生的警告信息
warnings.filterwarnings('ignore')

town = 'Town01'
index = '2'
img_path = f'E:/CARLADataset/{town}/{index}/img'
nav_path = 'D:/VSCodeProjects/AutoDrive/SimDataProcessingPM/nav_test/fake_nav/normal'

video_frame_width = 800
video_frame_height = 450

argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    '--model',
    type=str,
    default='L1',
    help='Model'
)
args = argparser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = args.model
date = '0622' if not model=='MA2' else '0608'
print('>> loading model...')
# G_model = GeneratorUNet()
G_model = GeneratorUNet() if not model=='CNN' else GeneratorCNN()
G_model.load_state_dict(torch.load(f'./model/{date}/{model}.pth'))
G_model.to(device)
G_model.eval()

def read_png(path, file_name:str):
    item_path = f'{path}/{file_name}.png'
    item = cv2.imread(item_path)
    item = cv2.resize(item,(256,128))
    return item

def get_pm(img, nav):
    global G_model, device
    
    img_height = 128
    img_width = 256
    _transforms = [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
     
    transform = transforms.Compose(_transforms)
    
    # 经过相同变换，便于cat拼接
    img = transform(Image.fromarray(img)).to(device)
    nav = transform(Image.fromarray(nav)).to(device)
    with torch.no_grad():
        x = torch.cat([img,nav],dim=0)
        x = x.view(1,6,img_height,img_width)
        x.requires_grad_(False)
        # pm = G_model(x) # tensor(1,1,128,256)
        pm = G_model(x)[0] # 兼容模型2元组输出
        pm = pm.view(img_height, img_width) # tensor(128,256) dtype = float32
        pm = pm.detach().cpu().numpy() # numpy array (128,256) dtype = uint8
        pm = (pm*255).astype(np.uint8)
    
    return pm

def insert_pm(img, pm):
    # Insert pm into img in Green channel 
    img[:,:,0] &= ~pm[:,:] # 按位与运算，将 轨迹 处的 R 通道与运算后为0（255取反为0）
    img[:,:,1] |= pm[:,:] # 按位或运算，将 轨迹 处的 G 通道或运算后为255（255附近）
    img[:,:,2] &= ~pm[:,:] # 按位与运算，将 轨迹 处的 B 通道或运算后为0（255取反为0）
    img = cv2.resize(img, (video_frame_width, video_frame_height))
    
    return img

def insert_nav(img, nav):
    # Insert the nav at the left-up corner
    nav = cv2.resize(nav,(256,128))
    img[0:128,0:256,:] = nav
    
    return img

def main():
    result_save_path = f'./result/lab4/'
    os.makedirs(f'{result_save_path}', exist_ok=True)
    print('>> Loading the png...')
    nav_list = ['st', 'r2', 'wh']
    print('>> Generating pm and insert into imgs...')
    times = ['1687417630.9916444', '1687417689.485599', '1687417650.9037354']
    
    try:
        img_inserted = []
        for i in range(len(times)): 
            img = read_png(img_path, file_name=times[i])
            nav = read_png(nav_path, file_name=nav_list[i])
            
            # Calculate the pm.png
            pm = get_pm(img, nav)
            img = insert_pm(img, pm)
            if model == 'MA2':
                img = insert_nav(img, nav)
            
            # cv2.imwrite(f'{result_save_path}pm_insert/{file}-{nav_file_path}.png', img)
            img_inserted.append(img)
            
        img_row1 = np.hstack((img_inserted[0], img_inserted[1], img_inserted[2]))

        
        cv2.imwrite(f'{result_save_path}/{model}.png', img_row1)

    except KeyboardInterrupt:
        # interrupt but release video stiil
        print('Exit by usr !')
    finally:
        print('>> Done')
        
        
if __name__ == '__main__':
    main()