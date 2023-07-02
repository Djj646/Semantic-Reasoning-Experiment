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
# img_path = f'E:/DATASET/KITTI/data_odometry_color/dataset/sequences/0{index}/image_2'
nav_path = 'D:/VSCodeProjects/AutoDrive/SimDataProcessingPM/nav_test/fake_nav/normal'

video_frame_width = 800
video_frame_height = 450
# video_frame_width = 1000
# video_frame_height = 400

argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    '--fps',
    type=int,
    default=30,
    help='FPS of video'
)
argparser.add_argument(
    '-n','--num',
    type=int,
    default=5000,
    help='Generate total nums'
)
argparser.add_argument(
    '--model',
    type=str,
    default='SD',
    help='Model'
)
args = argparser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = args.model

print('>> loading model...')
# G_model = GeneratorUNet()
G_model = GeneratorUNet() if not model=='CNN' else GeneratorCNN()
G_model.load_state_dict(torch.load(f'./model/0622/{model}.pth'))
G_model.to(device)
G_model.eval()

def get_filelist(path):
    files = glob.glob(f'{path}/*.png')
    file_names = []
    for file in files:
        file = file.split('/')[-1]
        file_name = file.split('\\')[-1][:-4] # win文件路径 '\\' 修改 舍去.png
        file_names.append(file_name)
    file_names.sort()
    return file_names

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
    img = cv2.resize(img, (video_frame_width,video_frame_height))
    
    return img

def insert_nav(img, nav):
    # Insert the nav at the left-up corner
    nav = cv2.resize(nav,(256,128))
    img[0:128,0:256,:] = nav
    
    return img

def main(): 
    print('>> Loading the png...')
    img_list = get_filelist(img_path)
    nav_list = ['l2', 'st', 'r2', 'l3', 'wh', 'r3']
    print('>> Generating pm and insert into imgs...')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'./video/lab4/{model}.mp4', fourcc, args.fps, (video_frame_width*3, video_frame_height*2))
    try:
        for i in tqdm(range(min(len(img_list), args.num))):
            file = img_list[i]
            # Reading the img and nav
            img_raw = read_png(img_path, file_name=file)
            img_inserted = []
            
            for i in range(len(nav_list)):
                img = img_raw.copy()
                nav_file_path = nav_list[i]
                nav = read_png(nav_path, file_name=nav_file_path)
                
                # Calculate the pm.png
                pm = get_pm(img, nav)
                img = insert_pm(img, pm)
                img = insert_nav(img, nav)

                img_inserted.append(img)
            
            img_row1 = np.hstack((img_inserted[0], img_inserted[1], img_inserted[2]))
            img_row2 = np.hstack((img_inserted[3], img_inserted[4], img_inserted[5]))
            
            img_merged = np.vstack((img_row1, img_row2))
            cv2.putText(img_merged, f'{model}', (video_frame_width*3-200, 30), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img_merged, file.split('.')[0], (video_frame_width*3-200, 60), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (202, 207, 219), 2, cv2.LINE_AA)

            # 写入视频
            video.write(img_merged)
    except KeyboardInterrupt:
        # interrupt but release video stiil
        print('Exit by usr !')
    finally:
        print('>> End writing the video and release...')
        video.release()
        print('>> Done')
        
        
if __name__ == '__main__':
    main()