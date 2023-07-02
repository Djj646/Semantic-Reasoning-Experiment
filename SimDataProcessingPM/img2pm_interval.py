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
from model_train_costmap import GeneratorUNet, GeneratorUNet2 # GAN模型
import warnings

# nav干扰
from utils.distortion import apply_distortion
from utils.blur import motion_blur

# 忽略 Python 产生的警告信息
warnings.filterwarnings('ignore')

path = 'E:/CARLADataset/Town01'

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
    default=30,
    help='FPS of video'
)
argparser.add_argument(
    '--model',
    type=str,
    default='L1',
    help='Model'
)
args = argparser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = args.model

print('>> loading model...')
# G_model = GeneratorUNet()
G_model = GeneratorUNet2()
G_model.load_state_dict(torch.load(f'./model/0618/{model}.pth'))
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

def get_pm(img,nav):
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


def main():
    img_width = 1024
    img_height = 512
    
    result_save_path = path+'/'+str(args.data)+f'/model_{model}_result_bz/'
    os.makedirs(result_save_path , exist_ok=True)
    # os.makedirs(result_save_path + 'pm/', exist_ok=True)
    os.makedirs(result_save_path + 'pm_insert/', exist_ok=True)
    
    print('>> Loading the png...')
    file_names = get_filelist(args.data)
    # 指定区间
    start = '1687104482.9653032'
    end = '1687104645.6909263'
    start_index = file_names.index(f'{start}')
    end_index = file_names.index(f'{end}')
    eval_img_list = file_names[start_index:end_index+1]
    print('>> Generating pm and insert into imgs...')
    video = cv2.VideoWriter(f'./video/{model}_bz.mp4', cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (img_width,img_height))
    try:
        for file in tqdm(eval_img_list):
            # Reading the img and nav
            img = read_png(file_name=file,kind='img',index=args.data)
            nav = read_png(file_name=file,kind='nav',index=args.data)
            
            # nav施加干扰 easy: 0.1, moderate: 0.2, hard: 0.3
            # nav = apply_distortion(nav, error_rate)
            # img运动模糊 easy: 0.3, moderate: 0.6, hard: 0.9
            # img = motion_blur(img, error_rate)
            
            # Calculate the pm.png
            pm = get_pm(img,nav)
            # cv2.imwrite(result_save_path+'pm/'+file+'.png', pm)
            
            # Insert pm into img in Green channel 
            img[:,:,0] &= ~pm[:,:] # 按位与运算，将 轨迹 处的 R 通道与运算后为0（255取反为0）
            img[:,:,1] |= pm[:,:] # 按位或运算，将 轨迹 处的 G 通道或运算后为255（255附近）
            img[:,:,2] &= ~pm[:,:] # 按位与运算，将 轨迹 处的 B 通道或运算后为0（255取反为0）
            img = cv2.resize(img,(img_width,img_height))
            
            # Insert the nav at the left-up corner
            # nav = cv2.resize(nav,(256,128))
            # img[0:128,0:256,:] = nav
            cv2.imwrite(result_save_path+'pm_insert/'+file+'.png', img)
            
            # 写入视频
            cv2.putText(img, file.split('.')[0], (img_width-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            video.write(img)
    except KeyboardInterrupt:
        # interrupt but release video stiil
        print('Exit by usr !')
    finally:
        print('>> End writing the video and release...')
        video.release()
        print('>> Done')
        
        
if __name__ == '__main__':
    main()