import cv2
import numpy as np
from utils.blur import gaussian

# 定义图片大小和噪声均值、方差
img_size = (360, 640, 3)
img = np.zeros(img_size, dtype=np.uint8)
cv2.randu(img, 0, 255)
img = cv2.add(img, gaussian.astype(np.uint8))
img = cv2.resize(img,(256,128))

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

argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    'model',
    type=str,
    default = 'L1',
    help='Model'
)
argparser.add_argument(
    'index',
    type=float,
    help='Generate img time index'
)
argparser.add_argument(
    'fake',
    type=str,
    default = 'st',
    help='fake nav kind'
)
args = argparser.parse_args()

model = args.model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'>> Loading model {model}...')
G_model = GeneratorUNet() if model=='Ma' else GeneratorUNet2()
G_model.load_state_dict(torch.load(f'./model/0608/{model}.pth'))
G_model.to(device)
G_model.eval()

def read_png(file_path:str):
    item = cv2.imread(file_path)
    item = cv2.resize(item,(256,128))
    return item

def get_pm(img,nav):
    global G_model, device
    
    img_height = 128
    img_width = 256
    _transforms = [
            transforms.Resize((img_height, img_width), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
     
    transform = transforms.Compose(_transforms)
    
    # 经过相同变换，便于cat拼接
    img = transform(Image.fromarray(img)).to(device)
    nav = transform(Image.fromarray(nav)).to(device)
    with torch.no_grad():
        x = torch.cat([img,nav],dim=0)
        print(x.shape)
        x = x.view(1,6,img_height,img_width)
        x.requires_grad_(False)
        # pm = G_model(x) # tensor(1,1,128,256)
        pm = G_model(x)[0] # 兼容模型2元组输出
        pm = pm.view(img_height, img_width) # tensor(128,256) dtype = float32
        pm = pm.detach().cpu().numpy() # numpy array (128,256) dtype = uint8
        pm = (pm*255).astype(np.uint8)
    
    return pm


def main():
    global img
    nav_path = './lab5a/fake_nav'
    index = args.index
    nav_kind = args.fake
    result_save_path = f'./test/gaussian/{index}'
    os.makedirs(result_save_path , exist_ok=True)

    # Reading the img and nav
    # img = read_png(f'{img_path}/{index}.png')
    nav = read_png(f'{nav_path}/{nav_kind}.png')
    
    # Calculate the pm.png
    pm = get_pm(img,nav)
    print('>> Generating pm...')
    
    # Insert pm into img in Green channel 
    img[:,:,0] &= ~pm[:,:] # 按位与运算，将 轨迹 处的 R 通道与运算后为0（255取反为0）
    img[:,:,1] |= pm[:,:] # 按位或运算，将 轨迹 处的 G 通道或运算后为255（255附近）
    img[:,:,2] &= ~pm[:,:] # 按位与运算，将 轨迹 处的 B 通道或运算后为0（255取反为0）
    
    # Insert the nav at the left-up corner
    nav = cv2.resize(nav,(256,128))
    # img[0:128,0:256,:] = nav
    
    cv2.imwrite(f'{result_save_path}/{index}-{nav_kind}.png', img)
    
    print('>> Done')
        
if __name__ == '__main__':
    main()