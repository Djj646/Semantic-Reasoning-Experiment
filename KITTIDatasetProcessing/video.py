import cv2
import os
from tqdm import tqdm
import argparse

argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    '--index',
    type = int,
    default=0,
    help='Sequence number(default: 0)'
)
argparser.add_argument(
    '--side',
    default='left',
    help='Camera side(default: left)'
)
args = argparser.parse_args()

index = str(args.index).zfill(2)
side = str(2) if args.side=='left' else str(3)
# 图片文件夹路径和输出视频路径
image_folder = 'E:/DATASET/KITTI/data_odometry_color/dataset/sequences/'+index+'/image_'+side
output_video = index+'_'+args.side+'.mp4'

# 获取图片文件列表并按名字数字排序
images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
images.sort(key=lambda x: int(os.path.splitext(x)[0]))

# 获取第一张图片的尺寸作为视频帧尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = frame.shape

# 设置视频编码器和输出视频对象，帧率默认30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

try:
    # 逐帧读取图片并写入视频
    for image in tqdm(images):
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        # 在右上角添加文字
        cv2.putText(frame, image.split('.')[0], (width-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 将帧写入视频
        video.write(frame)

except KeyboardInterrupt:
    print('Exit by usr !')
finally:
    # 释放视频对象
    video.release()
