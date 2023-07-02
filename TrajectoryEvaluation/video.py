import cv2
import os
from tqdm import tqdm
import argparse

argparser = argparse.ArgumentParser(
    description=__doc__)
# 添加参数
argparser.add_argument('range', type=str, help='Range in the format start-end(for all: total)')
args = argparser.parse_args()

# 图片文件夹路径
image_folder = 'E:/CARLADataset/Town03/1/model_Ma_result_dis006/pm_insert'

# 获取图片文件列表并按名字数字排序
images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
images.sort(key=lambda x: float(os.path.splitext(x)[0]))

if args.range == 'total':
    range_str = args.range
else:
# 解析参数
    range_str = args.range
    start, end = map(float, range_str.split('-'))

    # 指定区间
    start_index = images.index(f'{start}.png')
    end_index = images.index(f'{end}.png')
    images = images[start_index:end_index+1]

# 视频输出路径
output_video = f'./video/{range_str}.mp4'

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
        cv2.putText(frame, image.split('.')[0], (width-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 将帧写入视频
        video.write(frame)

except KeyboardInterrupt:
    print('Exit by usr !')
finally:
    # 释放视频对象
    video.release()
