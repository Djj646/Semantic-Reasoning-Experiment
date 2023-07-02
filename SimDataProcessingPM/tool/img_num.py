import glob
import os

# 指定图片文件夹路径
folder_path = 'E:/CARLADataset/Town01/1/model_CNN_result_pure/pm_insert'

# 定义支持的图片文件扩展名
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']

# 组合文件夹路径和图片扩展名模式
search_patterns = [os.path.join(folder_path, f'**/{ext}') for ext in image_extensions]

# 使用 glob 模块匹配文件名模式，获取匹配到的文件列表
image_files = []
for pattern in search_patterns:
    image_files.extend(glob.glob(pattern, recursive=True))

# 统计图片数量
num_images = len(image_files)

# 打印结果
print(f"Number of images: {num_images}")
