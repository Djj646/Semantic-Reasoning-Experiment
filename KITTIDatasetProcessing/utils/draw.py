import cv2
import random
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

scale = 12
map_width = 10000
map_height = 10000
x_offset = 0
y_offset = 0
map_border = 1000
global_nav_scale = 1

def get_map(xy_poses):
    global map_width, map_height, x_offset, y_offset, map_border
    
    x_max = np.max(xy_poses[:, 0])
    x_min = np.min(xy_poses[:, 0])
    y_max = np.max(xy_poses[:, 1])
    y_min = np.min(xy_poses[:, 1])
    
    x_offset = (-x_min if x_min<0 else x_min)*scale
    y_offset = (-y_min if y_min<0 else y_min)*scale
    
    map_width = int((x_max-x_min)*scale + x_offset + map_border*2)
    map_height = int((y_max-y_min)*scale + x_offset + map_border*2)

    origin_map = np.zeros((map_width, map_height, 3), dtype="uint8")
    origin_map.fill(255)
    origin_map = Image.fromarray(origin_map)
    
    return origin_map
    
def draw_route(poses, origin_map):
    coord_list = []
    for i in range(len(poses)):
        x = scale*poses[i][0] + x_offset + map_border
        y = scale*poses[i][1] + y_offset + map_border
        
        coord_list.append(x)
        coord_list.append(y)
        
    draw = ImageDraw.Draw(origin_map)
    draw.line(coord_list, 'red', width=30)
        
    return origin_map
    
def get_nav(plan_map, pos, yaw):
    global map_width, map_height, x_offset, y_offset

    x = int(scale*pos[0] + x_offset + map_border)
    y = int(scale*pos[1] + y_offset + map_border)
    _nav = plan_map.crop((x-400,y-400, x+400, y+400))

    im_rotate = _nav.rotate(-yaw+180)
    nav = im_rotate.crop((_nav.size[0]//2-150, _nav.size[1]//2-2*120, _nav.size[0]//2+150, _nav.size[1]//2)) # 300*240
    nav = cv2.cvtColor(np.asarray(nav), cv2.COLOR_BGR2RGB)
    return nav

def get_global_nav_map(plan_map):
    global map_width, map_height, global_nav_scale
    global_nav_scale = 640.0/map_height
    global_nav_map = plan_map.resize((int(map_height*global_nav_scale), int(map_width*global_nav_scale)), Image.ANTIALIAS)
    return global_nav_map

def get_global_nav(pos, global_nav_map):
    global map_width, map_height, global_nav_scale, x_offset, y_offset, map_border
    
    r = 2
    draw = ImageDraw.Draw(global_nav_map)
    x = int((scale*pos[0] + x_offset + map_border)*global_nav_scale)
    y = int((scale*pos[1] + y_offset + map_border)*global_nav_scale)
    draw.ellipse((x-r, y-r, x+r, y+r), fill='green', outline='green', width=2)

    global_nav_map = cv2.cvtColor(np.asarray(global_nav_map), cv2.COLOR_RGB2BGR)
    
    return global_nav_map

def draw_angles(euler_angles):
    x = np.arange(euler_angles.shape[0])
    y1 = euler_angles[:, 0]
    y2 = euler_angles[:, 1]
    y3 = euler_angles[:, 2]

    # 绘制折线图
    # 创建子图1
    plt.subplot(3, 1, 1)
    plt.plot(x, y1, color='red', label='roll')
    plt.title('Roll')

    # 创建子图2
    plt.subplot(3, 1, 2)
    plt.plot(x, y2, color='green', label='yaw')
    plt.title('Yaw')

    # 创建子图3
    plt.subplot(3, 1, 3)
    plt.plot(x, y3, color='blue', label='pitch')
    plt.title('Pitch')

    # 调整子图之间的间距
    plt.tight_layout()

    # 添加标题和标题
    plt.xlabel('Row Number')
    plt.ylabel('Value')

    # 显示图形
    plt.show()