import matplotlib.pyplot as plt
import pyperclip
import numpy as np
from utils.turning import read_pos, find_turning_index
from utils.io import get_filelist

pos_path = 'E:/CARLADataset/Town03/1/state/pos.txt'
path_eval = 'E:/CARLADataset/Town03/1/model_Ma_result/pm_insert'

# 设置数轴的长度
axis_length = 100

img_list = get_filelist(path_eval) # 根据实际参与评价的图片确定数轴长度
yaw_list= read_pos(file_path=pos_path)[:len(img_list), :] # (time, yaw)
turning_index, _, turning_list, _ = find_turning_index(yaw_list) # np数组
# _, turning_index, _, turning_list = find_turning_index(yaw_list)
turning_index_list = list(turning_index*axis_length/len(img_list)) # 缩放后列表

# 创建数轴
fig, ax = plt.subplots(figsize=(10, 1))
ax.plot([0, axis_length], [0, 0], color='blue', linewidth=2)

plt.annotate('start:'+img_list[0], xy=(0, 0), xytext=(0, 15),
        textcoords='offset points', ha='center', color='green')
plt.annotate('end:'+img_list[-1], xy=(100, 0), xytext=(0, 15),
        textcoords='offset points', ha='center', color='green')

y_points = list(np.zeros(len(turning_index)))
ax.plot(turning_index_list, y_points, marker='.', markersize=4, color='red', linestyle='none') # 取消点间红线绘制

# 添加事件处理器，用于显示文本和复制到剪贴板
def on_motion(event):
    if event.inaxes == ax:
        x = event.xdata
        index = int(x / axis_length * len(img_list))
        if 0 <= index < len(img_list):
            # 清除之前的文本标注
            for annotation in ax.texts:
                annotation.remove()
            # 显示新的文本标注
            ax.text(x, 3, img_list[index], ha='center', va='bottom', color='green',
                    bbox=None)
            fig.canvas.draw_idle()  # 重绘图形

def on_click(event):
    if event.inaxes == ax and event.button == 1:
        x = event.xdata
        index = int(x / axis_length * len(img_list))
        if 0 <= index < len(img_list):
            # 复制文本到剪贴板
            text = img_list[index]
            pyperclip.copy(text)
            print(f"'{text}' copied.")

# 添加事件处理器
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_press_event', on_click)

# 设置坐标轴范围
ax.set_xlim(0, axis_length)
ax.set_ylim(-20, 20)

# 隐藏坐标轴刻度
ax.axis('off')

# 显示图形
plt.show()
