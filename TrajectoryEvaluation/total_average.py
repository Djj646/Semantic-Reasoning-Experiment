import os

rate = '003_NoNPC'
# 定义需要处理的文件夹名
folder_name = f'./result0618/lab2_new/SD2/{rate}/'

# 初始化三个指标的和
sum_iou, sum_cover_rate, sum_delta_yaw = 0, 0, 0
subdirs = []

# 遍历两个子文件夹
for subdir in os.listdir(folder_name):

    # 构造 average_evaluation.txt 的路径
    eval_path = os.path.join(folder_name, subdir, 'average_evaluation.txt')

    # 读取并解析 average_evaluation.txt 文件中的指标
    with open(eval_path, 'r') as f:
        line = f.readline()     # 读取一行
        avg_iou = float(line.split()[-1])     # 提取 IoU 平均值
        line = f.readline()     # 读取一行
        avg_cover_rate = float(line.split()[-1])     # 提取 cover rate 平均值
        line = f.readline()     # 读取一行
        avg_delta_yaw = float(line.split()[-1])     # 提取 delta yaw 平均值

    # 累加三个指标的和
    sum_iou += avg_iou
    sum_cover_rate += avg_cover_rate
    sum_delta_yaw += avg_delta_yaw

# 计算两个子文件夹下指标的平均值
avg_iou = sum_iou / 2
avg_cover_rate = sum_cover_rate / 2
avg_delta_yaw = sum_delta_yaw / 2

# 将平均值写入 fold_n 目录下的 total.txt 文件
output_path = os.path.join(folder_name, 'total.txt')
with open(output_path, 'w') as f:
    f.write("average iou: {}\n".format(avg_iou))
    f.write("average cover_rate: {}\n".format(avg_cover_rate))
    f.write("average delta_yaw: {}\n".format(avg_delta_yaw))
