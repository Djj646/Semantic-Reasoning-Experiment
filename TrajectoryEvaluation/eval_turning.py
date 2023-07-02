from tqdm import tqdm
import os
from utils.evaluation import Evaluation
from utils.turning import read_pos, find_turning_index
from utils.io import get_filelist, read_png, read_png_true, save_average_evaluation

import argparse
import csv
import random

argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    '--fuzzy',
    action='store_true',
    default=False,
    help='Quick Evaluate Mode Not Fuzzy Evaluate Mode(default: False)'
)
argparser.add_argument(
    '--turning',
    action='store_true',
    default=False,
    help='Choose Turning Path(default: False)'
)
argparser.add_argument(
    '--num',
    type=int,
    default=1000,
    help='Step Size of Evaluation(default: 1)'
)
argparser.add_argument(
    '--model',
    type=str,
    default='CNN',
    help='Model(default: CNN)'
)
argparser.add_argument(
    '--rate',
    type=str,
    default='003',
    help='Model(default: 003)'
)
args = argparser.parse_args()

model = args.model
rate = args.rate
model_result = f'model_{model}_result_nav_dis{rate}'
name_turn = 'turning' if args.turning else 'st'
town = 'Town01'
index = '2'
path_true = f'E:/CARLADataset/{town}/{index}/pm'
path_eval = f'E:/CARLADataset/{town}/{index}/{model_result}/pm_insert'
path_pos = f'E:/CARLADataset/{town}/{index}/state/pos.txt'

save_folder = f'./result0622/lab2/{model}'
save_path = f'{save_folder}/{town}/{rate}/{name_turn}/'
os.makedirs(save_path , exist_ok=True)

def main():
    global path_true, path_eval
    
    img_list = get_filelist(path_eval)
    print('>> 待评价图片总数: ', len(img_list))
    yaw_list= read_pos(file_path=path_pos)[:len(img_list), :] # (time, yaw)
    turning_indexes, straight_indexes, _, _ = find_turning_index(yaw_list)
    
    mode = '快速模式' if not args.fuzzy else '模糊匹配模式'
    print(f'>> 评价模式: {mode}')
    
    if args.turning:
        eval_indexes = turning_indexes
        print('>> 评价类型: 转弯')
    else:
        eval_indexes = straight_indexes
        print('>> 评价类型: 直行')

    eval_num = min(args.num, len(eval_indexes))
    print(f'>> 评价数量: {eval_num}')
    evaluator = Evaluation(rgb_mode=True, quick=not args.fuzzy) # rgb_mode取决于待评价图片颜色模式
    
    # 图片评价指标
    iou_list = []
    cover_rate_list = []
    delta_yaw_list = []
    file = open(f'{save_path}evaluation.csv', mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Time_Index', 'IOU', 'Cover_Rate', 'Delta_Yaw'])
    
    try:
        # 每 step 选择一张评价
        for i in tqdm(range(eval_num)):
            index = random.choice(eval_indexes)
            img_true = read_png_true(path_true, img_list[index])
            img_eval = read_png(path_eval, img_list[index])
            
            if img_true is None:
                print('no pm found!')
                continue
            # 更新
            evaluator.update(img_true, img_eval)

            # 保存评价
            iou = evaluator.iou()
            cover_rate = evaluator.cover_rate()
            delta_yaw = evaluator.delta_yaw()
            iou_list.append(iou)
            cover_rate_list.append(cover_rate)
            delta_yaw_list.append(delta_yaw)

            # 将 evaluation 写入文件
            writer.writerow([img_list[index], iou, cover_rate, delta_yaw])
    except KeyboardInterrupt:
        print('>> Exit by user !')   
    finally:
        file.close()
        with open(f'{save_path}average_evaluation.txt', 'w') as file:
            save_average_evaluation(file, iou_list, cover_rate_list, delta_yaw_list)
            print(f">> 评价结果保存 {save_path} 成功！")
    
if __name__ == '__main__':
    main()