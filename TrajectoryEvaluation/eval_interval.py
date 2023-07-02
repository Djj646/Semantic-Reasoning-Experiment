from tqdm import tqdm
import os
from utils.evaluation import Evaluation
from utils.io import get_filelist, read_png, read_png_true, save_average_evaluation

import argparse
import csv

argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    '--fuzzy',
    action='store_true',
    default=False,
    help='Quick Evaluate Mode Not Fuzzy Evaluate Mode(default: False)'
)
argparser.add_argument(
    'range', 
    type=str, 
    help='Range in the format start-end(for all: total)'
)
argparser.add_argument(
    '--step',
    type=int,
    default=1,
    help='Step Size of Evaluation(default: 1)'
)
argparser.add_argument(
    '--name',
    type=str,
    default='temp3',
    help='Result Save Folder(default: temp1)'
)
args = argparser.parse_args()

path_true = 'E:/CARLADataset/Town03/1/pm'
path_eval = 'E:/CARLADataset/Town03/1/model_L2_500_result_dis000/pm_insert'

os.makedirs(f'./{args.name}/' , exist_ok=True)

def main():
    global path_true, path_eval
    
    img_list = get_filelist(path_eval)
    print('>> 待评价图片总数: ', len(img_list))
    
    range_str = args.range
    start, end = map(float, range_str.split('-'))
    
    # 指定区间
    start_index = img_list.index(f'{start}')
    end_index = img_list.index(f'{end}')
    eval_img_list = img_list[start_index:end_index+1]
    
    mode = '快速模式' if not args.fuzzy else '模糊匹配模式'
    print('>> 评价模式: '+mode)
    print('>> 评价范围: '+eval_img_list[0]+'-'+eval_img_list[-1])
    print('>> 评价数量: ', len(eval_img_list))
    evaluator = Evaluation(rgb_mode=True, quick=not args.fuzzy) # rgb_mode取决于待评价图片颜色模式
    
    # 图片评价指标
    iou_list = []
    cover_rate_list = []
    delta_yaw_list = []
    file = open(f'./{args.name}/evaluation.csv', mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Time_Index', 'IOU', 'Cover_Rate', 'Delta_Yaw'])
    
    try:
        # 每 step 选择一张评价
        for i in tqdm(range(0, len(eval_img_list), args.step)):
            img_true = read_png_true(path_true, eval_img_list[i])
            img_eval = read_png(path_eval, eval_img_list[i])
            
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
            writer.writerow([eval_img_list[i], iou, cover_rate, delta_yaw])
    except KeyboardInterrupt:
        print('>> Exit by user !')   
    finally:
        file.close()
        with open(f'./{args.name}/average_evaluation.txt', 'w') as file:
            save_average_evaluation(file, iou_list, cover_rate_list, delta_yaw_list)
            print(f">> 评价结果保存 {args.name} 成功！")
    
if __name__ == '__main__':
    main()