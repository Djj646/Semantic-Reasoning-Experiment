import cv2
import argparse
from utils.io import read_png_true, read_png
from utils.evaluation import Evaluation

town = 'Town10HD'
# 计算单张图片评价以及显示融合图片
path_true = f'E:/CARLADataset/{town}/1/pm'
path_eval = f'E:/CARLADataset/{town}/1/model_SD_result_nav_dis016/pm_insert'

argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    't',
    default=1685429080.9128702,
    help='Caculate average evaluation(default: False)'
)
argparser.add_argument(
    '--fuzzy',
    action='store_true',
    default=False,
    help='Fuzzy Evaluate Mode(default: False)'
)
args = argparser.parse_args()

evaluator = Evaluation(rgb_mode=True, quick=not args.fuzzy)

def eval(time_index):
    global evaluator
    
    img_time_index = str(time_index)
    
    img_true = read_png_true(path_true, img_time_index)
    img_eval = read_png(path_eval, img_time_index)
    
    evaluator.update(img_true, img_eval)
    
    iou = evaluator.iou()
    cr = evaluator.cover_rate()
    dy = evaluator.delta_yaw()
    
    print('>> 正在分析 '+img_time_index+'...')
    print('iou: ', iou)
    print('cover_rate: ', cr)
    print('delta_yaw: ', dy)
    
    return iou, cr, dy

def blend(time_index, eval_result):
    global evaluator
    
    img_time_index = str(time_index)

    img_true_rgb = read_png(path_true, img_time_index)
    img_eval = cv2.resize(read_png(path_eval, img_time_index), (img_true_rgb.shape[1], img_true_rgb.shape[0]))

    # 为两张图像创建一个融合后的图像
    blended_image = cv2.addWeighted(img_true_rgb, 0.5, img_eval, 0.5, 0)
    
    image_marked = blended_image.copy()

    # 将绿色中心线的位置用红色标记
    for coord in evaluator.centerline_eval:
        x, y = coord
        x = int(x)  # 将坐标转换为整数类型
        y = int(y)
        cv2.circle(image_marked, (x, y), 1, (255, 0, 0), -1)
    
    for coord in evaluator.centerline_true:
        x, y = coord
        x = int(x)  # 将坐标转换为整数类型
        y = int(y)
        cv2.circle(image_marked, (x, y), 1, (0, 0, 255), -1)
    
    text_w = image_marked.shape[1]-150
    cv2.putText(image_marked, 'iou:'+str(eval_result[0]).split('.')[0], (text_w, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image_marked, 'cover_rate:'+str(eval_result[1]).split('.')[0], (text_w, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image_marked, 'delta_yaw:'+str(eval_result[2]).split('.')[0], (text_w, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('EVAL-'+img_time_index, image_marked)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    eval_result = eval(args.t)
    blend(args.t, eval_result)