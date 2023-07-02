import glob
import cv2

# 读取文件 true: pos_pm(360x640), eval: pm_insert(512x1024)
def get_filelist(path):
    files = glob.glob(path+'/*.png')
    file_names = []
    for file in files:
        file = file.split('/')[-1]
        file_name = file.split('\\')[-1][:-4] # win文件路径 '\\' 修改
        file_names.append(file_name)
    file_names.sort()
    return file_names

def read_png_true(path, file_name:str):
    item_path = path+'/'+str(file_name)+'.png'
    item = cv2.imread(item_path, cv2.IMREAD_GRAYSCALE) # 黑白模式
    return item

def read_png(path, file_name:str):
    item_path = path+'/'+str(file_name)+'.png'
    item = cv2.imread(item_path)  
    return item

def save_average_evaluation(file, iou, cover_rate, delta_yaw):
    iou_average = sum(iou)/len(iou)
    cover_rate_average = sum(cover_rate)/len(cover_rate)
    delta_yaw_average = sum(delta_yaw)/len(delta_yaw)
    
    # 将 t1、t2 和 t3 写入文件
    file.write('average iou: ' + str(iou_average) + '\n')
    file.write('average cover_rate: ' + str(cover_rate_average) + '\n')
    file.write('average delta_yaw: ' + str(delta_yaw_average) + '\n')
    file.close()