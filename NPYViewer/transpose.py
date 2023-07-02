import sys
import os
import numpy as np

def trans(filename):
    try:
        # 加载原始的 .npy 文件数据
        data = np.load(filename)
        new_filename = os.path.basename(filename).split('.')[0] + '_T' + '.npy'

        # 调用 transpose 方法重新排列数据
        new_data = data.transpose()

        # 保存新的数据到 .npy 文件中
        np.save(new_filename , new_data)
    
        print(new_filename + " exported successfully!")
    except:
        print("Error")
    
if __name__ == '__main__':
    if len(sys.argv) ==2:
        trans(sys.argv[1])
    else:
        print("Enter the name of the file to be converted")
