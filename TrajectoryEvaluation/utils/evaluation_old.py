import cv2
import math

# ---------------------------
# *本版本由于运行效率过低而废弃
# --------------------------

class Param():
    isPathDist = 30 # 判断为绿色、白色颜色欧式距离
    matchCoordDist = 1 # 坐标是否重合最小匹配距离
    minTrajWidth = 20 # 轨迹最小宽度
    

class Evaluation():
    def __init__(self, img_true, img_eval, rgb_mode=True):
        self.param = Param()
        self.rgb_mode = rgb_mode
        self.img_height = img_true.shape[0]
        self.img_width = img_true.shape[1]
        # print('img_height: ', self.img_height, 'img_width: ', self.img_width)
        self.update_imgs(img_true, img_eval)
        
    def get_traj(self, img, rgb_mode=True):
        # 初始化轨迹列表
        traj = []

        # 遍历每个像素
        for x in range(self.img_width):
            for y in range(self.img_height):
                # print('x:',x,'y',y)
                b, g, r = img[y, x]
                d = math.sqrt((b - 0)**2 + (g - 255)**2 + (r - 0)**2) if rgb_mode \
                    else math.sqrt((b - 255)**2 + (g - 255)**2 + (r - 255)**2)
                # 如果像素是绿色，则将其坐标保存到轨迹列表中

                if d < self.param.isPathDist:
                        traj.append((x, y))
                    
        return traj
    
    def get_traj_centerline(self, traj):
        # 字典y2x用于存储相同y坐标下所有的x坐标
        y2x = {}
        for coord in traj:
            if coord[1] not in y2x:
                y2x[coord[1]] = []
            y2x[coord[1]].append(coord[0])

        # 计算中心点
        centerline = []
        for y, x_list in y2x.items():
            if len(x_list) > self.param.minTrajWidth:
                center_x = sum(x_list) / len(x_list)
                centerline.append((center_x, y))
        
        return centerline
    
    def update_imgs(self, img_true, img_eval):
        self.img_true = img_true
        self.img_eval = cv2.resize(img_eval, (self.img_width, self.img_height))
        self.traj_true = self.get_traj(self.img_true, False)
        self.traj_eval = self.get_traj(self.img_eval, self.rgb_mode)
        self.centerline_true = self.get_traj_centerline(self.traj_true)
        self.centerline_eval = self.get_traj_centerline(self.traj_eval)
    
    def iou(self):
        num_matching_coords = 0
        
        for coord_true in self.traj_true:
            for coord_eval in self.traj_eval:
                # 计算两个坐标的欧几里得距离
                dist = math.sqrt((coord_true[0]-coord_eval[0])**2 \
                    + (coord_true[1]-coord_eval[1])**2)
                
                # 如果距离小于 matchCoordDist，则表示两个坐标重合
                if dist < self.param.matchCoordDist:
                    num_matching_coords += 1
                    break # 找到一个即可，不重复
                
        # 计算重合坐标占总 traj_true 坐标数量的百分比
        iou = num_matching_coords / len(self.traj_true) * 100.0
        
        return iou
    
    def cover_rate(self):
        num_matching_centerline_coords = 0
        
        for coord_centerline_eval in self.centerline_eval:
            for coord_true in self.traj_true:
                dist = math.sqrt((coord_centerline_eval[0]-coord_true[0])**2 \
                    + (coord_centerline_eval[1]-coord_true[1])**2)
                
                if dist < self.param.matchCoordDist:
                    num_matching_centerline_coords += 1
                    break
        
        # 计算重合坐标占总轨迹中心线坐标数量的百分比
        cover_rate = num_matching_centerline_coords / len(self.centerline_eval) * 100.0
        
        return cover_rate
    
    def delta_yaw(self):
        start_x_true, start_y_true = max(self.centerline_true, key=lambda item: item[1])
        end_x_true, end_y_true = min(self.centerline_true, key=lambda item: item[1])
        
        start_x_eval, start_y_eval = max(self.centerline_eval, key=lambda item: item[1])
        end_x_eval, end_y_eval = min(self.centerline_eval, key=lambda item: item[1])
        
        yaw_true = math.atan2((end_y_true-start_y_true), (end_x_true-start_x_true))
        yaw_eval = math.atan2((end_y_eval-start_y_eval), (end_x_eval-start_x_eval))
        
        delta_yaw = abs(yaw_true-yaw_eval)
        
        return delta_yaw

    def evalate(self):
        iou = self.iou()
        cover_rate = self.cover_rate()
        delta_yaw = self.delta_yaw()
        
        return iou, cover_rate, delta_yaw