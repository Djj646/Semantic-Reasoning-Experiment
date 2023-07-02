import cv2
import math
import numpy as np

class Param():
    matchCoordDist = 2 # 模糊匹配模式坐标是否重合最小匹配距离
    minTrajWidth = 20 # 轨迹计算中心线最小宽度
    green_lower = [0, 245, 0] # 判断为绿色颜色范围
    green_upper = [10, 255, 10]
    alpha = 1 # iou
    beta = 1 # cover_rate
    gamma = 1 # delta_yaw
    
class Evaluation():
    def __init__(self, img_true=None, img_eval=None, rgb_mode=True, quick=False):
        self.param = Param()
        self.rgb_mode = rgb_mode
        self.img_true = img_true
        self.img_eval = img_eval
        self.mask_true = None
        self.mask_eval = None
        self.quick = quick # 是否快速，快速则不进行坐标模糊匹配计算
        
        if img_true is not None and img_eval is not None:
            self.img_height = img_true.shape[0]
            self.img_width = img_true.shape[1]
            self.update(img_true, img_eval)
     
    def get_true_traj(self, img):
        # 将图片转换为二值化形式，设置阈值为 127（假设大于阈值的像素为白色，小于阈值的像素为黑色）
        _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # self.mask_true = binary_image//250
        
        # 获取白色像素的位置
        white_pixel_indices = np.where(binary_image == 255)
        white_pixel_positions = list(zip(white_pixel_indices[1], white_pixel_indices[0])) # x,y
        
        return white_pixel_positions
    
    # rgb mode
    def get_eval_traj(self, img):
        # 定义绿色的RGB范围（以BGR格式表示）
        green_lower = np.array(self.param.green_lower, dtype=np.uint8)
        green_upper = np.array(self.param.green_upper, dtype=np.uint8)

        # 创建掩膜，提取绿色像素
        mask = cv2.inRange(img, green_lower, green_upper)
        
        # self.mask_eval = mask//250

        # 获取绿色像素的位置
        green_pixel_indices = np.where(mask == 255)
        green_pixel_positions = list(zip(green_pixel_indices[1], green_pixel_indices[0]))

        return green_pixel_positions

    def get_centerline(self, traj):
        centerline = []
        # y 为coord[0], x为coord[1]
        for y in np.unique([coord[1] for coord in traj]):
            row_positions = [pos[0] for pos in traj if pos[1] == y]
            if len(row_positions) < self.param.minTrajWidth:
                continue
            center_x = sum(row_positions) / len(row_positions)
            center_x = int(center_x)
            centerline.append((center_x, y))
        return centerline
    
    def update(self, img_true, img_eval):
        self.img_true = img_true
        self.img_height = img_true.shape[0]
        self.img_width = img_true.shape[1]
        self.img_eval = cv2.resize(img_eval, (self.img_width, self.img_height))
        self.traj_true = self.get_true_traj(self.img_true)
        self.traj_eval = self.get_eval_traj(self.img_eval) if self.rgb_mode \
            else self.get_true_traj(self.img_eval)
        self.centerline_true = self.get_centerline(self.traj_true)
        self.centerline_eval = self.get_centerline(self.traj_eval)
    
    # 模糊匹配模式，检查某点坐标是否在列表内
    @staticmethod
    def matchCoord(coord, coord_list, dist):
        x, y = coord
        x_range = range(x - dist, x + dist)
        y_range = range(y - dist, y + dist)
        fuzzy_coords = [(i, j) for i in x_range for j in y_range]

        for fuzzy_coord in fuzzy_coords:
            if fuzzy_coord in coord_list:
                return True
        
        return False
    
    # 计算交并比
    def iou(self):
        # 列表化成集合
        set_true = set(self.traj_true)
        set_eval = set(self.traj_eval)
        
        if not self.quick:
            # 坐标距离计算匹配法（弃用）
            # overlapping_positions = [pos for pos in self.traj_true if pos in self.traj_eval or\
            # any(np.linalg.norm(np.array(pos) - np.array(green_pos)) < self.param.matchCoordDist for green_pos in self.traj_eval)]
            traj_inter = [pos for pos in self.traj_true if self.matchCoord(pos, self.traj_eval, self.param.matchCoordDist)]
        else:
            # 列表匹配法（弃用）
            # traj_inter = [pos for pos in self.traj_true if pos in self.traj_eval]
            # 集合匹配法
            traj_inter = list(set_true.intersection(set_eval))
        
        
        # 计算重合部分像素的数量和比例
        inter_count = len(traj_inter)

        # 计算并集
        traj_union = list(set_true.union(set_eval))
        
        IoU = inter_count / len(traj_union) * 100.0

        return IoU
    
    # 并行与或运算 速度与iou一致
    def iou2(self):
        mask1 = self.mask_true
        mask2 = self.mask_eval
        
        i_mask = mask1 & mask2
        o_mask = mask1 | mask2
        
        i_sum = np.sum(i_mask)
        o_sum = np.sum(o_mask)
        
        return i_sum/o_sum * 100.0
    
    def cover_rate(self):
        # 计算 green_centerline 中坐标包含在 traj_true 的比例
        green_in_white_count = 0
        for coord in self.centerline_eval:
            if not self.quick:
                # if any(np.linalg.norm(np.array(coord) - np.array(white_pos)) < self.param.matchCoordDist for white_pos in self.traj_true):
                if self.matchCoord(coord, self.traj_true, self.param.matchCoordDist):
                    green_in_white_count += 1
            else:
                if coord in self.traj_true:
                    green_in_white_count += 1

        green_in_white_ratio = green_in_white_count / len(self.centerline_eval) * 100.0 if not len(self.centerline_eval)==0 else 0
        
        return green_in_white_ratio
    
    @staticmethod
    def cal_yaw(centerline):
        sorted_centerline = sorted(centerline, key=lambda coord: coord[1], reverse=False)
        
        # 计算最高点和最低点的坐标
        end_point = sorted_centerline[-1]
        start_point = sorted_centerline[0] # y最小为起点

        # 计算最高点和最低点之间的角度
        angle = math.atan2((end_point[1]-start_point[1]),(end_point[0]-start_point[0]))

        return angle
    
    def delta_yaw(self):
        yaw_true = self.cal_yaw(self.centerline_true)
        yaw_eval = self.cal_yaw(self.centerline_eval)
        
        # 将弧度转换为角度
        angle_deg = math.degrees(abs(yaw_true-yaw_eval))
        
        return angle_deg
    
    # 总体评价
    def evalate(self):
        iou = self.iou()
        cover_rate = self.cover_rate()
        delta_yaw = self.delta_yaw()
        
        return iou*self.param.alpha+cover_rate*self.param.beta+ self.param.gamma/delta_yaw