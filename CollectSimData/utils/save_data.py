import os
import cv2
import time
import numpy as np
from utils import debug

class SaveData():
    def __init__(self, save_path, enable_save=False):
        self.img = None
        self.seg_img = None
        self.pcd = None
        self.nav = None
        self.control = None
        self.pos = None
        self.acceleration = None
        self.angular_velocity = None
        self.vel = None
        
        self.path = str(save_path)
        self.enable_save = enable_save
        self.perepare_save()
        time.sleep(1) # 延时保证初始化成功
        
    def perepare_save(self):
        if not self.enable_save:
            return
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + 'img/', exist_ok=True)
        os.makedirs(self.path + 'segimg/', exist_ok=True)
        os.makedirs(self.path + 'pcd/', exist_ok=True)
        os.makedirs(self.path + 'nav/', exist_ok=True)
        os.makedirs(self.path + 'state/', exist_ok=True)
        os.makedirs(self.path + 'cmd/', exist_ok=True)
        
        # 文件写入指针
        self.cmd_file = open(self.path + 'cmd/cmd.txt', 'w+')
        self.pos_file = open(self.path + 'state/pos.txt', 'w+')
        self.vel_file = open(self.path + 'state/vel.txt', 'w+')
        self.acc_file = open(self.path + 'state/acc.txt', 'w+')
        self.angular_vel_file = open(self.path + 'state/angular_vel.txt', 'w+')
    
    def save(self, time):
        # RGB和NAV
        if not self.enable_save:
            return
        cv2.imwrite(self.path + 'img/' + str(time) + '.png', self.img)
        cv2.imwrite(self.path + 'nav/' + str(time) + '.png', self.nav)
        
        if self.seg_img is not None:
            cv2.imwrite(self.path + 'segimg/' + str(time) + '.png', self.seg_img)
        
        if self.pcd is not None:
            np.save(self.path + 'pcd/' + str(time) + '.npy', self.pcd)
        
        self.cmd_file.write(time + '\t' +
                   str(self.control.throttle) + '\t' +
                   str(self.control.steer) + '\t' +
                   str(self.control.brake) + '\n')
        # 使用时采用0, 1, 2, 3, 5，即ts, x, y, z, yaw
        self.pos_file.write(time + '\t' +
                    str(self.pos.location.x) + '\t' +
                    str(self.pos.location.y) + '\t' +
                    str(self.pos.location.z) + '\t' +
                    str(self.pos.rotation.pitch) + '\t' +
                    str(self.pos.rotation.yaw) + '\t' +
                    str(self.pos.rotation.roll) + '\t' + '\n')
        self.vel_file.write(time + '\t' +
                    str(self.vel.x) + '\t' +
                    str(self.vel.y) + '\t' +
                    str(self.vel.z) + '\t' + '\n')
        self.acc_file.write(time + '\t' +
                    str(self.acceleration.x) + '\t' +
                    str(self.acceleration.y) + '\t' +
                    str(self.acceleration.z) + '\t' + '\n')
        self.angular_vel_file.write(time + '\t' +
                            str(self.angular_velocity.x) + '\t' +
                            str(self.angular_velocity.y) + '\t' +
                            str(self.angular_velocity.z) + '\t' + '\n')
    
    def close_all(self):
        if not self.enable_save:
            return
        try:
            self.acc_file.close()
            self.angular_vel_file.close()
            self.cmd_file.close()
            self.pos_file.close()
            self.vel_file.close()
            
            debug(info='close all files', info_type='success')
        except:
            debug(info='failed to close all files', info_type='error')