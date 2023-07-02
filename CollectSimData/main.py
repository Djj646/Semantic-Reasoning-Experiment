import sys
import glob
import os

from utils import debug

# from os.path import join, dirname
# sys.path.insert(0, join(dirname(__file__), '..')) # 0表示python解释器搜索的优先级最高

path = 'D:\CARLA_0.9.13\WindowsNoEditor\PythonAPI'
sys.path.append(path) # 包括carla
sys.path.append(path+'/carla') # 包括agents

try:
    sys.path.append(glob.glob(path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("导入包失败，请检查路径 "+path+" 是否正确")

import carla
from carla import VehicleLightState as vls

from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

from utils.sensor_manager import SensorManager, visualize_data
from utils.navigator_sim import get_random_destination, get_map, get_nav, get_global_nav_map, get_global_nav, replan, close2dest
from utils.save_data import SaveData
from utils.spawn_actor import add_vehicle, add_npc_vehicles, add_npc_walkers, DestroyActor, SetVehicleLightState

import os
import cv2
import time
import copy
import numpy as np

import argparse
from tqdm import tqdm

from manual.manual_control import manual_control, ManualControl

world_config = {
    'host': 'localhost',
    'port': 2000,
    'timeout': 10.0,
    'town': 'Town01',
    'weather': carla.WeatherParameters.ClearNoon, # HardRainSunset, ClearNoon, ClearSunset, HardRainNoon
    'check_arrival_distance': 10,
    'min_route_lenth': 200,
}

# 传感器回馈函数暂存数据
global_img = None
global_pcd = None
global_pcd_unfilted = None
global_seg_img = None
global_lidar_viz = None
global_lidar_viz_unfilted = None

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
    '--enable-save',
    action='store_true',
    help='Save the data(default False)')
argparser.add_argument(
    '-M','--map-number',
    metavar='SERIAL NUMBER',
    default=1,
    type=int,
    help='Map Serial Number (1-5, default: 1)')
argparser.add_argument(
    '--vel',
    metavar='SPEED',
    default=30,
    type=int,
    help='speed of the host vehilce (default: 30)')
argparser.add_argument(
    '--vehicles',
    metavar='NUMBER',
    default=0,
    type=int,
    help='number of vehicles (default: 6)')
argparser.add_argument(
    '--walkers',
    metavar='NUMBER',
    default=0,
    type=int,
    help='number of walkers (default: 10)')
argparser.add_argument(
    '--safe',
    action='store_true',
    help='avoid spawning vehicles prone to accidents')
argparser.add_argument(
    '--filterv',
    metavar='Vehicle PATTERN',
    default='vehicle.*',
    help='vehicles filter (default: "vehicle.*")')
argparser.add_argument(
    '--filterw',
    metavar='Walker PATTERN',
    default='walker.pedestrian.*',
    help='pedestrians filter (default: "walker.pedestrian.*")')
argparser.add_argument(
    '--tm-port',
    metavar='P',
    default=8000,
    type=int,
    help='port to communicate with TM (default: 8000)')
argparser.add_argument(
    '--sync',
    action='store_false',
    default=True,
    help='Not Synchronous mode execution(default: True)')
argparser.add_argument(
    '--hybrid',
    action='store_true',
    help='Enanble Hybrid Mode')
argparser.add_argument(
    '--car-lights-on',
    action='store_true',
    default=False,
    help='Enanble car lights')
argparser.add_argument(
    '--traffic-lights',
    action='store_true',
    default=False,
    help='Ignore Traffic Lights(only ego vehicle)')
argparser.add_argument(
    '-n', '--data-num',
    type=int,
    default=20000,
    help='Total Number'
)
argparser.add_argument(
    '--data-index',
    type=int,
    default=1,
    help='Data Index'
)
args = argparser.parse_args()

# 图像反馈函数 将相机data存入全局变量global_img中，NumPy数组形状(h, w, 4)
def image_callback(data):
    global global_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    # (samples, channels, height, width)
    # 4 channels: R, G, B, A
    array = np.reshape(array, (data.height, data.width, 4))  # RGBA format
    global_img = array
    
# 激光雷达反馈函数 将激光雷达data的points中选择符合两个坐标条件的
# 存入全局变量global_pcd中
def lidar_callback(data):
    global global_pcd, global_pcd_unfilted, global_lidar_viz, global_lidar_viz_unfilted
    # (samples, (dim1, dim2, dim3)) NumPy二维数组 点集points
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 4])
    # np.stack用于堆叠数组
    # 第一维度为sample 分别取第1，0，2列坐标，对应y，x，z，变成3行的二维数组point_cloud
    point_cloud = np.stack([-lidar_data[:, 1], -lidar_data[:, 0], -lidar_data[:, 2]])
    mask = \
        np.where((point_cloud[0] > 1.0) | (point_cloud[0] < -4.0) | (point_cloud[1] > 1.2) | (point_cloud[1] < -1.2))[0]
    # np.where(condition)返回符合第一行坐标条件的序号（以元组的形式） 取元组第一位保存到mask中
    # 取满足条件的sample序号mask 对x, y坐标做出维度坐标限定
    point_cloud = point_cloud[:, mask]
    # 在满足第一行条件下（只取一列）
    # 取满足第三行坐标条件的序号（最多一列）
    mask = np.where(point_cloud[2] > -1.95)[0]
    point_cloud = point_cloud[:, mask]
    
    global_lidar_viz = visualize_data(point_cloud.transpose())
    global_lidar_viz_unfilted = visualize_data(lidar_data)
    global_pcd = point_cloud
    global_pcd_unfilted = lidar_data
    
def seg_image_callback(data):
    global global_seg_img
    data.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    # (samples, channels, height, width)
    # 4 channels: R, G, B, A
    array = np.reshape(array, (data.height, data.width, 4))  # RGBA format
    global_seg_img = array[:, :, :3]


def main():
    global args, global_pcd, global_pcd_unfilted, global_img, global_seg_img, \
        global_lidar_viz, global_lidar_viz_unfilted
    
    world_config['town'] = 'Town' + '{:02d}'.format(args.map_number)
    # world_config['town'] = 'Town10HD'
    debug(info='Collect From Map: '+world_config['town'], info_type='message')

    debug(info='Save Enabled' if args.enable_save else 'Save Disabled', info_type='message')
    sdata = SaveData(save_path='E:/CARLADataset/'\
        +world_config['town']+f'/{args.data_index}/', enable_save=args.enable_save)

    client = carla.Client(world_config['host'], world_config['port'])
    client.set_timeout(world_config['timeout'])
    world = client.load_world(world_config['town'])
    
    # 天气
    weather = carla.WeatherParameters(
        # cloudiness=90.0,
        # precipitation=0, # 0-100 rain
        sun_altitude_angle=70, # <0 night
        fog_density = 100, # fog
        fog_distance = 40, 
        # wetness = 20,
        )
    # world_config['weather'] = weather
    world.set_weather(world_config['weather'])
    
    blueprints = world.get_blueprint_library()
    world_map = world.get_map()

    ego_vehicle = add_vehicle(world, blueprints, vehicle_type='vehicle.audi.tt')
    ego_vehicle.set_simulate_physics(True)
    ego_light_state = vls(vls.Position | vls.LowBeam | vls.LowBeam)
    if args.car_lights_on:
        ego_vehicle.set_light_state(ego_light_state)
    
    sensor_dict = {
        'camera': {
            'transform': carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback': image_callback,
        },
        # 'lidar': {
        #     'transform': carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
        #     'callback': lidar_callback,
        # },
        # 'semantic': {
        #     'transform': carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
        #     'callback': seg_image_callback,
        # },
    }
    
    sm = SensorManager(world, blueprints, ego_vehicle, sensor_dict)
    sm.init_all()
    
    des_spawn_points = world_map.get_spawn_points()

    # 返回openDRIVE文件的拓扑的最小图元祖列表
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)
    
    if args.traffic_lights:
        debug(info='Ignore Traffic Lights (Only With No Npc)', info_type='message')
        
    agent = BasicAgent(ego_vehicle, target_speed=args.vel, opt_dict={'ignore_traffic_lights': args.traffic_lights})
    
    # 第一次导航
    destination = get_random_destination(des_spawn_points)
    plan_map, route_lenth = replan(agent, destination, copy.deepcopy(origin_map))
    global_nav_map = get_global_nav_map(plan_map)
    
    settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    
    if args.hybrid:
        traffic_manager.set_hybrid_physics_mode(True)
    
    if args.sync:
        traffic_manager.set_synchronous_mode(True)
        debug(info='Synchronous Mode', info_type='message')
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
    else:
        debug(info='Not Synchronous Mode', info_type='message')
    
    vehicles_list = add_npc_vehicles(client, world, traffic_manager, args.filterv, args.safe, args.sync, args.car_lights_on, args.vehicles, False)
    walkers_list, all_actors, all_id = add_npc_walkers(client, world, args.filterw, args.sync, args.walkers)

    debug(info='spawned %d vehicles and %d walkers !' % (len(vehicles_list), len(walkers_list)), info_type='success')
    
    try:
        m_control = ManualControl()
        manual_control(m_control)
        for cnt in tqdm(range(args.data_num)):
            # world.tick() # sync_mode
            # world.wait_for_tick() # not sync_mode
            (world.tick() if args.sync else world.wait_for_tick())
            
            if close2dest(ego_vehicle, destination, world_config['check_arrival_distance']):
                destination = get_random_destination(des_spawn_points)
                plan_map, route_lenth = replan(agent, destination, copy.deepcopy(origin_map))
                
                while route_lenth<world_config['min_route_lenth']:
                    # debug(info='New destination too close ! Replaning...', info_type='warning')
                    destination = get_random_destination(des_spawn_points)
                    plan_map, route_lenth = replan(agent, destination, copy.deepcopy(origin_map))

                global_nav_map = get_global_nav_map(plan_map)

            control = agent.run_step()
            
            if m_control.is_manual:
                control.throttle = m_control.throttle
                control.steer = m_control.steer
                control.brake = m_control.brake
                control.reverse = m_control.reverse
            
            # control = carla.VehicleControl(throttle=30, steer=0)
            ego_vehicle.apply_control(control)
            sdata.control = control
            # 获得卫星导航图
            sdata.nav = get_nav(ego_vehicle, plan_map)
            # 获得全局位置
            global_nav = get_global_nav(ego_vehicle, global_nav_map)
            # 位置与姿态信息：x, y, z, pitch(y), yaw(z), roll(x) 
            sdata.pos = ego_vehicle.get_transform()
            sdata.vel = ego_vehicle.get_velocity()
            sdata.acceleration = ego_vehicle.get_acceleration()
            sdata.angular_velocity = ego_vehicle.get_angular_velocity()
            sdata.img = global_img
            sdata.pcd = global_pcd
            sdata.seg_img = global_seg_img
            
            # cv2.imshow('Nav', sdata.nav)
            cv2.imshow('Global_Nav', global_nav)
            cv2.imshow('Vision', sdata.img)
            # cv2.imshow('SegVision', sdata.seg_img)
            # cv2.imshow('Lidar', global_lidar_viz)
            # cv2.imshow('Lidar_Unfilted', global_lidar_viz_unfilted)
            
            # --------------------------------
            # 调节实际采集帧率，最高帧需取消监视
            # --------------------------------
            cv2.waitKey(1)
            time_index = str(time.time())
            if m_control.enable_record:
                sdata.save(time_index)
    except KeyboardInterrupt:
        print("Exit by user !")
    finally:
        debug(info='destroying ego_vehicle', info_type='message')
        client.apply_batch([DestroyActor(ego_vehicle)])
        
        debug(info='destroying %d vehicles and %d walkers' % (len(vehicles_list),len(walkers_list)), info_type='message')
        client.apply_batch([DestroyActor(x) for x in vehicles_list])

        for i in range(0, len(all_actors), 2):
            all_actors[i].stop()

        client.apply_batch([DestroyActor(x) for x in all_id])

        sm.close_all()
        sdata.close_all()
        cv2.destroyAllWindows()
        
        # 重置world
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(False)
        
        time.sleep(0.5)


if __name__ == '__main__':
    try: 
        main()
    except KeyboardInterrupt:
        print("Exit by user !")
