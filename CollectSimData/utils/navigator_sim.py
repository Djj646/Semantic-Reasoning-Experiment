#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw

import carla

scale = 12.0
map_width = 8000
map_height = 8000
global_nav_scale = 0.05
x_offset = 0
y_offset = 0
map_border = 800

def get_random_destination(spawn_points):
    return random.sample(spawn_points, 1)[0]
    
def get_map(waypoint_tuple_list):
    global map_width, map_height, x_offset, y_offset, map_border
    max_x = 0
    min_x = 0
    max_y = 0
    min_y = 0
    
    for i in range(len(waypoint_tuple_list)):
        # lane segment entry waypoint
        _x1 = waypoint_tuple_list[i][0].transform.location.x
        _y1 = waypoint_tuple_list[i][0].transform.location.y
        # lane segment exit waypoint
        _x2 = waypoint_tuple_list[i][1].transform.location.x
        _y2 = waypoint_tuple_list[i][1].transform.location.y

        max_x = max(max_x, _x1, _x2)
        min_x = min(min_x, _x1, _x2)
        
        max_y = max(max_y, _y1, _y2)
        min_y = min(min_y, _y1, _y2)

    x_offset = abs(min_x)*scale if min_x<0 else -min_x*scale
    y_offset = abs(min_y)*scale if min_y<0 else -min_y*scale
    map_height = int((max_x-min_x)*scale + map_border*3)# 宽度，注意H,W顺序
    map_width = int((max_y-min_y)*scale + map_border*3)
    
    # print('x_offset: %f, y_offset: %f'%(x_offset, y_offset))
    # print('map_height: %d, map_width: %d'%(map_height, map_width))
    
    origin_map = np.zeros((map_width, map_height, 3), dtype="uint8")
    origin_map.fill(255)
    origin_map = Image.fromarray(origin_map)
    
    return origin_map

def draw_route(agent, destination, origin_map):
    global x_offset, y_offset
    
    start_waypoint = agent._map.get_waypoint(agent._vehicle.get_location())
    end_waypoint = agent._map.get_waypoint(destination.location)

    route_trace = agent.trace_route(start_waypoint, end_waypoint)
    route_trace_list = []
    for i in range(len(route_trace)):
        x = scale*route_trace[i][0].transform.location.x+x_offset+map_border
        y = scale*route_trace[i][0].transform.location.y+y_offset+map_border
        route_trace_list.append(x)
        route_trace_list.append(y)
    draw = ImageDraw.Draw(origin_map)
    draw.line(route_trace_list, 'red', width=30)
    
    route_lenth = int(len(route_trace_list)/2) # 检查路点长度

    return origin_map, route_lenth

def get_nav(vehicle, plan_map):
    global map_width, map_height, x_offset, y_offset

    x = int(scale*vehicle.get_location().x + x_offset + map_border)
    y = int(scale*vehicle.get_location().y + y_offset + map_border)
    _nav = plan_map.crop((x-400,y-400, x+400, y+400))

    im_rotate = _nav.rotate(vehicle.get_transform().rotation.yaw+90)
    nav = im_rotate.crop((_nav.size[0]//2-150, _nav.size[1]//2-2*120, _nav.size[0]//2+150, _nav.size[1]//2)) # 300*240
    nav = cv2.cvtColor(np.asarray(nav), cv2.COLOR_BGR2RGB)
    return nav

def get_global_nav_map(plan_map):
    global global_nav_scale, map_width, map_height
    # resize((h,w))
    global_nav_scale = 640.0/map_height
    global_nav_map = plan_map.resize((int(map_height*global_nav_scale), int(map_width*global_nav_scale)), Image.ANTIALIAS)
    # print("plan_map: %d, %d"%(plan_map.size[0], plan_map.size[1]))
    # print("global_nav_map: %d, %d"%(global_nav_map.size[0], global_nav_map.size[1]))
    return global_nav_map

def get_global_nav(vehicle, global_nav_map):
    global map_width, map_height, global_nav_scale, x_offset, y_offset, map_border
    
    r = 2
    draw = ImageDraw.Draw(global_nav_map)
    x = int((scale*vehicle.get_location().x + x_offset + map_border)*global_nav_scale)
    y = int((scale*vehicle.get_location().y + y_offset + map_border)*global_nav_scale)
    draw.ellipse((x-r, y-r, x+r, y+r), fill='green', outline='green', width=2)

    global_nav_map = cv2.cvtColor(np.asarray(global_nav_map), cv2.COLOR_RGB2BGR)
    
    return global_nav_map
    
def replan(agent, destination, origin_map):
    agent.set_destination(carla.Location(destination.location.x,
                           destination.location.y,
                           destination.location.z))
    # agent.set_destination(destination)
    plan_map, route_lenth = draw_route(agent, destination, origin_map)
    
    return plan_map, route_lenth

def replan2(agent, destination, origin_map):
    agent.set_destination(agent.vehicle.get_location(), destination.location, clean=True)
    plan_map = draw_route(agent, destination, origin_map)
    return plan_map
    
def close2dest(vehicle, destination, dist=30):
    return destination.location.distance(vehicle.get_location()) < dist