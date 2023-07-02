import keyboard

Throttle = 0
Steer = 0

doc = 'Q:\t manual/auto\n\
R:\t record\n\
W:\t throttle\n\
S:\t brake\n\
A/D:\t steer left/right\n\
E: \t reverse\n'

class ManualControl():
    def __init__(self):
        self.throttle = 0
        self.steer = 0
        self.brake = 0
        self.reverse = False # 倒车状态
        self.is_manual = False # 是否手动状态
        self.enable_record = True # 是否记录

def throttle_increase(control):
    global Throttle
    Throttle += 0.05 + (0.7-Throttle)*0.2
    Throttle = min(Throttle, 1.0)
    control.throttle = Throttle
    control.brake = 0
    
def brake(control):
    global Throttle
    Throttle = Throttle/2
    control.brake = 1.0
    control.throttle = Throttle

def steer_left_increase(control):
    global Steer
    Steer -= (0.1 +(1-abs(Steer))*0.1)
    Steer = max(Steer, -1.0)
    control.steer = Steer

def steer_right_increase(control):
    global Steer
    Steer += (0.1 +(1-abs(Steer))*0.1)
    Steer = min(Steer, 1.0)
    control.steer = Steer
    
def forward_left(control):
    throttle_increase(control)
    steer_left_increase(control)
    
def forward_right(control):
    throttle_increase(control)
    steer_right_increase(control)

def brake_left(control):
    brake(control)
    steer_left_increase(control)
    
def brake_right(control):
    brake(control)
    steer_right_increase(control)

def reverse(control):
    control.reverse = not control.reverse

def reset(event, control):
    global Throttle, Steer
    
    last_key = event.name
    if last_key == 'w':
        Throttle -= 0.02
        control.throttle = min(Throttle, 0)
    elif last_key == 's':
        control.brake = 0
    elif last_key == 'a' or last_key == 'd':
        Steer = 0
        control.steer = 0

def manual(control):
    control.is_manual = not control.is_manual

def record(control):
    control.enable_record = not control.enable_record

def manual_control(control):
    print(doc)
    keyboard.add_hotkey("q", lambda: manual(control))
    keyboard.add_hotkey("r", lambda: record(control))
    keyboard.add_hotkey("e", lambda: reverse(control))
    keyboard.add_hotkey("w", lambda: throttle_increase(control))
    keyboard.add_hotkey("a", lambda: steer_left_increase(control))
    keyboard.add_hotkey("d", lambda: steer_right_increase(control))
    keyboard.add_hotkey("s", lambda: brake(control))
    keyboard.add_hotkey("w+a", lambda: forward_left(control))
    keyboard.add_hotkey("w+d", lambda: forward_right(control))
    keyboard.add_hotkey("s+a", lambda: brake_left(control))
    keyboard.add_hotkey("s+d", lambda: brake_right(control))
    keyboard.on_release(lambda event: reset(event, control))