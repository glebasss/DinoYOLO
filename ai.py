import cv2
from ultralytics import YOLO
import numpy as np
from mss import mss
from pynput.keyboard import Key, Controller
import threading
import time

keyboard = Controller()
in_space = False

short_space = ["birdlow", "birdmid", "gameover", "small1", "small2", ]
long_space = ["bigthree", "bigtwo", "bigone", "small3"]
my_map = {
    0: "bigone",
    1: "bigthree",
    2: "bigtwo",
    3: "birdlow",
    4: "birdmid",
    5: "dino",
    6: "gameover",
    7: "highbird",
    8: "small1",
    9: "small2",
    10: "small3"
}


def press_space(duration):
    global in_space
    in_space = True
    keyboard.press(Key.space)
    time.sleep(duration)
    keyboard.release(Key.space)
    in_space = False


def screendetect(arg, result):
    """
    unpack with x1,y1,x2,y2
    """
    x1, y1, x2, y2 = result.boxes.xyxy[arg]
    x1, y1, x2, y2 = x1.round(), y1.round(), x2.round(), y2.round()
    x1, y1, x2, y2 = x1-50, y1+60, x2+550, y2-130
    return int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())


def get_dict(result, for_distance=False):
    my_dict = dict()
    if not for_distance:
        for n in range(len(result.boxes.cls)):
            if int(result.boxes.cls[n]) in my_dict.keys() and my_dict[int(result.boxes.cls[n])][1] > float(result.boxes.conf[n]):
                pass
            else:
                my_dict[int(result.boxes.cls[n])] = [n, float(
                    result.boxes.conf[n]), result.boxes.xyxy[n]]
    if for_distance:
        for n in range(len(result.boxes.cls)):
            my_dict[n] = [int(result.boxes.cls[n]), result.boxes.xyxy[n]]
    return my_dict


def getdino(dict_):
    return dict_[5][0]


def dino_and_near_distance(dict_):
    key_dino = ''
    for key, val in dict_.items():
        if val[0] == 5:
            key_dino = key
            break
    x_dino = int(dict_[key_dino][1][0].item() + dict_[key_dino][1][2].item())/2
    dict_of_x_distances = dict()
    for key, val in dict_.items():
        dist = int(dict_[key][1][0])-x_dino
        if key != key_dino and dist > 0:
            dict_of_x_distances[dict_[key][0]] = dist
    minn = 100000
    name = ''
    for key, val in dict_of_x_distances.items():
        if val < minn:
            minn = val
            name = my_map[key]
    return name, minn


model = YOLO("modeldino.pt", verbose=False)
sct = mss()
monitor = sct.monitors[2]
stop = False
while True:
    if stop:
        break
    screen_shot = sct.grab(monitor)
    img = np.array(screen_shot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    results = model.predict(source=img, imgsz=1920, conf=0.5, verbose=False)
    result = results[0]
    for clss in result.boxes.cls:
        print(int(clss))
        if int(clss) == 5:
            dict_ = get_dict(result)
            dino_present = getdino(dict_)
            x1, y1, x2, y2 = screendetect(dino_present, result)
            stop = True
            break
monitor = {
    'top': y2,
    'left': x1,
    'width': x2-x1,
    'height': y1-y2,
}
while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    results = model(frame, verbose=False)
    result = results[0]
    dict_ = get_dict(result, for_distance=True)
    name, dist = dino_and_near_distance(dict_)
    print(name)
    if dist <= 100 and not in_space:
        if name in long_space:
            threading.Thread(target=press_space, args=(0.2,)).start()
        if name in short_space:
            threading.Thread(target=press_space, args=(0.01,)).start()
    cv2.imshow('aaaa', result.plot())
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
