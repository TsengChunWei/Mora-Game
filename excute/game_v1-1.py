import os
import time
import cv2
import torch
import random
from ultralytics import YOLO

def who_win(you, computer):
    if (you=='paper' and computer=='paper') or (you=='scissors' and computer=='scissors') or (you=='stone' and computer=='stone'):
        return 'equal'
    elif (you=='paper' and computer=='scissors') or (you=='scissors' and computer=='stone') or (you=='stone' and computer=='paper') or (you == ''):
        return 'computer'
    else:
        return 'you'

def get_result_image(result):
    img = result.orig_img.copy()
    coor = result.boxes.xyxy.to(torch.int).tolist()
    label_name = result.names
    classes = result.boxes.cls.tolist()

    font = cv2.FONT_HERSHEY_DUPLEX

    maxi, maxi_idx = 0, 0
    w_min, w_max, h_min, h_max = 300, 550, 200, 460
    is_exist = False

    cv2.rectangle(img, (w_min, h_min), (w_max, h_max), (0, 255, 255), 2)
    for id in range(len(coor)):
        if result.boxes.conf[id].item() > maxi:
            maxi = result.boxes.conf[id].item()
            w1, h1, w2, h2 = coor[id][0], coor[id][1], coor[id][2], coor[id][3]
            maxi_idx = id
            is_exist = True

    if is_exist:
        global your_mora
        your_mora = label_name[classes[maxi_idx]]
        if (w_min < (w1+w2)//2 < w_max) and (h_min < (h1+h2)//2 < h_max):
            cv2.putText(img, 'you: '+your_mora, (10, 100), font , 1, (50, 255, 50), 2, cv2.LINE_AA, False)
            
    return img

def timer(seconds):
    start_time = time.time()
    count = 0
    while count < seconds: 
        count = time.time() - start_time
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return True
    return False


currPath = os.path.dirname(__file__)
model = YOLO(currPath+"/../model_pt/first_model.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


counting = 5
endWindows = False
while True:
    begin = time.time()
    while counting > 0:
        your_mora = ''
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            track_results = model.track(frame, persist=True)
            annotated_frame = get_result_image(track_results[0])
            counting = 5-round(time.time()-begin)
            cv2.putText(annotated_frame, str(counting), (10, 50), cv2.FONT_HERSHEY_DUPLEX , 2, (0, 0, 255), 2, cv2.LINE_AA, False)
            cv2.imshow('frame', annotated_frame)
            # cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            endWindows = True
            break
    
    if counting == 0:
        # print(your_mora)
        computer_s_mora = random.choice(['paper', 'scissors', 'stone'])

        winner = who_win(your_mora, computer_s_mora)

        cv2.putText(annotated_frame, 'computer: '+computer_s_mora, (10, 150), cv2.FONT_HERSHEY_DUPLEX , 1, (50, 255, 50), 2, cv2.LINE_AA, False)
        cv2.putText(annotated_frame, 'winner: '+winner, (10, 200), cv2.FONT_HERSHEY_DUPLEX , 1, (0, 0, 255), 2, cv2.LINE_AA, False)
        cv2.imshow('frame', annotated_frame)
        endWindows = timer(5)
        counting = 5

    if endWindows: 
        break

cv2.destroyAllWindows()
