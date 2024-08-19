import os
import time
import cv2
import torch
from ultralytics import YOLO


def get_result_image(result):
    img = result.orig_img.copy()    
    coor = result.boxes.xyxy.to(torch.int).tolist()
    label_name = result.names
    classes = result.boxes.cls.tolist()

    font = cv2.FONT_HERSHEY_DUPLEX
    text_color, text_color_bg = (0,0,0), (0,255,255)
    font_scale = 0.8
    font_thickness = 1
    
    for id in range(len(coor)):
        label = label_name[classes[id]]
        # print((coor[id][0], coor[id][1]), (coor[id][2], coor[id][3]))

        if result.boxes.id == None:
            msg = f"{label} {result.boxes.conf[id].item():.2f}"
        else:
            msg = f"{label} id:{result.boxes.id[id]:.0f} {result.boxes.conf[id].item():.2f}"

        if label == 'paper':
            text_color, text_color_bg = (220,20,20), (255,255,255)
        elif label == 'rock':
            text_color, text_color_bg = (220,20,20), (100,255,255)
        else:
            text_color, text_color_bg = (240,240,240), (255,0,0)
            
        cv2.rectangle(img, (coor[id][0], coor[id][1]), (coor[id][2], coor[id][3]), text_color_bg, 2)

        text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, (coor[id][0], coor[id][1]-text_h), (coor[id][0] + text_w, coor[id][1]), text_color_bg, -1)
        cv2.putText(img, msg, (coor[id][0], int(coor[id][1]+ font_scale - 1)), font , font_scale, text_color, font_thickness, cv2.LINE_AA, False) 
    return img


def main():
    # Load the YOLOv8 model
    currPath = os.path.dirname(__file__)

    model = YOLO(currPath+"/200_ep_700_img.pt")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        begin_time = time.time()
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if ret:
            track_results = model.track(frame, persist=True)
            annotated_frame = get_result_image(track_results[0]) #自己寫的畫圖
            cv2.imshow('frame', annotated_frame)
            print(f'{1 / (time.time() - begin_time):.3f}')
    
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()


main()