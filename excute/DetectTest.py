import os
import time
import cv2
import torch
from ultralytics import YOLO


class WindowSize():
    def __init__(self, width, height) -> None:
        self.shape = (width, height)

class myclass():
    def __init__(self, winsize) -> None:
        self.win = winsize

    def confident_setting(self, max_conf_take_or_not, minimum_val=0):
        self.max_conf_take_or_not = max_conf_take_or_not
        self.minimum_val = minimum_val

    def detect_cv2_setting(self):
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.text_color = (0,0,0)
        self.text_color_bg = (0,255,255)
        self.font_scale = 0.8
        self.font_thickness = 1

    def after_detect_frame(self, result, id, label, Cr, img):
        if result.boxes.conf[id].item() < self.minimum_val:
            return img
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
            
        cv2.rectangle(img, (Cr[0], Cr[1]), (Cr[2], Cr[3]), text_color_bg, 2)

        text_size, _ = cv2.getTextSize(msg, self.font, self.font_scale, self.font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, (Cr[0], Cr[1]-text_h), (Cr[0] + text_w, Cr[1]), text_color_bg, -1)
        cv2.putText(img, msg, (Cr[0], int(Cr[1]+ self.font_scale - 1)), self.font , self.font_scale, text_color, self.font_thickness, cv2.LINE_AA, False) 
        return img
    
    def get_result_image(self, result):
        img = result.orig_img.copy()    
        coor = result.boxes.xyxy.to(torch.int).tolist()
        label_name = result.names
        classes = result.boxes.cls.tolist()
        
        if self.max_conf_take_or_not:
            is_exit = False
            max_, max_id = 0, 0
            for id in range(len(coor)):
                if max_ < result.boxes.conf[id].item():
                    max_ = result.boxes.conf[id].item()
                    max_id = id
                    is_exit = True
            if is_exit:
                label = label_name[classes[id]]
                img = self.after_detect_frame(result, max_id, label, coor[max_id], img)

        else:
            for id in range(len(coor)):
                label = label_name[classes[id]]
                img = self.after_detect_frame(result, id, label, coor[id], img)

        return img

    def detect(self, model_pt):
        # Load the YOLOv8 model
        model = YOLO(model_pt)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.win.shape[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.win.shape[1])
        self.detect_cv2_setting()
        while True:
            begin_time = time.time()
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if ret:
                track_results = model.track(frame, persist=True)
                annotated_frame = self.get_result_image(track_results[0]) #自己寫的畫圖
                cv2.imshow('frame', annotated_frame)
                # print(f'{1 / (time.time() - begin_time):.3f}')
        
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()


winsize = WindowSize(1280, 720)
detect_test = myclass(winsize)  # 不用動
detect_test.confident_setting(False, 0)    # 變數1:要不要只偵測最大信心度的label, 變數2: 信心度的最小值
folder = "C:/Users/User/Desktop/image-recognition/Project/model_pt"
model_pt = f'{folder}/8x_436_ep_2378_img(2).pt' # .py 和 .pt同一個資料夾
# model_pt = 'C:/Users/User/Desktop/image-recognition/Project/excute/1002.pt' # 絕對路徑
detect_test.detect(model_pt)
