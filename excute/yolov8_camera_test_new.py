import time
import cv2
import torch
from ultralytics import YOLO


def get_result_image(result):
    img = result.orig_img.copy()
    # cv2.imwrite('result.jpg', img)
    
    coor = result.boxes.xyxy.to(torch.int).tolist()
    label_name = result.names
    classes = result.boxes.cls.tolist()

    font = cv2.FONT_HERSHEY_DUPLEX

    maxi, maxi_idx = 0, 0
    w_min, w_max, h_min, h_max = 300, 550, 200, 460
    is_exist = False

    cv2.rectangle(img, (w_min, h_min), (w_max, h_max), (0, 255, 255), 2)
    for id in range(len(coor)):
        Cr = coor[id]
        cv2.rectangle(img, (Cr[0], Cr[1]), (Cr[2], Cr[3]), (0, 255, 0), 2)
        if result.boxes.conf[id].item() > maxi:
            maxi = result.boxes.conf[id].item()
            w1, h1, w2, h2 = coor[id][0], coor[id][1], coor[id][2], coor[id][3]
            maxi_idx = id
            is_exist = True

    if is_exist:
        label = label_name[classes[maxi_idx]]
        # if (w_min < (w1+w2)//2 < w_max) and (h_min < (h1+h2)//2 < h_max):
        cv2.putText(img, label, (50, 50), font , 1, (0, 0, 255), 1, cv2.LINE_AA, False)
    return img


model = YOLO("C:/Users/User/Desktop/image-recognition/Project/excute/200_ep_700_imga.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

n = 0
begin = time.time()
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret:
        counting = 5-round(time.time()-begin)%6
        track_results = model.track(frame, persist=True)
        annotated_frame = get_result_image(track_results[0])
        cv2.imshow('frame', annotated_frame)
        cv2.putText(frame, str(counting), (50, 50), cv2.FONT_HERSHEY_DUPLEX , 2, (0, 0, 255), 2, cv2.LINE_AA, False)
 
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break


cv2.destroyAllWindows()
