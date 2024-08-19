import os

import cv2
from DetectTest_img import WindowSize, myclass

mydict = {"rock": 0, "paper": 1, "scissors": 2}

folder = "C:/Users/User/Desktop/image-recognition/Project/model_pt/1005_x478_4.pt"
model_pt = f'{folder}/1002.pt' # 絕對路徑

winsize = WindowSize(1280, 720)
detect_test = myclass(winsize, model_pt)  # 不用動

folder = "C:/Users/User/Desktop/image-recognition/Project/image-captures/_mix_"
for file in os.listdir(folder):
    detect_test.confident_setting(False, 0)    # 變數1:要不要只偵測最大信心度的label, 變數2: 信心度的最小值
    detect_test.detect(model_pt, f'{folder}/{file}', False)
    arr = detect_test.labelImg_txt(detect_test.position_arr)
    # print(file, detect_test.label_, arr)
    with open(f"{folder}/{os.path.splitext(file)[0]}.txt", "w") as file:
        # Write the string followed by the array elements
        file.write(f"{mydict[detect_test.label_]} " + " ".join(map(str, arr)) + "/n")
