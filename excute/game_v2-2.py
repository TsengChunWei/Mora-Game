import os
import time
import cv2
import numpy as np
import torch
import random
from ultralytics import YOLO

class WindowSize():
    def __init__(self) -> None:
        self.width = 1280
        self.height = 720

class GameInitVal():
    def __init__(self) -> None:
        # self.moras = ['paper', 'scissors', 'rock']
        self.moras = ['rock']
        self.tolRound = 5   # n戰(n+1)/2勝(n是奇數)
        self.currRound = 1
        self.playerScore = 0
        self.comScore = 0
        

class ComMoraImgInsert():
    def __init__(self) -> None:
        self.winSize = WindowSize()
        self.rock_img = cv2.imread(f'{os.path.dirname(__file__)}/../images/rock_alpha2.png', -1)
        self.paper_img = cv2.imread(f'{os.path.dirname(__file__)}/../images/paper_alpha2.png', -1)
        self.scissors_img = cv2.imread(f'{os.path.dirname(__file__)}/../images/scissors_alpha2.png', -1)

    def commoraimg(self, commora):
        if commora == 'rock':
            return self.rock_img
        elif commora == 'paper':
            return self.paper_img
        elif commora == 'scissors':
            return self.scissors_img


    def extend_alpha(self, image, interval, imgsize):
        right = imgsize[0]-interval[0]-image.shape[0]
        bottom = imgsize[1]-interval[1]-image.shape[1]
        return cv2.copyMakeBorder(image, interval[1], bottom, interval[0], right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])


    def mask_operate(self, Frame, commora):
        overlay_img = self.commoraimg(commora)
        overlay_img = cv2.resize(overlay_img, (300, 300))
        overlay_img = self.extend_alpha(overlay_img, (10, 200), (self.winSize.width, self.winSize.height))
        overlay_bgr = overlay_img[:, :, 0:3]
        overlay_img = overlay_img[:, :, 3:]
        overlay_out = cv2.bitwise_and(overlay_bgr, overlay_bgr, mask=overlay_img)
        original_out = cv2.bitwise_and(Frame, Frame, mask=cv2.bitwise_not(overlay_img))
        return cv2.add(original_out, overlay_out)


class MoraGame():
    def __init__(self, model_pt) -> None:
        self.model = YOLO(f"{os.path.dirname(__file__)}/../model_pt/{model_pt}")
        self.initVal = GameInitVal()
        self.winSize = WindowSize()
        self.w_min = (self.winSize.width*200)//640
        self.w_max = (self.winSize.width*550)//640
        self.h_min = (self.winSize.height*100)//480
        self.h_max = (self.winSize.height*400)//480
        self.confident_level = 0

    def cv2_puttext_style(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        lineType = cv2.LINE_AA
        self.counting_puttext = ((10, 50), font, 2, (0, 0, 255), 2, lineType, False)
        self.playerMora_puttext = ((10, 150), font, 1, (50, 255, 50), 2, lineType, False)
        self.comMora_puttext = ((10, 200), font, 1, (50, 255, 50), 2, lineType, False)
        self.winnerMora_puttext = ((10, 100), font, 1, (0, 0, 255), 2, lineType, False)
        self.Rounds_puttext = ((self.winSize.width-50, 50), font, 2, (255, 20, 20), 2, lineType, False)
        self.playerScore_puttext = ((10, self.winSize.height-100), font, 1, (255, 20, 20), 2, lineType, False)
        self.comScore_puttext = ((10, self.winSize.height-50), font, 1, (255, 20, 20), 2, lineType, False)
        self.victor_puttext = ((100, 200), font, 5, (240, 110, 70), 5, lineType, False)

    def who_win(self, player, com):
        if (player=='paper' and com=='paper') or (player=='scissors' and com=='scissors') or (player=='rock' and com=='rock'):
            return 'equal'
        elif (player=='paper' and com=='scissors') or (player=='scissors' and com=='rock') or (player=='rock' and com=='paper') or (player==''):
            return 'COM'
        else:
            return 'player'

    def WinResult(self, victor):
        frame = np.zeros((self.winSize.height, self.winSize.width, 3), dtype=np.uint8)
        frame[:] = (105, 155, 255)
        cv2.putText(frame, victor, *self.victor_puttext)
        cv2.imshow('frame', frame)

    def get_result_image(self, result):
        img = result.orig_img.copy()
        coor = result.boxes.xyxy.to(torch.int).tolist()
        label_name = result.names
        classes = result.boxes.cls.tolist()

        maxi, maxi_idx = 0, 0
        
        is_exist = False

        # cv2.rectangle(img, (self.w_min, self.h_min), (self.w_max, self.h_max), (0, 255, 255), 2)
        for id in range(len(coor)):
            if result.boxes.conf[id].item() > maxi:
                maxi = result.boxes.conf[id].item()
                # w1, h1, w2, h2 = coor[id][0], coor[id][1], coor[id][2], coor[id][3]
                maxi_idx = id
                if maxi > self.confident_level:
                    is_exist = True

        if is_exist:
            self.playerMora = label_name[classes[maxi_idx]]
            # if (self.w_min < (w1+w2)//2 < self.w_max) and (self.h_min < (h1+h2)//2 < self.h_max):
            cv2.putText(img, 'player: '+self.playerMora, *self.playerMora_puttext)
                
        return img

    def Timer(self, seconds):
        start_time = time.time()
        count = 0
        while count < seconds: 
            count = time.time() - start_time
            if cv2.waitKey(1) == 27:
                return True
        return False

    def play(self):
        commoraimginsert = ComMoraImgInsert()
        timeCount = 5
        counting = timeCount
        endWindows = False

        self.cv2_puttext_style()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.winSize.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.winSize.height)

        while self.initVal.currRound <= self.initVal.tolRound:
            begin = time.time()
            while counting > 0:
                self.playerMora = ''
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                # cv2.imwrite(os.path.dirname(__file__)+'/frame.jpg', frame)
                if ret:
                    track_results = self.model.track(frame, persist=True)
                    annotated_frame = self.get_result_image(track_results[0])
                    counting = timeCount-round(time.time()-begin)
                    cv2.putText(annotated_frame, str(self.initVal.currRound), *self.Rounds_puttext)
                    cv2.putText(annotated_frame, str(counting), *self.counting_puttext)
                    cv2.putText(annotated_frame, 'Player score: '+str(self.initVal.playerScore), *self.playerScore_puttext)
                    cv2.putText(annotated_frame, 'COM score: '+str(self.initVal.comScore), *self.comScore_puttext)
                    cv2.imshow('frame', annotated_frame)

                if cv2.waitKey(1) == 27:
                    endWindows = True
                    break
            
            if counting == 0:
                comMora = random.choice(self.initVal.moras)
                annotated_frame = commoraimginsert.mask_operate(annotated_frame, comMora)

                winner = self.who_win(self.playerMora, comMora)

                if winner == 'equal':
                    self.initVal.currRound -= 1
                elif winner == 'COM':
                    self.initVal.comScore += 1
                else:
                    self.initVal.playerScore += 1

                # cv2.putText(annotated_frame, 'COM: '+comMora, *self.comMora_puttext)
                cv2.putText(annotated_frame, 'winner: '+winner, *self.winnerMora_puttext)
                cv2.imshow('frame', annotated_frame)
                # cv2.imwrite('C:/Users/User/Desktop/frame.png', annotated_frame)
                endWindows = self.Timer(3)
                counting = timeCount
                if self.initVal.comScore == (self.initVal.tolRound+1)//2:
                    self.WinResult('COM')
                    self.Timer(3)
                    break
                elif self.initVal.playerScore == (self.initVal.tolRound+1)//2:
                    self.WinResult('player')
                    self.Timer(3)
                    break                

                self.initVal.currRound += 1

            if endWindows:
                break

        cv2.destroyAllWindows()

moragame = MoraGame("1002.pt")
moragame.play()
