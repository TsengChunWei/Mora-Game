import os
import time
import cv2
import numpy as np
import torch
import random
from ultralytics import YOLO

class MoraGame():
    def __init__(self) -> None:
        self.window_width = 1280
        self.window_height = (self.window_width*9)//16
        self.w_min = (self.window_width*300)//640
        self.w_max = (self.window_width*550)//640
        self.h_min = (self.window_height*200)//480
        self.h_max = (self.window_height*460)//480
        self.confident_level = 0


    def cv2_puttext_style(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        lineType = cv2.LINE_AA
        self.counting_puttext = ((10, 50), font, 2, (0, 0, 255), 2, lineType, False)
        self.playerMora_puttext = ((10, 100), font, 1, (50, 255, 50), 2, lineType, False)
        self.comMora_puttext = ((10, 150), font, 1, (50, 255, 50), 2, lineType, False)
        self.winnerMora_puttext = ((10, 200), font, 1, (0, 0, 255), 2, lineType, False)
        self.Rounds_puttext = ((self.window_width-50, 50), font, 2, (255, 20, 20), 2, lineType, False)
        self.playerScore_puttext = ((10, self.window_height-100), font, 1, (255, 20, 20), 2, lineType, False)
        self.comScore_puttext = ((10, self.window_height-50), font, 1, (255, 20, 20), 2, lineType, False)
        self.victor_puttext = ((100, 200), font, 5, (240, 110, 70), 5, lineType, False)


    def who_win(self, player, com):
        if (player=='paper' and com=='paper') or (player=='scissors' and com=='scissors') or (player=='rock' and com=='rock'):
            return 'equal'
        elif (player=='paper' and com=='scissors') or (player=='scissors' and com=='rock') or (player=='rock' and com=='paper') or (player==''):
            return 'COM'
        else:
            return 'player'
        

    def WinResult(self, victor):
        frame = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
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

        cv2.rectangle(img, (self.w_min, self.h_min), (self.w_max, self.h_max), (0, 255, 255), 2)
        for id in range(len(coor)):
            if result.boxes.conf[id].item() > maxi:
                maxi = result.boxes.conf[id].item()
                w1, h1, w2, h2 = coor[id][0], coor[id][1], coor[id][2], coor[id][3]
                maxi_idx = id
                if maxi > self.confident_level:
                    is_exist = True

        if is_exist:
            self.playerMora = label_name[classes[maxi_idx]]
            if (self.w_min < (w1+w2)//2 < self.w_max) and (self.h_min < (h1+h2)//2 < self.h_max):
                cv2.putText(img, 'player: '+self.playerMora, *self.playerMora_puttext)
                
        return img

    def timer(self, seconds):
        start_time = time.time()
        count = 0
        while count < seconds: 
            count = time.time() - start_time
            if cv2.waitKey(1) == 27:
                return True
        return False

    def main(self):
        currPath = os.path.dirname(__file__)
        model = YOLO(currPath+"/../model_pt/first_model.pt")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)

        moras = ['paper', 'scissors', 'rock']
        counting = 5
        endWindows = False

        self.cv2_puttext_style()

        rounds = 1
        Round = 5

        playerScore = 0
        comScore = 0

        while rounds <= Round:
            begin = time.time()
            while counting > 0:
                self.playerMora = ''
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                if ret:
                    track_results = model.track(frame, persist=True)
                    annotated_frame = self.get_result_image(track_results[0])
                    counting = 5-round(time.time()-begin)
                    cv2.putText(annotated_frame, str(rounds), *self.Rounds_puttext)
                    cv2.putText(annotated_frame, str(counting), *self.counting_puttext)
                    cv2.putText(annotated_frame, 'Player score: '+str(playerScore), *self.playerScore_puttext)
                    cv2.putText(annotated_frame, 'COM score: '+str(comScore), *self.comScore_puttext)
                    cv2.imshow('frame', annotated_frame)

                if cv2.waitKey(1) == 27:
                    endWindows = True
                    break
            
            if counting == 0:
                comMora = random.choice(moras)

                winner = self.who_win(self.playerMora, comMora)

                if winner == 'equal':
                    rounds -= 1
                elif winner == 'COM':
                    comScore += 1
                else:
                    playerScore += 1

                cv2.putText(annotated_frame, 'COM: '+comMora, *self.comMora_puttext)
                cv2.putText(annotated_frame, 'winner: '+winner, *self.winnerMora_puttext)
                cv2.imshow('frame', annotated_frame)
                endWindows = self.timer(5)
                counting = 5
                if comScore == (Round+1)//2:
                    self.WinResult('COM')
                    self.timer(3)
                    break
                elif playerScore == (Round+1)//2:
                    self.WinResult('player')
                    self.timer(3)
                    break                

                rounds += 1

            if endWindows:
                break

        cv2.destroyAllWindows()

moragame = MoraGame()
moragame.main()
