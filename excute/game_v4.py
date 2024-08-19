import time
import cv2
import numpy as np
import torch
import random
from ultralytics import YOLO
from game_v4_pack import *


class MoraGame():
    def __init__(self, model_pt) -> None:
        self.model = YOLO(model_pt)
        self.initVal = GameInitVal()
        self.window = Window()
        self.confident_level = 0

    def cv2_puttext_style(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        lineType = cv2.LINE_AA
        self.counting_puttext = ((560, 112), cv2.FONT_HERSHEY_TRIPLEX, 4, (0, 0, 255), 7, lineType, False)
        self.playerMora_puttext = ((730, 150), font, 1, (50, 255, 50), 2, lineType, False)
        self.comMora_puttext = ((10, 200), font, 1, (50, 255, 50), 2, lineType, False)
        self.winnerMora_puttext = ((500, 700), font, 1, (0, 0, 255), 2, lineType, False)
        self.playerScore_puttext = ((730, 80), font, 1, (255, 20, 20), 2, lineType, False)
        self.comScore_puttext = ((320, 80), font, 1, (255, 20, 20), 2, lineType, False)

    def who_win(self, player, com):
        if player == com:
            return 'equal'
        elif (player=='paper' and com=='scissors') or (player=='scissors' and com=='rock') or (player=='rock' and com=='paper') or (player==''):
            return 'COM'
        else:
            return 'player'

    def WinResult(self, victor, imgInsert, img):
        frame = np.zeros((self.window.shape[1], self.window.shape[0], 3), dtype=np.uint8)
        frame[:] = (105, 155, 255)
        self.victor_puttext = ((100, 200), cv2.FONT_HERSHEY_DUPLEX, 5, (240, 110, 70), 5, cv2.LINE_AA, False)
        overlay_img = imgInsert.extend_alpha(img, (0, self.window.shape[1]-512), self.window.shape)
        frame = imgInsert.mask_operate(frame, overlay_img)
        cv2.putText(frame, victor, *self.victor_puttext)
        cv2.imshow('frame', frame)

    def get_result_image(self, result):
        img = result.orig_img.copy()
        coor = result.boxes.xyxy.to(torch.int).tolist()
        label_name = result.names
        classes = result.boxes.cls.tolist()

        maxi, maxi_idx = 0, 0
        
        is_exist = False

        for id in range(len(coor)):
            if result.boxes.conf[id].item() > maxi:
                maxi = result.boxes.conf[id].item()
                maxi_idx = id
                if maxi > self.confident_level:
                    is_exist = True

        if is_exist:
            self.playerMora = label_name[classes[maxi_idx]]
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

    def GamingPlay(self, video_file = None):
        imgInsert = ImageInsert()
        commoraimg = ComMoraImg()
        character = ComCharacter("schoolgirl")
        timeCount = 5
        counting = timeCount
        endWindows = False

        imgShow = ScoreImgShow()
        OP = Opening(character.entrance)

        currRound = 1
        playerScore = 0
        comScore = 0

        self.cv2_puttext_style()


        if video_file:
            cap = cv2.VideoCapture(video_file)
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window.shape[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window.shape[1])
        
        # frame1 = cv2.imread("C:/Users/User/Desktop/image-recognition/Project/images/scene/opening_cut1.png")
        # frame2 = cv2.imread("C:/Users/User/Desktop/image-recognition/Project/images/scene/opening_cut2.png")
        # frame3 = cv2.imread("C:/Users/User/Desktop/image-recognition/Project/images/scene/opening_cut3.png")
        # frame1 = cv2.resize(frame1, (1280, 720))
        # frame2 = cv2.resize(frame2, (1280, 720))
        # frame3 = cv2.resize(frame3, (1280, 720))
        # cv2.imshow('frame', frame1)
        # self.Timer(2)
        # cv2.imshow('frame', frame2)
        # self.Timer(2)
        # cv2.imshow('frame', frame3)
        # cv2.waitKey(0)

        tempFrame = []
        OP.animation(cap)
    
        while currRound <= self.initVal.tolRound:
            begin = time.time()
            while counting > 0:
                self.playerMora = ''
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                # cv2.imwrite(os.path.dirname(__file__)+'/frame.jpg', frame)
                if ret:
                    track_results = self.model.track(frame, persist=True)
                    outputFrame = self.get_result_image(track_results[0])
                    counting = timeCount-round(time.time()-begin)
                    cv2.putText(outputFrame, f"{counting:02}", *self.counting_puttext)
                    outputFrame = imgShow.scoreShow(outputFrame, "COM", comScore)
                    outputFrame = imgShow.scoreShow(outputFrame, "player", playerScore)

                    tempFrame = outputFrame

                    overlay_img = imgInsert.extend_alpha(character.normal, (0, self.window.shape[1]-512), self.window.shape)
                    outputFrame = imgInsert.mask_operate(outputFrame, overlay_img)
                    cv2.imshow('frame', outputFrame)

                if cv2.waitKey(1) == 27:
                    endWindows = True
                    break
            
            if counting == 0:
                comMora = random.choice(self.initVal.moras)
                overlay_img = imgInsert.extend_alpha(commoraimg.imgresult(comMora), (self.window.shape[0]//2-256, self.window.shape[1]//2-256), (self.window.shape[0], self.window.shape[1]))
                outputFrame = imgInsert.mask_operate(tempFrame, overlay_img)

                winner = self.who_win(self.playerMora, comMora)

                if winner == 'equal':
                    currRound -= 1
                    overlay_img = imgInsert.extend_alpha(character.tie, (0, self.window.shape[1]-512), self.window.shape)
                    outputFrame = imgInsert.mask_operate(outputFrame, overlay_img)
                elif winner == 'COM':
                    comScore += 1
                    overlay_img = imgInsert.extend_alpha(character.win, (0, self.window.shape[1]-512), self.window.shape)
                    outputFrame = imgInsert.mask_operate(outputFrame, overlay_img)
                else:
                    playerScore += 1
                    overlay_img = imgInsert.extend_alpha(character.lose, (0, self.window.shape[1]-512), self.window.shape)
                    outputFrame = imgInsert.mask_operate(outputFrame, overlay_img)

                cv2.putText(outputFrame, 'winner: '+winner, *self.winnerMora_puttext)
                temp = outputFrame
                start_time = time.time()
                count = 0
                if winner == "equal":
                    while count < self.initVal.break_time: 
                        count = time.time() - start_time
                        cv2.imshow('frame', outputFrame)
                        if cv2.waitKey(1) == 27:
                            endWindows = True
                        # outputFrame = temp
                elif winner == "COM":
                    while count <= 0.8: 
                        count = time.time() - start_time
                        imgShow.comWinAnime(round(100*count), outputFrame, temp, imgShow.scorePosition("COM", comScore-1))
                        if cv2.waitKey(1) == 27:
                            endWindows = True
                    self.Timer(self.initVal.break_time-0.8)
                else:
                    while count <= 0.8: 
                        count = time.time() - start_time
                        imgShow.comWinAnime(round(100*count), outputFrame, temp, imgShow.scorePosition("player", playerScore-1))
                        if cv2.waitKey(1) == 27:
                            endWindows = True
                    self.Timer(self.initVal.break_time-0.8)

                counting = timeCount
                if comScore == (self.initVal.tolRound+1)//2:
                    endWindows = self.Timer(3)
                    self.WinResult('COM', imgInsert, character.victory)
                elif playerScore == (self.initVal.tolRound+1)//2:
                    endWindows = self.Timer(3)
                    self.WinResult('player', imgInsert, character.defeat)
                else:
                    currRound += 1
                    continue         
                endWindows = self.Timer(3)
                break

            if endWindows:
                break

        cv2.destroyAllWindows()


PF = PROJECT_FOLDER()
video_file = f'{PF.folder}/test_video/test_video.mp4'
moragame = MoraGame(f'{PF.folder}/model_pt/1007_x322_4.pt')
moragame.GamingPlay(video_file)
