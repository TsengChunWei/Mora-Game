import time
import cv2
import torch
import random
from ultralytics import YOLO
from ImageInsert import extend_alpha, mask_operate
from game_v7_pack import *


class GameInitVal():
    def __init__(self) -> None:
        self.moras = ['paper', 'scissors', 'rock']
        self.tolRound = 5  # n戰(n+1)/2勝(n是奇數)
        self.break_time = 10 
        self.timeCount = 5
        self.confident_level = 0
        self.winCondition = 3

class MoraGame():
    def __init__(self, model_pt) -> None:
        self.model = YOLO(model_pt)
        self.initVal = GameInitVal()
        self.window = Window()
        self.commoraimg = ComMoraImg()
        self.imgShow = ScoreImgShow()

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
                if maxi > self.initVal.confident_level:
                    is_exist = True
        if is_exist:
            self.playerMora = label_name[classes[maxi_idx]]
            cv2.putText(img, 'player: '+self.playerMora, *self.playerMora_puttext)
        return img
    
    def Timer(self, seconds):
        start_time = time.time()
        while time.time()-start_time < seconds: 
            if cv2.waitKey(1) == 27:
                return True
        return False
    
    def currResult(self):
        comMora = random.choice(self.initVal.moras)
        overlay_img = extend_alpha(self.commoraimg.imgresult(comMora), (self.window.shape[0]//2-256, self.window.shape[1]//2-256), (self.window.shape[0], self.window.shape[1]))
        outputFrame = mask_operate(self.tempFrame, overlay_img)

        winner = self.who_win(self.playerMora, comMora)

        pre_frame = outputFrame
        cv2.putText(outputFrame, 'winner: '+winner, *self.winnerMora_puttext)
        if winner == 'equal':
            self.currRound -= 1
            start_time = time.time()
            count = time.time() - start_time
            while count < self.initVal.break_time: 
                count = time.time() - start_time
                _ = self.character.tie1 if (100*count)%50 > 25 else self.character.tie2
                overlay_img = extend_alpha(_, (0, self.window.shape[1]-512), self.window.shape)
                outputFrame = mask_operate(pre_frame, overlay_img)
                cv2.imshow('frame', outputFrame)
                if cv2.waitKey(1) == 27:
                    self.endWindows = True
                    return
        
        elif winner == 'COM':
            self.comScore += 1
            start_time = time.time()
            count = time.time() - start_time
            temp = outputFrame
            while count <= 0.8: 
                count = time.time() - start_time
                if 0.5 < count:
                    temp = self.imgShow.comWinAnime(round(100*count), pre_frame, self.imgShow.scorePosition("COM", self.comScore-1))
                _ = self.character.win1 if (100*count)%50 > 25 else self.character.win2
                overlay_img = extend_alpha(_, (0, self.window.shape[1]-512), self.window.shape)
                outputFrame = mask_operate(temp, overlay_img)
                cv2.imshow('frame', outputFrame)
                if cv2.waitKey(1) == 27:
                    self.endWindows = True
                    return
            while count < self.initVal.break_time: 
                count = time.time() - start_time
                _ = self.character.win1 if (100*count)%50 > 25 else self.character.win2
                overlay_img = extend_alpha(_, (0, self.window.shape[1]-512), self.window.shape)
                outputFrame = mask_operate(temp, overlay_img)
                cv2.imshow('frame', outputFrame)
                if cv2.waitKey(1) == 27:
                    self.endWindows = True
                    return

        else:
            self.playerScore += 1
            start_time = time.time()
            count = time.time() - start_time
            temp = outputFrame
            while count <= 0.8: 
                count = time.time() - start_time
                if 0.5 < count:
                    temp = self.imgShow.comWinAnime(round(100*count), pre_frame, self.imgShow.scorePosition("player", self.playerScore-1))
                _ = self.character.lose1 if (100*count)%50 > 25 else self.character.lose2
                overlay_img = extend_alpha(_, (0, self.window.shape[1]-512), self.window.shape)
                outputFrame = mask_operate(temp, overlay_img)
                cv2.imshow('frame', outputFrame)
                if cv2.waitKey(1) == 27:
                    self.endWindows = True
                    return
            while count < self.initVal.break_time: 
                count = time.time() - start_time
                _ = self.character.lose1 if (100*count)%50 > 25 else self.character.lose2
                overlay_img = extend_alpha(_, (0, self.window.shape[1]-512), self.window.shape)
                outputFrame = mask_operate(temp, overlay_img)
                cv2.imshow('frame', outputFrame)
                if cv2.waitKey(1) == 27:
                    self.endWindows = True
                    return

        return
    
    def mora_process(self, cap, begin):
        self.playerMora = ''
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            track_results = self.model.track(frame, persist=True)
            outputFrame = self.get_result_image(track_results[0])
            self.counting = self.initVal.timeCount-round(time.time()-begin)
            cv2.putText(outputFrame, f"{self.counting:02}", *self.counting_puttext)
            outputFrame = self.imgShow.scoreShow(outputFrame, "COM", self.comScore)
            outputFrame = self.imgShow.scoreShow(outputFrame, "player", self.playerScore)

            self.tempFrame = outputFrame

            overlay_img = extend_alpha(self.character.normal, (0, self.window.shape[1]-512), self.window.shape)
            outputFrame = mask_operate(outputFrame, overlay_img)
            cv2.imshow('frame', outputFrame)

        if cv2.waitKey(1) == 27:
            self.endWindows = True
            return 
        return


    def GamingPlay(self, video_file = None):
        self.character = ComCharacter("schoolgirl")
        self.cv2_puttext_style()

        self.endWindows = False
        self.counting = self.initVal.timeCount
        self.currRound, self.playerScore, self.comScore = 1, 0, 0

        if video_file:
            cap = cv2.VideoCapture(video_file)
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window.shape[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window.shape[1])

        self.tempFrame = []
        OP = Opening(self.character.entrance)
        OP.op_init_val(4, (25, 10), 0.4, 1.6)
        OP.animation(cap)
    
        while self.currRound <= self.initVal.tolRound:
            begin = time.time()
            while self.counting > 0:
                self.mora_process(cap, begin)
            
            if self.counting <= 0:
                self.currResult()

                self.counting = self.initVal.timeCount
                if self.comScore < self.initVal.winCondition and self.playerScore < self.initVal.winCondition:
                    self.currRound += 1
                    continue

                ED = ending()
                ED.ed_show(self.comScore, self.initVal.winCondition, 3)
                self.endWindows = True
                break

            if self.endWindows:
                break

        cv2.destroyAllWindows()


PF = PROJECT_FOLDER()
video_file = f'{PF.folder}/test_video/test_video.mp4'
moragame = MoraGame(f'{PF.folder}/model_pt/1007_x322_4.pt')
moragame.GamingPlay(video_file)
