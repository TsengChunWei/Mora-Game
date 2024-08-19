from time import time
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT,flip, imshow, waitKey, destroyAllWindows
from cv2 import putText, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_TRIPLEX, LINE_AA
from torch import int
from random import choice
from ultralytics import YOLO
from ImageInsert import extend_alpha, mask_operate
from ProjectPackage import *


class GameInitVal():
    def __init__(self) -> None:
        self.moras = ['paper', 'scissors', 'rock']
        self.tolRound = 5  # n戰(n+1)/2勝(n是奇數)
        self.break_time = 3 
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
        self.animng = AnimeManage()

    def cv2_puttext_style(self):
        font = FONT_HERSHEY_DUPLEX
        lineType = LINE_AA
        self.counting_puttext = ((560, 112), FONT_HERSHEY_TRIPLEX, 4, (0, 0, 255), 7, lineType, False)
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
        coor = result.boxes.xyxy.to(int).tolist()
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
            putText(img, 'player: '+self.playerMora, *self.playerMora_puttext)
        return img
    
    def currResult(self):
        comMora = choice(self.initVal.moras)
        overlay_img = extend_alpha(self.commoraimg.imgresult(comMora), (self.window.shape[0]//2-256, self.window.shape[1]//2-256), (self.window.shape[0], self.window.shape[1]))
        outputFrame = mask_operate(self.tempFrame, overlay_img)

        winner = self.who_win(self.playerMora, comMora)

        pre_frame = outputFrame
        temp = outputFrame
        count = 0
        putText(outputFrame, 'winner: '+winner, *self.winnerMora_puttext)
        if winner == 'equal':
            self.currRound -= 1
            start_time = time()
            while count < self.initVal.break_time: 
                count = time() - start_time
                outputFrame = self.animng.character_shake(count, temp, self.character.tie1, self.character.tie2)
                imshow('frame', outputFrame)
                if waitKey(1) == 27:
                    self.endWindows = True
                    return
            return
        
        if winner == 'COM':
            self.comScore += 1
            temp_score = self.comScore
            temp_shake1 = self.character.win1
            temp_shake2 = self.character.win2
        else:
            self.playerScore += 1
            temp_score = self.playerScore
            temp_shake1 = self.character.lose1
            temp_shake2 = self.character.lose2

        start_time = time()
        while count <= 0.8: 
            count = time() - start_time
            if 0.5 < count:
                temp = self.animng.star_appear(count, pre_frame, self.imgShow.star_mask, self.imgShow.scorePosition(winner, temp_score-1))
            outputFrame = self.animng.character_shake(count, temp, temp_shake1, temp_shake2)
            imshow('frame', outputFrame)
            if waitKey(1) == 27:
                self.endWindows = True
                return
        while count < self.initVal.break_time: 
            count = time() - start_time
            outputFrame = self.animng.character_shake(count, temp, temp_shake1, temp_shake2)
            imshow('frame', outputFrame)
            if waitKey(1) == 27:
                self.endWindows = True
                return
        return
    
    def mora_process(self, cap, begin):
        self.playerMora = ''
        ret, frame = cap.read()
        frame = flip(frame, 1)
        if ret:
            track_results = self.model.track(frame, persist=True)
            outputFrame = self.get_result_image(track_results[0])
            self.counting = self.initVal.timeCount-round(time()-begin)
            putText(outputFrame, f"{self.counting:02}", *self.counting_puttext)
            outputFrame = self.imgShow.scoreShow(outputFrame, "COM", self.comScore)
            outputFrame = self.imgShow.scoreShow(outputFrame, "player", self.playerScore)

            self.tempFrame = outputFrame

            overlay_img = extend_alpha(self.character.normal, (0, self.window.shape[1]-512), self.window.shape)
            outputFrame = mask_operate(outputFrame, overlay_img)
            imshow('frame', outputFrame)

        if waitKey(1) == 27:
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
            cap = VideoCapture(video_file)
        else:
            cap = VideoCapture(0)
            cap.set(CAP_PROP_FRAME_WIDTH, self.window.shape[0])
            cap.set(CAP_PROP_FRAME_HEIGHT, self.window.shape[1])

        self.tempFrame = []
        OP = Opening(self.character.entrance)
        OP.op_init_val(4, (25, 10), 0.2, 1.6)
        OP.animation(cap)
    
        while self.currRound <= self.initVal.tolRound:
            begin = time()
            while self.counting > 0:
                self.mora_process(cap, begin)
            
            if self.counting == 0:
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

        destroyAllWindows()
