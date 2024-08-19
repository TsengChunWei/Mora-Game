import os
import time
import cv2
import numpy as np
import torch
import random
from ultralytics import YOLO


class PROJECT_FOLDER():
    def __init__(self) -> None:
        self.folder = os.path.dirname(__file__) + "/.."

class Window:
    def __init__(self):
        self.shape = (1280, 720)

class GameInitVal():
    def __init__(self) -> None:
        self.moras = ['paper', 'scissors', 'rock']
        self.tolRound = 5  # n戰(n+1)/2勝(n是奇數)
        self.break_time = 3 

class ComCharacter():
    def __init__(self, person) -> None:
        PF = PROJECT_FOLDER()
        self.normal = cv2.imread(f'{PF.folder}/images/character/{person}/normal.png', -1)
        self.win = cv2.imread(f'{PF.folder}/images/character/{person}/win.png', -1)
        self.lose = cv2.imread(f'{PF.folder}/images/character/{person}/lose.png', -1)
        self.tie = cv2.imread(f'{PF.folder}/images/character/{person}/tie.png', -1)
        self.victory = cv2.imread(f'{PF.folder}/images/character/{person}/victory.png', -1)
        self.defeat = cv2.imread(f'{PF.folder}/images/character/{person}/defeat.png', -1)

class ComMoraImg():
    def __init__(self) -> None:
        PF = PROJECT_FOLDER()
        self.rock_img = cv2.imread(f'{PF.folder}/images/base/rock.png', -1)
        self.paper_img = cv2.imread(f'{PF.folder}/images/base/paper.png', -1)
        self.scissors_img = cv2.imread(f'{PF.folder}/images/base/scissors.png', -1)
    def imgresult(self, commora):
        if commora == 'rock':
            return self.rock_img
        elif commora == 'paper':
            return self.paper_img
        elif commora == 'scissors':
            return self.scissors_img

class ImageInsert():
    def __init__(self) -> None:
        pass

    def extend_alpha(self, image, interval, imgsize):
        right = imgsize[0]-interval[0]-image.shape[0]
        bottom = imgsize[1]-interval[1]-image.shape[1]
        return cv2.copyMakeBorder(image, interval[1], bottom, interval[0], right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

    def mask_operate(self, Frame, overlay_img):
        overlay_bgr = overlay_img[:, :, 0:3]
        overlay_img = overlay_img[:, :, 3:]
        overlay_out = cv2.bitwise_and(overlay_bgr, overlay_bgr, mask=overlay_img)
        original_out = cv2.bitwise_and(Frame, Frame, mask=cv2.bitwise_not(overlay_img))
        return cv2.add(original_out, overlay_out)

class ScoreImage():
    def __init__(self) -> None:
        self.window = Window()
        PF = PROJECT_FOLDER()
        self.star_mask = cv2.imread(f"{PF.folder}/images/base/star2.png", -1)
        self.mask_size = (self.star_mask.shape[1], self.star_mask.shape[0])
        self.II = ImageInsert()

    def comscore(self, gc, score):
        overlay_img = self.II.extend_alpha(self.star_mask, ((550-10*score)-(score+1)*self.mask_size[0], self.window.shape[1]//10-self.mask_size[1]//2), self.window.shape)
        return self.II.mask_operate(gc, overlay_img)
    
    def playerscore(self, gc, score):
        overlay_img = self.II.extend_alpha(self.star_mask, ((730+10*score)+score*self.mask_size[0], self.window.shape[1]//10-self.mask_size[1]//2), self.window.shape)
        return self.II.mask_operate(gc, overlay_img)

class TimeOperator():
    def __init__(self) -> None:
        pass
    def Timer(self, seconds):
        start_time = time.time()
        count = 0
        while count < seconds: 
            count = time.time() - start_time
            if cv2.waitKey(1) == 27:
                return True
        return False

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
        self.Rounds_puttext = ((self.window.shape[0]-50, 50), font, 2, (255, 20, 20), 2, lineType, False)
        self.playerScore_puttext = ((730, 80), font, 1, (255, 20, 20), 2, lineType, False)
        self.comScore_puttext = ((320, 80), font, 1, (255, 20, 20), 2, lineType, False)

    def who_win(self, player, com):
        if (player=='paper' and com=='paper') or (player=='scissors' and com=='scissors') or (player=='rock' and com=='rock'):
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

    def GamingPlay(self, video_file = None):
        imgInsert = ImageInsert()
        commoraimg = ComMoraImg()
        character = ComCharacter("schoolgirl")
        timeCount = 5
        counting = timeCount
        endWindows = False

        star_mask = cv2.imread(f"{PROJECT_FOLDER().folder}/images/base/star2.png", -1)
        star_size = (star_mask.shape[1], star_mask.shape[0])
        star_anime_time = star_size[0]+100*self.initVal.break_time

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

        tempFrame = []
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
                    cv2.putText(outputFrame, str(currRound), *self.Rounds_puttext)
                    cv2.putText(outputFrame, f"{counting:02}", *self.counting_puttext)

                    for i in range(comScore):
                        overlay_img = imgInsert.extend_alpha(star_mask, ((550-10*i)-(i+1)*star_size[0], self.window.shape[1]//10-star_size[1]//2), self.window.shape)
                        outputFrame = imgInsert.mask_operate(outputFrame, overlay_img)
                    for i in range(playerScore):
                        overlay_img = imgInsert.extend_alpha(star_mask, ((730+10*i)-i*star_size[0], self.window.shape[1]//10-star_size[1]//2), self.window.shape)
                        outputFrame = imgInsert.mask_operate(outputFrame, overlay_img)
                    tempFrame = outputFrame

                    overlay_img = imgInsert.extend_alpha(character.normal, (0, self.window.shape[1]-512), self.window.shape)
                    outputFrame = imgInsert.mask_operate(outputFrame, overlay_img)
                    cv2.imshow('frame', outputFrame)

                if cv2.waitKey(1) == 27:
                    endWindows = True
                    break
            
            if counting == 0:
                comMora = random.choice(self.initVal.moras)
                overlay_img = imgInsert.extend_alpha(commoraimg.imgresult(comMora), (128, 0), (self.window.shape[0], self.window.shape[1]))
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
                        outputFrame = temp
                elif winner == "COM":
                    while count <= 0.8: 
                        count = time.time() - start_time
                        count_i = round(100*count)
                        # print(count_i)
                        if 50 < count_i:
                            resize_mask = cv2.resize(star_mask, (176-count_i, 176-count_i))
                            overlay_img = imgInsert.extend_alpha(resize_mask, ((550-10*(comScore-1))-(comScore)*star_size[0], self.window.shape[1]//10-star_size[1]//2), self.window.shape)
                            outputFrame = imgInsert.mask_operate(outputFrame, overlay_img)
                        cv2.imshow('frame', outputFrame)
                        
                        if cv2.waitKey(1) == 27:
                            endWindows = True
                        outputFrame = temp
                    while count < self.initVal.break_time: 
                        count = time.time() - start_time
                        if cv2.waitKey(1) == 27:
                            endWindows = True

                else:
                    while count < self.initVal.break_time: 
                        count = time.time() - start_time
                        count_i = round(100*count)
                        # print(count_i)
                        resize_mask = cv2.resize(star_mask, (star_anime_time-count_i, star_anime_time-count_i))
                        overlay_img = imgInsert.extend_alpha(resize_mask, ((730+10*(playerScore-1))+(playerScore-1)*star_size[0], self.window.shape[1]//10-star_size[1]//2), self.window.shape)
                        outputFrame = imgInsert.mask_operate(outputFrame, overlay_img)
                        cv2.imshow('frame', outputFrame)
                        if cv2.waitKey(1) == 27:
                            endWindows = True
                        outputFrame = temp

                counting = timeCount
                if comScore == (self.initVal.tolRound+1)//2:
                    while count < 3: 
                        count = time.time() - start_time
                        if cv2.waitKey(1) == 27:
                            endWindows = True
                    self.WinResult('COM', imgInsert, character.victory)
                elif playerScore == (self.initVal.tolRound+1)//2:
                    while count < 3: 
                        count = time.time() - start_time
                        if cv2.waitKey(1) == 27:
                            endWindows = True
                    self.WinResult('player', imgInsert, character.defeat)
                else:
                    currRound += 1
                    continue         
                start_time = time.time()
                count = 0
                while count < 3: 
                    count = time.time() - start_time
                    if cv2.waitKey(1) == 27:
                        endWindows = True
                break

            if endWindows:
                break

        cv2.destroyAllWindows()


PF = PROJECT_FOLDER()
video_file = f'{PF.folder}/test_video/test_video.mp4'
moragame = MoraGame(f'{PF.folder}/model_pt/1002.pt')
moragame.GamingPlay()
