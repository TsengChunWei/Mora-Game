import os
import time
import cv2
from ImageInsert import extend_alpha, mask_operate

class PROJECT_FOLDER():
    def __init__(self) -> None:
        self.folder = os.path.dirname(__file__)+"/.."

class Window:
    def __init__(self):
        self.shape = (1280, 720)

class ComCharacter():
    def __init__(self, person) -> None:
        PF = PROJECT_FOLDER()
        self.normal = cv2.imread(f'{PF.folder}/images/character/{person}/normal.png', -1)
        self.win = cv2.imread(f'{PF.folder}/images/character/{person}/win.png', -1)
        self.lose = cv2.imread(f'{PF.folder}/images/character/{person}/lose.png', -1)
        self.tie = cv2.imread(f'{PF.folder}/images/character/{person}/tie.png', -1)
        self.victory = cv2.imread(f'{PF.folder}/images/character/{person}/victory.png', -1)
        self.defeat = cv2.imread(f'{PF.folder}/images/character/{person}/defeat.png', -1)
        self.entrance = cv2.imread(f'{PF.folder}/images/character/{person}/entrance.png', -1)

class ComMoraImg():
    def __init__(self) -> None:
        PF = PROJECT_FOLDER()
        self.rock_img = cv2.imread(f'{PF.folder}/images/base/rock_heart.png', -1)
        self.paper_img = cv2.imread(f'{PF.folder}/images/base/paper_heart.png', -1)
        self.scissors_img = cv2.imread(f'{PF.folder}/images/base/scissors_heart.png', -1)
    def imgresult(self, commora):
        if commora == 'rock':
            return self.rock_img
        elif commora == 'paper':
            return self.paper_img
        elif commora == 'scissors':
            return self.scissors_img

class ScoreImgShow():
    def __init__(self) -> None:
        self.window = Window()
        self.star_mask = cv2.imread(f"{PROJECT_FOLDER().folder}/images/base/star2.png", -1)
        self.star_size = (self.star_mask.shape[1], self.star_mask.shape[0])
    
    def scoreShow(self, outputFrame, p, score):
        for i in range(score):
            overlay_img = extend_alpha(self.star_mask, self.scorePosition(p, i), self.window.shape)
            outputFrame = mask_operate(outputFrame, overlay_img)
        return outputFrame
    
    def scorePosition(self, p, score):
        if p == "COM":
            return ((550-10*(score))-(score+1)*self.star_size[0], self.window.shape[1]//10-self.star_size[1]//2)
        elif p == "player":
            return ((730+10*(score))+(score)*self.star_size[0], self.window.shape[1]//10-self.star_size[1]//2)
        return (0, 0)

    def comWinAnime(self, count_i, outputFrame, pre_frame, point):
        if 50 < count_i:
            resize_mask = cv2.resize(self.star_mask, (176-count_i, 176-count_i))
            overlay_img = extend_alpha(resize_mask, point, self.window.shape)
            outputFrame = mask_operate(outputFrame, overlay_img)
        cv2.imshow('frame', outputFrame)            
        outputFrame = pre_frame

class Opening():
    def __init__(self, charac_ent) -> None:
        ScFolder = PROJECT_FOLDER().folder+"/images/scene"
        self.window = Window()
        self.charac_ent = charac_ent
        self.ready_mask = cv2.imread(f"{ScFolder}/op_ready_go/ready.png", -1)
        self.go_mask    = cv2.imread(f"{ScFolder}/op_ready_go/go.png", -1)
        self.boxer_down = cv2.imread(f"{ScFolder}/op_cut1/boxer_down.png")
        self.boxer_up   = cv2.imread(f"{ScFolder}/op_cut1/boxer_up.png")
        self.boxer_hit1 = cv2.imread(f"{ScFolder}/op_cut2/boxer_hit1.png")
        self.boxer_hit2 = cv2.imread(f"{ScFolder}/op_cut2/boxer_hit2.png")
        self.boxer_hit3 = cv2.imread(f"{ScFolder}/op_cut2/boxer_hit3.png")
        self.start_bg1  = cv2.imread(f"{ScFolder}/op_cut3/start_bg1.png")
        self.start_bg2  = cv2.imread(f"{ScFolder}/op_cut3/start_bg2.png")
        self.hit_time = 1
        self.jump_times = 1
        self.Jump = (10, 10)
        self.start_flase_second = 2
    
    def op_init_val(self, jump_times, Jump, hit_time, start_flase_second):
        self.hit_time = hit_time
        self.jump_times = jump_times
        self.Jump = Jump
        self.start_flase_second = start_flase_second
        
    def animation(self, cap):
        i = 0
        for _ in range(self.jump_times*(self.Jump[0]+self.Jump[1])):
            if i%(self.Jump[0]+self.Jump[1]) < self.Jump[1]:
                cv2.imshow('frame', self.boxer_up)
                if i == (self.Jump[0]+self.Jump[1]):
                    i = -1
            else:
                cv2.imshow('frame', self.boxer_down)
            if cv2.waitKey(1) == 27:
                return
            i += 1
        
        for hit in [self.boxer_hit1, self.boxer_hit2, self.boxer_hit3]:
            t = time.time()
            while time.time()-t < self.hit_time:
                cv2.imshow('frame', hit)
                if cv2.waitKey(1) == 27:
                    return
        
        i = 0
        START = False
        while not START:
            t = time.time()
            while time.time()-t < self.start_flase_second/2 and not START:
                cv2.imshow('frame', self.start_bg1)
                if cv2.waitKey(1) == 32:
                    START = True
                    break
            t = time.time()
            while time.time()-t < self.start_flase_second/2 and not START:
                cv2.imshow('frame', self.start_bg2)
                if cv2.waitKey(1) == 32:
                    START = True
                    break

        minsize = 0
        fq = self.window.shape[1]-minsize    # 720
        T = 20
        for i in range(fq//T):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                overlay_img = extend_alpha(self.charac_ent, (0, self.window.shape[1]-512), self.window.shape)
                frame = mask_operate(frame, overlay_img)
                newsize = (minsize+fq-T*i, minsize+fq-T*i)
                resize_mask = cv2.resize(self.ready_mask, newsize)
                overlay_img = extend_alpha(resize_mask, (self.window.shape[0]//2-newsize[0]//2, self.window.shape[1]//2-newsize[1]//2), self.window.shape)
                frame = mask_operate(frame, overlay_img)
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) == 27:
                return
        T = 35
        minsize = 20
        for i in range(fq//T+1):
            ret, frame_temp = cap.read()
            if ret:
                frame = cv2.flip(frame_temp, 1)
                overlay_img = extend_alpha(self.charac_ent, (0, self.window.shape[1]-512), self.window.shape)
                frame = mask_operate(frame, overlay_img)
                newsize = (minsize+T*i, minsize+T*i)
                resize_mask = cv2.resize(self.go_mask, newsize)
                overlay_img = extend_alpha(resize_mask, (self.window.shape[0]//2-newsize[0]//2, self.window.shape[1]//2-newsize[1]//2), self.window.shape)
                frame = mask_operate(frame, overlay_img)
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) == 27:
                return
        return

class ending():
    def __init__(self) -> None:
        ScFolder = PROJECT_FOLDER().folder+"/images/scene"
        self.vic = cv2.imread(f"{ScFolder}/ending/ending_win.png")
        self.dft = cv2.imread(f"{ScFolder}/ending/ending_lose.png")

    def ed_show(self, comScore, winCondition, show_time):
        if comScore == winCondition:
            t = time.time()
            while time.time()-t < show_time:
                cv2.imshow('frame', self.dft)
                if cv2.waitKey(1) == 27:
                    return

        else:
            t = time.time()
            while time.time()-t < show_time:
                cv2.imshow('frame', self.vic)
                if cv2.waitKey(1) == 27:
                    return

        return