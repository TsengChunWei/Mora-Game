from os.path import dirname
from time import time
from cv2 import flip, imread, resize, imshow, waitKey
from ImageInsert import extend_alpha, mask_operate

class PROJECT_FOLDER():
    def __init__(self) -> None:
        self.folder = dirname(__file__)

class Window:
    def __init__(self):
        self.shape = (1280, 720)

class ComCharacter():
    def __init__(self, person) -> None:
        PF = PROJECT_FOLDER()
        self.entrance = imread(f'{PF.folder}/images/character/{person}/entrance.png', -1)
        self.normal = imread(f'{PF.folder}/images/character/{person}/normal.png', -1)
        self.win1 = imread(f'{PF.folder}/images/character/{person}/action/win_act_1.png', -1)
        self.win2 = imread(f'{PF.folder}/images/character/{person}/action/win_act_2.png', -1)
        self.lose1 = imread(f'{PF.folder}/images/character/{person}/action/lose_act_1.png', -1)
        self.lose2 = imread(f'{PF.folder}/images/character/{person}/action/lose_act_2.png', -1)
        self.tie1 = imread(f'{PF.folder}/images/character/{person}/action/tie_act_1.png', -1)
        self.tie2 = imread(f'{PF.folder}/images/character/{person}/action/tie_act_2.png', -1)
        

class ComMoraImg():
    def __init__(self) -> None:
        PF = PROJECT_FOLDER()
        self.rock_img = imread(f'{PF.folder}/images/base/rock_heart.png', -1)
        self.paper_img = imread(f'{PF.folder}/images/base/paper_heart.png', -1)
        self.scissors_img = imread(f'{PF.folder}/images/base/scissors_heart.png', -1)
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
        self.star_mask = imread(f"{PROJECT_FOLDER().folder}/images/base/star.png", -1)
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
    
class AnimeManage():
    def __init__(self) -> None:
        self.window = Window()

    def star_appear(self, count, inputFrame, star_mask, point):
        resize_mask = resize(star_mask, (176-round(100*count), 176-round(100*count)))
        overlay_img = extend_alpha(resize_mask, point, self.window.shape)
        return mask_operate(inputFrame, overlay_img)
    
    def character_shake(self, count, inputFrame, shake1, shake2):
        _ = shake1 if (100*count)%50 > 25 else shake2
        overlay_img = extend_alpha(_, (0, self.window.shape[1]-512), self.window.shape)
        return mask_operate(inputFrame, overlay_img)

class Opening():
    def __init__(self, charac_ent) -> None:
        ScFolder = PROJECT_FOLDER().folder+"/images/scene"
        self.window = Window()
        self.charac_ent = charac_ent
        self.ready_mask = imread(f"{ScFolder}/op_ready_go/ready.png", -1)
        self.go_mask    = imread(f"{ScFolder}/op_ready_go/go.png", -1)
        self.boxer_down = imread(f"{ScFolder}/op_cut1/boxer_down.png")
        self.boxer_up   = imread(f"{ScFolder}/op_cut1/boxer_up.png")
        self.boxer_hit1 = imread(f"{ScFolder}/op_cut2/boxer_hit1.png")
        self.boxer_hit2 = imread(f"{ScFolder}/op_cut2/boxer_hit2.png")
        self.boxer_hit3 = imread(f"{ScFolder}/op_cut2/boxer_hit3.png")
        self.start_bg1  = imread(f"{ScFolder}/op_cut3/start_bg1.png")
        self.start_bg2  = imread(f"{ScFolder}/op_cut3/start_bg2.png")
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
                imshow('frame', self.boxer_up)
                if i == (self.Jump[0]+self.Jump[1]):
                    i = -1
            else:
                imshow('frame', self.boxer_down)
            if waitKey(1) == 27:
                return
            i += 1
        
        for hit in [self.boxer_hit1, self.boxer_hit2]:
            t = time()
            while time()-t < self.hit_time:
                imshow('frame', hit)
                if waitKey(1) == 27:
                    return
        t = time()
        while time()-t < 1.2:
            imshow('frame', self.boxer_hit3)
            if waitKey(1) == 27:
                return
        
        i = 0
        START = False
        while not START:
            t = time()
            while time()-t < self.start_flase_second/2 and not START:
                imshow('frame', self.start_bg1)
                if waitKey(1) == 32:
                    START = True
                    break
            t = time()
            while time()-t < self.start_flase_second/2 and not START:
                imshow('frame', self.start_bg2)
                if waitKey(1) == 32:
                    START = True
                    break

        fq = 512
        T = 8
        minsize = 0
        for i in range(fq//T):
            ret, frame = cap.read()
            if ret:
                frame = flip(frame, 1)
                overlay_img = extend_alpha(self.charac_ent, (0, self.window.shape[1]-512), self.window.shape)
                frame = mask_operate(frame, overlay_img)
                newsize = (minsize+fq-T*i, minsize+fq-T*i)
                resize_mask = resize(self.ready_mask, newsize)
                overlay_img = extend_alpha(resize_mask, (self.window.shape[0]//2-newsize[0]//2, self.window.shape[1]//2-newsize[1]//2), self.window.shape)
                frame = mask_operate(frame, overlay_img)
                imshow('frame', frame)
            if waitKey(1) == 27:
                return
        T = 125
        minsize = 12
        for i in range(fq//T+1):
            ret, frame_temp = cap.read()
            if ret:
                frame = flip(frame_temp, 1)
                overlay_img = extend_alpha(self.charac_ent, (0, self.window.shape[1]-512), self.window.shape)
                frame = mask_operate(frame, overlay_img)
                newsize = (minsize+T*i, minsize+T*i)
                resize_mask = resize(self.go_mask, newsize)
                overlay_img = extend_alpha(resize_mask, (self.window.shape[0]//2-newsize[0]//2, self.window.shape[1]//2-newsize[1]//2), self.window.shape)
                frame = mask_operate(frame, overlay_img)
                imshow('frame', frame)
            if waitKey(1) == 27:
                return
        t = time()
        while time() - t < 0.5:
            imshow('frame', frame)
            if waitKey(1) == 27:
                return
        return

class ending():
    def __init__(self) -> None:
        ScFolder = PROJECT_FOLDER().folder+"/images/scene"
        self.vic = imread(f"{ScFolder}/ending/ending_win.png")
        self.dft = imread(f"{ScFolder}/ending/ending_lose.png")

    def ed_show(self, comScore, winCondition, show_time):
        if comScore == winCondition:
            t = time()
            while time()-t < show_time:
                imshow('frame', self.dft)
                if waitKey(1) == 27:
                    return

        else:
            t = time()
            while time()-t < show_time:
                imshow('frame', self.vic)
                if waitKey(1) == 27:
                    return

        return