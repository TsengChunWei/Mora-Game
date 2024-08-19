import os
import cv2

class PROJECT_FOLDER():
    def __init__(self) -> None:
        self.folder = os.path.dirname(__file__)+"/.."

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

class ScoreImgShow():
    def __init__(self) -> None:
        self.imgInsert = ImageInsert()
        self.window = Window()
        self.star_mask = cv2.imread(f"{PROJECT_FOLDER().folder}/images/base/star2.png", -1)
        self.star_size = (self.star_mask.shape[1], self.star_mask.shape[0])
        self.star_anime_time = self.star_size[0]+100*GameInitVal().break_time

    
    def scoreShow(self, outputFrame, p, score):
        for i in range(score):
            overlay_img = self.imgInsert.extend_alpha(self.star_mask, self.scorePosition(p, i), self.window.shape)
            outputFrame = self.imgInsert.mask_operate(outputFrame, overlay_img)
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
            overlay_img = self.imgInsert.extend_alpha(resize_mask, point, self.window.shape)
            outputFrame = self.imgInsert.mask_operate(outputFrame, overlay_img)
        cv2.imshow('frame', outputFrame)            
        outputFrame = pre_frame

class Opening():
    def __init__(self, charac_ent) -> None:
        self.imgInsert = ImageInsert()
        self.window = Window()
        self.charac_ent = charac_ent
        self.ready_mask = cv2.imread(f"{PROJECT_FOLDER().folder}/images/base/ready.png", -1)
        self.go_mask = cv2.imread(f"{PROJECT_FOLDER().folder}/images/base/go.png", -1)

    def animation(self, cap):
        minsize = 0
        fq = self.window.shape[1]-minsize    # 720
        T = 20
        for i in range(fq//T):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                overlay_img = self.imgInsert.extend_alpha(self.charac_ent, (0, self.window.shape[1]-512), self.window.shape)
                frame = self.imgInsert.mask_operate(frame, overlay_img)
                newsize = (minsize+fq-T*i, minsize+fq-T*i)
                resize_mask = cv2.resize(self.ready_mask, newsize)
                overlay_img = self.imgInsert.extend_alpha(resize_mask, (self.window.shape[0]//2-newsize[0]//2, self.window.shape[1]//2-newsize[1]//2), self.window.shape)
                frame = self.imgInsert.mask_operate(frame, overlay_img)
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) == 27:
                return
        T = 35
        minsize = 20
        for i in range(fq//T+1):
            ret, frame_temp = cap.read()
            if ret:
                frame = cv2.flip(frame_temp, 1)
                overlay_img = self.imgInsert.extend_alpha(self.charac_ent, (0, self.window.shape[1]-512), self.window.shape)
                frame = self.imgInsert.mask_operate(frame, overlay_img)
                newsize = (minsize+T*i, minsize+T*i)
                resize_mask = cv2.resize(self.go_mask, newsize)
                overlay_img = self.imgInsert.extend_alpha(resize_mask, (self.window.shape[0]//2-newsize[0]//2, self.window.shape[1]//2-newsize[1]//2), self.window.shape)
                frame = self.imgInsert.mask_operate(frame, overlay_img)
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) == 27:
                return
        return