import os
import random
import shutil

class DataSplit():
    def __init__(self, split_rate) -> None:
        self.split_rate = split_rate

    def random_data(self, folder_path):
        files = os.listdir(folder_path)
        files.sort()

        filename = [os.path.splitext(file)[0] for file in files]
        len_ = len(filename)
        selected = set(random.sample(filename, len_-int(self.split_rate*len_)))
        return selected

    def move_(self, src_folder, dst_train_folder, dst_valid_folder, select_name):
        for file in os.listdir(src_folder):
            filename = os.path.splitext(file)[0]
            if filename not in select_name:
                continue
            fullDir = os.path.join(src_folder, file)
            if os.path.isfile(fullDir):
                shutil.move(fullDir, dst_valid_folder)

        for file in os.listdir(src_folder):
            fullDir = os.path.join(src_folder, file)
            if os.path.isfile(fullDir):
                shutil.move(fullDir, dst_train_folder)
        

    def splitting(self, temp_imgs, temp_labels, train_imgs, train_labels, valid_imgs, valid_labels):
        select_name = self.random_data(temp_imgs)
        self.move_(temp_imgs, train_imgs, valid_imgs, select_name)
        self.move_(temp_labels, train_labels, valid_labels, select_name)

class ClassifyFile():
    def __init__(self, folder) -> None:
        self.folder = folder

        self.paper_datas = folder + '/all_data/paper'
        self.scissors_datas = folder + '/all_data/scissors'
        self.stone_datas = folder + '/all_data/rock'
        self.mix_datas = folder + '/all_data/_mix_'

        self.train_imgs = folder + '/yolo_data/train/images'
        self.train_labels = folder + '/yolo_data/train/labels'
        self.valid_imgs = folder + '/yolo_data/valid/images'
        self.valid_labels = folder + '/yolo_data/valid/labels'

        self.temp_imgs = folder + '/_temp_/images'
        self.temp_labels = folder + '/_temp_/labels'

    def make_folder(self):
        for nnn in [
            self.paper_datas,self.scissors_datas,self.stone_datas,self.mix_datas,self.train_imgs,self.train_labels,self.valid_imgs,self.valid_labels,self.temp_imgs,self.temp_labels
        ]:
            if not os.path.exists(nnn):
                os.makedirs(nnn)


    def classify_file_type(self, _data):
        for files in os.listdir(_data):
            ext = os.path.splitext(files)[1]
            fullDir = os.path.join(_data, files)
            if os.path.isfile(fullDir):
                if ext == '.txt':
                    shutil.copy(fullDir, self.temp_labels)
                elif ext.lower() in {'.jpg', '.jpeg', '.png'}:
                    shutil.copy(fullDir, self.temp_imgs)


def main():
    folder = 'C:/Users/User/Desktop/data'
    split_rate = 0.8

    classifyfile = ClassifyFile(folder)
    datasplit = DataSplit(split_rate)

    classifyfile.make_folder()
    (temp_imgs, temp_labels, train_imgs, train_labels, valid_imgs, valid_labels) = (
        classifyfile.temp_imgs, classifyfile.temp_labels, classifyfile.train_imgs, classifyfile.train_labels, classifyfile.valid_imgs, classifyfile.valid_labels
        )

    hand_data = [classifyfile.scissors_datas, classifyfile.stone_datas, classifyfile.paper_datas, classifyfile.mix_datas]
    for hand in hand_data:
        classifyfile.classify_file_type(hand)
        datasplit.splitting(temp_imgs, temp_labels, train_imgs, train_labels, valid_imgs, valid_labels)

main()
