
from ProjectPackage import PROJECT_FOLDER
from MoraGame import MoraGame

PF = PROJECT_FOLDER()
video_file = f'{PF.folder}/video/video.mp4'
moragame = MoraGame(f'{PF.folder}/model_pt/main_pt.pt')
moragame.GamingPlay()