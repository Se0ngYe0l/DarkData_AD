import schedule
import torch
import time
import os

from utils.model_func import model_func
from network.networks import NetG
import random

########################################################
# 메모리
path = "AD/network/75_netG.pth"
device = torch.device("cuda:0")
netg = NetG().to(device)
pretrained_dict = torch.load(path)['state_dict']
netg.load_state_dict(pretrained_dict)
########################################################

def AD():
    data_dict = []

    # AD/save_img 폴더에서 5장씩
    imglist = os.listdir("AD/save_img")
    a = random.sample(range(0,len(imglist)),5)

    # 모델 Inference
    for i in range(5):
        diagnosis = model_func(imglist[a[i]], netg, device)
        data_dict.append(diagnosis)

    # 결과 출력
    print(data_dict)

########################################################
# 1초마다 반복
schedule.every(1).seconds.do(AD)

while True:
    schedule.run_pending()
    time.sleep(1)
########################################################