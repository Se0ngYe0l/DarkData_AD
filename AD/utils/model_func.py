import numpy as np
import torch
from utils.transform import data_transform

def model_func(file_name, network, device):
    # 데이터 전처리
    t_data = data_transform(file_name, device)

    network.eval()
    with torch.no_grad():
        _, latent_i, latent_o = network(t_data)
        error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)

        error = error.cpu().numpy()
        error = np.squeeze(error)
        error = error.tolist()
        
        min = 0.0004461664648260921
        max = 0.029765909537672997
        threshold = 0.43434194
        
        an_score = (error - min) / (max - min)
        if an_score >= threshold:
            return 1
        else:
            return 0