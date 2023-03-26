import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from scipy import spatial



def swap_df_path(data_root, csv_name, dataset):
    csv_path = os.path.join(data_root, csv_name)
    data_path = os.path.join(data_root, dataset)
    df = pd.read_csv(csv_path, index_col=False)
    n = len(df)
    print('loading dataset...')
    for i in tqdm(range(n)):
        ImgId = Path(df.iloc[i,0]).stem + '.jpg'
        new_path = os.path.join(data_path, ImgId)
        df.iloc[i, 0] = new_path
    return df

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1)) 
            #loc是高斯分布的均值（中心）,scale是是标准差（宽度）random.normal从正态分布中随机挑选样本
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    def _init_fn(worker_id):
        np.random.seed(int(seed)+worker_id)
    return _init_fn

def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred) 
        for y_true, y_pred in zip(y_trues, y_preds)
    ])

def log_epoch(logs_filename, epoch):
    with open(logs_filename, 'a') as f:
        division_line = f'Epoch_{epoch}'.center(100, '-')
        f.writelines(f'{division_line} \n\n')

def log_train(logs_filename, loss_trn, cos_trn):
    with open(logs_filename, 'a') as f:
        f.writelines('trn/loss={:.4f}, trn/cos={:.4f} \n'.format(loss_trn, cos_trn))

def log_val(logs_filename, loss_val, cos_val):
    with open(logs_filename, 'a') as f:
        f.writelines('trn/loss={:.4f}, trn/cos={:.4f} \n\n'.format(loss_val, cos_val))