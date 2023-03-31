from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
 
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
 
        self.in_features = in_features    # 特征输入通道数
        self.out_features = out_features  # 特征输出通道数
        self.s = s                        # 输入特征范数 ||x_i||
        self.m = m                        # 加性角度边距 m (additive angular margin)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))  # FC 权重
        nn.init.xavier_uniform_(self.weight)  # Xavier 初始化 FC 权重
 
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
 
    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # 分别归一化输入特征 xi 和 FC 权重 W, 二者点乘得到 cosθ, 即预测值 Logit  
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # 由 cosθ 计算相应的 sinθ
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # 展开计算 cos(θ+m) = cosθ*cosm - sinθ*sinm, 其中包含了 Target Logit (cos(θyi+ m)) (由于输入特征 xi 的非真实类也参与了计算, 最后计算新 Logit 时需使用 One-Hot 区别)
        phi = cosine * self.cos_m - sine * self.sin_m  
        # 是否松弛约束??
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将 labels 转换为独热编码, 用于区分是否为输入特征 xi 对应的真实类别 yi
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # 计算新 Logit
        #  - 只有输入特征 xi 对应的真实类别 yi (one_hot=1) 采用新 Target Logit cos(θ_yi + m)
        #  - 其余并不对应输入特征 xi 的真实类别的类 (one_hot=0) 则仍保持原 Logit cosθ_j
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # can use torch.where if torch.__version__  > 0.4
        # 使用 s rescale 放缩新 Logit, 以馈入传统 Softmax Loss 计算
        output *= self.s


class ArcFace1(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace1, self).__init__()
        self.scale = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s        
        return logits


class criterion(nn.Module):
    def __init__(self, type, margin=0.1):
        super(criterion, self).__init__()
        self.type = type
        if type == 'Cosine':
            self.loss_func = nn.CosineEmbeddingLoss()
        elif type == 'Hinge':
            self.loss_func = nn.HingeEmbeddingLoss(margin)
        elif type == 'ArcFace':
            self.loss_func = ArcFace()
        else:
            self.loss_func = None
    
    def forward(self, emb_pred, emb_true):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.type == 'Cosine':
            target = torch.ones(emb_pred.size(0)).to(device)
            return self.loss_func(emb_pred, emb_true, target)
        elif self.type == 'Hinge':
            x = 1 - torch.cosine_similarity(emb_pred, emb_true)
            y = torch.ones(emb_pred.size(0)).to(device)
            return self.loss_func(x, y)
        else:
            return None