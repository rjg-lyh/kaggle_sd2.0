import torch
import torch.nn as nn

class criterion(nn.Module):
    def __init__(self, type, margin=0.1):
        super(criterion, self).__init__()
        self.type = type
        if type == 'Cosine':
            self.loss_func = nn.CosineEmbeddingLoss()
        elif type == 'Hinge':
            self.loss_func = nn.HingeEmbeddingLoss(margin)
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