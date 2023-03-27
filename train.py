import os
import argparse
import random
import utils
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import timm
from timm.utils import AverageMeter
from data import DiffusionDataset
from utils import swap_df_path, AddGaussianNoise, seed_everything, cosine_similarity, log_epoch, log_train, log_val
from sklearn.model_selection import train_test_split
import warnings


def get_argparser():
    parser = argparse.ArgumentParser()
    
    #Experiment number
    parser.add_argument('-expnum', '--expnum',  type=int, default=3,
                        help='the number of my train')

    #Dataset Options
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp',
                        help='path to Dataset')
    parser.add_argument('--csv_name', type=str, default='diffusiondb_15W_add_embedding_mine.csv',
                        help='path to csv')
    parser.add_argument('--dataset', type=str, default='dataset_15W',
                        choices=['dataset_3W', 'dataset_15W', 'dataset_200W'],
                        help='Name of Dataset')

    #Model Options
    parser.add_argument('-model', '--model_name', type=str, default='vit_large_patch16_224',
                        choices=['vit_large_patch16_384', 'vit_base_patch32_224','vit_base_patch16_224', 'vit_large_patch16_224'],
                        help='model name')  
    parser.add_argument('--num_classes', type=int, default=384,
                        help='the dimension of embedding vector ')

    #Train Options
    parser.add_argument('-epoch', '--num_epochs', type=int, default=4,
                        help='epoch number')
    parser.add_argument('-batch', "--batch_size", type=int, default=64,
                        help='batch size')
    parser.add_argument('-input', "--input_size", type=int, default=224)
    parser.add_argument('-seed', "--random_seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--loss_type", type=str, default='Cosine', choices=['Cosine', 'Hinge'], 
                        help="loss type")
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'],
                        help='the type of optimizer')
    parser.add_argument('--scheduler', type=str, default='Cosine', choices=['Step', 'Cosine'],
                        help='learning rate scheduler policy')
    parser.add_argument('-lr', '--init_lr', type=float, default=1e-4,
                        help='init learning rate')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--step_size', type=int, default=1,
                        help='when to change LR')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='number of CPU workers, cat /proc/cpuinfo| grep "cpu cores"| uniq 查看cpu核心数')
    return parser


def get_dataset(opts):
    csv_path = os.path.join(opts.data_root, opts.csv_name)
    df = pd.read_csv(csv_path)
    trn_df, val_df = train_test_split(df, test_size=0.1, random_state=opts.random_seed)
    transform1 = transforms.Compose([
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        AddGaussianNoise(mean=random.uniform(0.5,1.5), variance=0.5, amplitude=random.uniform(0, 45),p = 0.5),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(opts.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform2 = transforms.Compose([
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # AddGaussianNoise(mean=random.uniform(0.5,1.5), variance=0.5, amplitude=random.uniform(0, 45),p = 0.5),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(opts.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trn_dataset = DiffusionDataset(trn_df, transform1)
    val_dataset = DiffusionDataset(val_df, transform2)
    return trn_dataset, val_dataset 


def get_dataloader(opts, trn_dataset, val_dataset):
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        dataset=trn_dataset,
        shuffle=True,
        batch_size=opts.batch_size,
        pin_memory=True,
        num_workers=opts.num_workers,
        drop_last=True,
        collate_fn=None,
        worker_init_fn=_init_fn
    )
    dataloaders['val'] = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=opts.batch_size,
        pin_memory=True,
        num_workers=opts.num_workers,
        drop_last=False,
        collate_fn=None,
        worker_init_fn=_init_fn
    )
    len1 = len(dataloaders['train'])
    len2 = len(dataloaders['val'])
    print(f'dataset: {opts.dataset}  train: {len1}  val: {len2}')
    return dataloaders


def validate(loader, model, device):
    val_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
    model.eval()
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            X_out = model(X)
            #target = torch.ones(X.size(0)).to(device)
            loss = criterion(X_out, y)

            val_loss = loss.item()
            val_cos = cosine_similarity(
                X_out.detach().cpu().numpy(), 
                y.detach().cpu().numpy()
            )

        val_meters['loss'].update(val_loss, n=X.size(0))
        val_meters['cos'].update(val_cos, n=X.size(0))
    return val_meters['loss'].avg, val_meters['cos'].avg


def train(opts, loader, model, device, optimizer, scheduler):
        train_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
        model.train()
        for X, y in tqdm(loader, leave=False):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            X_out = model(X)
            #target = torch.ones(X.size(0)).to(device)
            loss = criterion(X_out, y)
            loss.backward()

            optimizer.step()
            if opts.scheduler == 'Cosine':
                scheduler.step()

            trn_loss = loss.item()
            trn_cos = cosine_similarity(
                X_out.detach().cpu().numpy(), 
                y.detach().cpu().numpy()
            )

            train_meters['loss'].update(trn_loss, n=X.size(0))
            train_meters['cos'].update(trn_cos, n=X.size(0))
        if opts.scheduler == 'Step':
            scheduler.step()
        return train_meters['loss'].avg, train_meters['cos'].avg




if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    #Configs
    opts = get_argparser().parse_args()
    print(f'model: {opts.model_name}  input_size:{opts.input_size}  batchsize:{opts.batch_size}  num_epoch:{opts.num_epochs}')

    #Setup log.txt
    dir_path = '../record_sd2.0/checkpoints_%d'%opts.expnum
    Path(dir_path).mkdir(parents=True, exist_ok=False)
    logs_filename = dir_path + '/log.txt'
    Path(logs_filename).touch(exist_ok=False)

    with open(logs_filename, 'a') as f:
        f.writelines(f'{vars(opts)}\n\n\n')

    # Setup random seed
    _init_fn = seed_everything(opts.random_seed)

    ##Setup dataloader
    trn_dataset, val_dataset = get_dataset(opts)
    dataloaders = get_dataloader(opts, trn_dataset, val_dataset)

    #Setup model
    model = timm.create_model(
        opts.model_name,
        pretrained=True,
        num_classes=opts.num_classes
    )
    model.set_grad_checkpointing()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    #Setup criterion
    criterion = utils.criterion(opts.loss_type)

    #Setup optimizer
    if opts.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opts.init_lr)
    elif opts.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.init_lr, momentum=0.9, weight_decay=opts.weight_decay)

    #Setup scheduler
    if opts.scheduler == 'Cosine':
        ttl_iters = opts.num_epochs * len(dataloaders['train'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-6)
    elif opts.scheduler == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    
    best_score = -1.0
    for epoch in range(opts.num_epochs):
        log_epoch(logs_filename, epoch+1)
        #train
        loss_trn, cos_trn = train(opts, dataloaders['train'], model, device, optimizer, scheduler)
        print('Epoch {:d} / trn/loss={:.4f}, trn/cos={:.4f}'.format(epoch + 1, loss_trn, cos_trn))
        log_train(logs_filename, loss_trn, cos_trn)
        #val
        loss_val, cos_val = validate(dataloaders['val'], model, device)
        print('Epoch {:d} / val/loss={:.4f}, val/cos={:.4f}'.format(epoch + 1, loss_val, cos_val))
        log_val(logs_filename, loss_val, cos_val)
        #Save best_model
        if cos_val > best_score:
            best_score = cos_val
            torch.save(model.state_dict(), f'{dir_path}/best_{opts.model_name}.pth')
    
    print('finish all tasks! ! !')