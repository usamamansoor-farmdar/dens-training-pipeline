import glob, os, pickle, random, shutil, time, gc
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import getpass
import rasterio as rio

from tqdm import tqdm
from pathlib import Path
from random import choice
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from typing import List, Any, Callable, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import ttach as tta
import albumentations as A
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter

import os
import zipfile

def unzip_file(file_path, extract_dir):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

file_name = 'skywatch_field_boundary.zip'  # Specify the name of your ZIP file
file_path = os.path.join(os.getcwd(), file_name)  # Construct the file path
extract_dir = './'  # Specify the directory where you want to extract the files

unzip_file(file_path, extract_dir)

IMG_WIDTH = 256 
IMG_HEIGHT = 256 
IMG_CHANNELS = 4
BATCH_SIZE = 2
SEED = 2023
is_train = True
n_accumulate = 1
EPOCH = 80
n_folds = 3
lr = 2e-3
num_classes = 1
MONTHS = ['2023_05']
gpus = '0'
SCHEDULER = 'CosineAnnealingWarmRestarts'
decoder = 'UnetPlusPlus'
encoder = 'timm-efficientnet-l2'
OUTPUT_DIR = '7folds_m1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_id = 'skywatch_field_boundary'
archives = ['source_train', 'source_test', 'source_labels_train']
train_source_items = f"{dataset_id}/{dataset_id}_source_train"
print("train_source_items",train_source_items)
train_label_items = f"{dataset_id}/{dataset_id}_source_labels_train"
print("train_label_items",train_label_items)
output_dir = './logs'  # Define the output directory for logs
os.makedirs(output_dir, exist_ok=True)
writer = SummaryWriter(log_dir='./logs')

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize(array: np.ndarray):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def clean_string(s: str) -> str:
    s = s.replace(f"{dataset_id}_source_", '').split('_')[1:]
    return '_'.join(s)

set_seed(SEED)

train_tiles = [clean_string(s) for s in next(os.walk(train_source_items))[1] if 'source_train' in s]
train_tile_ids = []
for tile in train_tiles:
    print(tile)
    train_tile_ids.append(tile.split('_')[0])
train_tile_ids = sorted(set(train_tile_ids))
print(train_tile_ids)


# Datasets
class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, tiles, label=True, transforms=None):
        self.label = label
        self.tiles = tiles
        self.transforms = transforms

    def __len__(self):
        return len(self.tiles)
    
    def get_image(self, tile_id):
        X = np.empty((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS*len(MONTHS)), dtype=np.float32)
        idx = 0
        source = train_source_items if self.label else test_source_items
        txt = 'train' if self.label else 'test'
        months = MONTHS.copy()
        for month in months:
            bd1 = rio.open(f"{source}/{dataset_id}_source_{txt}_{tile_id}_{month}/B01.tif")
            bd1_array = bd1.read(1)
            bd2 = rio.open(f"{source}/{dataset_id}_source_{txt}_{tile_id}_{month}/B02.tif")
            bd2_array = bd2.read(1)
            bd3 = rio.open(f"{source}/{dataset_id}_source_{txt}_{tile_id}_{month}/B03.tif")
            bd3_array = bd3.read(1)
            bd4 = rio.open(f"{source}/{dataset_id}_source_{txt}_{tile_id}_{month}/B04.tif")
            bd4_array = bd4.read(1)
            b01_norm = normalize(bd1_array)
            b02_norm = normalize(bd2_array)
            b03_norm = normalize(bd3_array)
            b04_norm = normalize(bd4_array)

            field = np.dstack((b04_norm, b03_norm, b02_norm, b01_norm))
 
        return field  
    def __getitem__(self, index):
        img = self.get_image(self.tiles[index])
        if self.label:
            msk  = rio.open(Path.cwd() / f"{train_label_items}/{dataset_id}_source_labels_train_{tile}/B01.tif").read(1)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img, msk = data['image'], data['mask']
            return torch.tensor(np.transpose(img, (2, 0, 1))), str(self.tiles[index]).zfill(2), torch.tensor(np.transpose(np.expand_dims(msk, axis=2), (2, 0, 1)))
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']
            return torch.tensor(np.transpose(img, (2, 0, 1))), str(self.tiles[index]).zfill(2)

data_transforms = {
    "train": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.PiecewiseAffine(p=0.5),
    ], p=1.0),
    "valid": A.Compose([
    ], p=1.0)
}

BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)

def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

def f1_score(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold)*1.0
    prec = (y_pred*y_true).sum()/(1e-6 + y_pred.sum())
    rec = (y_pred*y_true).sum()/(1e-6 + y_true.sum())
    f1 = 2*prec*rec/(1e-6 + prec + rec)
    return f1

# model
def build_model(encoder, decoder):
    model = smp.UnetPlusPlus(
        encoder_name=encoder, 
        encoder_weights='noisy-student',
        in_channels=IMG_CHANNELS*len(MONTHS),
        classes=num_classes,
        activation=None,
        decoder_attention_type='scse'
    )

    if len(gpus.split(',')) > 1:
        model = nn.DataParallel(model)
    model.to(device)
    return model

def load_model(encoder, decoder, path):
    model = build_model(encoder, decoder)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, tiles, masks) in pbar:         
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = criterion(y_pred, masks)
            loss   = loss / n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            # zero the parameter gradients
            optimizer.zero_grad()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}')
        # Log the training loss to TensorBoard
        writer.add_scalar('Train Loss', epoch_loss, epoch * len(dataloader) + step)
    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    val_f1s = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, tiles, masks) in pbar:
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        y_pred  = model(images)
        loss    = criterion(y_pred, masks)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        val_f1s.append(f1_score(y_true=masks.cpu().detach().numpy(), y_pred=y_pred.cpu().detach().numpy()))
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',)
         # Log the validation loss to TensorBoard
        writer.add_scalar('Valid Loss', epoch_loss, epoch * len(dataloader) + step)
    val_f1  = np.mean(val_f1s, axis=0)
    
    
    return epoch_loss, val_f1

def get_loaders(
    train_ids, val_ids,
    batch_size: int = 2,
    num_workers: int = 2,
    train_transforms_fn = None,
    valid_transforms_fn = None,
) -> dict:
    train_dataset = BuildDataset(tiles=train_ids, transforms=train_transforms_fn)
    valid_dataset = BuildDataset(tiles=val_ids, transforms=valid_transforms_fn)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size*2, 
                              num_workers=num_workers, shuffle=False, pin_memory=True)
    return train_loader, valid_loader

def run_training(model, train_loader, valid_loader, device, fold, OUTPUT_DIR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=EPOCH, eta_min=1e-6)
    best_dice      = -np.inf
    best_f1        = -np.inf
    best_epoch     = -1

    for epoch in range(1, EPOCH + 1): 
        print(f'Epoch {epoch}/{EPOCH}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=device, epoch=epoch)

        val_loss, val_f1 = valid_one_epoch(model, valid_loader, device=device, epoch=epoch)
        scheduler.step()
        print(f'Valid Loss: {val_loss:0.4f} | Valid F1: {val_f1:0.4f}')
        if val_f1 >= best_f1:
            print(f"Valid F1 Improved ({best_f1:0.4f} ---> {val_f1:0.4f})")
            best_f1    = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'fold{fold}_f2_best.pth'))
    return best_f1

import random
my_seeded_random = random.Random(SEED)

nb_rows = len(train_tile_ids)
index_all = list(range(nb_rows))
my_seeded_random.shuffle(index_all)
fold_size = nb_rows // n_folds

dict_folds = {}
for fold in range(n_folds):
    if fold == 0:
        index_val = index_all[:fold_size]
        index_train = index_all[fold_size:]
    elif fold == (n_folds - 1):
        index_val = index_all[fold_size*(n_folds-1)+1:]
        index_train = index_all[:fold_size*(n_folds-1)+1]
    else:
        index_val = index_all[fold_size*fold:fold_size*(fold+1)]
        index_train = index_all[:fold_size*fold] + index_all[fold_size*(fold+1):]
        
    dict_folds[fold] = (index_train, index_val)
    print(fold, len(index_train), len(index_val))


fold_score={}
fold_score[f'{encoder}_{decoder}'] = []
for fold in range(n_folds):
    print(f'#'*15)
    print(f'### Fold: {fold}')
    print(f'#'*15)
    (index_train, index_val) = dict_folds[fold]
    fold_train_tile_ids = [train_tile_ids[i] for i in index_train]
    fold_val_tile_ids = [train_tile_ids[i] for i in index_val]
    train_loader, valid_loader = get_loaders(fold_train_tile_ids, fold_val_tile_ids, BATCH_SIZE, 4, data_transforms['train'], data_transforms['valid'])
    comment = f'{decoder}_{encoder}'
    model_path = os.path.join(OUTPUT_DIR, comment)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model = build_model(encoder=encoder, decoder=decoder)
    best_f1 = run_training(model, train_loader, valid_loader, device, fold, model_path)
    fold_score[f'{encoder}_{decoder}'].append({f'{fold}fold' : best_f1})

