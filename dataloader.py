import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision import transforms

import random
# import openslide
import h5py

from PIL import Image

def train_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    th_img, rep_tensors, caps, caplens, n_imgs = zip(*data)
    
    th_img = torch.vstack([x.unsqueeze(0) for x in th_img])
    rep_tensors = torch.vstack(rep_tensors)
    caps = torch.vstack([x.unsqueeze(0) for x in caps])
    caplens = torch.vstack([x.unsqueeze(0) for x in caplens])
    n_imgs = torch.vstack([x.unsqueeze(0) for x in n_imgs])

    return th_img, rep_tensors, caps, caplens, n_imgs

def val_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    th_img, rep_tensors, caps, caplens, allcaps, n_imgs = zip(*data)
    
    th_img = torch.vstack([x.unsqueeze(0) for x in th_img])
    rep_tensors = torch.vstack(rep_tensors)
    caps = torch.vstack([x.unsqueeze(0) for x in caps])
    caplens = torch.vstack([x.unsqueeze(0) for x in caplens])
    allcaps = torch.vstack([x.unsqueeze(0) for x in allcaps])
    n_imgs = torch.vstack([x.unsqueeze(0) for x in n_imgs])

    return th_img, rep_tensors, caps, caplens, allcaps, n_imgs
   
class PreLoadedReps_v2(Dataset):

    def __init__(self, df_path, dtype='train', th_transform=None, pid_list=None):
        self.df_path = df_path
        df = pd.read_pickle(self.df_path)
        self.dtype=dtype
        if pid_list==None:
            self.df=df[df.dtype==dtype]
        else:
            self.df=df[df.pid.isin(pid_list)]
        self.th_transform = th_transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        svs_path=self.df.iloc[idx]['svs_path']
        patch_path=self.df.iloc[idx]['patch_path']
        reps_path=self.df.iloc[idx]['reps_path']
        idx_tokens=self.df.iloc[idx]['idx_tokens']
        caplens=self.df.iloc[idx]['caplens']
        thumb_path=self.df.iloc[idx]['thumb_paths']
        pid=self.df.iloc[idx]['pid']

        rep_tensors = torch.load(reps_path)
        n_patches = torch.LongTensor([rep_tensors.shape[0]])
        caption_tensor = torch.LongTensor(idx_tokens)
        caplen = torch.LongTensor([caplens])
        th_img = Image.open(thumb_path)

        if self.th_transform is not None:
            th_img = self.th_transform(th_img)

        coords = h5py.File(patch_path, 'r')['coords']
        cord_dict={svs_path:[list(x) for x in coords]}
        
        if self.dtype=='train':
            rep_tensor=rep_tensors
            return rep_tensor, caption_tensor, caplen, n_patches, th_img, pid
        else:
            rep_tensor=rep_tensors
            all_captions=torch.LongTensor([idx_tokens])
            return rep_tensor, caption_tensor, caplen, all_captions, cord_dict, n_patches, th_img

## Data loader for custom PIDS
class PreLoadedReps_v3(Dataset):

    def __init__(self, df_path, dtype='train', transform=None, target_transform=None, pids=None):
        self.df_path = df_path
        df = pd.read_pickle(self.df_path)
        self.dtype=dtype
        self.df=df[df.dtype==dtype]
        if pids != None:
            self.df=df[df.pid.isin(pids)]
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        svs_path=self.df.iloc[idx]['svs_path']
        patch_path=self.df.iloc[idx]['patch_path']
        reps_path=self.df.iloc[idx]['reps_path']
        idx_tokens=self.df.iloc[idx]['idx_tokens']
        caplens=self.df.iloc[idx]['caplens']

        rep_tensors = torch.load(reps_path)
        caption_tensor = torch.LongTensor(idx_tokens)
        caplen = torch.LongTensor([caplens])

        coords = h5py.File(patch_path, 'r')['coords']
        cord_dict={svs_path:[list(x) for x in coords]}
        
        if self.dtype=='train':
            rep_tensor=rep_tensors
            return rep_tensor, caption_tensor, caplen
        else:
            rep_tensor=rep_tensors
            all_captions=torch.LongTensor([idx_tokens])
            return rep_tensor, caption_tensor, caplen, all_captions, cord_dict


class ResnetDataset(Dataset):

    def __init__(self, df_path, dtype='train', th_transform=None, pid_list=None):
        self.df_path = df_path
        df = pd.read_pickle(self.df_path)
        self.dtype=dtype
        if pid_list==None:
            self.df=df[df.dtype==dtype]
        else:
            self.df=df[df.pid.isin(pid_list)]
        self.th_transform = th_transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx_tokens=self.df.iloc[idx]['idx_tokens']
        caplens=self.df.iloc[idx]['caplens']
        thumb_path=self.df.iloc[idx]['thumb_paths']

        caption_tensor = torch.LongTensor(idx_tokens)
        caplen = torch.LongTensor([caplens])
        th_img = Image.open(thumb_path)

        if self.th_transform is not None:
            th_img = self.th_transform(th_img)
        
        if self.dtype=='train':
            return th_img, caption_tensor, caplen
        else:
            all_captions=torch.LongTensor([idx_tokens])
            return th_img, caption_tensor, caplen, all_captions
        
class ResnetPlusVitDataset(Dataset):

    def __init__(self, df_path, dtype='train', th_transform=None, pid_list=None, vit_img_size=64):
        self.df_path = df_path
        df = pd.read_pickle(self.df_path)
        self.dtype=dtype
        if pid_list==None:
            self.df=df[df.dtype==dtype]
        else:
            self.df=df[df.pid.isin(pid_list)]
        self.th_transform = th_transform
        self.vit_img_size = vit_img_size
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx_tokens=self.df.iloc[idx]['idx_tokens']
        caplens=self.df.iloc[idx]['caplens']
        thumb_path=self.df.iloc[idx]['thumb_paths']
        reps_path=self.df.iloc[idx]['reps_path']

        caption_tensor = torch.LongTensor(idx_tokens)
        caplen = torch.LongTensor([caplens])
        th_img = Image.open(thumb_path)
        
        rep_tensors = torch.load(reps_path)
        n_imgs = rep_tensors.shape[0]
        # if n_imgs < self.vit_img_size:
        #     print("here")
        #     zero_pad_tensor = torch.zeros((self.vit_img_size-n_imgs, 256, 384))
        #     rep_tensors = torch.vstack((rep_tensors, zero_pad_tensor))
        # else:
        #     rep_tensors = rep_tensors[:self.vit_img_size,:,:]

        # tensorize number if 4096 patches
        n_imgs = torch.LongTensor([n_imgs])

        if self.th_transform is not None:
            th_img = self.th_transform(th_img)
        
        if self.dtype=='train':
            return th_img, rep_tensors, caption_tensor, caplen, n_imgs
        else:
            all_captions=torch.LongTensor([idx_tokens])
            return th_img, rep_tensors, caption_tensor, caplen, all_captions, n_imgs