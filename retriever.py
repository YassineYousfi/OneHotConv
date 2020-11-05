import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import jpegio as jio
import pandas as pd
import numpy as np
import pickle
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
import sys
import random
sys.path.insert(1,'./')
from tools.jpeg_utils import *

def rot_and_flip_jpeg(S, rot, flip=False):
    """Lossless rotation and flips in the DCT domain"""
    assert S.coef_arrays[0].shape[0] % 8 == 0, 'Image height is not a multiple of 8!'
    assert S.coef_arrays[0].shape[1] % 8 == 0, 'Image width is not a multiple of 8!'
    I = np.empty([S.coef_arrays[0].shape[0], S.coef_arrays[0].shape[1], S.image_components],dtype=np.int32)
    for ch in range(I.shape[-1]):
        C = np.copy(S.coef_arrays[ch])  
        C = np.rot90(C, k=rot)
        fun = lambda x: np.rot90(x,k=4-rot, axes=(2,3))
        C = segmented_stride(C, fun)
        # transpose
        if rot in [1,3]:
            fun = lambda x: np.copy(np.transpose(x, axes=[0,1,3,2]))
            C = segmented_stride(C, fun)
            for q in range(len(S.quant_tables)):
                S.quant_tables[q] = S.quant_tables[q].T
        # multiply even rows by -1
        if rot in [1,2]:
            C[1::2,:] *= -1
        # multiply even columns by -1
        if rot in [2,3]:
            C[:,1::2] *= -1  
        # flip
        if flip == True:
            C = np.fliplr(C)
            fun = lambda x: np.flip(x, axis=3)
            C = segmented_stride(C, fun)
            C[:,1::2] *= -1 
        S.coef_arrays[ch] = C[:,:]
    return S

def abs_bounded_onehot(I,T=5):
    """Threshold + abs + onehot encoding"""
    I = np.abs(I)
    I = np.clip(I, a_min=0,a_max=T)
    I_oh = np.zeros((I.shape)+(T+1,))
    for ch in range(I.shape[-1]):
        I_oh[:,:,ch,:] = (I == range(T+1))
    return I_oh.reshape((I.shape[0],I.shape[1], -1))

class TrainRetrieverPaired(Dataset):

    def __init__(self, data_path, kinds, folds, image_names, labels, num_classes=2, transforms=True, T=5):
        super().__init__()
        
        self.data_path = data_path
        self.kinds = kinds
        self.folds = folds
        self.image_names = image_names
        self.labels = labels
        self.num_classes = num_classes
        self.transforms = transforms
        self.T = T
        
    def __preprocess_strcture(self, tmp, label):
        
        image_dct = np.dstack(tmp.coef_arrays).astype(np.float32)
        image = decompress_structure(tmp).astype(np.float32)
        image = torch.from_numpy(image.transpose(2,0,1))
        
        image_dct = abs_bounded_onehot(image_dct, T=self.T).astype(np.float32)
        image_dct = torch.from_numpy(image_dct.transpose(2,0,1))
        
        label = torch.tensor(label, dtype=torch.long)
        return image, image_dct, label
    
    def __getitem__(self, index: int):
        
        kind, fold, image_name, label = self.kinds[index], self.folds[index], self.image_names[index], self.labels[index]
        
        if self.transforms:
            rot = random.randint(0,3)
            flip = random.random() < 0.5
        else:
            rot = 0
            flip = False
        path = self.data_path/kind[0]/fold/image_name
        tmp = jio.read(str(path))
        tmp = rot_and_flip_jpeg(tmp, rot, flip)
        cover, cover_dct, target_cover = self.__preprocess_strcture(tmp, label[0])
        
        i = np.random.randint(low=1, high=self.num_classes)
        path = self.data_path/kind[i]/fold/image_name
        tmp = jio.read(str(path))
        tmp = rot_and_flip_jpeg(tmp, rot, flip)
        stego, stego_dct, target_stego = self.__preprocess_strcture(tmp, label[i])

        return torch.stack([cover,stego]), torch.stack([cover_dct,stego_dct]), torch.stack([target_cover, target_stego])

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)