import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np 
import pickle
import argparse
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Generator, Union
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
import random
from retriever import *
import SRNet
from pytorch_lightning.metrics.converters import _sync_ddp_if_available

class LitModel(pl.LightningModule):
    """Transfer Learning
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 backbone: str = 'OneHotSRNet',
                 batch_size: int = 16,
                 lr: float = 1e-3,
                 eps: float = 1e-8,
                 lr_scheduler_name: str = 'MultiStepLR',
                 qf: str = 'QF100',
                 optimizer_name: str = 'Adamax',
                 num_workers: int = 6, 
                 epochs: int = 300, 
                 milestone: int = 150,
                 gpus: int = 1, 
                 weight_decay: float = 1e-4,
                 payload: str = '0.1_bpnzac',
                 stego_scheme: str = 'nsf5_simulation',
                 loss_weights: list = [1.0, 1.0, 1.0],
                 threshold: int = 5
                 ,**kwargs) -> None:
        
        super().__init__()
        self.data_path = data_path
        self.epochs = epochs
        self.milestone = milestone
        self.backbone = backbone
        self.batch_size = batch_size
        self.lr = lr
        self.stego_scheme = stego_scheme
        self.payload = payload
        self.qf = qf
        self.num_workers = num_workers
        self.lr_scheduler_name = lr_scheduler_name
        self.optimizer_name = optimizer_name
        self.gpus = gpus
        self.weight_decay = weight_decay
        self.eps = eps
        self.threshold = threshold
        self.loss_weights = loss_weights
        self.save_hyperparameters()
        self.data_path = Path(self.data_path)/self.qf
        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        self.net = getattr(SRNet, self.backbone)(1, 2, self.threshold)
        
        self.loss_func = F.cross_entropy

    def forward(self, x, x_dct):
        """Forward pass. Returns logits."""

        logits = self.net(x, x_dct)
        
        return logits

    def loss(self, logits, labels, loss_weights):
        
        labels = torch.flatten(labels)
        losses = [self.loss_func(logits[i], labels) for i in range(3)]
        loss = sum(l*w for l,w in zip(losses,loss_weights))
        return losses+[loss] #torch.cat((losses,loss.reshape(1)))
    
    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def training_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, x_dct, y = batch
        logits = self.forward(x, x_dct)

        # 2. Compute loss & accuracy:
        srnet_loss, oh_loss, fc_loss, train_loss = self.loss(logits, y, self.loss_weights)        
        #train_loss.requires_grad = True
        
        losses = {'trn_SRNet_loss': srnet_loss,
                  'trn_OH_loss': oh_loss,
                  'trn_FC_loss': fc_loss}      
        
        metrics = {'trn_SRNet_acc': self.__accuracy(logits[0], y)[0],
                   'trn_OH_acc': self.__accuracy(logits[1], y)[0],
                   'trn_FC_acc': self.__accuracy(logits[2], y)[0]
                  }

        # 3. Outputs:        
        output = OrderedDict({'loss': train_loss,
                              'progress_bar': losses,
                              'log': {**metrics, **losses}})

        return output

    def validation_step(self, batch, batch_idx):
        
        # 1. Forward pass:
        x, x_dct, y = batch
        logits = self.forward(x, x_dct)

        # 2. Compute loss & accuracy:
        srnet_loss, oh_loss, fc_loss, val_loss = self.loss(logits, y, self.loss_weights)
        
        losses = {'val_SRNet_loss': srnet_loss,
                  'val_OH_loss': oh_loss,
                  'val_FC_loss': fc_loss,
                  'val_loss': val_loss}      
        
        metrics = {'val_SRNet_acc': self.__accuracy(logits[0], y)[0],
                   'val_OH_acc': self.__accuracy(logits[1], y)[0],
                   'val_FC_acc': self.__accuracy(logits[2], y)[0]
                  }
                
        return {**metrics, **losses}
    
    def validation_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""
        metrics = {}
        
        for key in outputs[0].keys():
            metrics[key] = torch.stack([output[key] for output in outputs]).mean()
            metrics[key] = _sync_ddp_if_available(metrics[key], reduce_op='avg')
                        
        metrics['step'] = self.current_epoch    
            
        return {'log': metrics}
    

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)
                
        optimizer = optimizer(self.parameters(), 
                              lr=self.lr, 
                              weight_decay=self.weight_decay, 
                              eps=self.eps)
        
        scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)        
        scheduler = scheduler(optimizer, milestones=[self.milestone])
        interval = 'epoch'

        return [optimizer], [{'scheduler': scheduler, 'interval': interval, 'name': 'lr'}]


    def prepare_data(self):
        """Download images and prepare images datasets."""
        
        print('Data Preparation is not part of this script, make sure your datasets are ready')

    def setup(self, stage: str): 
                
        stego = self.stego_scheme+'_'+self.payload
        cover_dir = self.data_path/'COVER'
        
        IL_train = os.listdir(cover_dir/'TRN')
        IL_val =  os.listdir(cover_dir/'VAL')
        
        dataset = []    
        for path in IL_train:
            dataset.append({
                'kind': ('COVER', stego),
                'image_name': path,
                'label': (0,1),
                'fold': 'TRN',
                })
            
        for path in IL_val:
            dataset.append({
                'kind': ('COVER', stego),
                'image_name': path,
                'label': (0,1),
                'fold': 'VAL',
                })
        
        random.shuffle(dataset)
        dataset = pd.DataFrame(dataset)
        
        self.train_dataset = TrainRetrieverPaired(
            data_path=self.data_path,
            kinds=dataset[dataset['fold'] != 'VAL'].kind.values,
            folds=dataset[dataset['fold'] != 'VAL'].fold.values,
            image_names=dataset[dataset['fold'] != 'VAL'].image_name.values,
            labels=dataset[dataset['fold'] != 'VAL'].label.values,
            transforms=True,
            num_classes=2,
            T=self.threshold,
        )
        
        self.valid_dataset = TrainRetrieverPaired(
            data_path=self.data_path,
            kinds=dataset[dataset['fold'] == 'VAL'].kind.values,
            folds=dataset[dataset['fold'] == 'VAL'].fold.values,
            image_names=dataset[dataset['fold'] == 'VAL'].image_name.values,
            labels=dataset[dataset['fold'] == 'VAL'].label.values,
            transforms=False,
            num_classes=2,
            T=self.threshold,
        )
    
    
    def __dataloader(self, train):
        """Train/validation loaders."""

        _dataset = self.train_dataset if train else self.valid_dataset
        
        def collate_fn(data):
            images, images_dct, labels = zip(*data)
            images = torch.cat(images)
            labels = torch.cat(labels)
            images_dct = torch.cat(images_dct)
            return images, images_dct, labels
        
        loader = DataLoader(dataset=_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            collate_fn=collate_fn,
                            shuffle=True if train else False)

        return loader

    def train_dataloader(self):
        log.info('Training data loaded.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loaded.')
        return self.__dataloader(train=False)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--backbone',
                            default='OneHotSRNet',
                            type=str,
                            metavar='BK',
                            help='Name as in the SRNet.py file')
        parser.add_argument('--data-path',
                            default='/media/ONEHOT-DCT/BOSS/JPEG_standard/',
                            type=str,
                            metavar='dp',
                            help='data_path')
        parser.add_argument('--stego-scheme',
                            default='nsf5_simulation',
                            type=str,
                            help='Stego scheme')
        parser.add_argument('--payload',
                            default='0.1_bpnzac',
                            type=str,
                            help='Payload')
        parser.add_argument('--epochs',
                            default=300,
                            type=int,
                            metavar='N',
                            help='total number of epochs')
        parser.add_argument('--milestone',
                            default=150,
                            type=int,
                            help='drop LR milestone')
        parser.add_argument('--batch-size',
                            default=16,
                            type=int,
                            metavar='B',
                            help='batch size',
                            dest='batch_size')
        parser.add_argument('--threshold',
                            type=int,
                            default=5,
                            help='DCT domain threshold')
        parser.add_argument('--gpus',
                            type=int,
                            default=1,
                            help='number of gpus to use')
        parser.add_argument('--lr',
                            '--learning-rate',
                            default=1e-3,
                            type=float,
                            metavar='LR',
                            help='initial learning rate',
                            dest='lr')
        parser.add_argument('--loss-weights',
                            default=[1.0, 1.0, 1.0],
                            nargs='+',
                            type=float,
                            help='loss weights')
        parser.add_argument('--eps',
                            default=1e-8,
                            type=float,
                            help='eps for adaptive optimizers',
                            dest='eps')
        parser.add_argument('--num-workers',
                            default=6,
                            type=int,
                            metavar='W',
                            help='number of CPU workers',
                            dest='num_workers')
        parser.add_argument('--lr-scheduler-name',
                            default='MultiStepLR',
                            type=str,
                            metavar='LRS',
                            help='Name of LR scheduler')
        parser.add_argument('--optimizer-name',
                            default='Adamax',
                            type=str,
                            metavar='OPTI',
                            help='Name of optimizer')
        parser.add_argument('--qf',
                            default='QF100',
                            type=str,
                            help='quality factor')
        parser.add_argument('--weight-decay',
                            default=1e-4,
                            type=float,
                            metavar='wd',
                            help='Optimizer weight decay')


        return parser
