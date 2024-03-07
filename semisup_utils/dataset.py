from semisup_utils.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Load unlabeled samples
class SemiCDDataset(Dataset):
    def __init__(self, root, unlabel_fl, size=None, mode="unlabel"):
        #self.name = name
        self.root = root
        self.size = size   #512
        self.mode = mode
        self.unlabel_fl = unlabel_fl   

    def __getitem__(self, item):
        fn = self.unlabel_fl[item]
        
        imgA = Image.open(os.path.join(fn)).convert('RGB')
        imgB = Image.open(os.path.join(fn.replace('_pre_','_post_'))).convert('RGB')
        mask = Image.fromarray(np.zeros_like(imgA)[:,:,0])

        if self.mode == 'test':
            mask = np.array(Image.open(fn.replace('/images/','/masks/').replace('_pre_','_post_')))
            mask = Image.fromarray(mask.astype(np.uint8))
           
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask
        
        imgA, imgB, mask = resize(imgA, imgB, mask, (0.8, 1.2))
        imgA, imgB, mask = crop(imgA, imgB, mask, self.size)
        imgA, imgB, mask = hflip(imgA, imgB, mask, p=0.5)

        imgA_w, imgB_w = deepcopy(imgA), deepcopy(imgB)   # weakly-augmented data
        return normalize(imgA_w), normalize(imgB_w)

    def __len__(self):
        return len(self.unlabel_fl)
