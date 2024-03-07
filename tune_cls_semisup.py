import os
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

from apex import amp

from torch.optim import AdamW
from zoo.losses import dice_round, ComboLoss

import pandas as pd
from tqdm import tqdm
import timeit
import cv2
import gc
import pandas as pd
import yaml
import copy

from zoo.models import SeResNext50_Unet_Double

from imgaug import augmenters as iaa

from utils import *

from skimage.morphology import square, dilation
from semisup_utils.dataset import SemiCDDataset
from semisup_utils.loss_helper import (
    compute_contra_memobank_loss
)

from semisup_utils.utils import (
    AverageMeter,
    label_onehot
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

disaster = 'PAKISTAN_FLOODING'  

train_dirs = ['data/{}/train'.format(disaster)]  
unlabel_dirs = ['data/{}/unlabel'.format(disaster)]
test_dirs = ['data/{}/test'.format(disaster)]
loc_folder = 'data/{}/unlabel/loc'.format(disaster)

models_folder = 'weights'
models_tune_folder = 'weights_tune'
log_folder = 'log/{}'.format(disaster)

log_file = '{}.log'.format(disaster)

logger = get_logger(os.path.join(log_folder, log_file))
input_shape = (512, 512)


label_file = "./data/{}/all_labels.csv".format(disaster)
df = pd.read_csv(label_file,engine = 'python')

key = []
value = []
for i in df['file_name']:
    key.append(i)
for j in df['set_type']:
    value.append(j)
    
set_dict = dict(zip(key,value))

train_files = []
test_files = []
for d in train_dirs:
    for f in sorted(listdir(path.join(d, 'images'))):
         if '_pre_' in f  and disaster in f: 
            if f not in set_dict:    
                continue
            
            if disaster in f and set_dict[f] == 'train':
                train_files.append(path.join(d, 'images', f))

            elif disaster in f and set_dict[f] == 'test':
                test_files.append(path.join(d, 'images', f))
                
                
unlabel_files = []
for d in unlabel_dirs:
    for f in sorted(listdir(path.join(d, 'images'))):
        if '_pre_' in f:  
            if f in set_dict and set_dict[f] == 'train':   
                continue
            unlabel_files.append(path.join(d, 'images', f))


# set random seed
seed = 0
np.random.seed(seed + 1234)
random.seed(seed + 1234)

train_len, test_len, unlabel_len = len(train_files), len(test_files), len(unlabel_files)

class TrainData(Dataset):
    def __init__(self, train_idxs, low, high):
        super().__init__()
        self.train_idxs = train_idxs
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.low =low
        self.high = high

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]

        fn = train_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img2 = cv2.imread(fn.replace('_pre_', '_post_'), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_', '_post_'), cv2.IMREAD_UNCHANGED)
        msk0[msk0>0] = 255

        lbl_msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_', '_post_'), cv2.IMREAD_UNCHANGED)
        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255
        msk1[lbl_msk1 == 1] = 255

        if random.random() > 0.87:  
            lam = np.random.beta(2, 1.8)
            rand_inx = torch.randint(low=self.low,high=self.high,size=(1,))
            ttt = self.train_idxs[rand_inx]
            fn_rand = train_files[ttt]
            img_random = cv2.imread(fn_rand, cv2.IMREAD_COLOR)
            img2_random = cv2.imread(fn_rand.replace('_pre_', '_post_'), cv2.IMREAD_COLOR)
            msk0_random = cv2.imread(fn_rand.replace('/images/', '/masks/').replace('_pre_', '_post_'), cv2.IMREAD_UNCHANGED)
            msk0_random[msk0_random>0] = 255
            lbl_msk1_random = cv2.imread(fn_rand.replace('/images/', '/masks/').replace('_pre_', '_post_'), cv2.IMREAD_UNCHANGED)
            bbx1, bby1, bbx2, bby2 = rand_bbox((1024, 1024), lam)
            img[bbx1:bbx2, bby1:bby2, :] = img_random[bbx1:bbx2, bby1:bby2, :]
            img2[bbx1:bbx2, bby1:bby2, :] = img2_random[bbx1:bbx2, bby1:bby2, :]
            msk0[bbx1:bbx2, bby1:bby2] = msk0_random[bbx1:bbx2, bby1:bby2]
            lbl_msk1[bbx1:bbx2, bby1:bby2] = lbl_msk1_random[bbx1:bbx2, bby1:bby2]

        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255
        msk1[lbl_msk1 == 1] = 255

        if random.random() > 0.5:
            img = img[::-1, ...]
            img2 = img2[::-1, ...]
            msk0 = msk0[::-1, ...]
            msk1 = msk1[::-1, ...]
            msk2 = msk2[::-1, ...]
            msk3 = msk3[::-1, ...]
            msk4 = msk4[::-1, ...]

        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                img2 = np.rot90(img2, k=rot)
                msk0 = np.rot90(msk0, k=rot)
                msk1 = np.rot90(msk1, k=rot)
                msk2 = np.rot90(msk2, k=rot)
                msk3 = np.rot90(msk3, k=rot)
                msk4 = np.rot90(msk4, k=rot)
                    
        if random.random() > 0.8:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            img2 = shift_image(img2, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)
            msk1 = shift_image(msk1, shift_pnt)
            msk2 = shift_image(msk2, shift_pnt)
            msk3 = shift_image(msk3, shift_pnt)
            msk4 = shift_image(msk4, shift_pnt)
            
        if random.random() > 0.2:
            rot_pnt =  (img.shape[0] // 2 + random.randint(-320, 320), img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                img2 = rotate_image(img2, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)
                msk1 = rotate_image(msk1, angle, scale, rot_pnt)
                msk2 = rotate_image(msk2, angle, scale, rot_pnt)
                msk3 = rotate_image(msk3, angle, scale, rot_pnt)
                msk4 = rotate_image(msk4, angle, scale, rot_pnt)

        crop_size = input_shape[0]
        if random.random() > 0.1:
            crop_size = random.randint(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 10)
        for i in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk2[y0:y0+crop_size, x0:x0+crop_size].sum() * 5 + msk3[y0:y0+crop_size, x0:x0+crop_size].sum() * 5 + msk4[y0:y0+crop_size, x0:x0+crop_size].sum() * 2 + msk1[y0:y0+crop_size, x0:x0+crop_size].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]
        msk1 = msk1[y0:y0+crop_size, x0:x0+crop_size]
        msk2 = msk2[y0:y0+crop_size, x0:x0+crop_size]
        msk3 = msk3[y0:y0+crop_size, x0:x0+crop_size]
        msk4 = msk4[y0:y0+crop_size, x0:x0+crop_size]
        
        if crop_size != input_shape[0]:
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, input_shape, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, input_shape, interpolation=cv2.INTER_LINEAR)
            msk1 = cv2.resize(msk1, input_shape, interpolation=cv2.INTER_LINEAR)
            msk2 = cv2.resize(msk2, input_shape, interpolation=cv2.INTER_LINEAR)
            msk3 = cv2.resize(msk3, input_shape, interpolation=cv2.INTER_LINEAR)
            msk4 = cv2.resize(msk4, input_shape, interpolation=cv2.INTER_LINEAR)
            

        if random.random() > 0.96:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.96:
            img2 = shift_channels(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.96:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.96:
            img2 = change_hsv(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.9:
            if random.random() > 0.96:
                img = clahe(img)
            elif random.random() > 0.96:
                img = gauss_noise(img)
            elif random.random() > 0.96:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.9:
            if random.random() > 0.96:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img = contrast(img, 0.9 + random.random() * 0.2)

        if random.random() > 0.9:
            if random.random() > 0.96:
                img2 = clahe(img2)
            elif random.random() > 0.96:
                img2 = gauss_noise(img2)
            elif random.random() > 0.96:
                img2 = cv2.blur(img2, (3, 3))
        elif random.random() > 0.9:
            if random.random() > 0.96:
                img2 = saturation(img2, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img2 = brightness(img2, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img2 = contrast(img2, 0.9 + random.random() * 0.2)

                
        if random.random() > 0.96:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        if random.random() > 0.96:
            el_det = self.elastic.to_deterministic()
            img2 = el_det.augment_image(img2)

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        msk[..., 0] = True
        msk[..., 1] = dilation(msk[..., 1], square(5))
        msk[..., 2] = dilation(msk[..., 2], square(5))
        msk[..., 3] = dilation(msk[..., 3], square(5))
        msk[..., 4] = dilation(msk[..., 4], square(5))
        msk[..., 1][msk[..., 2:].max(axis=2)] = False
        msk[..., 3][msk[..., 2]] = False
        msk[..., 4][msk[..., 2]] = False
        msk[..., 4][msk[..., 3]] = False
        msk[..., 0][msk[..., 1:].max(axis=2)] = False
        msk = msk * 1

        lbl_msk = msk.argmax(axis=2)

        img = np.concatenate([img, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}
        return sample

class ValData(Dataset):
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]

        fn = test_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img2 = cv2.imread(fn.replace('_pre_', '_post_'), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_', '_post_'), cv2.IMREAD_UNCHANGED)
        msk0[msk0>0] = 255

        lbl_msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_', '_post_'), cv2.IMREAD_UNCHANGED)
        msk_loc = cv2.imread(path.join(loc_folder, '{0}'.format(fn.split('/')[-1].replace('.png', '_part1.png'))), cv2.IMREAD_UNCHANGED) > 140 #(0.3*255)
        
        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk1[lbl_msk1 == 1] = 255
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        msk = msk * 1

        lbl_msk = msk[..., 1:].argmax(axis=2)
        
        img = np.concatenate([img, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn, 'msk_loc': msk_loc}
        return sample


def validate(model, data_loader, logger):
    dices0 = []

    tp = np.zeros((5,))
    fp = np.zeros((5,))
    fn = np.zeros((5,))
    _thr = 0.3

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msks = sample["msk"].numpy()
            lbl_msk = sample["lbl_msk"].numpy()
            imgs = sample["img"].cuda(non_blocking=True)
            msk_loc = sample["msk_loc"].numpy() * 1
            out, _ = model(imgs)

            msk_pred = msk_loc
            msk_damage_pred = torch.softmax(out, dim=1).cpu().numpy()[:, 1:, ...]
            
            for j in range(msks.shape[0]):      

                targ = lbl_msk[j][msks[j, 0] > 0]
                pred = msk_damage_pred[j].argmax(axis=0)
                pred = pred * (msk_pred[j] > _thr)
                pred = pred[msks[j, 0] > 0]
                for c in range(4):
                    tp[c] += np.logical_and(pred == c, targ == c).sum()
                    fn[c] += np.logical_and(pred != c, targ == c).sum()
                    fp[c] += np.logical_and(pred == c, targ != c).sum()
    
    oa = 2 * np.sum(tp) / (2 * np.sum(tp) + np.sum(fp) + np.sum(fn))

    f1_sc = np.zeros((4,))
    for c in range(4):
        f1_sc[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])

    f1 = 4 / np.sum(1.0 / (f1_sc + 1e-6))

    sc = 0.3 * oa + 0.7 * f1
    logger.info("Val Score: F1: {}, F1_0: {}, F1_1: {}, F1_2: {}, F1_3: {}".format(f1, f1_sc[0], f1_sc[1], f1_sc[2], f1_sc[3]))
    print("Val Score: {}, OA: {}, F1: {}, F1_0: {}, F1_1: {}, F1_2: {}, F1_3: {}".format(sc, oa, f1, f1_sc[0], f1_sc[1], f1_sc[2], f1_sc[3]))
    #return oa
    return sc



def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch, logger):
    model = model.eval()
    d = validate(model, data_loader=data_val, logger=logger)

    if d > best_score:  
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, path.join(models_tune_folder, snapshot_name + '_best'))
        best_score = d

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(log_folder, exist_ok=True)
    makedirs(models_folder, exist_ok=True)
    
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'
    cudnn.benchmark = True

    batch_size = 5
    batch_size_unlabel = 5
    val_batch_size = 5
    
    snapshot_name = 'res50_cls_cce_{}_tuned_{}'.format(seed, disaster)

    file_classes = []
    for fn in tqdm(train_files):
        fl = np.zeros((4,), dtype=bool) 
        msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_', '_post_'), cv2.IMREAD_UNCHANGED)
        
        for c in range(1, 5):
            fl[c-1] = c in msk1
        file_classes.append(fl)  
    file_classes = np.asarray(file_classes)

    train_idxs0 = np.arange(train_len)
    val_idxs = np.arange(test_len)

    train_idxs = []
    for i in train_idxs0:
        train_idxs.append(i)
        if file_classes[i, 1:].max():
            train_idxs.append(i)
    low1 = len(train_idxs)

    for i in train_idxs0:
        if file_classes[i, 1:3].max():
            train_idxs.append(i)
    high1 =len(train_idxs)

    train_idxs = np.asarray(train_idxs)


    # supervised dataloader
    trainset_l = TrainData(train_idxs, low1, high1)
    train_data_loader = DataLoader(trainset_l, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=False, drop_last=True)


    # unlabeled self-supervised dataloader
    trainset_u = SemiCDDataset(root = "data/{}/unlabel".format(disaster), unlabel_fl = unlabel_files,
                             size=512,mode="unlabel")
    semisup_loader = DataLoader(trainset_u, batch_size=batch_size_unlabel, num_workers=1, shuffle=True, pin_memory=False)
    
    data_semival = ValData(val_idxs)
    val_data_loader = DataLoader(data_semival, batch_size=val_batch_size, num_workers=1, shuffle=False, pin_memory=False)
    
    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(test_files) // val_batch_size

    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)


    model = SeResNext50_Unet_Double().cuda()
    params = model.parameters()

    optimizer = AdamW(params, lr=0.00005, weight_decay=1e-6)
    
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1, 5, 11, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.5)

    model = nn.DataParallel(model).cuda()
    
    snap_to_load = 'res50_cls_cce_{}_tuned_best'.format(seed)
    
    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict) 
    print("loaded checkpoint '{}' (epoch {}, best_score {})"
            .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
    del loaded_dict
    del sd
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    # model_teacher
    model_teacher = copy.deepcopy(model)
    for p in model_teacher.parameters():
        p.requires_grad = False


    cfg = yaml.load(open("semisup_utils/config.yaml"), Loader=yaml.Loader)

    # supervised loss
    loss_type = cfg["trainer"]["supervised"]["loss_type"]
    seg_loss = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).cuda()
    ce_loss = nn.CrossEntropyLoss().cuda()

    # unsupervised loss
    criterion_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()

    loader = zip(train_data_loader, semisup_loader, semisup_loader)

    best_score = 0
    torch.cuda.empty_cache()


    # build class-wise memory bank
    memobank = []
    queue_ptrlis = []
    queue_size = []
    for i in range(cfg["net"]["num_classes"]):
        memobank.append([torch.zeros(0, 256)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # build prototype
    prototype = torch.zeros(
        (
            cfg["net"]["num_classes"],
            cfg["trainer"]["contrastive"]["num_queries"],
            1,
            256,
        )
    ).cuda()

    total_epoch = cfg["trainer"]["epochs"]
    sup_only_epoch = cfg["trainer"]["supervised"]["sup_only_epoch"]

    for epoch in range(total_epoch):
        
        total_loss = AverageMeter()
        total_loss_sup = AverageMeter() 
        total_loss_contra = AverageMeter() 

        losses = AverageMeter()
        losses1 = AverageMeter()
        dices = AverageMeter()

        loader = zip(train_data_loader, semisup_loader)
        iterator = tqdm(loader, total=len(train_data_loader))
        for step, ((img_sup), (imgA_u_w, imgB_u_w)) in enumerate(iterator):
            
            i_iter = epoch * len(train_data_loader) + step
            
            imgA_u_w, imgB_u_w = imgA_u_w.cuda(), imgB_u_w.cuda()
            img_u_w = torch.cat((imgA_u_w, imgB_u_w), dim=1)   
            model.train()
            
            imgs_x = img_sup["img"].cuda(non_blocking=True)
            msks_x = img_sup["msk"].cuda(non_blocking=True)
            lbl_msk = img_sup["lbl_msk"].cuda(non_blocking=True)
            
            image_all = torch.cat((imgs_x, img_u_w), dim=0) 
            num_labeled = len(imgs_x)    
            pred_all, rep_all = model(image_all)
            preds_x = pred_all[:num_labeled]
            rep_x= rep_all[:num_labeled]
            
            out = preds_x
            loss0 = seg_loss(out[:, 0, ...], msks_x[:, 0, ...])
            loss1 = seg_loss(out[:, 1, ...], msks_x[:, 1, ...])
            loss2 = seg_loss(out[:, 2, ...], msks_x[:, 2, ...])
            loss3 = seg_loss(out[:, 3, ...], msks_x[:, 3, ...])
            loss4 = seg_loss(out[:, 4, ...], msks_x[:, 4, ...])

            loss_ce = ce_loss(out, lbl_msk)

            loss_combo = 0.1 * loss0 + 0.2 * loss1 + 6 * loss2 + 3 * loss3 + 2 * loss4     
            
            if loss_type == "CrossEntropy":
                loss_sup = loss_ce     
            else:
                loss_sup = loss_combo

            with torch.no_grad():
                _probs = 1 - torch.sigmoid(out[:, 0, ...])
                dice_sc = 1 - dice_round(_probs, 1 - msks_x[:, 0, ...])

            losses.update(loss_sup.item(), imgs_x.size(0))
            dices.update(dice_sc, imgs_x.size(0))
         
            with torch.no_grad():
                # unsupervised loss
                drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)
                percent_unreliable = (100 - drop_percent) * (1 - epoch / total_epoch)  
                drop_percent = 100 - percent_unreliable

                # contrastive loss using unreliable pseudo labels
                image_all = torch.cat((imgs_x, img_u_w), dim=0) 
                num_labeled = len(imgs_x)    
                pred_all_teacher, rep_all_teacher = model_teacher(image_all)
                prob_all_teacher = F.softmax(pred_all_teacher, dim=1)   # softmax probabilities
                prob_l_teacher, prob_u_teacher = (
                    prob_all_teacher[:num_labeled],
                    prob_all_teacher[num_labeled:],
                )

                pred_u_teacher = pred_all_teacher[num_labeled:]
                pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
                logits_u, label_u = torch.max(pred_u_teacher, dim=1) 

            contra_flag = "none"
            if cfg["trainer"].get("contrastive", False): 
                cfg_contra = cfg["trainer"]["contrastive"]

                alpha_t = cfg_contra["low_entropy_threshold"] * (
                    1 - epoch / cfg["trainer"]["epochs"])

                model_teacher.train()
                with torch.no_grad():
                    prob_u = torch.softmax(pred_u_teacher, dim=1)   
                    entropy = -torch.sum(prob_u * torch.log(prob_u + 1e-10), dim=1)  

                    low_thresh = np.percentile(
                        entropy[label_u != 255].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (  
                        entropy.le(low_thresh).float() * (label_u != 255).bool()
                    )

                    high_thresh = np.percentile(
                        entropy[label_u != 255].cpu().numpy().flatten(),
                        100 - alpha_t,
                    )
                    high_entropy_mask = (  
                        entropy.ge(high_thresh).float() * (label_u != 255).bool()
                    )

                    label_l = lbl_msk   # labeled
                    low_mask_all = torch.cat(  
                        (
                            (label_l.unsqueeze(1) != 255).float(),
                            low_entropy_mask.unsqueeze(1),
                        )
                    )
                    
                    # entropy-high pixels 
                    if cfg_contra.get("negative_high_entropy", True):   #
                        contra_flag += " high"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    else: 
                        contra_flag += " low"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                torch.ones(logits_u.shape)
                                .float()
                                .unsqueeze(1)
                                .cuda(),
                            ),
                        )
                    high_mask_all = F.interpolate(
                        high_mask_all, size=preds_x.shape[2:], mode="nearest"
                    )  

                    #concat
                    label_l_small = label_onehot(label_l, cfg["net"]["num_classes"])
                    label_u_small = label_onehot(label_u, cfg["net"]["num_classes"])
                
                if not cfg_contra.get("anchor_ema", False):  # anchors for each batch 
                    new_keys, loss_contra = compute_contra_memobank_loss(
                        rep_all,
                        label_l_small.long(),
                        label_u_small.long(),
                        prob_l_teacher.detach(),  
                        prob_u_teacher.detach(), 
                        low_mask_all,
                        high_mask_all,
                        cfg_contra,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                    )
                else: 
                    prototype, new_keys, loss_contra = compute_contra_memobank_loss(   # momentum anchors -> prototype
                        rep_all,
                        label_l_small.long(),
                        label_u_small.long(),
                        prob_l_teacher.detach(),   
                        prob_l_teacher.detach(), 
                        low_mask_all,
                        high_mask_all,
                        cfg_contra,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                        prototype,
                        i_iter
                    )

                loss_contra = (
                    loss_contra
                    * cfg["trainer"]["contrastive"].get("loss_weight", 1)
                )

            else:
                loss_contra = 0 * rep_all.sum()

            del preds_x
            del rep_x
            del rep_all
            del rep_all_teacher
            del pred_all

            if epoch < sup_only_epoch:
                loss = loss_sup 
            else:
                loss = loss_sup + loss_contra

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 0.999)
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_sup.update(loss_sup.item())
            total_loss_contra.update(loss_contra.item())
            
            iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Loss Sup {loss_sup.val:.4f} ({loss_sup.avg:.4f}); Loss Unsup {loss_unsup.val:.4f} ({loss_unsup.avg:.4f})".format(
                    epoch, scheduler.get_lr()[-1], loss=total_loss, loss_sup=total_loss_sup, loss_unsup=total_loss_contra))
            
            # update teacher model with EMA
            ema_decay_origin = cfg["net"]["ema_decay"]
            i_iter = epoch * len(train_data_loader) + step
            if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
                with torch.no_grad():
                    ema_decay = min(1 - 1 / ( i_iter
                            - len(train_data_loader) * cfg["trainer"].get("sup_only_epoch", 1) + 1
                            ), ema_decay_origin,)
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = (
                            ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                        )
            
        scheduler.step(epoch)

        print("epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Loss Sup {loss_sup.val:.4f} ({loss_sup.avg:.4f}); Loss Unsup {loss_unsup.val:.4f} ({loss_unsup.avg:.4f})".format(
                    epoch, scheduler.get_lr()[-1], loss=total_loss, loss_sup=total_loss_sup, loss_unsup=total_loss_contra))

        torch.cuda.empty_cache()

        logger.info("====Epoch====: {}".format(epoch))
        best_score = evaluate_val(val_data_loader, best_score, model_teacher, snapshot_name, epoch, logger)


    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
