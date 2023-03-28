#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import albumentations as A
from tqdm import tqdm
import cv2
import matplotlib as mpl
import random
from colorama import Fore, Style
import segmentation_models_pytorch as smp
from models import ResUnet
from multiprocessing import Process
import warnings

# mpl.use('TkAgg')
warnings.simplefilter('ignore')

c_ = Fore.GREEN
sr_ = Style.RESET_ALL


# In[2]:
# plan 1: folds=1, epochs=10, 目的查看loss下降情况 log:
# plan 2: 增加数据集 10张 步长[1, 2, 3]

class CFG:
    train_rate = 0.9
    vaild_rate = 0.1
    test_rate = 0.0
    train_bs = 2
    valid_bs = 2
    learning_rate = 0.000001
    pre_size = (512, 512)
    img_size = [512, 512]
    folds = 50
    epochs = 259
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


'''DATASETS'''
root = r"E:\DATASETS\UNZIP_datasets\data_kaggle\vesuvius-challenge-ink-detection"
train_path = root + '/train'
# train1 = root + '/train/1'
# surface_volume = train1 + '/surface_volume'
'''
1,2,3 train's surface_volume number is 64
deal 1 first and the other the same
'''


# In[4]:


def make_group(ls, start, group_num=10, step=1):
    group = []
    idx = start
    while group_num:
        group.append(ls[idx])
        group_num -= 1
        idx = (idx + step) % len(ls)
    return group


# In[5]:


def make_sorted_list(train_path, group_len=10, sets="1", step=1):
    groups = []
    path = train_path + os.sep + sets + "/surface_volume"
    path = Path(path)
    files = sorted([f for f in path.iterdir()])
    end = len(files) - 1
    for i in range(end):
        groups.append(make_group(files, i, group_len, step))
    return groups


# In[6]:


# 将一组group合成(b, c=10, h, w) 
def make_one_img(group):
    gp = list(map(lambda img: np.array(Image.open(img)), group))
    return gp


def make_datasets(train_path, groups, label="1"):
    lable_path = train_path + os.sep + str(label) + os.sep + "inklabels.png"
    # label = np.array(Image.open(lable_path))
    label = cv2.imread(lable_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    ret, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
    label = cv2.resize(label, CFG.pre_size)
    label = Image.fromarray(label)

    for group in groups:
        img = make_one_img(group)
        img = list(map(lambda image: cv2.resize(image, CFG.pre_size), img))
        img = list(map(lambda image: Image.fromarray(image), img))
        yield (img, label)



# In[7]:


gc.collect()

# In[8]:


# make 1, 2, 3 datasets
print("loading all datasets...")
groups1_1 = make_sorted_list(train_path, 10, "1", 1)
groups2_1 = make_sorted_list(train_path, 10, "2", 1)
groups3_1 = make_sorted_list(train_path, 10, "3", 1)

ds1_1 = make_datasets(train_path, groups1_1, "1")
ds2_1 = make_datasets(train_path, groups2_1, "2")
ds3_1 = make_datasets(train_path, groups3_1, "3")

groups1_2 = make_sorted_list(train_path, 10, "1", 2)
groups2_2 = make_sorted_list(train_path, 10, "2", 2)
groups3_2 = make_sorted_list(train_path, 10, "3", 2)

ds1_2 = make_datasets(train_path, groups1_2, "1")
ds2_2 = make_datasets(train_path, groups2_2, "2")
ds3_2 = make_datasets(train_path, groups3_2, "3")

groups1_3 = make_sorted_list(train_path, 10, "1", 3)
groups2_3 = make_sorted_list(train_path, 10, "2", 3)
groups3_3 = make_sorted_list(train_path, 10, "3", 3)

ds1_3 = make_datasets(train_path, groups1_3, "1")
ds2_3 = make_datasets(train_path, groups2_3, "2")
ds3_3 = make_datasets(train_path, groups3_3, "3")


# print(next(iter(ds1))[0])
# print(next(iter(ds1))[1])

# In[9]:


# show ds
def show_img_label(ds):
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(ds[0][0])  # 0.1 img not the whole img
    ax1.set_title("img")
    ax1.axis("off")
    ax2.imshow(ds[1])
    ax2.set_title("label")
    ax2.axis("off")
    plt.show()


# In[10]:


# show_img_label(next(iter(ds1)))


# In[11]:


# show_img_label(next(iter(ds2)))


# In[12]:


# show_img_label(next(iter(ds3)))


# In[13]:


gc.collect()

# In[14]:


print("Combine all datasets into one")

# print("test one datasets")
datasets = list(iter(ds1_1))
datasets.extend(list(iter(ds1_2)))
datasets.extend(list(iter(ds1_3)))
datasets.extend(list(iter(ds2_1)))
datasets.extend(list(iter(ds2_2)))
datasets.extend(list(iter(ds2_3)))
datasets.extend(list(iter(ds3_1)))
datasets.extend(list(iter(ds3_2)))
datasets.extend(list(iter(ds3_3)))
print(f"datasets number: {len(datasets)}")
random.shuffle(datasets)


# np_ds = np.array(datasets)
# np.save('test.npy',np_ds)
# test_npds = np.load('test.npy')

# In[15]:


class MyLazyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def com_imgs(self, imgs):
        imgs = list(map(lambda img: np.array(img, dtype="float32") // 255., imgs))
        return np.stack(imgs, axis=-1)

    def __getitem__(self, index):
        imgs = self.dataset[index][0]
        img = self.com_imgs(imgs)  # (H, W, 10)
        msk = np.array(self.dataset[index][1], dtype="float32") // 255.  # (H, W)
        if self.transform:
            data = self.transform(image=img, mask=msk)
            img = data['image']
            msk = data['mask']
        else:
            img = img
            msk = msk

        img = np.transpose(img, (2, 0, 1))
        img = torch.Tensor(img)
        msk = torch.Tensor(msk)
        return img, msk

    def __len__(self):
        return len(self.dataset)


# In[16]:


data_transforms = {
    "train": A.Compose([
        A.Resize(*CFG.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0], std=[1]),
    ]),

    "valid": A.Compose([
        A.Resize(*CFG.img_size),
        A.Normalize(mean=[0], std=[1]),
    ])
}

# In[17]:


ds_num = len(datasets)
train_num = int(ds_num * CFG.train_rate)
valid_num = int(ds_num * CFG.vaild_rate)
test_num = ds_num - train_num - valid_num
print(f"number of train:{train_num}")
print(f"number of valid:{valid_num}")
print(f"number of test:{test_num}")
train_ds = datasets[:train_num]
valid_ds = datasets[train_num: train_num + valid_num]
test_ds = datasets[train_num + valid_num:]

train_ds = MyLazyDataset(train_ds, data_transforms['train'])
valid_ds = MyLazyDataset(valid_ds, data_transforms['valid'])
test_ds = MyLazyDataset(test_ds)
print("Making dataloader...")
train_loader = DataLoader(train_ds, batch_size=CFG.train_bs, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=CFG.valid_bs, shuffle=False)
test_loader = DataLoader(test_ds)
# In[18]:
'''model'''


class InkDetection(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        # self.layer1_ = nn.Sequential(
        #     nn.Conv2d(in_channel, 1, kernel_size=1, stride=1),
        #     nn.ReLU()
        # )
        self.layer2_ = ResUnet(10, 1)

    def forward(self, inputs):
        # x = self.layer1_(inputs)
        x = self.layer2_(inputs)
        return x


def build_model():
    model = InkDetection(10)
    return model.to(CFG.device)


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# In[19]:


'''loss function'''


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


BCELoss = smp.losses.SoftBCEWithLogitsLoss()


def criterion(y_pred, y_true):
    return BCELoss(y_pred, y_true)


# In[20]:


def train(model):
    model.train()

    running_loss = 0.0
    dataset_size = 0

    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.learning_rate,
                                                    total_steps=CFG.epochs)
    pbar = tqdm(enumerate(train_loader), total=CFG.epochs)
    for i, (subvolumes, inklabels) in pbar:
        if i >= CFG.epochs:
            break
        optimizer.zero_grad()
        outputs = model(subvolumes.to(CFG.device, dtype=torch.float32))
        # outputs = torch.tensor(outputs, dtype=torch.float32)
        inklabels = torch.unsqueeze(inklabels, 1)

        loss = criterion(outputs, inklabels.to(CFG.device, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = subvolumes.size(0)
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')
        if i % 259 == 259 - 1:
            print(f"Loss:{(running_loss / dataset_size)}")
            running_loss = 0
            dataset_size = 0

    torch.cuda.empty_cache()
    return epoch_loss


def eval(model):
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    val_scores = []
    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for i, (subvolumes, inklabels) in pbar:
        outputs = model(subvolumes.to(CFG.device, dtype=torch.float32))
        # outputs = torch.tensor(outputs, dtype=torch.float32)
        inklabels = torch.unsqueeze(inklabels, 1)
        inklabels = inklabels.to(CFG.device, dtype=torch.float32)

        loss = criterion(outputs, inklabels)
        batch_size = subvolumes.size(0)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        val_dice = dice_coef(inklabels, outputs).cpu().detach().numpy()
        val_jaccard = iou_coef(inklabels, outputs).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')

    val_scores = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()

    return epoch_loss, val_scores


def run_training():
    model = build_model()
    # for p in model.parameters():
    #     print(p.requires_grad)
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    best_dice = -np.inf
    for fold in range(CFG.folds):
        # print(f'#' * 15)
        # print(f'### Fold: {fold}')
        # print(f'#' * 15)
        train(model)
        val_loss, val_scores = eval(model)

        val_dice, val_jaccard = val_scores
        print(f"val_dice:{val_dice}")
        print(f"val_jaccard:{val_jaccard}")
        # print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
        # if val_dice >= best_dice:
        #     print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
        #     best_dice = val_dice
        #     best_jaccard = val_jaccard
        #     torch.save(model.state_dict(), PATH)
        #     print(f"Model Saved{sr_}")
    PATH = f"last_epoch-{fold:02d}.pt"
    torch.save(model.state_dict(), PATH)
    # print("Best Score: {:.4f}".format(best_jaccard))


'''prediction'''
# test_path = r"E:\DATASETS\UNZIP_datasets\data_kaggle\vesuvius-challenge-ink-detection\test"
# def make_test_datasets(test_path):
#     mask = test_path + os.sep + "mask.png"
#
#
# def prediction(model, test_path):
#     model.eval()
#     a_groups = make_sorted_list(test_path, 10, "a", 2)



if __name__ == '__main__':
    run_training()
