import argparse
import time
import nltk
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from data_aug.data_aug import *
from model import *
from utils import *
from datasets import *
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

hyp = {'xy': 0.167,  # xy loss gain
       'wh': 0.09339,  # wh loss gain
       'cls': 0.03868,  # cls loss gain
       'conf': 4.546,  # conf loss gain
       'iou_t': 0.2454,  # iou target-anchor training threshold
       'lr0': 0.000198,  # initial learning rate
       'lrf': -5.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.95,  # SGD momentum
       'weight_decay': 0.0007838}  # optimizer weight decay


device = select_device()
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((416,416)),
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1)),
    torchvision.transforms.ToTensor()
])

det_tran = Sequence([RandomHorizontalFlip(1), RandomScale(0.3, diff=True), RandomHSV(10, 100, 100), Resize(416)])

# dataset = torchvision.datasets.CocoDetection(root='train2014', annFile='annotations/instances_train2014.json', transform=transforms, target_transform=transforms)
dataset = LoadCoCoDataset(root='train2014', annFile='annotations/instances_train2014.json', transforms=None, target_transform=None, det_transforms=det_tran)
dataloader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False, pin_memory=True, collate_fn=dataset.collate_fn)

img = None
target = None

for i, (imgs, targets) in enumerate(dataloader):
    img = imgs
    target = targets
    print(targets)
    break

img = torchvision.transforms.ToPILImage()(img[4])
img = np.asarray(img)
target = target[4].numpy()
print(target, target[:, 1:5])
plotted_img = draw_rect(img, target[:, 1:5])
plt.imshow(plotted_img)
plt.show()

