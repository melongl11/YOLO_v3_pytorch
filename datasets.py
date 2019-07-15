import os
import nltk
import torch
from torch.utils.data import Dataset
from torchvision.datasets import *
from pycocotools.coco import COCO
from utils import *
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import random


def transform_annotation(x):
    # convert the PIL image to a numpy array
    image = np.array(x[0])

    # get the bounding boxes and convert them into 2 corners format
    boxes = [a["bbox"] for a in x[1]]

    boxes = np.array(boxes)

    boxes = boxes.reshape(-1, 4)

    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    # grab the classes
    category_ids = np.array([coco91_to_coco80(a["category_id"]) for a in x[1]]).reshape(-1, 1)

    ground_truth = np.concatenate([boxes, category_ids], 1).reshape(-1, 5)

    nL = len(ground_truth)
    target = np.zeros((nL, 6))
    target[:, 1:] = ground_truth
    return image, ground_truth


class LoadCoCoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms, target_transform, det_transforms):
        super(LoadCoCoDataset, self).__init__(root, annFile, transforms, target_transform)
        self.det_transform = det_transforms

    def __getitem__(self, index):
        img, bboxes = super(LoadCoCoDataset, self).__getitem__(index)
        img, bboxes = transform_annotation([img, bboxes])
        img, bboxes = self.det_transform(img, bboxes)
        nL = len(bboxes)
        target = np.zeros((nL, 6))
        target[:, 1:] = bboxes
        img = transforms.ToTensor()(img)
        target = torch.from_numpy(target)
        nL = len(target)
        new_target = torch.zeros((nL, 6))
        if nL:
            target[:, 1:5] = xyxy2xywh(target[:, 1:5])
            target[:, [2, 4]] /= float(img.shape[1])
            target[:, [1, 3]] /= float(img.shape[2])
            new_target[:, 0] = target[:, 0].float()
            new_target[:, 1] = target[:, 5].float()
            new_target[:, 2:6] = target[:, 1:5].float()


        return img, new_target

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate_fn(batch):
        img, bboxes = list(zip(*batch))
        for i, l in enumerate(bboxes):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(bboxes, 0)

class TestLoader(DatasetFolder):
    def __init__(self, root, loader, extensions, det_transform, transform=None, target_transform=None):
        super(TestLoader, self).__init__(root, loader, extensions, transform, target_transform)
        self.det_transform = det_transform

    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = self.loader(path)
        bbox = [[0, 0, 0, 0]]
        bbox = np.array(bbox)
        bbox = bbox.reshape(-1, 4)
        cat = np.array([0]).reshape(-1, 1)

        gt = np.concatenate([bbox, cat]).reshape(-1, 5)
        nL = len(gt)
        target = np.zeros((nL, 6))
        target[:, 1:] = gt

        img, _ = self.det_transform(img, target)

        img = transforms.ToTensor()(img)

        return img, 0

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        img, _ = list(zip(*batch))

        return torch.stack(img, 0)
