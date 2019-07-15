from PIL import Image
import torch
import torchvision.transforms as transforms
from model import *
from utils import *
from data_aug.bbox_util import *
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import random

hyp = {'xy': 0.2,  # xy loss gain
     'wh': 0.1,  # wh loss gain
     'cls': 0.04,  # cls loss gain
     'conf': 4.5,  # conf loss gain
     'iou_t': 0.5,  # iou target-anchor training threshold
     'lr0': 0.001,  # initial learning rate
     'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
     'momentum': 0.90,  # SGD momentum
     'weight_decay': 0.0005}  # optimizer weight decay

device = select_device()
test_set = torchvision.datasets.ImageFolder('./test_set')
test_loader = DataLoader(test_set, batch_size=16, num_workers=0, shuffle=False, pin_memory=True)

im = Image.open('test_image.jpg')
im = transforms.Resize((416, 416))(im)
im = transforms.ToTensor()(im)
im = im.unsqueeze(0).to(device)
print(im.shape)

chkpt = torch.load('./sex', map_location=device)
model = YOLO_v3(BasicBlock, [1,2,8,8,4], hyp)
model.load_state_dict(chkpt['model'])
model.eval()
model.cuda(device)

inf_out, _ = model(im)
output = non_max_suppression(inf_out, conf_thres=0.2, nms_thres=0.4)[0]
classes = load_classes('./data/coco.names')
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
im = im.to('cpu')
im = transforms.ToPILImage()(im[0])
im = np.asarray(im)
for *xyxy, conf, cls_conf, cls in output:
	print(xyxy, int(cls), conf, colors[int(cls)])
	im = draw_rect(im, xyxy, color=colors[int(cls)], label=classes[int(cls)])

plt.imshow(im)
plt.show()


