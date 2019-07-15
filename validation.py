from utils import *
from model import *
from torch.utils.data import DataLoader

import torch
import torchvision



def validation(model, valloader, ):
	loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
	jdict, stats, ap, ap_class = [], [], [], []
	iou_thres = 0.5
	conf_thres = 0.001
	nms_thres = 0.5
	model.eval()
    for i, (img, target) in enumerate(valloader):
    	imgs = img.to(args['gpu'])
    	targets = targe.to(args['gpu'])
    	_, _, width, height = imgs.shape
    	inf_out, train_out = model(imgs)
        loss, _ = compute_loss(train_out, targets, model, args['gpu'])

        output = non_max_suppression(inf_out, conf_thres=0.001, nms_thres=0.5)

        for si, pred in enumerate(output):
        	labels = targets[targets[:, 0] == si, 1:]
        	nl = len(labels)
        	tcls = labels[:, 0].tolist() if nl else []
        	seen += 1

        	if pred is None:
        		if nl:
        			stats.append(([], torch.Tensor(), torch.Tensor() tcls))
        		continue

        	correct = [0] * len(pred)
        	if nl:
        		detected = []
        		tcls_tensor = labels[:, 0]

        		tbox = wywh2xyxy(labels[:, 1:5])
        		tbox[:, [0, 2]] *= width
        		tbox[:, [1, 3]] *= height

        		for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):
        			if len(detected) == nl:
        				break
        			if pcls.item() not in tcls:
        				continue
        			# Best iou, index between pred and targets
        			m = (pcls == tcls_tensor).nonzero().view(-1)
        			iou, bi = bbox_iou(pbox, tbox[m]).max(0)

        			if iou > iou_thres and m[bi] not in detected:
        				correct[i] += 1
        				detected.append(m[bi])

        	stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]
    nt = np.bincount(stats[3].astype(np.int64), minlenth=nc)

    if len(stats):
    	p, r, ap, f1, ap_class = ap_per_class(*stats)
    	mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
	print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

	maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
	return (mp, mr, map, mf1, loss / len(dataloader)), maps