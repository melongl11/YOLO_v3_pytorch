import time
import torchvision
import torch.optim as optim
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from model import *
from utils import *
from datasets import *
from data_aug.data_aug import *
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
import torch.multiprocessing as mp

hyp = {'xy': 0.2,  # xy loss gain
       'wh': 0.1,  # wh loss gain
       'cls': 0.04,  # cls loss gain
       'conf': 4.5,  # conf loss gain
       'iou_t': 0.5,  # iou target-anchor training threshold
       'lr0': 0.001,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay


def main_worker(gpu, ngpus_per_node, args):
    start_epoch = 0
    epochs = 20
    args['gpu'] = gpu
    print(args['gpu'])
    # Model
    model = YOLO_v3(BasicBlock, [1, 2, 8, 8, 4], hyp)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in (0.8, 0.9)], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # load weight
    chkpt = torch.load('./weights/2019-07-27', map_location=gpu)
    model.load_state_dict(chkpt['model'])
    model.eval()
    if args['distributed']:
        args['rank'] = args['rank'] * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8888', world_size=args['world_size'],
                                rank=args['rank'])

        if args['gpu'] is not None:
            torch.cuda.set_device(args['gpu'])
            model.cuda(args['gpu'])

            args['batch_size'] = int(args['batch_size'] / ngpus_per_node)
            args['workers'] = int(args['workers'] / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']])
    else:
        torch.cuda.set_device(args['gpu'])
        model = model.cuda(args['gpu'])
    # Load data
    det_tran = Sequence([RandomScale(0.3), RandomHorizontalFlip(), RandomHSV(25, 100, 100), Resize(416)])
    trainset = LoadCoCoDataset(root=args['dataset_root'], annFile=args['ann_root'],
                               transforms=None, target_transform=None, det_transforms=det_tran)
    # valset = LoadCoCoDataset(root='coco/images/val2014', annFile='coco/annotations/instances_val2014.json', transforms=None, target_transform=None, det_transforms=det_tran)
    if args['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    else:
        train_sampler = None
        # val_sampler = None

    trainloader = DataLoader(trainset, batch_size=args['batch_size'], num_workers=args['workers'],
                             shuffle=(train_sampler is None), collate_fn=trainset.collate_fn, drop_last=True,
                             sampler=train_sampler)
    # valloader = DataLoader(valset, batch_size=args['batch_size'], num_workers=args['workers'], shuffle=(val_sampler is None), collate_fn=valset.collate_fn, drop_last=True, sampler=val_sampler)
    # Training
    total_time = time.time()
    best_loss = float('inf')
    mloss = torch.zeros(5).to(args['gpu'])
    t = time.time()
    for epoch in range(0, epochs):
        if args['distributed']:
            train_sampler.set_epoch(epoch)
            # val_sampler.set_epoch(epoch)

        model.train()
        scheduler.step()  # Update scheduler
        epoch_time = time.time()
        loss = 0
        for i, (img, target) in enumerate(trainloader):
            imgs = img.to(args['gpu'])
            targets = target.to(args['gpu'])

            # Model prediction
            pred = model(imgs)
            print(targets)
            # Compute loss, gradient
            loss, loss_items = compute_loss(pred, targets, model, args['gpu'])
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            mloss = (mloss * i + loss_items) / (i + 1)
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, len(trainloader) - 1), *mloss, len(targets), time.time() - t)
            t = time.time()
            print(s)

        # validation

        print('')
        print('%g epochs: %.3f hours.' % (epoch, (time.time() - total_time) / 3600))
    chkpt = {'epoch': epoch,
             'best_loss': best_loss,
             'model': model.module.state_dict() if type(
                 model) is nn.parallel.DistributedDataParallel else model.state_dict(),
             'optimizer': optimizer.state_dict()}

    # Save latest checkpoint
    torch.save(chkpt, './weights/2019-07-27')


if __name__ == '__main__':
    args = {'world_size': 1,
            'rank': 0,
            'workers': 1,
            'gpu': None,
            'distributed': False,
            'batch_size': 8,
            'validation_split': 0.1,
            'dataset_root': './train2014',
            'ann_root': './annotations/instances_train2014.json'}
    ngpus_per_node = torch.cuda.device_count()
    args['world_size'] = ngpus_per_node * args['world_size']
    if args['distributed']:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        gpu = select_device()
        main_worker(gpu, ngpus_per_node, args)
