import argparse
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.cuda.amp import GradScaler

from wann.data import CrystalGraphPreprocessedDataset, get_train_val_dataloader
from wann.model import Model
from wann.nn import ALIGNNPyG


def main():

    best_loss = 1e6
    best_mae = 1e6

    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()

    add = parser.add_argument

    add('--d_model', type=int, default=16)
    add('--h_dim', type=int, default=32)
    add('--num_layers', type=int, default=1)
    add('--conv_type', type=str, default='cgcnn')
    add('--n_heads', type=int, default=None)

    add('--num_workers', type=int, default=20)
    add('--drop_last', type=bool, default=False)
    add('--pin_memory', type=bool, default=True)
    add('--batch_size', '-bs', type=int, default=8)
    add('--dataset_ratio', '-dr', type=float, default=1)
    add('--train_ratio', type=float, default=0.8)
    add('--start_epoch', type=int, default=1)
    add('--epochs', '-e', type=int, default=200)

    add('--use_lr_scheduler', action='store_true')
    add('--weight_init', action='store_true')
    add('--warmup_ratio', '-wr', type=float, default=0.2)

    add('--optim', type=str, default='adamw')
    add('--learning_rate', '-lr', type=float, default=1e-3)
    add('--weight_decay', '-wd', type=float, default=0.0)

    add('--gpu_idx', type=int, default=0)
    add('--seed', type=int, default=0)
    add('--deterministic', type=bool, default=True)
    add('--resume', type=str, default=True)
    add('--log_wandb', action='store_true')
    add('--group', type=str, default=True)
    add('--no_cuda', action='store_true')

    args = parser.parse_args()

    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    torch.autograd.set_detect_anomaly(True)

    dataset = CrystalGraphPreprocessedDataset('dataset/MoS2_96/processed_dataset')
    # data0 = dataset[0]

    # model = Model(
    #     node_dim=data0.x.size(-1),
    #     edge_dim=data0.edge_attr.size(-1),
    #     d_model=args.d_model,
    #     h_dim=args.h_dim,
    #     num_layers=args.num_layers,
    #     conv_type=args.conv_type,
    #     n_heads=args.n_heads
    # )
    model = ALIGNNPyG(
        alignn_layers=4,
        gcn_layers=4,
        atom_input_features=92,
        edge_input_features=80,
        triplet_input_features=40,
        embedding_features=64,
        hidden_features=256,
        output_features=64
    )

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_idx))

    train_loader, val_loader = get_train_val_dataloader(
        dataset, args.batch_size, args.train_ratio, args.seed,
        args.num_workers, args.drop_last, args.pin_memory,
    )

    model.to(device)

    if args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-9, weight_decay=args.weight_decay)
    else:
        raise ValueError('Optimizer must be SGD, Adam or AdamW.')

    if args.resume:
        print("=> loading checkpoint")
        checkpoint = torch.load('checkpoints/model_best_114q0gb0.pth.tar')
        args.start_epoch = checkpoint['epoch'] + 1
        args.best_loss = checkpoint['best_loss']
        args.best_mae = checkpoint['best_mae']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    if args.log_wandb:
        wandb.init(project="wannier", entity="wayne833", config=args.__dict__)
        wandb.watch(model, log='all', log_freq=100)

    if args.use_lr_scheduler:
        total_steps = int(args.epochs * len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps, anneal_strategy='linear',
            div_factor=1e6, final_div_factor=1e6, pct_start=args.warmup_ratio
        )
    else:
        scheduler = None

    rid = wandb.run.id if args.log_wandb else None

    for epoch in range(args.start_epoch, args.epochs + 1):
        # train(model, train_loader, device, optimizer, epoch, args.log_wandb, scheduler)
        val_loss, val_mae = evaluate(model, val_loader, device, epoch, args.log_wandb)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        if val_mae < best_mae:
            best_mae = val_mae

        if args.log_wandb:
            wandb.log({'best_loss': best_loss, 'best_mae': best_mae, 'epoch': epoch})

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'best_mae': best_mae,
            'optimizer': optimizer.state_dict()}, is_best, rid)


def train(model, train_loader, device, optimizer, epoch, log_wandb, scheduler=None):
    batch_time = AverageMeter('Batch', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    maes = AverageMeter('MAE', ':.4f')

    if not log_wandb:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, maes],
            prefix='Epoch: [{}]'.format(epoch))

    model.train()

    end = time.time()

    for i, data in enumerate(train_loader):

        data_time.update(time.time() - end)

        data = data.to(device, non_blocking=True)

        # if scaler is not None:
        #     with torch.autocast(device_type=device, dtype=torch.float16):
        #         pred = model(data)
        #         loss = F.mse_loss(pred, data.y)
        #         mae = F.l1_loss(pred, data.y)
        #
        # else:
        pred = model(data)
        # loss = F.mse_loss(pred, data.y)
        # mae = F.l1_loss(pred, data.y)
        mask = torch.logical_and(data.target_mask, data.target.abs() > 0.1)
        pred = pred[mask]
        target = data.target[mask]
        loss = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)

        num = pred.numel()

        losses.update(loss.item(), num)
        maes.update(mae.item(), num)

        optimizer.zero_grad()
        # if scaler is not None:
        #     scaler.scale(loss).backward()
        #
        #     scaler.unscale_(optimizer)
        #     # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        #
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if log_wandb:
            wandb.log({'train_loss': losses.avg,
                       'train_mae': maes.avg,
                       'epoch': epoch,
                       'lr': optimizer.param_groups[0]['lr'],
                       # 'lr-scheduler': scheduler.get_last_lr()[0]
                       })
        else:
            if i % 10 == 0:
                progress.display(i)


def evaluate(model, val_loader, device, epoch, log_wandb):
    batch_time = AverageMeter('Batch', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    maes = AverageMeter('MAE', ':.4f')

    if not log_wandb:
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, maes],
            prefix='Val: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):

            data = data.to(device, non_blocking=True)
            pred = model(data)

            # loss = F.mse_loss(pred, data.y)
            # mae = F.l1_loss(pred, data.y)
            mask = torch.logical_and(data.target_mask, data.target.abs() > 0.1)
            pred = pred[mask]
            target = data.target[mask]
            loss = F.mse_loss(pred, target)
            mae = F.l1_loss(pred, target)
            num = pred.numel()

            losses.update(loss.item(), num)
            maes.update(mae.item(), num)

            batch_time.update(time.time() - end)
            end = time.time()

            if log_wandb:
                wandb.log({'val_loss': losses.avg,
                           'val_mae': maes.avg,
                           'epoch': epoch})
            else:
                if i % 10 == 0:
                    progress.display(i)

    return losses.avg, maes.avg


def plot(t, p, mae):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.scatter(t, p, s=2)
    plt.plot([-3, 2], [-3, 2], '--', c='grey')
    plt.title(f'MAE: {mae}')
    plt.show()


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, rid=None):
    if rid is not None:
        filename = 'checkpoints/checkpoint_{}.pth.tar'.format(rid)
        bestname = 'checkpoints/model_best_{}.pth.tar'.format(rid)
    else:
        filename = 'checkpoints/checkpoint.pth.tar'
        bestname = 'checkpoints/model_best.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


if __name__ == '__main__':
    main()
