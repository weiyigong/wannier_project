import argparse
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.cuda.amp import GradScaler

from data import CrystalGraphPreprocessedDataset, get_train_val_dataloader
from model import Model


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--drop_last', default=False, type=bool)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--dataset_ratio', '-dr', type=float, default=0.1)
    # training
    parser.add_argument('--train_ratio', default=0.8, type=float)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--epochs', '-e', default=1, type=int)
    parser.add_argument('--use_lr_scheduler', action='store_true')
    parser.add_argument('--warmup_ratio', '-wr', default=0.2, type=float)
    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=0., type=float)

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_idx', default=0, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--log_wandb', action='store_true')

    args = parser.parse_args()

    dataset = CrystalGraphPreprocessedDataset('../dataset/processed_dataset')
    data0 = dataset[0]

    model = Model(
        node_dim=data0.x.size(-1),
        edge_dim=data0.edge_attr.size(-1),
        d_model=64,
        h_dim=128,
        num_layers=1)

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_idx))

    best_loss = 1e6
    best_mae = 1e6

    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader = get_train_val_dataloader(
        dataset, args.batch_size, args.train_ratio, args.seed,
        args.num_workers, args.drop_last, args.pin_memory,
    )

    model.to(device)

    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.wd)
    elif args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-9, weight_decay=args.wd)
    else:
        raise ValueError('Optimizer must be SGD, Adam or AdamW.')

    if args.resume:
        print("=> loading checkpoint")
        checkpoint = torch.load('checkpoints/model_best_ook1dh8p.pth.tar')
        args.start_epoch = checkpoint['epoch'] + 1
        args.best_loss = checkpoint['best_loss']
        args.best_mae = checkpoint['best_mae']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    if args.log_wandb:
        wandb.init(project="bandformer", entity="wayne833", config=args.__dict__)
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

    scaler = GradScaler() if args.amp else None

    for epoch in range(args.start_epoch, args.epochs + 1):
        train(model, train_loader, device, optimizer, epoch, args.log_wandb, scheduler, scaler)
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


def train(model, train_loader, device, optimizer, epoch, log_wandb, scheduler=None, scaler=None):
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

        if scaler is not None:
            with torch.autocast(device_type=device, dtype=torch.float16):
                pred = model(data)
                loss = F.mse_loss(pred, data.y)
                mae = F.l1_loss(pred, data.y)

        else:
            pred = model(data)
            loss = F.mse_loss(pred, data.y)
            mae = F.l1_loss(pred, data.y)

        num = pred.numel()

        losses.update(loss.item(), num)
        maes.update(mae.item(), num)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            scaler.step(optimizer)
            scaler.update()
        else:
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
            if i % 100 == 0:
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

            loss = F.mse_loss(pred, data.y)
            mae = F.l1_loss(pred, data.y)
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
                if i % 100 == 0:
                    progress.display(i)

    return losses.avg, maes.avg


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
