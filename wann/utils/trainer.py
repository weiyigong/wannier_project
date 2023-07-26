import os
import shutil
import time
import wandb
import matplotlib.pyplot as plt

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .meter import AverageMeter, ProgressMeter


def get_optimizer(parameters, optim_type, learning_rate, weight_decay, momentum=None):
    if optim_type == 'SGD':
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optim_type == 'Adam':
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optim_type == 'AdamW':
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {optim_type}')


def run_epoch(model, loader, epoch, device, optimizer, scheduler=None, train=True, log_wandb=False):
    meters = [AverageMeter(name) for name in ['Batch', 'Data', 'Loss', 'MAE']]

    prefix = 'Epoch: [{}]' if train else 'Val: [{}]'
    progress = ProgressMeter(len(loader), meters, prefix=prefix.format(epoch))

    if train:
        model.train()
    else:
        model.eval()

    end = time.time()

    with torch.set_grad_enabled(train):
        for i, data in enumerate(loader):

            meters[1].update(time.time() - end)

            data = data.to(device, non_blocking=True)

            pred = model(data)
            mask = torch.logical_and(data.target_mask, data.target.abs() > 0.1)
            pred = pred[mask]
            target = data.target[mask]

            loss = F.mse_loss(pred, target)
            mae = F.l1_loss(pred, target)
            num = pred.numel()

            meters[2].update(loss.item(), num)
            meters[3].update(mae.item(), num)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            meters[0].update(time.time() - end)
            end = time.time()

            if scheduler is not None:
                scheduler.step()

            stage_str = 'train' if train else 'val'

            if log_wandb:
                wandb.log({
                    f'{stage_str}_loss': meters[2].avg, f'{stage_str}_mae': meters[3].avg,
                    'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']})
            else:
                if i % 10 == 0:
                    progress.display(i)

    return meters[2].avg, meters[3].avg


def save_checkpoint(state, is_best, rid=None):
    os.makedirs('checkpoints', exist_ok=True)

    filename = 'checkpoints/checkpoint.pth.tar' if rid is None else f'checkpoints/checkpoint_{rid}.pth.tar'
    bestname = 'checkpoints/model_best.pth.tar' if rid is None else f'checkpoints/model_best_{rid}.pth.tar'

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, bestname)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0, 1 / math.sqrt(m.embedding_dim))
