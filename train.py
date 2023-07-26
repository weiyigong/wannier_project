import random

import numpy as np
import torch
import wandb

from wann.dataset import CrystalGraphPreprocessedDataset
from wann.nn import ALIGNNPyG
from wann.utils import *

best_loss = 1e6
best_mae = 1e6


def main(config_file):
    global best_loss, best_mae

    config = load_yaml_config(config_file)

    if config['environment']['deterministic']:
        random.seed(config['environment']['seed'])
        np.random.seed(config['environment']['seed'])
        torch.manual_seed(config['environment']['seed'])
        torch.cuda.manual_seed(config['environment']['seed'])
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    if config['resume']['resume_from'] is None:
        dataset = CrystalGraphPreprocessedDataset('dataset/processed_dataset',
                                                  dataset_ratio=config['training']['dataset_ratio'])
        model = ALIGNNPyG(**config['model'])
        train_loader, test_loader = get_train_test_dataloader(
            dataset,
            batch_size=config['training']['batch_size'],
            train_ratio=config['training']['train_ratio'],
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory'],
            seed=config['environment']['seed'])

        optimizer = get_optimizer(model.parameters(), **config['optim'])

    else:
        checkpoint = torch.load('checkpoints/{}'.format(config['resume']['resume_from']))
        prev_config = checkpoint['config']

        dataset = CrystalGraphPreprocessedDataset('dataset/processed_dataset',
                                                  dataset_ratio=prev_config['training']['dataset_ratio'])
        model = ALIGNNPyG(**prev_config['model'])
        model.load_state_dict(checkpoint['state_dict'])

        train_loader, test_loader = get_train_test_dataloader(
            dataset,
            batch_size=prev_config['training']['batch_size'],
            train_ratio=prev_config['training']['train_ratio'],
            num_workers=prev_config['training']['num_workers'],
            pin_memory=prev_config['training']['pin_memory'],
            seed=prev_config['environment']['seed'])

        optimizer = get_optimizer(model.parameters(), **prev_config['optim'])
        print(
            'Successfully loaded checkpoint {} (epoch {})'.format(config['resume']['resume_from'], checkpoint['epoch']))

    if config['training']['weight_init']:
        model.apply(weight_init)

    if config['environment']['no_cuda']:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{config["environment"]["gpu_idx"]}')
    model.to(device)

    if config['logging']['log_wandb']:
        wandb.init(project="wannier", entity="wayne833", group=config['logging']['group'], config=config)
        wandb.watch(model, log='all', log_freq=100)

    if config['training']['use_lr_scheduler']:
        total_steps = int(config['training']['epochs'] * len(train_loader))

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['optim']['learning_rate'], total_steps=total_steps, anneal_strategy='linear',
            div_factor=1e6, final_div_factor=1e6, pct_start=config['training']['warmup_ratio'])
    else:
        scheduler = None

    run_id = wandb.run.id if config['logging']['log_wandb'] else None

    for epoch in range(config['training']['start_epoch'], config['training']['epochs'] + 1):
        train_loss, train_mae = run_epoch(model, train_loader, epoch, device, optimizer, scheduler,
                                          train=True, log_wandb=config['logging']['log_wandb'])
        test_loss, test_mae = run_epoch(model, test_loader, epoch, device, optimizer, scheduler,
                                        train=False, log_wandb=config['logging']['log_wandb'])

        print('Average: Loss {:.4f}\tMAE {:.4f}'.format(test_loss, test_mae))

        is_best = test_mae < best_mae
        if is_best:
            best_mae = test_mae

        if test_loss < best_loss:
            best_loss = test_loss

        if config['logging']['log_wandb']:
            wandb.log({'best_loss': best_loss, 'best_mae': best_mae, 'epoch': epoch})

        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_loss': best_loss,
                         'best_mae': best_mae, 'optimizer': optimizer.state_dict(), 'config': config}, is_best, run_id)


if __name__ == '__main__':
    main('configs/test_cfg.yaml')
