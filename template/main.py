import os
import sys
import math
import datetime
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.utils import ModelEmaV2
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from accelerate import Accelerator
from contexttimer import Timer
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb

import utils
from model import SimpleModel
from dataset import SimpleDataset


@hydra.main(config_path='config', config_name='default')
def main(cfg: DictConfig):

    # Accelerator
    accelerator = Accelerator(fp16=cfg.fp16, cpu=cfg.cpu)
    
    # Logging
    utils.setup_distributed_print(accelerator.is_local_main_process)
    if cfg.wandb and accelerator.is_local_main_process:
        wandb.init(project='template', name=cfg.name, job_type='train', config=cfg, save_code=True)

    # Configuration
    print(OmegaConf.to_yaml(cfg))
    print(f'Current working directory: {os.getcwd()}')

    # Set random seed
    utils.set_seed(cfg.seed)

    # Model
    model = SimpleModel(**cfg.model)
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')

    # Exponential moving average of model parameters
    model_ema = None
    if cfg.ema.enabled:
        model_ema = ModelEmaV2(model, decay=cfg.ema.decay, device=cfg.ema.device)
        print('Initialized model EMA')
    
    # Optimizer and scheduler
    optimizer = create_optimizer(cfg.optimizer, model)
    scheduler, _ = create_scheduler(cfg.scheduler, optimizer)

    # Resume from checkpoint
    start_epoch = 0
    if cfg.checkpoint.resume:
        start_epoch = utils.resume_from_checkpoint(cfg, model, optimizer, scheduler, model_ema)
        
    # Transforms
    train_transform = A.Compose([
        A.RandomResizedCrop(cfg.data.transform.crop_size, cfg.data.transform.crop_size),
        A.Normalize(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(cfg.data.transform.resize_size, cfg.data.transform.resize_size),
        A.CenterCrop(cfg.data.transform.crop_size, cfg.data.transform.crop_size),
        A.Normalize(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std),
        ToTensorV2(),
    ])

    # Datasets
    with Timer(prefix="Loading data..."):
        dataset_train = SimpleDataset(root=cfg.data.train.root, transform=train_transform)
        dataset_val = SimpleDataset(root=cfg.data.val.root, transform=val_transform)
        print(f'Dataset train / val sizes: {len(dataset_train):_} / {len(dataset_val):_}')

    # Dataloader
    dataloader_train = DataLoader(dataset_train, shuffle=True, drop_last=True, **cfg.data.loader)
    dataloader_val = DataLoader(dataset_val, shuffle=False, drop_last=False, **cfg.data.loader)
    print(f'Dataloader train size: {len(dataloader_train):_} ({dataloader_train.batch_size})')
    print(f'Dataloader val size: {len(dataloader_val):_} ({dataloader_val.batch_size})')
    
    # Setup
    model, optimizer, dataloader_train, dataloader_val = accelerator.prepare(
        model, optimizer, dataloader_train, dataloader_val)

    # Evaluation
    if cfg.eval:
        test_stats = evaluate(cfg=cfg, model=model, loader=dataloader_val, accelerator=accelerator)
        return test_stats['top1']

    # Training
    best_top1 = 0
    print(f"Starting training at {datetime.datetime.now()}")
    for epoch in range(start_epoch, cfg.scheduler.epochs):

        # Single epoch of training
        train_stats = train_one_epoch(
            cfg=cfg,
            model=model,
            loader=dataloader_train,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            epoch=epoch,
            model_ema=model_ema)

        # Learning rate scheduler - note that if you use a PyTorch scheduler
        # instead of a timm optimizer, you probably just want to call .step()
        # without the epoch argument
        scheduler.step(epoch)

        # Checkpoint
        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'model_ema': model_ema.state_dict() if model_ema else {},
            'cfg': cfg
        }
        
        # Save on only 1 process
        if accelerator.is_local_main_process:
            print('Saving checkpoint...')
            torch.save(checkpoint_dict, 'checkpoint-latest.pth')

        # Evaluate
        if True:  # (5 < epoch < 275) and (epoch) % 5 != 0:
            test_stats = evaluate(cfg=cfg, model=model, loader=dataloader_val, accelerator=accelerator, header=f'Val [{epoch}]')
            if accelerator.is_local_main_process:
                if test_stats['top1'] > best_top1:
                    best_top1 = test_stats['top1']
                    torch.save(checkpoint_dict, 'checkpoint-best.pth')
                if cfg.wandb:
                    wandb.log(test_stats)
                    wandb.run.summary["best_top1"] = best_top1


def train_one_epoch(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    accelerator: Accelerator,
    epoch: int,
    model_ema: Optional[ModelEmaV2] = None):

    # Train mode
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # Train
    for i, (input, target) in enumerate(metric_logger.log_every(loader, cfg.logging.print_freq, header)):
        
        # Forward
        output = model(input)
        loss = F.cross_entropy(output, target)

        # Measure accuracy
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        # Exit if loss is NaN
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Loss scaling and backward
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        # Model EMA
        if model_ema is not None:
            model_ema.update(model)

        # Logging
        log_dict = dict(lr=optimizer.param_groups[0]["lr"], train_loss=loss_value, train_top1=acc1[0], train_top5=acc5[0])
        metric_logger.update(**log_dict)
        if cfg.wandb and accelerator.is_local_main_process:
            wandb.log(log_dict)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes(device=accelerator.device)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    loader: Iterable,
    accelerator: Accelerator,
    header: str = 'Eval'):

    # Eval mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    progress_bar = enumerate(metric_logger.log_every(loader, cfg.logging.print_freq, header))
    for i, (input, target) in progress_bar:

        # Forward
        output = model(input)
        loss = F.cross_entropy(output, target)
        torch.cuda.synchronize()

        # Measure accuracy
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        # Logging
        log_dict = dict(val_loss=loss.item(), top1=acc1[0], top5=acc5[0])
        metric_logger.update(**log_dict, n=len(input))  # update with batch size

    # Gather stats from all processes
    metric_logger.synchronize_between_processes(device=accelerator.device)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()
