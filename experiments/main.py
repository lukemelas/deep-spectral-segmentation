import os
import sys
import math
import datetime
from pathlib import Path
from typing import Iterable, Optional
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from contexttimer import Timer
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb

import utils
from model import SimpleModel
from dataset import SimpleDataset, get_transforms


@hydra.main(config_path='config', config_name='default')
def main(cfg: DictConfig):

    # Accelerator
    accelerator = Accelerator(fp16=cfg.fp16, cpu=cfg.cpu)

    # Logging
    utils.setup_distributed_print(accelerator.is_local_main_process)
    if cfg.wandb and accelerator.is_local_main_process:
        wandb.init(project='template', name=cfg.name, job_type=cfg.job_type, config=cfg, save_code=True)

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
    if cfg.ema.enabled:
        from timm.utils import ModelEmaV2
        model_ema = ModelEmaV2(model, decay=cfg.ema.decay, device=cfg.ema.device)
        print('Initialized model EMA')
    else:
        model_ema = None
        print('Not using model EMA')

    # Optimizer and scheduler
    optimizer = utils.get_optimizer(cfg, model, accelerator)
    scheduler = utils.get_scheduler(cfg, optimizer)

    # Resume from checkpoint
    start_epoch = 0
    if cfg.checkpoint.resume:
        start_epoch = utils.resume_from_checkpoint(cfg, model, optimizer, scheduler, model_ema)

    # Transforms
    train_transform, val_transform = get_transforms(cfg)

    # Datasets
    with Timer(prefix="Loading data..."):
        dataset_train = SimpleDataset(root=cfg.data.train.root, transform=train_transform)
        dataset_val = SimpleDataset(root=cfg.data.val.root, transform=val_transform)
        dataset_vis = SimpleDataset(root=cfg.data.val.root, transform=val_transform)
        print(f'Dataset train size: {len(dataset_train):_}')
        print(f'Dataset val size: {len(dataset_val):_}')
        print(f'Dataset vis size: {len(dataset_val):_}')

        # # We can shuffle the visualization dataset if desired
        # indices = np.random.default_rng(seed=cfg.seed).permutation(len(dataset_val))
        # dataset_vis = Subset(dataset_vis, indices=indices)

    # Dataloader
    dataloader_train = DataLoader(dataset_train, shuffle=True, drop_last=True, **cfg.data.loader)
    dataloader_val = DataLoader(dataset_val, shuffle=False, drop_last=False, **cfg.data.loader)
    dataloader_vis = DataLoader(dataset_vis, shuffle=False, drop_last=False, **cfg.data.loader)
    print(f'Dataloader train size: {len(dataloader_train):_} (batch_size = {dataloader_train.batch_size})')
    print(f'Dataloader val size: {len(dataloader_val):_} (batch_size = {dataloader_val.batch_size})')
    print(f'Dataloader vis size: {len(dataloader_vis):_} (batch_size = {dataloader_vis.batch_size})')

    # Setup
    model, optimizer, dataloader_train, dataloader_val = accelerator.prepare(
        model, optimizer, dataloader_train, dataloader_val)

    # Shared training, evaluation, and visualization args
    kwargs = dict(
        cfg=cfg,
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        dataloader_vis=dataloader_vis,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        model_ema=model_ema)

    # Evaluation
    if cfg.job_type == 'eval':
        test_stats = evaluate(**kwargs)
        return test_stats['val_loss']

    # Training
    best_val_loss = 1e5
    print(f"Starting training at {datetime.datetime.now()}")
    for epoch in range(start_epoch, cfg.epochs):

        # Visualize (before training)
        visualize(**kwargs, num_batches=1, identifier=f'epoch_{epoch:04d}')

        # Single epoch of training
        train_one_epoch(**kwargs, epoch=epoch)

        # Learning rate scheduler - note that if you use a PyTorch scheduler
        # instead of a timm optimizer, you probably just want to call .step()
        # without the epoch argument
        scheduler.step(epoch + 1)

        # Save checkpoint on only 1 process
        if accelerator.is_local_main_process:
            print('Saving checkpoint...')
            checkpoint_dict = {
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': model_ema.state_dict() if model_ema else {},
                'cfg': cfg
            }
            torch.save(checkpoint_dict, 'checkpoint-latest.pth')

        # Evaluate
        if (epoch < 5) or (epoch % 5 == 0):
            test_stats = evaluate(**kwargs, header=f'Val [{epoch}]')
            if accelerator.is_local_main_process:
                if test_stats['val_loss'] < best_val_loss:
                    best_val_loss = test_stats['val_loss']
                    torch.save(checkpoint_dict, 'checkpoint-best.pth')
                if cfg.wandb:
                    wandb.log(test_stats)
                    wandb.run.summary["best_val_loss"] = best_val_loss


def train_one_epoch(
        *,
        cfg: DictConfig,
        model: torch.nn.Module,
        dataloader_train: Iterable,
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
        epoch: int,
        model_ema: Optional[object] = None,
        **_unused_kwargs):

    # Train mode
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    progress_bar = metric_logger.log_every(dataloader_train, cfg.logging.print_freq, header=f'Epoch: [{epoch}]')

    # Train
    for i, (inputs, target) in enumerate(progress_bar):
        if i >= cfg.get('limit_train_batches', math.inf):
            break

        # Forward
        output = model(inputs)
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
        if model_ema is not None and i % cfg.ema.update_every == 0:
            model_ema.update(model)

        # Logging
        log_dict = dict(lr=optimizer.param_groups[0]["lr"],
                        train_loss=loss_value, train_top1=acc1[0], train_top5=acc5[0])
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
        dataloader_val: Iterable,
        accelerator: Accelerator,
        header: str = 'Eval',
        **_unused_kwargs):

    # Eval mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    progress_bar = metric_logger.log_every(dataloader_val, cfg.logging.print_freq, header)

    # Evaluate
    for i, (inputs, target) in enumerate(progress_bar):
        if i >= cfg.get('limit_val_batches', math.inf):
            break

        # Forward
        output = model(inputs)
        loss = F.cross_entropy(output, target)
        torch.cuda.synchronize()

        # Measure accuracy
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        # Logging
        log_dict = dict(val_loss=loss.item(), top1=acc1[0], top5=acc5[0])
        metric_logger.update(**log_dict, n=len(inputs))  # update with batch size

    # Gather stats from all processes
    metric_logger.synchronize_between_processes(device=accelerator.device)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def visualize(
        *
        cfg: DictConfig,
        model: torch.nn.Module,
        dataloader_vis: Iterable,
        accelerator: Accelerator,
        identifier: str = '',
        num_batches: Optional[int] = None,
        **_unused_kwargs):

    # Eval mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    progress_bar = metric_logger.log_every(dataloader_vis, cfg.logging.print_freq, "Vis")

    # Visualize
    for batch_idx, (inputs, target) in enumerate(progress_bar):
        if num_batches is not None and batch_idx >= num_batches:
            break

        # Inverse normalization
        Inv = utils.NormalizeInverse(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std)
        image = Inv(inputs).clamp(0, 1)

        # Lists for logging
        img_list = [image]
        name_list = ['image']

        # Save only on the main single GPU
        if accelerator.is_local_main_process:
            wandb_log_dict = {}
            for x, name in zip(img_list, name_list):
                x = x.detach().cpu().requires_grad_(False)
                for i, x_i in enumerate(x):
                    ndarr = x_i.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
                    pil_img = Image.fromarray(ndarr)
                    filename = f'vis/{name}/img-{i}-{name}-{identifier}.png'
                    Path(filename).parent.mkdir(exist_ok=True, parents=True)
                    pil_img.save(filename)
                    if i < 2:  # log to weights and biases
                        wandb_log_dict[f'img-{i}-{name}'] = [wandb.Image(pil_img)]
            print(f'Saved visualization images (e.g. {filename})')
            if cfg.wandb:
                wandb.log(wandb_log_dict, commit=False)


if __name__ == '__main__':
    main()
