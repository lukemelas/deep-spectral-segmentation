import datetime
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Iterable, Optional

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import util as utils
from dataset import get_datasets
from model import get_model


@hydra.main(config_path='config', config_name='train')
def main(cfg: DictConfig):

    # Accelerator
    accelerator = Accelerator(fp16=cfg.fp16, cpu=cfg.cpu)

    # Logging
    utils.setup_distributed_print(accelerator.is_local_main_process)
    if cfg.wandb and accelerator.is_local_main_process:
        wandb.init(name=cfg.name, job_type=cfg.job_type, config=OmegaConf.to_container(cfg), save_code=True, **cfg.wandb_kwargs)
        cfg = DictConfig(wandb.config.as_dict())  # get the config back from wandb for hyperparameter sweeps

    # Configuration
    print(OmegaConf.to_yaml(cfg))
    print(f'Current working directory: {os.getcwd()}')

    # Set random seed
    utils.set_seed(cfg.seed)

    # Create model
    model = get_model(**cfg.model)

    # Freeze layers, if desired
    if cfg.unfrozen_backbone_layers >= 0:
        num_unfrozen = None if (cfg.unfrozen_backbone_layers == 0) else (-cfg.unfrozen_backbone_layers)
        for module in list(model.backbone.children())[:num_unfrozen]:
            for p in module.parameters():
                p.requires_grad_(False)

    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')
    print(f'Backbone parameters (total): {sum(p.numel() for p in model.backbone.parameters()):_d}')
    print(f'Backbone parameters (train): {sum(p.numel() for p in model.backbone.parameters() if p.requires_grad):_d}')

    # Optimizer and scheduler
    optimizer = utils.get_optimizer(cfg, model, accelerator)
    scheduler = utils.get_scheduler(cfg, optimizer)

    # Resume from checkpoint and create the initial training state
    if cfg.checkpoint.resume:
        train_state: utils.TrainState = utils.resume_from_checkpoint(cfg, model, optimizer, scheduler, model_ema=None)
    else:
        train_state = utils.TrainState()  # start training from scratch

    # Data
    dataset_train, dataset_val, collate_fn = get_datasets(cfg)
    dataloader_train = DataLoader(dataset_train, shuffle=True, drop_last=True, 
        collate_fn=collate_fn, **cfg.data.loader)
    dataloader_val = DataLoader(dataset_val, shuffle=False, drop_last=False, 
        collate_fn=collate_fn, **{**cfg.data.loader, 'batch_size': 1})
    total_batch_size = cfg.data.loader.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    # SyncBatchNorm
    if accelerator.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Setup
    model, optimizer, dataloader_train = accelerator.prepare(model, optimizer, dataloader_train)

    # Exponential moving average of model parameters
    if cfg.ema.use_ema:
        from torch_ema import ExponentialMovingAverage
        model_ema = ExponentialMovingAverage((p for p in model.parameters() if p.requires_grad), decay=cfg.ema.decay)
        print('Initialized model EMA')
    else:
        model_ema = None
        print('Not using model EMA')

    # Shared training, evaluation, and visualization args
    kwargs = dict(
        cfg=cfg,
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        model_ema=model_ema,
        train_state=train_state)

    # Evaluation
    if cfg.job_type == 'generate':
        test_stats = generate(**kwargs)
        return 0

    # Evaluation
    if cfg.job_type == 'eval':
        test_stats = evaluate(**kwargs)
        return test_stats['val_loss']

    # Info
    print(f'***** Starting training at {datetime.datetime.now()} *****')
    print(f'    Dataset train size: {len(dataset_train):_}')
    print(f'    Dataset val size: {len(dataset_val):_}')
    print(f'    Dataloader train size: {len(dataloader_train):_}')
    print(f'    Dataloader val size: {len(dataloader_val):_}')
    print(f'    Batch size per device = {cfg.data.loader.batch_size}')
    print(f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}')
    print(f'    Gradient Accumulation steps = {cfg.gradient_accumulation_steps}')
    print(f'    Max optimization steps = {cfg.max_train_steps}')
    print(f'    Max optimization epochs = {cfg.max_train_epochs}')
    print(f'    Training state = {train_state}')

    # Evaluate masks before training
    if cfg.get('eval_masks_before_training', True):
        print('Evaluating masks before training...')
        if accelerator.is_main_process:
            evaluate(**kwargs, evaluate_dataset_pseudolabels=True)  # <-- to evaluate the self-training masks
        torch.cuda.synchronize()

    # Training loop
    while True:

        # Single epoch of training
        train_state = train_one_epoch(**kwargs)

        # Save checkpoint on only 1 process
        if accelerator.is_local_main_process:
            checkpoint_dict = {
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': train_state.epoch,
                'step': train_state.step,
                'best_val': train_state.best_val,
                'model_ema': model_ema.state_dict() if model_ema else {},
                'cfg': cfg
            }
            print(f'Saved checkpoint to {str(Path(".").resolve())}')
            accelerator.save(checkpoint_dict, 'checkpoint-latest.pth')
            if (train_state.epoch > 0) and (train_state.epoch % cfg.checkpoint_every == 0):
                accelerator.save(checkpoint_dict, f'checkpoint-{train_state.epoch:04d}.pth')

        # Evaluate
        if train_state.epoch % cfg.get('eval_every', 1) == 0:
            test_stats = evaluate(**kwargs) 
            if accelerator.is_local_main_process:
                if (train_state.best_val is None) or (test_stats['mIoU'] > train_state.best_val):
                    train_state.best_val = test_stats['mIoU']
                    torch.save(checkpoint_dict, 'checkpoint-best.pth')
                if cfg.wandb:
                    wandb.log(test_stats)
                    wandb.run.summary["best_mIoU"] = train_state.best_val
        
        # End training
        if ((cfg.max_train_steps is not None and train_state.step >= cfg.max_train_steps) or 
            (cfg.max_train_epochs is not None and train_state.epoch >= cfg.max_train_epochs)):
            print(f'Ending training at: {datetime.datetime.now()}')
            print(f'Final train state: {train_state}')
            sys.exit()


def train_one_epoch(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    dataloader_train: Iterable,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    scheduler: Callable,
    train_state: utils.TrainState,
    model_ema: Optional[object] = None,
    **_unused_kwargs
):

    # Train mode
    model.train()
    log_header = f'Epoch: [{train_state.epoch}]'
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('step', utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    progress_bar = metric_logger.log_every(dataloader_train, cfg.logging.print_freq, header=log_header)

    # Train
    for i, (images, _, pseudolabels, _) in enumerate(progress_bar):
        if i >= cfg.get('limit_train_batches', math.inf):
            break

        # Forward
        output = model(images)  # (B, C, H, W)

        # Cross-entropy loss
        loss = F.cross_entropy(output, pseudolabels)

        # Measure accuracy
        acc1, acc5 = utils.accuracy(output, pseudolabels, topk=(1, 5))

        # Exit if loss is NaN
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Loss scaling and backward
        accelerator.backward(loss)

        # Gradient accumulation, optimizer step, scheduler step
        if i % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            if cfg.scheduler.stepwise:
                scheduler.step()
            train_state.step += 1

            # Model EMA
            if model_ema is not None and (train_state.epoch % cfg.ema.update_every) == 0:
                model_ema.update((p for p in model.parameters() if p.requires_grad))

        # Logging
        log_dict = dict(
            lr=optimizer.param_groups[0]["lr"], step=train_state.step,
            train_loss=loss_value, sup_loss=sup_loss, con_loss=con_loss,
            train_top1=acc1[0], train_top5=acc5[0],
        )
        metric_logger.update(**log_dict)
        if cfg.wandb and accelerator.is_local_main_process:
            wandb.log(log_dict)

    # Scheduler
    if not cfg.scheduler.stepwise:
        scheduler.step()

    # Epoch complete
    train_state.epoch += 1

    # Gather stats from all processes
    metric_logger.synchronize_between_processes(device=accelerator.device)
    print("Averaged stats:", metric_logger)
    return train_state


@torch.no_grad()
def evaluate(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    dataloader_val: Iterable,
    accelerator: Accelerator,
    model_ema: Optional[object] = None,
    evaluate_dataset_pseudolabels: bool = False,
    **_unused_kwargs
):
    
    # To avoid CUDA errors on my machine
    torch.backends.cudnn.benchmark = False

    # Eval mode
    model.eval()
    torch.cuda.synchronize()
    eval_context = model_ema.average_parameters if cfg.ema.use_ema else nullcontext

    # Add background class
    n_classes = cfg.data.num_classes + 1

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    # Check
    assert dataloader_val.batch_size == 1, 'Please use batch_size=1 for val to compute mIoU'

    # Load all pixel embeddings
    all_preds = np.zeros((len(dataloader_val) * 500 * 500), dtype=np.float32)
    all_gt = np.zeros((len(dataloader_val) * 500 * 500), dtype=np.float32)
    offset_ = 0

    # Add all pixels to our arrays
    with eval_context():
        for (inputs, targets, mask, _) in tqdm(dataloader_val, desc='Concatenating all predictions'):
            
            # Predict
            if evaluate_dataset_pseudolabels:
                mask = mask
            else:
                logits = model(inputs.to(accelerator.device).contiguous()).squeeze(0)  # (C, H, W)
                mask = torch.argmax(logits, dim=0)  # (H, W)
            
            # Convert
            target = targets.numpy().squeeze()
            mask = mask.cpu().numpy().squeeze()
            
            # Check where ground-truth is valid and append valid pixels to the array
            valid = (target != 255)
            n_valid = np.sum(valid)
            all_gt[offset_:offset_+n_valid] = target[valid]
            
            # Possibly reshape embedding to match gt.
            if mask.shape != target.shape:
                raise ValueError(f'{mask.shape=} != {target.shape=}')
            
            # Append the predicted targets in the array
            all_preds[offset_:offset_+n_valid, ] = mask[valid]
            all_gt[offset_:offset_+n_valid, ] = target[valid]
            
            # Update offset_
            offset_ += n_valid

    # Truncate to the actual number of pixels
    all_preds = all_preds[:offset_, ]
    all_gt = all_gt[:offset_, ]

    # TP, FP, and FN evaluation
    for i_part in range(0, n_classes):
        tmp_all_gt = (all_gt == i_part)
        tmp_pred = (all_preds == i_part)
        tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
        fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
        fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

    # Calculate Jaccard index
    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Print results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)
    print('Evaluation of semantic segmentation ')
    print(f'Full eval result: {eval_result}')
    print('mIoU is %.2f' % (100*eval_result['mIoU']))
    return eval_result


@torch.no_grad()
def generate(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    dataloader_val: Iterable,
    accelerator: Accelerator,
    model_ema: Optional[object] = None,
    **_unused_kwargs
):
    
    # To avoid CUDA errors on my machine
    torch.backends.cudnn.benchmark = False

    # Eval mode
    model.eval()
    torch.cuda.synchronize()
    eval_context = model_ema.average_parameters if cfg.ema.use_ema else nullcontext

    # Create paths
    preds_dir = Path('preds')
    gt_dir = Path('gt')
    preds_dir.mkdir(exist_ok=True, parents=True)
    gt_dir.mkdir(exist_ok=True, parents=True)

    # Generate and save
    with eval_context():
        for (inputs, targets, _, metadata) in tqdm(dataloader_val, desc='Concatenating all predictions'):
            # Predict
            logits = model(inputs.to(accelerator.device).contiguous()).squeeze(0)  # (C, H, W)
            # Convert
            preds = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
            gt = targets.squeeze().numpy().astype(np.uint8)
            # Save
            Image.fromarray(preds).convert('L').save(preds_dir / f"{metadata[0]['id']}.png")
            Image.fromarray(gt).convert('L').save(gt_dir / f"{metadata[0]['id']}.png")
        
    print(f'Saved to {Path(".").absolute()}')


if __name__ == '__main__':
    main()
