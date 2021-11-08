import os
import sys
import math
import datetime
from contextlib import nullcontext
from collections import namedtuple
from pathlib import Path
from typing import Callable, Iterable, Optional
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
from tqdm import tqdm

import util as utils
from model import get_model
from dataset import get_datasets


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
    if cfg.unfrozen_backbone_layers > 0:
        for module in list(model.backbone.children())[:-cfg.unfrozen_backbone_layers]:
            for p in module.parameters():
                p.requires_grad_(False)

    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')
    print(f'Backbone parameters (total): {sum(p.numel() for p in model.backbone.parameters()):_d}')
    print(f'Backbone parameters (train): {sum(p.numel() for p in model.backbone.parameters() if p.requires_grad):_d}')

    # Exponential moving average of model parameters
    if cfg.ema.use_ema:
        from torch_ema import ExponentialMovingAverage
        model_ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema.decay)
        print('Initialized model EMA')
    else:
        model_ema = None
        print('Not using model EMA')

    # Optimizer and scheduler
    optimizer = utils.get_optimizer(cfg, model, accelerator)
    scheduler = utils.get_scheduler(cfg, optimizer)

    # Resume from checkpoint and create the initial training state
    if cfg.checkpoint.resume:
        train_state: utils.TrainState = utils.resume_from_checkpoint(cfg, model, optimizer, scheduler, model_ema)
    else:
        train_state = utils.TrainState()  # start training from scratch

    # Data
    dataset_train, dataset_val, collate_fn = get_datasets(cfg)
    dataloader_train = DataLoader(dataset_train, shuffle=True, drop_last=True, 
        collate_fn=collate_fn, **cfg.data.loader)
    dataloader_val = DataLoader(dataset_val, shuffle=False, drop_last=False, 
        collate_fn=collate_fn, **{**cfg.data.loader, 'batch_size': 1})
    total_batch_size = cfg.data.loader.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    # Setup
    model, optimizer, dataloader_train = accelerator.prepare(model, optimizer, dataloader_train)

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

    # Training loop
    while True:

        # Single epoch of training
        train_state = train_one_epoch(**kwargs)

        # Save checkpoint on only 1 process
        if accelerator.is_local_main_process:
            print('Saving checkpoint...')
            checkpoint_dict = {
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': train_state.epoch,
                'steps': train_state.step,
                'best_val': train_state.best_val,
                'model_ema': model_ema.state_dict() if model_ema else {},
                'cfg': cfg
            }
            accelerator.save(checkpoint_dict, 'checkpoint-latest.pth')

        # Evaluate
        if train_state.epoch % cfg.get('eval_every', 1) == 0:
        # if (train_state.epoch < 5) or (train_state.epoch) % 1 == 0:
            test_stats = evaluate(**kwargs)
            if accelerator.is_local_main_process:
                if (train_state.best_val is None) or (test_stats['val_loss'] < train_state.best_val):
                    train_state.best_val = test_stats['val_loss']
                    torch.save(checkpoint_dict, 'checkpoint-best.pth')
                if cfg.wandb:
                    wandb.log(test_stats)
                    wandb.run.summary["best_val_loss"] = train_state.best_val
        
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
        **_unused_kwargs):

    # Train mode
    model.train().to(accelerator.device)  # TODO
    log_header = f'Epoch: [{train_state.epoch}]'
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('step', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
    progress_bar = metric_logger.log_every(dataloader_train, cfg.logging.print_freq, header=log_header)

    # Train
    for i, (inputs, _targets, pseudolabels, _metadata) in enumerate(progress_bar):
        if i >= cfg.get('limit_train_batches', math.inf):
            break

        # print(inputs.shape)  # TODO: remove
        # print(_targets.shape)
        # print(pseudolabels.shape)
        # import pdb
        # pdb.set_trace()

        # Forward
        output = model(inputs)
        sup_loss = F.cross_entropy(output, pseudolabels)
        loss = sup_loss

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
                model_ema.update(model.parameters())

        # Logging
        log_dict = dict(
            lr=optimizer.param_groups[0]["lr"], step=train_state.step,
            train_loss=loss_value, sup_loss=sup_loss,
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
        **_unused_kwargs):

    # Eval mode
    model.eval().to('cpu')  # TODO
    torch.cuda.synchronize()
    eval_context = model_ema.average_parameters if cfg.ema.use_ema else nullcontext

    # Add background class
    n_classes = n_clusters = cfg.data.num_classes + 1

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
        for (inputs, targets, mask, _metadata) in tqdm(dataloader_val, desc='Concatenating all predictions'):
            # Predict
            if evaluate_dataset_pseudolabels:
                mask = mask
            else:
                with torch.no_grad():
                    logits = model(inputs).squeeze(0)  # (C, H, W)  # TODO
                    # logits = model(inputs.to(accelerator.device)).squeeze(0)  # (C, H, W)  # TODO
                mask = torch.argmax(logits, dim=0)  # (H, W)
            # Convert
            target = targets.numpy().squeeze()
            mask = mask.cpu().numpy()
            # Check where ground-truth is valid and append valid pixels to the array
            valid = (target != 255)
            n_valid = np.sum(valid)
            all_gt[offset_:offset_+n_valid] = target[valid]
            # Possibly reshape embedding to match gt.
            if mask.shape != target.shape:
                raise ValueError(f'{mask.shape=} != {target.shape=}')
                # mask = cv2.resize(embedding, target.shape[::-1], interpolation=cv2.INTER_NEAREST)
            # Append the predicted targets in the array
            all_preds[offset_:offset_+n_valid, ] = mask[valid]
            all_gt[offset_:offset_+n_valid, ] = target[valid]
            # Update offset_
            offset_ += n_valid

    # Truncate to the actual number of pixels
    all_preds = all_preds[:offset_, ]
    all_gt = all_gt[:offset_, ]

    # # Do hungarian matching
    # num_elems = offset_
    # if n_clusters == n_classes:
    #     print('Using hungarian algorithm for matching')
    #     match = eval_utils.hungarian_match(all_preds, all_gt, preds_k=n_clusters, targets_k=n_classes, metric='iou')
    # else:
    #     print('Using majority voting for matching')
    #     match = eval_utils.majority_vote(all_preds, all_gt, preds_k=n_clusters, targets_k=n_classes)
    # print(f'Optimal matching: {match}')

    # # Remap predictions
    # reordered_preds = np.zeros(num_elems, dtype=all_preds.dtype)
    # for pred_i, target_i in match:
    #     reordered_preds[all_preds == int(pred_i)] = int(target_i)
    
    # TODO: check if reordering makes a difference after training 
    reordered_preds = all_preds

    # TP, FP, and FN evaluation
    for i_part in range(0, n_classes):
        tmp_all_gt = (all_gt == i_part)
        tmp_pred = (reordered_preds == i_part)
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
    print('mIoU is %.2f' % (100*eval_result['mIoU']))
    
    return eval_result


if __name__ == '__main__':
    main()