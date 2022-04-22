import os
from pathlib import Path
from typing import Iterable, Optional

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from matplotlib.cm import get_cmap
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from skimage.color import label2rgb
from tqdm import tqdm, trange

import eval_utils
import util as utils
from dataset.voc import VOCSegmentationWithPseudolabels


@hydra.main(config_path='config', config_name='eval')
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

    # Create dataset with segments/pseudolabels
    dataset_val = VOCSegmentationWithPseudolabels(
        **cfg.data.val_kwargs, 
        segments_dir=cfg.segments_dir,
        transform=None,  # no transform to evaluate at original resolution
    )

    # Evaluate
    eval_stats, match = evaluate(cfg=cfg, dataset_val=dataset_val, n_clusters=cfg.get('n_clusters', None))
    print(eval_stats)
    if cfg.wandb and accelerator.is_local_main_process:
        wandb.summary['mIoU'] = eval_stats['mIoU']

    # Visualize
    visualize(cfg=cfg, dataset_val=dataset_val)


def visualize(
        *,
        cfg: DictConfig,
        dataset_val: Iterable,
        vis_dir: str = './vis'):

    # Visualize
    num_vis = 40
    vis_dir = Path(vis_dir)
    colors = get_cmap('tab20', cfg.data.num_classes + 1).colors[:,:3]
    pbar = tqdm(dataset_val, total=num_vis, desc='Saving visualizations: ')
    for i, (image, target, mask, metadata) in enumerate(pbar):
        if i >= num_vis: break
        image = np.array(image)
        target = np.array(target)
        target[target == 255] = 0  # set the "unknown" regions to background for visualization
        # Overlay mask on image
        image_pred_overlay = label2rgb(label=mask, image=image, colors=colors[np.unique(target)[1:]], bg_label=0, alpha=0.45)
        image_target_overlay = label2rgb(label=target, image=image, colors=colors[np.unique(target)[1:]], bg_label=0, alpha=0.45)
        # Save
        image_id = metadata["id"]
        path_pred = vis_dir / 'pred' / f'{image_id}-pred.png'
        path_target = vis_dir / 'target' / f'{image_id}-target.png'
        path_pred.parent.mkdir(exist_ok=True, parents=True)
        path_target.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray((image_pred_overlay * 255).astype(np.uint8)).save(str(path_pred))
        Image.fromarray((image_target_overlay * 255).astype(np.uint8)).save(str(path_target))
    print(f'Saved visualizations to {vis_dir.absolute()}')


def evaluate(
        *,
        cfg: DictConfig,
        dataset_val: Iterable,
        n_clusters: Optional[int] = None):

    # Add background class
    n_classes = cfg.data.num_classes + 1
    if n_clusters is None:
        n_clusters = n_classes

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    # Load all pixel embeddings
    all_preds = np.zeros((len(dataset_val) * 500 * 500), dtype=np.float32)
    all_gt = np.zeros((len(dataset_val) * 500 * 500), dtype=np.float32)
    offset_ = 0

    # Add all pixels to our arrays
    _alread_warned = 0
    for i in trange(len(dataset_val), desc='Concatenating all predictions'):
        image, target, mask, metadata = dataset_val[i]
        # Check where ground-truth is valid and append valid pixels to the array
        valid = (target != 255)
        n_valid = np.sum(valid)
        all_gt[offset_:offset_+n_valid] = target[valid]
        # Append the predicted targets in the array
        all_preds[offset_:offset_+n_valid, ] = mask[valid]
        all_gt[offset_:offset_+n_valid, ] = target[valid]
        # Update offset_
        offset_ += n_valid

    # Truncate to the actual number of pixels
    all_preds = all_preds[:offset_, ]
    all_gt = all_gt[:offset_, ]

    # Do hungarian matching
    num_elems = offset_
    if n_clusters == n_classes:
        print('Using hungarian algorithm for matching')
        match = eval_utils.hungarian_match(all_preds, all_gt, preds_k=n_clusters, targets_k=n_classes, metric='iou')
    else:
        print('Using majority voting for matching')
        match = eval_utils.majority_vote(all_preds, all_gt, preds_k=n_clusters, targets_k=n_classes)
    print(f'Optimal matching: {match}')

    # Remap predictions
    reordered_preds = np.zeros(num_elems, dtype=all_preds.dtype)
    for pred_i, target_i in match:
        reordered_preds[all_preds == int(pred_i)] = int(target_i)

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
    return eval_result, match


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()
