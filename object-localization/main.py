import os
import sys
import argparse
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from typing import Union
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou
from visualizations import visualize_fms, visualize_predictions, visualize_seed_expansion
from object_discovery import lost, detect_box, dino_seg, get_eigenvectors_from_features, get_largest_cc_box, get_bbox_from_patch_mask


def parse_args(): 
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "resnet50",
            "vgg16_imagenet",
            "resnet50_imagenet",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["fms", "seed_expansion", "pred", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # For ResNet dilation
    parser.add_argument("--resnet_dilate", type=int, default=2, help="Dilation level of the resnet model.")

    # LOST parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )

    # Misc
    parser.add_argument("--name", type=str, default=None, help='Experiment name')
    parser.add_argument("--skip_if_exists", action='store_true', help='If results dir already exists , exit')

    # Use dino-seg proposed method
    parser.add_argument("--ganseg", action="store_true", help="Apply GAN model.")
    parser.add_argument("--ganseg_threshold", type=float, default=0.5)
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)
    
    # Use eigenvalue method
    parser.add_argument("--eigenseg", action='store_true', help='Apply eigenvalue method')
    parser.add_argument("--precomputed_eigs_dir", default=None, type=str, 
                        help='Apply eigenvalue method with precomputed bboxes')
    parser.add_argument("--precomputed_eigs_downsample", default=16, type=str)
    parser.add_argument("--which_matrix", choices=['affinity_torch', 'affinity', 'laplacian', 'matting_laplacian'],
                        default='affinity_torch', help='Which matrix to use for eigenvector calculation')

    # Parse
    args = parser.parse_args()

    # Modify
    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    return args


@torch.no_grad()
def main():

    # Args
    args = parse_args()

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # Transform
    if args.ganseg:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
            transforms.ToPILImage()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])        
        inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.ToPILImage()
        ])

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, transform)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard, transform)

    # Naming
    if args.name is not None:
        exp_name = args.name
    elif args.ganseg:
        exp_name = f"gan-exp"
    elif args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with LOST
        exp_name = f"LOST-{args.arch}"
        if "resnet" in args.arch:
            exp_name += f"dilate{args.resnet_dilate}"
        elif "vit" in args.arch:
            exp_name += f"{args.patch_size}_{args.which_features}"

    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)

    # Skip if already exists
    exp_dir = Path(args.output_dir) / exp_name
    if args.skip_if_exists and exp_dir.is_dir() and len(list(exp_dir.iterdir())) > 0:
        print(f'Directory already exists and is not empty: {str(exp_dir)}')
        print(f'Exiting...')
        sys.exit()
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.ganseg:
        model = torch.hub.load('greeneggsandyaml/uss', 'simple_unet').to(device).eval()
    else:
        model = get_model(args.arch, args.patch_size, args.resnet_dilate, device)

    print(f"Running LOST on the dataset {dataset.name} (exp: {exp_name})")
    print(f"Args:")
    print(pprint(args.__dict__))

    # Visualization 
    if args.visualize:
        vis_folder = f"{args.output_dir}/visualizations/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    gt_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))
    
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]
        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])

        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        if not args.ganseg:

            # For eigenseg, crop the image rather than padding it. Padding causes major issues. 
            if args.eigenseg:
                size_im = (
                    img.shape[0],
                    int(np.floor(img.shape[1] / args.patch_size) * args.patch_size),
                    int(np.floor(img.shape[2] / args.patch_size) * args.patch_size),
                )
                img = paded = img[:, :size_im[1], :size_im[2]]
            else:
                size_im = (
                    img.shape[0],
                    int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
                    int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
                )
                paded = torch.zeros(size_im)
                paded[:, : img.shape[1], : img.shape[2]] = img
                img = paded

            # Size for transformers
            w_featmap = img.shape[-2] // args.patch_size
            h_featmap = img.shape[-1] // args.patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------

        # Extract features from GAN
        if args.ganseg:
            img = img.cuda(non_blocking=True)
            img = img.unsqueeze(0)  # (1, 3, H, W)
            img_resized = F.interpolate(img, size=(128, 128))  # (1, 3, 128, 128)
            pred = model(img_resized, postprocess=False)  # (1, 2, 128, 128)
            pred = torch.softmax(pred, dim=1)  # (1, 2, H, W)
            pred = F.interpolate(pred, size=img.shape[-2:])  # (1, 2, H, W)
            try:
                mask = (pred > args.ganseg_threshold / 2).detach().cpu().numpy()[0, 1]  # (H, W)
                pred = get_largest_cc_box(mask)  # [xmin, ymin, xmax, ymax]
            except:
                try:
                    mask = (pred > args.ganseg_threshold / 2).detach().cpu().numpy()[0, 1]  # (H, W)
                    pred = get_largest_cc_box(mask)  # [xmin, ymin, xmax, ymax]
                except:
                    try:
                        mask = (pred > args.ganseg_threshold / 4).detach().cpu().numpy()[0, 1]  # (H, W)
                        pred = get_largest_cc_box(mask)  # [xmin, ymin, xmax, ymax]
                    except:
                        ymin, ymax = (0, img.shape[-2] + 1)  # entire image if we don't find anything
                        xmin, xmax = (0, img.shape[-1] + 1)  # entire image if we don't find anything
                        pred = [xmin, ymin, xmax, ymax]
            pred = np.array(pred)

        # Load precomputer bounding boxes
        elif args.eigenseg and args.precomputed_eigs_dir is not None:

            # Load
            fname = im_name.replace('.jpg', '.pth') if 'VOC07' in dataset.name else im_name.replace('.jpg', '.pth')
            precomputed_eigs_file = os.path.join(args.precomputed_eigs_dir, fname)
            precomputed_eigs = torch.load(precomputed_eigs_file, map_location='cpu')
            eigenvectors = precomputed_eigs['eigenvectors']  # tensor of shape (K, H_lr * W_lr)

            # Get eigenvectors 
            assert ('affinity' in args.which_matrix) ^ ('laplacian' in args.which_matrix)
            if 'affinity' in args.which_matrix:
                if eigenvectors.shape[0] > eigenvectors.shape[1]:  # HACK
                    patch_mask = (eigenvectors[:, 0] > 0)
                else:
                    patch_mask = (eigenvectors[0] > 0)
            else:
                patch_mask = (eigenvectors[1] > 0)
                # patch_mask = patch_mask[1:]  # NOTE: <-- if you're evaluating ['out'] features
            pred = get_bbox_from_patch_mask(patch_mask, init_image_size)
            # P = args.precomputed_eigs_downsample
            # dims_wh = (img.shape[-2] // P, img.shape[-1] // P)
            # scales = (P, P)
            # pred = get_bbox_from_patch_mask(patch_mask, dims_wh, scales, init_image_size)

            # TODO: Maybe think of a better way to do the background detection this?
            # TODO: Discuss the background detection issue in the paper
            # TODO: Discuss the largest connected component thing in the paper

        # Extract features from self-supervised model
        else:
            
            # Move to GPU
            img = img.cuda(non_blocking=True)
            
            # ------------ FORWARD PASS -------------------------------------------
            if "vit" in args.arch:
                # Store the outputs of qkv layer from the last attention layer
                feat_out = {}
                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output
                model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

                # Forward pass in the model
                attentions = model.get_last_selfattention(img[None, :, :, :])

                # Scaling factor
                scales = [args.patch_size, args.patch_size]

                # Dimensions
                nb_im = attentions.shape[0]  # Batch size
                nh = attentions.shape[1]  # Number of heads
                nb_tokens = attentions.shape[2]  # Number of tokens

                # Baseline: compute DINO segmentation technique proposed in the DINO paper
                # and select the biggest component
                if args.dinoseg:
                    pred = dino_seg(attentions, (w_featmap, h_featmap), args.patch_size, head=args.dinoseg_head)
                    pred = np.asarray(pred)
                else:
                    # Extract the qkv features of the last attention layer
                    qkv = (
                        feat_out["qkv"]
                        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

                    # Modality selection
                    if args.which_features == "k":
                        feats = k[:, 1:, :]
                    elif args.which_features == "q":
                        feats = q[:, 1:, :]
                    elif args.which_features == "v":
                        feats = v[:, 1:, :]
            elif "resnet" in args.arch:
                x = model.forward(img[None, :, :, :])
                d, w_featmap, h_featmap = x.shape[1:]
                feats = x.reshape((1, d, -1)).transpose(2, 1)
                # Apply layernorm
                layernorm = nn.LayerNorm(feats.size()[1:]).to(device)
                feats = layernorm(feats)
                # Scaling factor
                scales = [
                    float(img.shape[1]) / x.shape[2],
                    float(img.shape[2]) / x.shape[3],
                ]
            elif "vgg16" in args.arch:
                x = model.forward(img[None, :, :, :])
                d, w_featmap, h_featmap = x.shape[1:]
                feats = x.reshape((1, d, -1)).transpose(2, 1)
                # Apply layernorm
                layernorm = nn.LayerNorm(feats.size()[1:]).to(device)
                feats = layernorm(feats)
                # Scaling factor
                scales = [
                    float(img.shape[1]) / x.shape[2],
                    float(img.shape[2]) / x.shape[3],
                ]
            else:
                raise ValueError("Unknown model.")

            # Sizes
            dims_wh = [w_featmap, h_featmap]

            # ------------ Apply LOST -------------------------------------------
            if not args.dinoseg:
                if args.eigenseg:
                    # Get eigenvectors
                    eigenvectors = get_eigenvectors_from_features(feats, args.which_matrix)
                    # Get bounding box
                    assert ('affinity' in args.which_matrix) ^ ('laplacian' in args.which_matrix)
                    eig_index = 0 if 'affinity' in args.which_matrix else 1
                    patch_mask = (eigenvectors[:, eig_index] > 0)
                    pred = get_bbox_from_patch_mask(patch_mask, init_image_size)
                    # pred = get_bbox_from_patch_mask(patch_mask, dims_wh, scales, init_image_size)
                    # use_crf=True, img_np=np.array(inverse_transform(img))
                else:
                    pred, A, M, scores, seed = lost(feats, dims_wh, scales, init_image_size, k_patches=args.k_patches)

                # ------------ Visualizations -------------------------------------------
                if args.visualize == "fms":
                    visualize_fms(A.clone().cpu().numpy(), seed, scores, dims_wh, scales, vis_folder, im_name)

                elif args.visualize == "seed_expansion":
                    image = dataset.load_image(im_name)

                    # Before expansion
                    pred_seed, _ = detect_box(A[seed, :], seed, dims_wh, scales=scales, initial_im_size=init_image_size[1:])
                    visualize_seed_expansion(image, pred, seed, pred_seed, scales, dims_wh, vis_folder, im_name)

                elif args.visualize == "pred":
                    image = dataset.load_image(im_name)
                    visualize_predictions(image, pred, seed, scales, dims_wh, vis_folder, im_name)
            
        # Save the prediction
        preds_dict[im_name] = pred
        gt_dict[im_name] = gt_bbxs

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1

        cnt += 1
        if cnt % 10 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt} ({int(np.sum(corloc))/cnt * 100:.1f}%)")

    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "preds.pkl"), "wb") as f:
            pickle.dump(preds_dict, f)
        with open(os.path.join(folder, "gt.pkl"), "wb") as f:
            pickle.dump(gt_dict, f)
        print(f"Predictions saved to {folder}")

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        result_file = os.path.join(folder, 'results.txt')
        with open(result_file, 'w') as f:
            f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
        print('File saved at %s'%result_file)


if __name__ == "__main__":
    main()