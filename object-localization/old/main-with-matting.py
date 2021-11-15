import os
import sys
import argparse
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou
from visualizations import visualize_fms, visualize_predictions, visualize_seed_expansion
from object_discovery import lost, detect_box, dino_seg, eigen_lost, get_largest_cc_box


def erode_or_dilate_mask(x: Union[torch.Tensor, np.ndarray], r: int = 1, erode=True):
    fn = binary_erosion if erode else binary_dilation
    for _ in range(r):
        x_new = fn(x)
        if x_new.sum() > 0:
            x = x_new
    return x

inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.ToPILImage()
])


def perform_matting(img, patch_mask):
    """ Performs alpha matting given a preliminary patch mask and an image """

    import scipy
    from skimage.morphology import binary_erosion, binary_dilation
    from skimage.transform import resize
    from pymatting import cf_laplacian, ichol, cg
        
    # Reshape mask to 2D
    H, W = img.shape[-2:]
    patch_mask = patch_mask.reshape(H // 16, W // 16).cpu().numpy()

    # Get matting laplacian
    np_image = np.array(inverse_transform(img.cpu())) / 255
    L = cf_laplacian(np_image)

    # Create trimap
    is_fg = erode_or_dilate_mask(patch_mask, r=2, erode=True)
    is_bg = ~(erode_or_dilate_mask(patch_mask, r=2, erode=False))
    is_fg = resize(is_fg, output_shape=(H, W)).astype(bool).reshape(-1)
    is_bg = resize(is_bg, output_shape=(H, W)).astype(bool).reshape(-1)
    is_known = (is_fg | is_bg)
    
    # Linear system
    lambda_value = 100
    C_m = scipy.sparse.diags(lambda_value * is_known)
    A_m = L + C_m  # matting affinity matrix

    # Build ichol preconditioner for faster convergence
    A_m = A_m.tocsr()
    A_m.sum_duplicates()
    M_m = ichol(A_m)

    # Solve linear system
    b = (lambda_value * is_fg).astype(np.float64)
    x = cg(A_m, b, M=M_m)

    # Result
    alpha = np.clip(x, 0.0, 1.0).reshape(H, W)
    return alpha

######### END #########


if __name__ == "__main__":
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

    # Use dino-seg proposed method
    parser.add_argument("--ganseg", action="store_true", help="Apply GAN model.")
    parser.add_argument("--ganseg_threshold", type=float, default=0.5)
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)
    
    # Use eigenvalue method
    parser.add_argument("--eigenseg", action='store_true', help='Apply eigenvalue method')

    args = parser.parse_args()

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # Transform
    if args.ganseg:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, transform)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard, transform)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.ganseg:
        model = torch.hub.load('greeneggsandyaml/uss', 'simple_unet').to(device).eval()
    else:
        model = get_model(args.arch, args.patch_size, args.resnet_dilate, device)

    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    if args.ganseg:
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

    print(f"Running LOST on the dataset {dataset.name} (exp: {exp_name})")

    # Visualization 
    if args.visualize:
        vis_folder = f"{args.output_dir}/visualizations/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    alpha_dict = {}
    preds_dict = {}
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
            # size_im = (
            #     img.shape[0],
            #     int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),  # TODO TODO TODO MAKE THIS CEIL AGAIN
            #     int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),  # TODO TODO TODO MAKE THIS CEIL AGAIN
            # )
            # paded = torch.zeros(size_im)
            # paded[:, : img.shape[1], : img.shape[2]] = img
            # img = paded

            size_im = (
                img.shape[0],
                int(np.floor(img.shape[1] / args.patch_size) * args.patch_size),  # TODO TODO TODO MAKE THIS CEIL AGAIN
                int(np.floor(img.shape[2] / args.patch_size) * args.patch_size),  # TODO TODO TODO MAKE THIS CEIL AGAIN
            )
            img = paded = img[:, :size_im[1], :size_im[2]]
            
            # if args.image_path is None:
            #     paded = torch.zeros(size_im)
            #     paded[:, : img.shape[1], : img.shape[2]] = img
            #     img = paded
            # else:
            #     print(f'{img.shape=}')
            #     print(f'{size_im=}')
            #     img = paded = img[:, :size_im[1], :size_im[2]]
            #     print(f'{img.shape=}')

            # Size for transformers
            w_featmap = img.shape[-2] // args.patch_size
            h_featmap = img.shape[-1] // args.patch_size
        
        # Move to gpu
        img = img.cuda(non_blocking=True)

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        if not args.ganseg:
            with torch.no_grad():

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

            # ------------ Apply LOST -------------------------------------------
            if not args.dinoseg:
                if args.eigenseg:
                    pred = eigen_lost( 
                        feats,
                        [w_featmap, h_featmap],
                        scales,
                        init_image_size,
                        k_patches=args.k_patches,
                        img_np=np.array(inverse_transform(img))
                    )
                else:
                    pred, A, M, scores, seed = lost(
                        feats,
                        [w_featmap, h_featmap],
                        scales,
                        init_image_size,
                        k_patches=args.k_patches,
                    )

                # # NOTE: M is the correlations for the expanded seed
                # print(img.shape)
                # alpha = perform_matting(img=img, patch_mask=(M > 0))  # <-- alpha matting using unsupervised trimap

                # # # Save info
                # output_file = f'examples/info-{args.image_path.split("/")[-1].split(".")[0]}.pth'
                # torch.save(dict(feats=feats, pred=pred, A=A, scores=scores, seed=seed, im_name=im_name, img=img, w_featmap=w_featmap,
                #                 h_featmap=h_featmap, scales=scales, init_image_size=init_image_size), output_file)
                # import pdb
                # pdb.set_trace()

                # ------------ Visualizations -------------------------------------------
                if args.visualize == "fms":
                    visualize_fms(A.clone().cpu().numpy(), seed, scores, [w_featmap, h_featmap], scales, vis_folder, im_name)

                elif args.visualize == "seed_expansion":
                    image = dataset.load_image(im_name)

                    # Before expansion
                    pred_seed, _ = detect_box(
                        A[seed, :],
                        seed,
                        [w_featmap, h_featmap],
                        scales=scales,
                        initial_im_size=init_image_size[1:],
                    )
                    visualize_seed_expansion(image, pred, seed, pred_seed, scales, [w_featmap, h_featmap], vis_folder, im_name)

                elif args.visualize == "pred":
                    image = dataset.load_image(im_name)
                    visualize_predictions(image, pred, seed, scales, [w_featmap, h_featmap], vis_folder, im_name)
            
        if args.ganseg:
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

        # Save the prediction
        preds_dict[im_name] = pred
        # alpha_dict[im_name] = alpha

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1

        cnt += 1
        if cnt % 10 == 0:
            # pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt} ({int(np.sum(corloc))/cnt * 100:.1f}%)")
            # torch.save(alpha_dict, 'alphas.tmp.pth')

    # # Save
    # torch.save(alpha_dict, 'alphas.pth')

    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)
        print("Predictions saved at %s" % filename)

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        result_file = os.path.join(folder, 'results.txt')
        with open(result_file, 'w') as f:
            f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
        print('File saved at %s'%result_file)
