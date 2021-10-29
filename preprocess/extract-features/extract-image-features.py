from collections import defaultdict
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from torchvision import transforms
import torch.distributed as dist
from tqdm import tqdm
import fire

from dataset import get_transforms, ImagesDataset


@torch.no_grad()
def extract_features(
    prefix: str,
    images_list: str = "/work/lukemk/machine-learning-datasets/image-captioning/yfcc/yfcc-clip-image-files.txt",
    images_root: Optional[str] = None,
    model_name: str = 'dino_vits16',
    batch_size: int = 1024,
    output_dir: str = './features',
):
    """
    Example:
        python extract-image-features.py extract_features \
            --prefix VOC2007 \
            --images_list /home/luke/projects/experiments/active/found/preprocess/extract-features/image-lists/VOC2007.txt \
            --images_root /data_q1_d/machine-learning-datasets/semantic-segmentation/PASCAL_VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages \
            --model_name dino_vits16 \
            --batch_size 1
    """

    # Models
    model_name_lower = model_name.lower()
    if 'dino' in model_name:
        model = torch.hub.load('facebookresearch/dino:main', model_name)
        model.fc = torch.nn.Identity()
        val_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        patch_size = model.patch_embed.patch_size
        num_heads = model.blocks[0].attn.num_heads
    else:
        raise NotImplementedError()
    model = model.eval()

    # Add hook
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    # Dataset
    filenames = Path(images_list).read_text().splitlines()
    dataset = ImagesDataset(filenames=filenames, images_root=images_root, transform=val_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)
    print(f'Dataset size: {len(dataset)=}')
    print(f'Dataloader size: {len(dataloader)=}')

    # Prepare
    accelerator = Accelerator(fp16=True, cpu=False)
    model, dataloader = accelerator.prepare(model, dataloader)

    # Process
    for i, (images, files, indices) in enumerate(tqdm(dataloader, desc='Processing')):
        output_dict = {}

        # Reshape image
        P = patch_size
        B, C, H, W = images.shape
        H_patch, W_patch = H // P, W // P
        H_pad, W_pad = H_patch * P, W_patch * P
        T = H_patch * W_patch + 1  # number of tokens, add 1 for [CLS]
        # images = F.interpolate(images, size=(H_pad, W_pad), mode='bilinear')  # resize image
        images = images[:, :, :H_pad, :W_pad]

        # Forward and gather
        output_dict['out'] = accelerator.unwrap_model(model).get_intermediate_layers(images)
        output_qkv = (
            feat_out["qkv"]
            .reshape(B, T, 3, num_heads, -1 // num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = output_qkv[0], output_qkv[1], output_qkv[2]
        output_dict['k'] = k.transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
        output_dict['q'] = q.transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
        output_dict['v'] = v.transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
        output_dict['indices'] = indices
        output_dict['files'] = files
        output_dict['shape'] = (B, C, H, W)
        output_dict['images_resized'] = images
        output_dict = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in output_dict.items()}

        # Gather 
        if accelerator.num_processes > 1:
            raise NotImplementedError()  # I wrote it and it might work but I haven't tested it
            batch_output_dict = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(batch_output_dict, output_dict)
        else:
            batch_output_dict = output_dict
        # Save
        # output_dict = {'features': output.detach().cpu(), 'files': batch_files, 'indices': indices.detach().cpu()}
        output_file = str(Path(output_dir) / f'{prefix}-image-features-{model_name_lower}-{i:05d}.pth')
        accelerator.save(output_dict, output_file)
        accelerator.wait_for_everyone()
    print(f'Saved features to {output_dir}')


def combine_features(
    input_dir: str = '/home/luke/machine-learning-datasets/image-captioning/CC/features',
    output_file: str = '/home/luke/machine-learning-datasets/image-captioning/CC/features.pth',
):
    """
    Combines all the features into a single large file.
    
    Example:
        python extract-image-features.py combine_features \
            --input_dir ./features \
            --output_file ./VOC2007-image-features-dino_vits16.pth
    """

    # Combine tokenized image files
    combined_output_dict = defaultdict(list)
    for i, p in tqdm(list(enumerate(sorted(Path(input_dir).iterdir())))):
        output_dict = torch.load(p, map_location='cpu')
        for k, v in output_dict.items():
            combined_output_dict[k].append(v)
    combined_output_dict = dict(combined_output_dict)
    
    # Save
    torch.save(combined_output_dict, output_file)
    print(f'Saved file to {output_file}')


if __name__ == '__main__':
    fire.Fire(dict(
        extract_features=extract_features,
        combine_features=combine_features
    ))
