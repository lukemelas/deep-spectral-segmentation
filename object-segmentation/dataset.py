from pathlib import Path
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def get_paths_from_folders(images_dir):
    """Returns list of files in folders of input"""
    paths = []
    for folder in Path(images_dir).iterdir():
        for p in folder.iterdir():
            paths.append(p)
    return paths


def central_crop(x):
    dims = x.size
    crop = T.CenterCrop(min(dims[0], dims[1]))
    return crop(x)


class SegmentationDataset(Dataset):

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        image_size: Optional[int] = None,
        resize_image=True,
        resize_mask=None,
        crop=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        name: Optional[str] = None,
    ):
        self.name = name
        self.crop = crop
        
        # Find out if dataset is organized into folders or not
        has_folders = not any(str(next(Path(images_dir).iterdir())).endswith(ext) for ext in IMG_EXTENSIONS)

        # Get and sort list of paths
        if has_folders:
            image_paths = get_paths_from_folders(images_dir)
            label_paths = get_paths_from_folders(labels_dir)
        else:
            image_paths = Path(images_dir).iterdir()
            label_paths = Path(labels_dir).iterdir()
        self.image_paths = list(sorted(image_paths))
        self.label_paths = list(sorted(label_paths))
        assert len(self.image_paths) == len(self.label_paths)

        # Transformation
        resize_image = (image_size is not None) and resize_image
        resize_mask = resize_image if resize_mask is None else resize_mask
        image_transform = [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        mask_transform = [T.ToTensor()]
        if resize_image:
            image_transform.insert(0, T.Resize(image_size))
        if resize_mask:
            mask_transform.insert(0, T.Resize(image_size))
        if crop:
            image_transform.insert(0, central_crop)
            mask_transform.insert(0, central_crop)
        self.image_transform = T.Compose(image_transform)
        self.mask_transform = T.Compose(mask_transform)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Load
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.label_paths[idx])
        metadata = {'image_file': str(self.image_paths[idx])}

        # Transform
        image = image.convert('RGB')
        mask = mask.convert('RGB')
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5)[0].long()  # TODO: this could be improved
        return image, mask, metadata
