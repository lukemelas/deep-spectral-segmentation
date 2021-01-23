from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader


class SimpleDataset(Dataset):
    """ 
    Description...
    """

    def __init__(self, root, transform=None):
        """
        :param root: ...
        :param transform: ...
        """
        self.root = Path(root)
        self.samples_list = list(self.root.iterdir())
        self.transform = transform
        self.size = len(self.samples_list)

    def __getitem__(self, i):
        image_path = self.samples_list[i]
        image = default_loader(image_path)
        target = 0
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return self.size


if __name__ == "__main__":
    transform = transforms.ToTensor()
    dataset = SimpleDataset(
        root="/home/luke/machine-learning-datasets/semantic-segmentation/cityscapes/leftImg8bit/train/zurich", 
        transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    image = next(iter(dataloader))
    print('dataset loading test complete')
    print(image.shape)
    import pdb; pdb.set_trace()