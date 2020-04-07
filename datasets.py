import os
import glob
import torch
from PIL import Image
from torchvision import datasets, transforms

class CycleGANDatasets(torch.utils.data.Dataset):
    def __init__(self, root, img_shape=(128, 128), is_train=True):
        # Directory Name
        if is_train:
            root_A = root + os.sep + 'trainA/'
            root_B = root + os.sep + 'trainB/'
        else:
            root_A = root + os.sep + 'testA/'
            root_B = root + os.sep + 'testB/'
        # Init
        self.img_shape = img_shape
        self.transform = transforms.Compose([
            transforms.Resize(self.img_shape),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.dataset_A = self._read(root=root_A)
        self.dataset_B = self._read(root=root_B)
        self.data_num = len(self.dataset_A)

    def _read(self, root):
        return glob.glob(root+'*.jpg')

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        img_A = self.transform(Image.open(self.dataset_A[idx]).convert('RGB'))
        img_B = self.transform(Image.open(self.dataset_B[idx]).convert('RGB'))
        return img_A, img_B


if __name__ == "__main__":
    horse2zebra = CycleGANDatasets('./data/horse2zebra')
