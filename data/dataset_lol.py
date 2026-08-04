import os, random
import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class LOLDataReader(Dataset):
    """LOL dataset reader. Computes illumination map as auxiliary input.
    
    Returns: (input_rgb, illumination_map, target_rgb, filename)
    - input_rgb: low-light image [3, H, W]
    - illumination_map: max(R,G,B) of input [1, H, W]  (Retinex illumination)
    - target_rgb: normal-light image [3, H, W]
    """
    def __init__(self, img_dir, inp='input', tar='target', mode='train',
                 img_options=None):
        super().__init__()
        inp_files = sorted(os.listdir(os.path.join(img_dir, inp)))
        tar_files = sorted(os.listdir(os.path.join(img_dir, tar)))
        self.inp_filenames = [os.path.join(img_dir, inp, x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(img_dir, tar, x) for x in tar_files if is_image_file(x)]
        self.mode = mode
        self.img_options = img_options
        self.sizex = len(self.tar_filenames)
        
        if self.mode == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.RandomResizedCrop(size=(img_options['h'], img_options['w'])),
            ], additional_targets={'target': 'image'})
        else:
            self.transform = A.Compose([
                A.Resize(height=img_options['h'], width=img_options['w']),
            ], additional_targets={'target': 'image'})

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        
        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')
        inp_np = np.array(inp_img)
        tar_np = np.array(tar_img)
        
        transformed = self.transform(image=inp_np, target=tar_np)
        inp_img = F.to_tensor(transformed['image'])
        tar_img = F.to_tensor(transformed['target'])
        
        # Retinex illumination map: max(R,G,B) channel
        illum = torch.max(inp_img, dim=0, keepdim=True)[0]
        
        filename = os.path.basename(tar_path)
        return inp_img, illum, tar_img, filename
