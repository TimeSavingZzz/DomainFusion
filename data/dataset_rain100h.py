import os, random, re
import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ["jpeg", "JPEG", "jpg", "png", "JPG", "PNG", "gif"])


def _extract_num(fname):
    m = re.search(r'(\d+)', os.path.splitext(fname)[0])
    return int(m.group(1)) if m else 0


class Rain100HDataReader(Dataset):
    def __init__(self, img_dir, inp="input", tar="target", mode="train", img_options=None):
        super().__init__()
        inp_dir = os.path.join(img_dir, inp)
        tar_dir = os.path.join(img_dir, tar)
        inp_list = [x for x in os.listdir(inp_dir) if is_image_file(x)]
        tar_list = [x for x in os.listdir(tar_dir) if is_image_file(x)]
        tar_by_num = {_extract_num(x): x for x in tar_list}
        paired = []
        for x in inp_list:
            n = _extract_num(x)
            if n in tar_by_num:
                paired.append((x, tar_by_num[n]))
        paired.sort(key=lambda p: _extract_num(p[0]))
        self.inp_filenames = [os.path.join(inp_dir, p[0]) for p in paired]
        self.tar_filenames = [os.path.join(tar_dir, p[1]) for p in paired]
        self.mode = mode
        self.img_options = img_options
        self.sizex = len(self.tar_filenames)
        self.target_h = img_options["h"] if img_options else 256
        self.target_w = img_options["w"] if img_options else 256
        if self.mode == "train":
            self.transform = A.Compose([
                A.Resize(height=self.target_h, width=self.target_w),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
            ], additional_targets={"target": "image", "aux": "image"}, is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.Resize(height=self.target_h, width=self.target_w),
            ], additional_targets={"target": "image", "aux": "image"}, is_check_shapes=False)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        inp_img = Image.open(inp_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")
        inp_np = np.array(inp_img)
        tar_np = np.array(tar_img)
        transformed = self.transform(image=inp_np, target=tar_np, aux=inp_np)
        inp_tensor = F.to_tensor(transformed["image"])
        tar_tensor = F.to_tensor(transformed["target"])
        aux_rgb = F.to_tensor(transformed["aux"])
        aux_tensor = F.rgb_to_grayscale(aux_rgb, num_output_channels=1)
        filename = os.path.basename(tar_path)
        return inp_tensor, aux_tensor, tar_tensor, filename
