import os
import glob
import random
from PIL import Image
import torch

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG")

def list_imgs(d):
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(glob.glob(os.path.join(d, f"*{ext}")))
    return sorted(imgs)

class CatFolderDataset(torch.utils.data.Dataset):
    """
    Reads:
      data_root/
        id_000/
          *.jpg
        id_001/
          *.jpg
    Excludes folders ending with " 2" by default (caller should filter).
    """
    def __init__(self, id2imgs, split="train", split_ratio=0.8, transform=None, seed=42):
        self.transform = transform
        rng = random.Random(seed)
        self.samples = []  # (path, label)

        for y, imgs in enumerate(id2imgs):
            imgs = imgs[:]
            rng.shuffle(imgs)
            n_train = max(1, int(len(imgs) * split_ratio))
            if split == "train":
                chosen = imgs[:n_train]
            else:
                chosen = imgs[n_train:] if len(imgs[n_train:]) > 0 else imgs[:1]
            for p in chosen:
                self.samples.append((p, y))

        # indices_by_id for PK sampler
        self.indices_by_id = {}
        for i, (_, y) in enumerate(self.samples):
            self.indices_by_id.setdefault(y, []).append(i)

        self.num_ids = len(self.indices_by_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)

def build_datasets_from_folder(data_root: str, split_ratio: float,
                               train_transform, test_transform, seed: int = 42):
    # list identity dirs
    id_dirs = sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and d.startswith("id_")
        and not d.endswith(" 2")
    ])

    id2imgs = []
    for d in id_dirs:
        imgs = list_imgs(d)
        if len(imgs) >= 2:
            id2imgs.append(imgs)

    num_ids = len(id2imgs)

    train_set = CatFolderDataset(id2imgs, split="train", split_ratio=split_ratio,
                                 transform=train_transform, seed=seed)
    test_set  = CatFolderDataset(id2imgs, split="test",  split_ratio=split_ratio,
                                 transform=test_transform,  seed=seed)

    return train_set, test_set, num_ids
