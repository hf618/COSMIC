import math
import os

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image
import ipdb

class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None, dino=False):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        self.dino = dino
        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform and not self.dino:
            image = self.transform(image)
            return image, torch.tensor(label).long()
        if self.transform and self.dino:
            image1, image2 = self.transform(image)
            return image1, image2, torch.tensor(label).long()
        # ipdb.set_trace()  # 暂停执行



# fewshot_datasets = ['DTD', 'Flower102', 'Food101', 'Cars', 'SUN397',
#                     'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat']
fewshot_datasets = ['dtd', 'flower102', 'food101', 'cars', 'sun397',
                     'aircraft', 'pets', 'caltech101', 'ucf101', 'eurosat']
path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    "flower102": ["jpg", "/home/hfd24/TPTs/dataset/data_splits/split_zhou_OxfordFlowers.json"],
    "food101": ["images", "/home/hfd24/TPTs/dataset/data_splits/split_zhou_Food101.json"],
    "dtd": ["images", "/home/hfd24/TPTs/dataset/data_splits/split_zhou_DescribableTextures.json"],
    "pets": ["images", "/home/hfd24/TPTs/dataset/data_splits/split_zhou_OxfordPets.json"],
    "sun397": ["SUN397", "/home/hfd24/TPTs/dataset/data_splits/split_zhou_SUN397.json"],
    "caltech101": ["101_ObjectCategories", "/home/hfd24/TPTs/dataset/data_splits/split_zhou_Caltech101.json"],
    "ucf101": ["UCF-101-midframes", "/home/hfd24/TPTs/dataset/data_splits/split_zhou_UCF101.json"],
    "cars": ["", "/home/hfd24/TPTs/dataset/data_splits/split_zhou_StanfordCars.json"],
    "eurosat": ["2750", "/home/hfd24/TPTs/dataset/data_splits/split_zhou_EuroSAT.json"]
}

def build_fewshot_dataset(set_id, root, transform, mode='train', n_shot=None, dino=False):
    if set_id.lower() == 'aircraft':
        return Aircraft(root, mode, n_shot, transform)
    path_suffix, json_path = path_dict[set_id.lower()]
    image_path = os.path.join(root, path_suffix)
    return BaseJsonDataset(image_path, json_path, mode, n_shot, transform, dino=dino)


class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        #self.path = root
        self.path = os.path.join(root, "fgvc-aircraft-2013b/data")
        self.mode = mode
        #ipdb.set_trace()  # 暂停执行
        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()

