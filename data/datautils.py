import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations
from data.randaugment import RandAugmentMC
from collections import defaultdict
import random
import ipdb

# ImageNet code should change this value
IMAGE_SIZE = 224

ID_to_DIRNAME={
    'I': 'ImageNet',
    'A': 'imagenet-a/imagenet-a',
    'K': 'ImageNet-Sketch/sketch',
    'R': 'imagenet-r/imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val/imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False, dino=False):
    if set_id == 'I':
        if mode == 'train' and n_shot:
            traindir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'train')
            trainset = datasets.ImageFolder(traindir, transform=transform)
            split_by_label_dict = defaultdict(list)
            for i in range(len(trainset.imgs)):
                split_by_label_dict[trainset.targets[i]].append(trainset.imgs[i])
            imgs = []
            targets = []
            for label, items in split_by_label_dict.items():
                imgs = imgs + random.sample(items, n_shot)
                targets = targets + [label for i in range(n_shot)]

            trainset.imgs = imgs
            trainset.targets = targets
            trainset.samples = imgs

            testset = trainset
        else:
            # ImageNet validation set
            testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
            testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            # ipdb.set_trace()
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, dino=dino)
        #ipdb.set_trace()
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment(resolution=IMAGE_SIZE):
    return transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(0.5, 1)),
            # transforms.Resize(256, interpolation=BICUBIC),
            # transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1, resolution=224):
    preaugment = get_preaugment(resolution=resolution)
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False,
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            #self.aug_list = augmentations.augmentations
            #self.aug_list = augmentations.augmentations_all
            #self.aug_list = augmentations.augmentations_both
            self.aug_list = augmentations.augmentations_all
        else:
            self.aug_list = []
        self.severity = severity

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views


# 针对双管齐下的魔改版
class AugMixAugmenter2(object):
    def __init__(self, base_transform, base_transform2, preprocess, n_views=2, augmix=False,
                 severity=1, resolution_dino=224):
        self.base_transform = base_transform
        self.base_transform2 = base_transform2
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            # self.aug_list = augmentations.augmentations
            # self.aug_list = augmentations.augmentations_all
            # self.aug_list = augmentations.augmentations_both
            self.aug_list = augmentations.augmentations_new
        else:
            self.aug_list = []
        self.severity = severity
        self.resolution_dino = resolution_dino

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]

        image2 = self.preprocess(self.base_transform2(x))
        # print("image",image.shape)
        # print("image2",image2.shape)
        # ipdb.set_trace()
        views2 = [augmix(x, self.preprocess, self.aug_list, self.severity, self.resolution_dino) for _ in range(self.n_views)]
        return [image] + views, [image2] + views2

####################################### weak augmentation for memorized images.
def get_preaugment_augmem():
    return transforms.Compose([
            # transforms.Resize(230, interpolation=BICUBIC),
            # transforms.RandomCrop(224),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1)),
            transforms.RandomHorizontalFlip(),
        ])


def augmem(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment_augmem()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    return x_processed

class AugMemAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False,
                 severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmem(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views

def randaug(image, preprocess, strong_aug):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_orig = strong_aug(x_orig)
    x_processed = preprocess(x_orig)
    return x_processed

class StrongAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False,
                 severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        self.strong_aug = RandAugmentMC(n=2, m=10)

    def __call__(self, x):
        preaugment = get_preaugment()
        x_orig = preaugment(x)
        image = self.preprocess(x_orig)

        # image = augmix(x, self.preprocess, self.aug_list, self.severity)
        # image = randaug(x, self.preprocess, self.strong_aug)

        return image

class StrongAugmenterRand(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False,
                 severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        self.strong_aug = RandAugmentMC(n=2, m=10)

    def __call__(self, x):
        rand_num = random.random()
        if rand_num < 0.5:
            preaugment = get_preaugment()
            x_orig = preaugment(x)
            image = self.preprocess(x_orig)
        else:
            image = augmix(x, self.preprocess, self.aug_list, self.severity)
        # else:
        #     image = randaug(x, self.preprocess, self.strong_aug)

        return image
