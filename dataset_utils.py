# Copyright © 2025 Sony Research Inc.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------
# SLIP: https://github.com/facebookresearch/SLIP
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT License
# ----------------------------------------------------------
from collections import defaultdict
import json
import os

import numpy as np
from PIL import Image, ImageFile

import torch
from torchvision import transforms
from torchvision import datasets as t_datasets
from datasets import load_dataset as load_dataset_hf


ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata):
        self.dataset = dataset
        self.root = root
        if self.dataset == "coco":
            samples = defaultdict(list)
            with open(metadata) as f:
                annotations = json.load(f)["annotations"]
            for ann in annotations:
                samples[ann["image_id"]].append(ann["caption"])
            self.samples = [(k, v) for k, v in samples.items()]
        elif self.dataset == "cc12m" or self.dataset == "cc3m":
            self.samples = np.load(metadata, allow_pickle=True)
        elif self.dataset == "redcaps":
            with open(metadata) as f:
                annotations = json.load(f)
            self.samples = [(ann["image_id"], ann["subreddit"], ann["caption"]) for ann in annotations]
        else:
            raise ValueError(f"Invalid dataset: {self.dataset}")
        

    def get_raw_item(self, i):
        if self.dataset == "coco":
            index, captions = self.samples[i]
            path = os.path.join(self.root, "train2017", "{:012d}.jpg".format(index))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == "cc3m":
            ann = self.samples[i]
            filename, captions = ann["image_id"], ann["captions"]
            path = os.path.join(self.root, str(filename))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == "cc12m":
            ann = self.samples[i]
            filename, captions = ann["image_name"], ann["captions"]
            path = os.path.join(self.root, filename)
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == "redcaps":
            image_id, subreddit, caption = self.samples[i]
            path = os.path.join(self.root, subreddit, f"{image_id}.jpg")
            img = pil_loader(path)

        return img, caption

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class ImageCaptionDatasetCLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            image = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)
    
class HuggingFaceDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, path, cache_dir, transform, tokenizer, split="train"):
        self.dataset = load_dataset_hf(path, cache_dir=cache_dir, split=split)
        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = self.dataset[index]
        img = self.transform(item["jpg"].convert("RGB"))
        caption = self.tokenizer(item["txt"])
        return img, caption
    
    def __len__(self):
        return len(self.dataset)


def get_random_split(dataset, split, seed):
    assert split in ["train", "val", "test"]
    if split == "test":
        return dataset

    dataset_train, dataset_val = torch.utils.data.random_split(
            dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(seed),
            )
    if split == "train":
        return dataset_train
    else:  # val
        return dataset_val

def get_downstream_dataset(catalog, name, is_train=None, transform=None, split=None, seed=314):
    entry = catalog[name]
    root = entry["path"]
    if split is None:
        assert is_train is not None
        split = "train" if is_train else "test"
    assert split in ["train", "val", "test"]

    if entry["type"] == "imagefolder":
        if "val" in entry:
            dataset = t_datasets.ImageFolder(os.path.join(root, entry[split]), transform=transform)
        else:
            if split == "test":
                dataset = t_datasets.ImageFolder(os.path.join(root, entry[split]), transform=transform)
            else:
                dataset = t_datasets.ImageFolder(os.path.join(root, entry["train"]), transform=transform)
                dataset = get_random_split(dataset, split, seed=seed)
    elif entry["type"] == "special":
        if name == "cifar10":
            dataset = t_datasets.CIFAR10(root, train=(split != "test"), transform=transform, download=True)
            if split != "test":
                dataset = get_random_split(dataset, split, seed=seed)
        elif name == "cifar100":
            dataset = t_datasets.CIFAR100(root, train=(split != "test"), transform=transform, download=True)
            if split != "test":
                dataset = get_random_split(dataset, split, seed=seed)
        elif name == "stl10":
            if split == "test":
                dataset = t_datasets.STL10(root, split=split, transform=transform, download=True)
            else:
                dataset = t_datasets.STL10(root, split="train", transform=transform, download=True)
                dataset = get_random_split(dataset, split, seed=seed)
        elif name == "flowers":
            dataset = t_datasets.Flowers102(root, split=split, transform=transform, download=True)
        elif name == "dtd":
            dataset = t_datasets.DTD(root, split=split, transform=transform, download=True)
        elif name == "aircraft":
            dataset = t_datasets.FGVCAircraft(root, split=split, transform=transform, download=True)
    elif entry["type"] == "filelist":
        path = entry[split]
        val_images = os.path.join(root, path + "_images.npy")
        val_labels = os.path.join(root, path + "_labels.npy")
        target_transform = None
        dataset = FileListDataset(val_images, val_labels, transform, target_transform)
    else:
        raise Exception("Unknown dataset")

    return dataset


def get_dataset(train_transform, tokenizer, cfgs):
    args = cfgs.dataset

    if cfgs.model._target_.startswith("models.CLIP"):
        if args.name == "huggingface_dataset":
            return HuggingFaceDatasetWrapper(args.path, args.cache_dir, train_transform, tokenizer)
        else:
            return ImageCaptionDatasetCLIP(args.name, args.root, args.metadata, train_transform, tokenizer)
    else:
        raise ValueError(f"invalid model: {cfgs.model._target_}")


def get_img_transform(cfgs, mode="pretraining"):
    assert mode in ["pretraining", "zeroshot_classification", "downstream_classification"]

    args = cfgs.dataset.get("transform", dict())
    interpolation = args.get("interpolation", None)
    if interpolation is None:
        interpolation = transforms.InterpolationMode.BILINEAR
        print("BILINEAR is activated.")
    elif interpolation == "bicubic":
        interpolation = transforms.InterpolationMode.BICUBIC
        print("BICUBIC is activated.")
    else:
        raise ValueError(f"invalid interpolation: {interpolation}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if mode == "pretraining":
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=interpolation),
                transforms.ToTensor(),
                normalize
            ])
        val_transform = transforms.Compose([
                transforms.Resize(224, interpolation=interpolation),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        return train_transform, val_transform
    elif mode == "zeroshot_classification":
        val_transform = transforms.Compose([
            transforms.Resize(224, interpolation=interpolation),
            transforms.CenterCrop(224),
            lambda x: x.convert("RGB"),
            transforms.ToTensor(),
            normalize
        ])
        return val_transform
    elif mode == "downstream_classification":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            lambda x: x.convert("RGB"),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=interpolation),
            transforms.CenterCrop(224),
            lambda x: x.convert("RGB"),
            transforms.ToTensor(),
            normalize,
        ])
        return train_transform, val_transform