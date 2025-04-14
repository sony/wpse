# Copyright © 2025 Sony Research Inc.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import json
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import dataset_utils
import models_linear
import utils


@hydra.main(version_base=None, config_path="configs_linear", config_name="defaults_linear")
def main(args):
    args = utils.init_distributed_mode(args)

    if utils.get_world_size() != 1:
        raise ValueError("multi-gpu is currently not supported.")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create model
    print("=> creating model '{}'".format(args.model))
    model = instantiate(args.model)

    args.workers = args.workers
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    cudnn.benchmark = True

    # Data loading code
    with open("dataset_catalog.json") as f:
        catalog = json.load(f)

    if args.dataset.name is not None:
        task_list = [args.dataset.name]
    else:
        task_list = args.dataset.list
        if task_list is None:
            task_list = catalog.keys()


    _, val_transform = dataset_utils.get_img_transform(args, mode="downstream_classification")

    print(args)
    OmegaConf.save(args, os.path.join(args.output_dir, "config_dump_linear_feats.yaml"))

    os.makedirs(os.path.join(args.output_dir, "frozen_feats"), exist_ok=True)
    for task in task_list:
        for split in ("train", "val", "test"):
            print(f"task: {task}, split: {split}")
            output_npz = os.path.join(args.output_dir, "frozen_feats", f"{task}_{split}.npz")
            if os.path.exists(output_npz):
                raise FileExistsError(f"{output_npz} already exists")
            
            split_dataset = dataset_utils.get_downstream_dataset(catalog, task, split=split, transform=val_transform)

            if args.distributed:
                data_sampler = torch.utils.data.distributed.DistributedSampler(split_dataset)
            else:
                data_sampler = None

            data_loader = torch.utils.data.DataLoader(
                split_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, sampler=data_sampler, drop_last=False)

            features, labels = extract_features(model, data_loader, args)
            print(f"task: {task}, split: {split}, features.shape: {features.shape}, labels.shape: {labels.shape}")
            
            np.savez(output_npz, x=features, y=labels)


def extract_features(model, data_loader, args):
    assert isinstance(model, models_linear.LinearProbe)
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, target in tqdm(data_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            
            feature = model.feature_extractor(images)
            features.append(feature.detach().cpu().numpy())
            labels.append(target.detach().numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

if __name__ == "__main__":
    main()
