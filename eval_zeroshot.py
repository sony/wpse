# Copyright © 2025 Sony Research Inc.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------
# SLIP: https://github.com/facebookresearch/SLIP
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT License
# ----------------------------------------------------------
import argparse
from hydra.utils import instantiate
from omegaconf import OmegaConf
from collections import OrderedDict
import json
import os
import time
from sklearn import metrics
import pandas as pd
from tqdm import tqdm


import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import dataset_utils
import losses
from tokenizer import SimpleTokenizer
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(description="zero-shot evaluations", add_help=False)
    parser.add_argument("--output-dir", default="./", type=str, help="output dir")
    parser.add_argument("--batch-size", default=256, type=int, help="batch_size")
    parser.add_argument("-j", "--workers", default=10, type=int, metavar="N",
                        help="number of data loading workers per process")
    parser.add_argument("--resume", default="", type=str, help="path to latest checkpoint")
    parser.add_argument("--gpu", default=0, type=int, help="gpu id")
    parser.add_argument("--forced", action="store_true", help="When true, all evaluations will be performed even if some datasets were already evaluated.")
    parser.add_argument("--task-list", nargs="+", help="target datasets")
    parser.add_argument("--nrepeat", default=1, type=int, help="num of repetition")

    return parser


def main(args):
    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        ckpt_path = args.resume
        assert os.path.isfile(ckpt_path)
        csv_filename = os.path.join(
                        os.path.splitext(args.resume)[0],
                        "results_zeroshot.csv")
        os.makedirs(os.path.splitext(args.resume)[0], exist_ok=True)
    elif os.path.isfile(os.path.join(args.output_dir, "checkpoint_best.pt")):
        ckpt_path = os.path.join(args.output_dir, "checkpoint_best.pt")
        csv_filename = os.path.join(args.output_dir, "results_zeroshot.csv")
    else:
        raise Exception("no checkpoint found")
    
    if args.gpu < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v
    
    # create model
    old_args = OmegaConf.load(os.path.join(args.output_dir, "config.yaml"))
    print("=> creating model: {}".format(old_args.model))
    model = instantiate(old_args.model)
    model.to(device)
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt["epoch"]))

    # load criterion
    if hasattr(old_args, "distributed"):
        old_args.distributed = False
    criterion = instantiate(old_args.criterion)
    criterion.to(device)

    cudnn.benchmark = True

    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, "dataset_catalog.json")) as f:
        catalog = json.load(f)

    with open(os.path.join(cwd, "templates.json")) as f:
        all_templates = json.load(f)

    with open(os.path.join(cwd, "labels.json")) as f:
        all_labels = json.load(f)

    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    val_transform = dataset_utils.get_img_transform(old_args, mode="zeroshot_classification")

    if args.task_list is None:
        task_list = catalog
    else:
        task_list = args.task_list
    
    for i in range(args.nrepeat):
        oneloop(
            csv_filename, task_list, catalog, all_templates, all_labels,
            model, criterion, tokenizer, device, val_transform,
            )


def oneloop(
        csv_filename, task_list, catalog, all_templates, all_labels,
        model, criterion, tokenizer, device, val_transform,
        ):
    results = pd.DataFrame(columns=["task", "metric", "score"])

    for d in task_list:
        if d not in catalog:
            print(f"Invalid task: {d}. skipped.")
            continue
        if d in results["task"].unique() and not args.forced:
            print(f"{d} was already evaluated. skipped.")
            continue

        print("Evaluating {}".format(d))
        val_dataset = dataset_utils.get_downstream_dataset(catalog, name=d, is_train=False, transform=val_transform)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False)

        templates = all_templates[d]
        labels = all_labels[d]

        is_acc = d not in ["aircraft", "pets", "caltech101", "flowers"]
        if is_acc:
            topk = [1, 3, 5]  # top
        else:
            topk = None

        acc_or_outputs = validate_zeroshot(val_loader, templates, labels, model, criterion, tokenizer, topk, device)

        if d in ["aircraft", "pets", "caltech101", "flowers"]:
            metric = mean_per_class(*acc_or_outputs)
        else:
            metric = acc_or_outputs

        if topk is None:
            if d in ["aircraft", "pets", "caltech101", "flowers"]:
                metric_type = ["mean_per_class_acc"]
            else:
                metric_type = ["specific"]
            metric = [metric]
        else:
            metric_type = [f"acc{k}" for k in topk]
        new_record = pd.DataFrame.from_dict({
            "task": [d] * len(metric_type),
            "metric": metric_type,
            "score": metric,
            "timestamp": [time.ctime()] * len(metric_type),
        })
        print(new_record)
        results = pd.concat([results, new_record], ignore_index=True)

    print("all results:")
    print(results)
    if os.path.isfile(csv_filename):
        results_prev = pd.read_csv(csv_filename)
        assert {"task", "metric", "score"}.issubset(results_prev.columns)
        results = pd.concat([results_prev, results])
    results.to_csv(csv_filename, index=False)


def get_text_features(model, tokenizer, labels, templates, device):
    text_features = []
    for label in labels:
        if isinstance(label, list):
            texts = [t.format(l) for t in templates for l in label]
        else:
            texts = [t.format(label) for t in templates]
        texts = tokenizer(texts).to(device, non_blocking=True)
        texts = texts.view(-1, 77).contiguous()
        class_embeddings = utils.get_model(model).encode_text(texts)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.mean(dim=0)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        text_features.append(class_embeddings)
    text_features = torch.stack(text_features, dim=0)
    return text_features

def get_logits_per_image(model, images, text_features):
    # encode images
    image_features = utils.get_model(model).encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = image_features @ text_features.t()
    return logits_per_image


def get_text_features_WPSE(model, criterion, tokenizer, labels, templates, device):
    w, b = criterion.rff_trick.sample_fourier_weights(device=device)
    text_features = []
    for label in labels:
        if isinstance(label, list):
            texts = [t.format(l) for t in templates for l in label]
        else:
            texts = [t.format(label) for t in templates]
        texts = tokenizer(texts).to(device, non_blocking=True)
        texts = texts.view(-1, 77).contiguous()
        class_emb, class_w = utils.get_model(model).encode_text(texts)
        z_class = criterion.feature_oneside(class_emb, class_w, w, b)
        z_class = z_class.mean(dim=0)
        text_features.append(z_class)
    text_features = torch.stack(text_features, dim=0)
    return {"text_features": text_features,
            "w": w,
            "b": b}

def get_logits_per_image_WPSE(model, criterion, images, dict_text_features):
    text_features = dict_text_features["text_features"]
    w = dict_text_features["w"]
    b = dict_text_features["b"]
    # encode images
    image_features, image_w = utils.get_model(model).encode_image(images)
    image_features = criterion.feature_oneside(image_features, image_w, w, b)

    # similarity as logits
    logits_per_image = criterion.similarity(image_features, text_features)
    return logits_per_image

def validate_zeroshot(val_loader, templates, labels, model, criterion, tokenizer, topk, device):
    # switch to evaluate mode
    model.eval()
    total_correct = None if topk is None else [0] * len(topk)
    total_images = 0

    all_outputs = []
    all_targets = []

    print("=> encoding captions")
    with torch.no_grad():
        if isinstance(criterion, losses.CLIPLoss):
            text_features = get_text_features(model, tokenizer, labels, templates, device)
        elif isinstance(criterion, losses.CLIP_WPSE_Loss):
            dict_text_features = get_text_features_WPSE(model, criterion, tokenizer, labels, templates, device)
        else:
            raise ValueError(f"invalid criterion type: {type(criterion)}")

        for images, target in tqdm(val_loader, leave=False):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if isinstance(criterion, losses.CLIPLoss):
                logits_per_image = get_logits_per_image(model, images, text_features)
            elif isinstance(criterion, losses.CLIP_WPSE_Loss):
                logits_per_image = get_logits_per_image_WPSE(model, criterion, images, dict_text_features)
            if topk is not None:
                # measure accuracy and record loss
                maxk = max(topk)
                _, pred = logits_per_image.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.reshape(1, -1).expand_as(pred))
                
                for i, k in enumerate(topk):
                    correct_k = correct[:k].float().sum()
                    total_correct[i] += correct_k.item()

                total_images += images.size(0)
            else:
                all_outputs.append(logits_per_image.cpu())
                all_targets.append(target.cpu())
            
    if topk is not None:
        return [ 100 * score / total_images for score in total_correct]
    else:
        return torch.cat(all_outputs), torch.cat(all_targets)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_per_class(outputs, targets):
    pred = outputs.argmax(1)
    confusion_matrix = metrics.confusion_matrix(targets, pred)
    per_classes = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    return 100 * per_classes.mean()


def roc_auc(outputs, targets):
    pos_score = outputs[:, 1] - outputs[:, 0]
    metric = metrics.roc_auc_score(targets, pos_score)

    return 100 * metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser("zero-shot evaluations", parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
