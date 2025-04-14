# Copyright © 2025 Sony Research Inc.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------
# SLIP: https://github.com/facebookresearch/SLIP
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT License
# ----------------------------------------------------------
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from collections import OrderedDict
import json
import math
import os
import sys
import time
import mlflow

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.datasets import ImageFolder

import dataset_utils
import losses
from tokenizer import SimpleTokenizer
import utils
import eval_zeroshot


@hydra.main(version_base=None, config_path="configs")
def main_single_node(args):
    main(args)

def main(args):
    args = utils.init_distributed_mode(args)

    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    best_acc1 = 0

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model: {}".format(args.model._target_))
    model = instantiate(args.model)
    model.cuda(args.gpu)
    print(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)

    # define loss function (criterion) and optimizer
    criterion = instantiate(args.criterion).cuda(args.gpu)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_scaler)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            epoch = checkpoint["epoch"] if "epoch" in checkpoint else 0
            args.start_epoch = epoch
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                result = model.module.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                result = model.load_state_dict(checkpoint["state_dict"], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint["optimizer"]) if "optimizer" in checkpoint else ()
            scaler.load_state_dict(checkpoint["scaler"]) if "scaler" in checkpoint else ()
            best_acc1 = checkpoint["best_acc1"]
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, "checkpoint.pt")
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location="cpu")
            args.start_epoch = latest_checkpoint["epoch"]
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                result = model.module.load_state_dict(latest_checkpoint["state_dict"])
            else:
                result = model.load_state_dict(latest_checkpoint["state_dict"])
            optimizer.load_state_dict(latest_checkpoint["optimizer"])
            scaler.load_state_dict(latest_checkpoint["scaler"])
            best_acc1 = latest_checkpoint["best_acc1"]
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint["epoch"]))

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    train_transform, val_transform = dataset_utils.get_img_transform(args, mode="pretraining")

    train_dataset = dataset_utils.get_dataset(train_transform, tokenizer, args)
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, "dataset_catalog.json")) as f:
        root = json.load(f)["imagenet"]["path"]
    val_split = args.get("imagenet_valsplit", "val")
    val_dataset = ImageFolder(os.path.join(root, val_split), val_transform)

    # dist eval resamples data to pad uneven batch sizes
    # make sure num_samples = 0 mod num_gpus for exact acc
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    if args.evaluate:
        assert RuntimeError("args.evaluate is no longer available")

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    if utils.is_main_process():
        if args.mlflow:
            mlflow.set_tracking_uri(args.mlflow.tracking_uri)
            mlflow.set_experiment(args.mlflow.experiment)
            mlflow.start_run(run_name=args.mlflow.run_name)


    print(args)
    if utils.is_main_process():
        OmegaConf.save(args, os.path.join(args.output_dir, "config.yaml"))

    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)

        if (epoch + 1) % args.eval_freq != 0:
            continue

        val_stats = validate_zeroshot(val_loader, model, tokenizer, criterion, args)
        acc1 = val_stats["acc1"]

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        print("=> saving checkpoint")
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        if args.get("ckpt_save_interval", None) is not None:
            if (epoch+1) % args.ckpt_save_interval == 0:
                ckpt_suffix = f"ep{epoch+1}"
            else:
                ckpt_suffix = None
        else:
            ckpt_suffix = None
        utils.save_on_master({
                "epoch": epoch + 1,
                "state_dict": model_state_dict,
                "optimizer" : optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc1": best_acc1,
                "args": args,
            }, is_best, args.output_dir,
            suffix=ckpt_suffix
            )

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     **{f"test_{k}": v for k, v in val_stats.items()},
                     "epoch": epoch}

        if utils.is_main_process():
            if args.mlflow:
                for key, value in log_stats.items():
                    mlflow.log_metric(key, value, step=epoch+1)
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    if utils.is_main_process() and args.mlflow:
        mlflow.end_run()

    if utils.is_dist_avail_and_initialized():
        dist.destroy_process_group()


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter("Time", ":6.2f")
    data_time = AverageMeter("Data", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    metric_names = criterion.get_metric_names()
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ":.2e")) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    assert args.amp_dtype in [None, "bfloat16", "float16"]
    if args.amp_dtype == "float16":
        amp_dtype = torch.float16
        enable_amp = True
        print("Use FP16.")
    elif args.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
        enable_amp = True
        print("Use BF16.")
    else:  # None
        amp_dtype = None
        enable_amp = False
        print("Disable AMP training.")
        assert args.disable_scaler
    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

        # compute output
        with amp.autocast(enabled=enable_amp, dtype=amp_dtype):
            outputs = model(*inputs)
            loss_dict = criterion(outputs)
            loss = loss_dict["loss"]
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            torch.save(
                {"inputs": inputs,
                 "outputs": outputs,
                },
                os.path.join(args.output_dir, "dump_loss_nan_inout_rank{}.pgz".format(utils.get_rank()))
            )
            if utils.is_main_process():
                torch.save(
                    {"losses": loss_dict,
                     "state_dict": model.state_dict()
                    },
                    os.path.join(args.output_dir, "dump_loss_nan_model.pgz")
                )
                print("Loss is {}, stopping training".format(loss.item()))
            time.sleep(5)
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]

        # gradient clipping
        if args.clip_grad:
            scaler.unscale_(optimizer)
            if args.clip_grad.type == "norm":
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad.max_norm)
            elif args.clip_grad.type == "value":
                torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad.max_value)
            else:
                raise ValueError(f"invalid gradient clipping type: {args.clip_grad.type}")

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale
        utils.get_model(model).clamp_logit_scale()
        logit_scale = utils.get_model(model).get_logit_scale().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process():
                log_stats = {**{k: v.item() for k, v in loss_dict.items()},
                            "scaler": scaler.get_scale(),
                            "logit": logit_scale}
                if args.mlflow:
                    current_step = iters_per_epoch * epoch + data_iter
                    for key, value in log_stats.items():
                        mlflow.log_metric(key, value, step=current_step)

            progress.synchronize()
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "logit_scale": logit_scale}

def get_prompts(dataset_name):
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, "templates.json")) as f:
        templates = json.load(f)[dataset_name]
    with open(os.path.join(cwd, "labels.json")) as f:
        labels = json.load(f)[dataset_name]
    return templates, labels

def validate_zeroshot(val_loader, model, tokenizer, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    print("=> encoding captions")
    templates, labels = get_prompts("imagenet")

    with torch.no_grad():
        if isinstance(criterion, losses.CLIPLoss):
            text_features = eval_zeroshot.get_text_features(model, tokenizer, labels, templates, f"cuda:{args.gpu}")
        elif isinstance(criterion, losses.CLIP_WPSE_Loss):
           dict_text_features = eval_zeroshot.get_text_features_WPSE(
                model, criterion, tokenizer, labels, templates, f"cuda:{args.gpu}")
        else:
            raise ValueError(f"invalid criterion type: {type(criterion)}")
        
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            if isinstance(criterion, losses.CLIPLoss):
                logits_per_image = eval_zeroshot.get_logits_per_image(model, images, text_features)
            elif isinstance(criterion, losses.CLIP_WPSE_Loss):
                logits_per_image = eval_zeroshot.get_logits_per_image_WPSE(
                                    model, criterion, images, dict_text_features)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits_per_image, target, topk=(1, 5))
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.synchronize()
                progress.display(i)

    progress.synchronize()
    print("0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}"
          .format(top1=top1, top5=top5))
    return {"acc1": top1.avg, "acc5": top5.avg}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        world_size = utils.get_world_size()
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device="cuda")
        t = t / world_size
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = round(t[1])
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


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


if __name__ == "__main__":
    main_single_node()
