# Weighted Point Set Embedding for Multimodal Contrastive Learning Toward Optimal Similarity 

This repository houses the official implementation of the paper titled "Weighted Point Set Embedding for Multimodal Contrastive Learning Toward Optimal Similarity", which is presented at ICLR 2025.
[[`OpenReview`](https://openreview.net/forum?id=uSz2K30RRd)]

Contact: Toshimitsu Uesaka toshimitsu.uesaka@sony.com

## Installation
### Docker
```
docker build -t <image_name> installation
```
### venv+pip
```
python -m venv <env_name>
source <env_name>/bin/activate
pip install -r installation/requirements.txt
pip install -r installation/requirements_torch.txt
pip install -r installation/requirements_rapids.txt
```

## Trained model parameters
We release the following model parameters trained on CC12M.
* WPSE ViT-B/16 Gaussian ($\sigma=0.5$):  [parameters and config](https://zenodo.org/records/15210698)
* WPSE ViT-B/16 IMQ ($c=0.5$): [parameters and config](https://zenodo.org/records/15210795)

## Datasets Setup
### Conceptual Captions Setup
We use HuggingFace datasets for [CC3M](https://huggingface.co/datasets/pixparse/cc3m-wds) and [CC12M](https://huggingface.co/datasets/pixparse/cc12m-wds).
Before use them, rewrite `cache_dir` in yaml configuration files.
When a dataset is not in `cache_dir`, HuggingFace [`datasets.load_dataset()`](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/loading_methods#datasets.load_dataset) downloads the dataset.

**CC3M** ([`configs/dataset/example_cc3m.yaml`](configs/dataset/example_cc3m.yaml))
```yaml
name: huggingface_dataset
path: pixparse/cc3m-wds
cache_dir: /path/to/huggingface/cache/directory
...
```

**CC12M** ([`configs/dataset/example_cc12m.yaml`](configs/dataset/example_cc12m.yaml))
```yaml
name: huggingface_dataset
path: pixparse/cc12m-wds
cache_dir: /path/to/huggingface/cache/directory
...
```

### ImageNet Setup
ImageNet is used as a validation dataset in training. The path to ImageNet is read from `dataset_catalog.json`.
Copy [dataset_catalog_example.json](dataset_catalog_example.json) to `dataset_catalog.json` and rewrite dataset paths.
```
cp dataset_catalog_example.json dataset_catalog.json
```

The scripts requires the following directory structure for using `torchvision.datasets.ImageFolder`:
* /path/to/imagenet/
  - train/
    * n01440764/
      - n01440764_10026.JPEG
      - ...
      - n01440764_9981.JPEG
    * n01443537/
    * ...
    * n15074101/
  - val/
    * n01440764/
      - ILSVRC2012_val_00002138.JPEG
      - ...
      - ILSVRC2012_val_00048969.JPEG
    * ...
    * n15074101/
  - your_own_split_if_you_have/
    * n01440764/
    * ...
    * n15074104/

If you use customized data split (such as a subset of the train split) for the validation in training, please designate it as `imagenet_valsplit` in a configuration yaml for training.
```yaml
...
imagenet_valsplit: your_own_split_if_you_have
...
```

### Downstream Datasets Setup
The scripts read dataset paths from [dataset_catalog.json](#imagenet-setup).
The class labels and caption templates for zero-shot evaluation are read from [labels.json](labels.json) and [templates.json](templates.json).

**CIFAR10, CIFAR100, STL-10, Flowers102, DTD, Aircraft, and MNIST** are loaded by [`torchvision.datasets`](https://pytorch.org/vision/stable/datasets.html).
**For other datasets**, please use scripts from [VISSL](https://github.com/facebookresearch/vissl/tree/main/extra_scripts/datasets).

## Training
Configuration yaml files are placed in `configs/`. We use [Hydra](https://hydra.cc/) as a configuration management tool.
As results of training, following files are created in a directory designated by `output_dir` in a configuration yaml.
* config.yaml
  - A copy of the configuration file used in the training.
* log.txt
  - A log file of, for example, training losses and validation results.
* checkpoint.pt
  - The checkpoint file at the last epoch.
* checkpoint_best.pt
  - The checkpoint file that achieved the best score in the validation.

In addition, `mlflow` also serves as loggers if you specify it in the configuration yaml.

### Single-GPU training
The following examples run trainings of CLIP models on CC3M. For WPSE models (Gaussian kernel, $\sigma = 0.5, (\alpha_1, \alpha_2) = (0.667, 0.333)$), please replace the config name, `example_cc3m_clip`, with `example_cc3m_wpse`.

```bash
config_name=example_cc3m_clip

python main.py --config-name $config_name
```

### Single-node Multi-GPU training (4 GPUs)
```bash
config_name=example_cc3m_clip

torchrun --standalone --nnodes=1 \
         --nproc_per_node=4 main.py \
         --config-name $config_name
```

### Multi-node Multi-GPU training (4 nodes)
```bash
config_name=example_cc3m_clip
hostfile=<host file>
hostname=<host address>
port_num=<port num>

mpirun -np 4 -map-by ppr:1:node -hostfile $hostfile \
        python main_multi_nodes.py \
        --config-name $config_name \
        hostname=$hostname \
        port_num=$port_num
```

## Evaluation
### Zero-shot classification
For the zero-shot classification evaluation, `eval_zeroshot.py` is used.
In the following examples, `/path/to/directory` is supposed to contain `checkpoint_best.pt` and `config.yaml`.
As a result of zero-shot evaluation, `results_zeroshot.csv` will be created in `/path/to/directory`.

The dataset paths are read from [dataset_catalog.json](#imagenet-setup), and the class labels and caption templates are read from [labels.json](labels.json) and [templates.json](templates.json).

```bash
tgt_dir=/path/to/directory
gpuid=0
python eval_zeroshot.py --output-dir $tgt_dir --gpu $gpuid
```

You can also conduct evaluations on a subset of datasets as follows:
```bash
python eval_zeroshot.py --output-dir $tgt_dir --gpu $gpuid \
       --tasklist cifar10 cifar100 stl10
```

### Linear probing
The scripts conduct linear probing evaluations in a two-stage manner.
First, features for linear classifiers are extracted.
After that, linear classifiers are fit, using extracted features.

#### Extracting features
To extract features after the last projection layer, run the following scripts.
Here, `/path/to/model/dir` is supposed to contain `checkpoint_best.pt` and `config.yaml`.
After running the above scripts, the extracted features are placed in `/path/to/workspace/frozen_feats/`

**For CLIP models**
```bash
workspace=/path/to/workspace
model_dir=/path/to/model/dir
python dump_linear_feats.py --config-name example_clip_dump \
       output_dir=$workspace \
       model.feature_extractor.model_dir=$model_dir
```
**For WPSE models**
```bash
workspace=/path/to/workspace
model_dir=/path/to/model/dir
python dump_linear_feats.py --config-name example_wpse_d1024_dump \
       output_dir=$workspace \
       model.feature_extractor.model_dir=$model_dir
```

To extract features before the last projection layer, run the following scripts.

**For CLIP models**
```bash
python dump_linear_feats.py --config-name example_clip_bef_proj_dump \
       output_dir=$workspace \
       model.feature_extractor.model_dir=$model_dir
```
**For WPSE models**
```bash
python dump_linear_feats.py --config-name example_wpse_bef_proj_dump \
       output_dir=$workspace \
       model.feature_extractor.model_dir=$model_dir
```

#### Fitting linear classifiers
After extracting features, `rapids_linear_probe.py` conducts fitting linear classifiers.
The information about datasets and data splits are read from [dataset_catalog.json](#imagenet-setup).
`/path/to/workspace` is supposed to contain `frozen_feats/` created by `dump_linear_feats.py`
```bash
workspace=/path/to/workspace
task=cifar10
python rapids_linear_probe.py $workspace $task
```

## License
This repository is licensed under the MIT license. See [LICENSE](LICENSE) for details.

This repository includes work from the following repositories:
* SLIP (https://github.com/facebookresearch/SLIP)
  - Copyright (c) Meta Platforms, Inc. and affiliates.
  - Distributed under the MIT License. 
* SSL-HSIC (https://github.com/google-deepmind/ssl_hsic)
  - Copyright 2021 DeepMind Technologies Limited.
  - Distributed under the Apache License 2.0.

## Citation
```
@inproceedings{
uesaka2025weighted,
title={Weighted Point Set Embedding for Multimodal Contrastive Learning Toward Optimal Similarity Metric},
author={Toshimitsu Uesaka and Taiji Suzuki and Yuhta Takida and Chieh-Hsin Lai and Naoki Murata and Yuki Mitsufuji},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=uSz2K30RRd}
}
```
