# Copyright © 2025 Sony Research Inc.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------
# SLIP: https://github.com/facebookresearch/SLIP
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT License
# ----------------------------------------------------------

import os
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
from torch import nn
from torch.nn import functional as F
import timm

from models import CLIP, CLIP_WPSE_two_proj
from losses import CLIP_WPSE_Loss

def sanity_check(state_dict, pretrained_weights, linear_keyword, visual_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint["state_dict"]

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if "%s.weight" % linear_keyword in k or "%s.bias" % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = visual_keyword + k[len("module."):] \
            if k.startswith("module.") else visual_keyword + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            "{} is changed in linear classifier training.".format(k)

    print("=> sanity check passed.")

def remove_prefix(state_dict):
    prefix = "module."
    # remove effects of DistributedDataParallel
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            # remove prefix
            state_dict[k[len(prefix):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    return state_dict

def load_representation_model(model_dir, load_criterion=False):
    conf = OmegaConf.load(f"{model_dir}/config.yaml")
    model = instantiate(conf.model)
    ckpt_data = torch.load(f"{model_dir}/checkpoint_best.pt", map_location="cpu")
    model.load_state_dict(remove_prefix(ckpt_data["state_dict"]))

    # freeze parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    if not load_criterion:
        return model
    
    criterion = instantiate(conf.criterion)
    return model, criterion


class EncoderWrapper(nn.Module):
    linear_keyword = "head"
    def __init__(self, arch, pretrained, visual_keyword="module.visual."):
        super().__init__()
        self.pretrained = pretrained
        self.visual_keyword = visual_keyword
        self.model = timm.models.create_model(arch, num_classes=1000)
        self.load_checkpoint(pretrained)
        self.freeze()
        self.init_fc_layer()

    def forward(self, x):
        return self.model(x)

    def load_checkpoint(self, pretrained):
        if not os.path.isfile(pretrained):
            raise Exception("Missing pretrained model checkpoint: {}".format(pretrained))
        
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename CLIP pre-trained keys
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith(self.visual_keyword) and not k.startswith(self.visual_keyword + self.linear_keyword):
                # remove prefix
                state_dict[k[len(self.visual_keyword):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = self.model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"%s.weight" % self.linear_keyword, "%s.bias" % self.linear_keyword}

    def freeze(self):
        # freeze all layers but the last fc
        for name, param in self.model.named_parameters():
            if name not in ["%s.weight" % self.linear_keyword, "%s.bias" % self.linear_keyword]:
                param.requires_grad = False

    def init_fc_layer(self):
        getattr(self.model, self.linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
        getattr(self.model, self.linear_keyword).bias.data.zero_()

    def sanity_check(self):
        sanity_check(self.model.state_dict(), self.pretrained, self.linear_keyword, self.visual_keyword)


class CLIPWrapper(nn.Module):
    def __init__(self, model_dir, out_type="before_proj", logit_scaling=False):
        super().__init__()
        self.model_dir = model_dir
        self.model, criterion = load_representation_model(model_dir, load_criterion=True)
        assert type(self.model) is CLIP, f"the model is not CLIP: {type(self.model)}"
        self.eps = criterion.eps
        self.out_type = out_type
        assert self.out_type in ["before_proj", "after_proj", "normed"], f"invalid out_type: {self.out_type}"
        self.logit_scaling = logit_scaling
        if logit_scaling:
            assert self.out_type == "normed", "logit_scale is only supported for out_type=='normed'"
    def forward(self, x):
        if self.out_type == "before_proj":
            return self.model.visual(x)
        elif self.out_type == "after_proj":
            return self.model.encode_image(x)
        else:  # normed
            embed = self.model.encode_image(x)
            embed = F.normalize(embed, dim=-1, p=2, eps=self.eps)
            if self.logit_scaling:
                #embed = embed * (self.model.logit_scale / 2).exp()  # sqrt{logit_scale.exp()}
                embed = embed * self.model.get_logit_scale().sqrt()
            return embed


class CLIPWPSETwoProjWrapper(nn.Module):
    def __init__(self, model_dir, dim_rff, logit_scaling=False):
        super().__init__()
        self.model_dir = model_dir
        self.model, self.criterion = load_representation_model(model_dir, load_criterion=True)
        assert type(self.model) is CLIP_WPSE_two_proj, f"the model is not CLIP_WPSE_two_proj: {type(self.model)}"
        assert type(self.criterion) is CLIP_WPSE_Loss, f"the criterion is not CLIP_WPSE_Loss: {type(self.criterion)}"
        
        self.dim_rff = dim_rff
        self.logit_scaling = logit_scaling
        self.init_rff_coef()
    
    def forward(self, x):
        image_features, image_w = self.model.encode_image(x)
        image_features = self.criterion.feature_oneside(image_features, image_w, self.rff_w, self.rff_b)
        if self.logit_scaling:
            image_features = image_features * self.model.get_logit_scale().sqrt()
        
        return image_features

    def init_rff_coef(self):
        w, b = self.criterion.rff_trick.sample_fourier_weights(device=None, dim_out=self.dim_rff)
        self.rff_w = nn.Parameter(w, requires_grad=False)
        self.rff_b = nn.Parameter(b, requires_grad=False)

class CLIPWPSETwoProjWrapperOptional(nn.Module):
    def __init__(self, model_dir, dim_rff, logit_scaling=False, linfeat_type="before_proj"):
        super().__init__()
        self.model_dir = model_dir
        self.model, self.criterion = load_representation_model(model_dir, load_criterion=True)
        assert type(self.model) is CLIP_WPSE_two_proj
        assert type(self.criterion) is CLIP_WPSE_Loss
        
        self.linfeat_type = linfeat_type
        assert linfeat_type == "before_proj"
        self.dim_rff = dim_rff
        self.logit_scaling = logit_scaling
        self.init_rff_coef()
    
    def forward(self, x):
        x = self.model.visual.forward_features(x)
        x = self.model.visual.fc_norm(x)
        linfeats = self.model.visual.head(x)
            
        v_emb = linfeats @ self.model.image_projection
        w_emb = linfeats @ self.model.image_weight_proj
        w_emb = self.model.weight_act(w_emb)

        linfeats = torch.mean(linfeats * w_emb, dim=1)

        if self.dim_rff > 0:
            #image_features, image_w = self.model.encode_image(x)
            if self.criterion.normalize_weights_p:
                w_emb = F.normalize(w_emb, p=self.criterion.normalize_weights_p, dim=1, eps=self.criterion.eps)
            rffs = self.criterion.rff_trick.forward_with_weights_oneside(v_emb, w_emb, self.rff_w, self.rff_b)

            if self.logit_scaling:
                rffs = rffs * self.model.get_logit_scale().sqrt()

            res  = torch.cat((linfeats, rffs), dim=1)
        else:
            res = linfeats
        
        return res

    def init_rff_coef(self):
        if self.dim_rff > 0:
            w, b = self.criterion.rff_trick.sample_fourier_weights(device=None, dim_out=self.dim_rff)
            self.rff_w = nn.Parameter(w, requires_grad=False)
            self.rff_b = nn.Parameter(b, requires_grad=False)
        else:
            self.rff_w = None
            self.rff_b = None


class LinearProbe(nn.Module):
    def __init__(self, feature_extractor, dim_feat, dim_class):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc_layer = nn.Linear(dim_feat, dim_class)
        self.init_fc_layer()

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.detach()
        return self.fc_layer(x)

    def init_fc_layer(self):
        self.fc_layer.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_layer.bias.data.zero_()

    def sanity_check(self):
        pass
