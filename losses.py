# Copyright © 2025 Sony Research Inc.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------
# SLIP: https://github.com/facebookresearch/SLIP
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT License
# Modified from github.com/openai/CLIP
# ----------------------------------------------------------
# SSL-HSIC: https://github.com/google-deepmind/ssl_hsic
# Copyright 2021 DeepMind Technologies Limited
# Licensed under the Apache License 2.0.
# ----------------------------------------------------------
import numpy as np
import mpmath

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


import utils


class CLIPLoss(nn.Module):
    def __init__(self, eps=1e-12, gather_batch_with_grad=True):  # 1e-12: default eps of F.normalize
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.eps = eps
        self.gather_batch_with_grad = gather_batch_with_grad

    def get_metric_names(self):
        return ["loss", "clip_loss", "clip_acc"]

    def forward(self, outputs):
        image_embed = outputs["image_embed"]
        text_embed = outputs["text_embed"]
        logit_scale = outputs["logit_scale"]
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2, eps=self.eps)
        text_embed = F.normalize(text_embed, dim=-1, p=2, eps=self.eps)

        # gather features from all GPUs
        if self.gather_batch_with_grad:
            image_embed_all, text_embed_all = utils.all_gather_batch_with_grad([image_embed, text_embed])
        else:
            image_embed_all, text_embed_all = utils.all_gather_batch([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {"loss": loss, "clip_loss": loss, "clip_acc": acc}

    def sim_mat(self, img_emb, txt_emb):
        return self.similarity(img_emb, txt_emb)
    
    def similarity(self, img_emb, txt_emb):  # cosine similarity
        img_emb = F.normalize(img_emb, dim=-1, p=2, eps=self.eps)
        txt_emb = F.normalize(txt_emb, dim=-1, p=2, eps=self.eps)
        return img_emb @ txt_emb.t()



class RFFFeatures(nn.Module):
    def __init__(self, embed_dim, dim_out, first_normalize, parallel=False, eps=1e-12):
        super().__init__()
        self.dim_in = embed_dim
        self.dim_out = dim_out
        self.first_normalize = first_normalize
        self.eps = eps  # default 1e-12 is the default of F.normalize
        self.parallel = parallel  # In a parallel training, frequency vectors must be the same among processes.

    def sample_fourier_weights(self, device=None):
        raise NotImplementedError()

    def rff_features(self, x, w, b):
        z_x = np.sqrt(2 / self.dim_out) * torch.cos(torch.matmul(x, w.T) + b)
        return z_x

    def forward(self, x_img, x_txt):
        """
        x_img.shape: (batch, token, dim), not normalized
        x_txt.shape: (batch, token, dim), not normalized
        """
        if self.first_normalize:
            x_img = F.normalize(x_img, p=2, dim=2, eps=self.eps)
            x_txt = F.normalize(x_txt, p=2, dim=2, eps=self.eps)

        w, b = self.sample_fourier_weights(device=x_img.device)
        z_img = self.rff_features(x_img, w, b)
        z_txt = self.rff_features(x_txt, w, b)

        z_img = torch.mean(z_img, dim=1)
        z_txt = torch.mean(z_txt, dim=1)

        return z_img, z_txt

    def forward_with_weights(self, x_img, w_img, x_txt, w_txt):
        """
        x_img.shape: (batch, token, dim), not normalized
        x_txt.shape: (batch, token, dim), not normalized
        w_img.shape: (batch, token, 1)
        w_txt.shape: (batch, token, 1)
        """
        if self.first_normalize:
            x_img = F.normalize(x_img, p=2, dim=2, eps=self.eps)
            x_txt = F.normalize(x_txt, p=2, dim=2, eps=self.eps)

        w, b = self.sample_fourier_weights(device=x_img.device)
        z_img = self.rff_features(x_img, w, b)
        z_txt = self.rff_features(x_txt, w, b)

        z_img = z_img * w_img
        z_txt = z_txt * w_txt

        z_img = torch.mean(z_img, dim=1)
        z_txt = torch.mean(z_txt, dim=1)

        return z_img, z_txt

    def forward_with_weights_oneside(self, x_feat, w_feat, w_rff, b_rff):
        """
        x.shape: (batch, token, dim)
        w.shape: (batch, tokne, 1)
        w_rff, b_rff: values for the rff featuers
        """
        if self.first_normalize:
            x_feat = F.normalize(x_feat, p=2, dim=2, eps=self.eps)
        z_feat = self.rff_features(x_feat, w_rff, b_rff)
        z_feat = z_feat * w_feat
        z_feat = torch.mean(z_feat, dim=1)

        return z_feat

class RFFFeaturesGaussian(RFFFeatures):
    def __init__(self, embed_dim, dim_out, sigma, first_normalize, parallel=False, eps=1e-12):
        super().__init__(embed_dim, dim_out, first_normalize, parallel=parallel, eps=eps)
        self.inv_sigma = 1. / sigma

    def sample_fourier_weights(self, device=None, dim_out=None):
        if dim_out is None:
            dim_out = self.dim_out
        b = torch.rand(1, dim_out) * 2 * np.pi
        w = torch.normal(0, self.inv_sigma, size=(dim_out, self.dim_in))
        if device is None:
            return w, b
        elif self.parallel:
            w = w.cuda(device)
            b = b.cuda(device)
            dist.broadcast(w, src=0)
            dist.broadcast(b, src=0)
            return w, b
        else:
            return w.to(device, non_blocking=True), b.to(device, non_blocking=True)


class RFFFeaturesIMQ(RFFFeatures):
    """
    This class includes code from SSL-HSIC: https://github.com/google-deepmind/ssl_hsic
    """
    def __init__(self, embed_dim, dim_out, imq_c, first_normalize, parallel=False, eps=1e-12, seed=42):
        super().__init__(embed_dim, dim_out, first_normalize, parallel=parallel, eps=eps)
        self.imq_c = imq_c
        self.amp, self.amp_probs = self.imq_amplitude_frequency_and_probs(self.dim_in)
        self.rng = np.random.default_rng(seed=seed)

    @staticmethod
    def compute_prob(n, x_range):
        """Compute the probablity to sample the random fourier features."""
        probs = [mpmath.besselk((n - 1) / 2, x) * mpmath.power(x, (n - 1) / 2)
                 for x in x_range]
        normalized_probs = [float(p / sum(probs)) for p in probs]
        return np.array(normalized_probs)

    @staticmethod
    def imq_amplitude_frequency_and_probs(n):
        """Returns the range and probablity for sampling RFF."""
        x = np.linspace(1e-12, 100, 10000)  # int(n * 10 / c)
        p = RFFFeaturesIMQ.compute_prob(n, x)
        return x, p

    def sample_fourier_weights(self, device=None, dim_out=None):
        if dim_out is None:
            dim_out = self.dim_out
        amp = self.rng.choice(self.amp, size=(dim_out, 1), p=self.amp_probs)
        directions = self.rng.normal(size=(dim_out, self.dim_in))
        b = self.rng.uniform(size=(1, dim_out)) * 2 * np.pi
        w = directions / np.linalg.norm(directions, axis=-1, keepdims=True) * amp
        w = w / self.imq_c

        w = torch.from_numpy(w.astype(np.float32))
        b = torch.from_numpy(b.astype(np.float32))
        if device is None:
            return w, b
        elif self.parallel:
            w = w.cuda(device)
            b = b.cuda(device)
            dist.broadcast(w, src=0)
            dist.broadcast(b, src=0)
            return w, b
        else:
            return w.to(device, non_blocking=True), b.to(device, non_blocking=True)


class CLIP_WPSE_Loss(nn.Module):
    def __init__(self,
                 rff_trick,
                 comb_weights = None,
                 norm_linear_feat = False,
                 eps = None,
                 normalize_weights_p = None,
                 gather_batch_with_grad = True,
                 ):
        super().__init__()
        self.rff_trick = rff_trick
        self.labels = None
        self.last_local_batch_size = None
        self.norm_linear_feat = norm_linear_feat
        self.eps = self.rff_trick.eps if eps is None else eps
        self.normalize_weights_p = normalize_weights_p
        self.gather_batch_with_grad = gather_batch_with_grad
        if comb_weights is None:
            self.enable_linear_kernel = False
        else:
            self.enable_linear_kernel = True
            assert len(comb_weights) == 2
            self.comb_weights = comb_weights  # weights for (linear kernel, nonlinear kernel)

    def get_metric_names(self):
        return ["loss", "clip_loss", "clip_acc"]

    @staticmethod
    def similarity(feat_image, feat_text):  # inner product
        return feat_image @ feat_text.t()
    
    def forward(self, outputs):
        img_emb = outputs["image_embed"]  # (batch, token, dim)
        img_w = outputs["image_weight"]   # (batch, token, 1)
        txt_emb = outputs["text_embed"]   # (batch, token, dim)
        txt_w = outputs["text_weight"]    # (batch, token, 1)
        logit_scale = outputs["logit_scale"]
        local_batch_size = img_emb.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=img_emb.device
            )
            self.last_local_batch_size = local_batch_size

        if self.enable_linear_kernel:
            z_image, z_text = self.comb_feature(img_emb, img_w, txt_emb, txt_w)
        else:
            z_image, z_text = self.rff_trick.forward_with_weights(img_emb, img_w, txt_emb, txt_w)

        # gather features from all GPUs
        if self.gather_batch_with_grad:
            z_image_all, z_text_all = utils.all_gather_batch_with_grad([z_image, z_text])
        else:
            z_image_all, z_text_all = utils.all_gather_batch([z_image, z_text])

        # similarity as logits
        logits_per_image = logit_scale * self.similarity(z_image, z_text_all)
        logits_per_text = logit_scale * self.similarity(z_text, z_image_all)

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {"loss": loss, "clip_loss": loss, "clip_acc": acc}
    
    def comb_feature(self, img_emb, img_w, txt_emb, txt_w):
        if self.normalize_weights_p:
            img_w = F.normalize(img_w, p=self.normalize_weights_p, dim=1, eps=self.eps)
            txt_w = F.normalize(txt_w, p=self.normalize_weights_p, dim=1, eps=self.eps)

        z_image, z_text = self.rff_trick.forward_with_weights(img_emb, img_w, txt_emb, txt_w)

        # normalize embed. The normalization for z_{domain} is performed inside of rff_trick.forward_with_weights if rff_trick.first_normalize is True.
        img_emb = F.normalize(img_emb, p=2, dim=2, eps=self.eps)
        txt_emb = F.normalize(txt_emb, p=2, dim=2, eps=self.eps)

        x_image = torch.mean(img_emb * img_w, dim=1)  # (batch, dim)
        x_text  = torch.mean(txt_emb * txt_w, dim=1)  # (batch, dim)
        if self.norm_linear_feat:
            x_image = F.normalize(x_image, p=2, dim=1)
            x_text  = F.normalize(x_text,  p=2, dim=1)

        feat_image = torch.cat((x_image*np.sqrt(self.comb_weights[0]), z_image*np.sqrt(self.comb_weights[1])), dim=1)
        feat_text  = torch.cat((x_text *np.sqrt(self.comb_weights[0]), z_text *np.sqrt(self.comb_weights[1])), dim=1)

        return feat_image, feat_text
    
    def feature_oneside(self, x_emb, w_emb, w_rff, b_rff):
        """
        x_emb.shape: (batch, token, dim)
        w_emb.shape: (batch, token, 1)
        """
        if self.normalize_weights_p:
            w_emb = F.normalize(w_emb, p=self.normalize_weights_p, dim=1, eps=self.eps)

        if self.enable_linear_kernel:
            z_emb = self.rff_trick.forward_with_weights_oneside(x_emb, w_emb, w_rff, b_rff)

            x_emb = F.normalize(x_emb, p=2, dim=2, eps=self.eps)
            x_emb = torch.mean(x_emb * w_emb, dim=1)
            if self.norm_linear_feat:
                x_emb = F.normalize(x_emb, p=2, dim=1)
            
            res = torch.cat((x_emb*np.sqrt(self.comb_weights[0]), z_emb*np.sqrt(self.comb_weights[1])), dim=1)
        else:
            res = self.rff_trick.forward_with_weights_oneside(x_emb, w_emb, w_rff, b_rff)

        return res

    def sim_mat(self, img_emb, img_w, txt_emb, txt_w):
        if self.enable_linear_kernel:
            z_image, z_text = self.comb_feature(img_emb, img_w, txt_emb, txt_w)
        else:
            z_image, z_text = self.rff_trick.forward_with_weights(img_emb, img_w, txt_emb, txt_w)
        sim = self.similarity(z_image, z_text)
        return sim
    