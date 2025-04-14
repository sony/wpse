# Copyright © 2025 Sony Research Inc.
#
# This source code is licensed under the license found in the
# LICENSE in the root directory of this source tree.
# ----------------------------------------------------------
# SLIP: https://github.com/facebookresearch/SLIP
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT License
# ----------------------------------------------------------

from collections import OrderedDict
from functools import partial
from typing import Optional, Union

import numpy as np

import timm
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: Optional[torch.Tensor] = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


########################################
# activation functions for logit_scale
class LogitScaleExp(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.exp()
    def inverse(self, x: float) -> float:
        # This module is used for the initialization of logit_scale
        return np.log(x)

class LogitScaleIdentity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    def inverse(self, x: float) -> float:
        # This module is used for the initialization of logit_scale
        return x

########################################

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 transformer_null_token_emb: Optional[str] = None,
                 config_logit_scale: Optional[dict] = None,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        if transformer_null_token_emb is None:
            transformer_null_token_emb = "none"
        self.transformer_null_token_emb = transformer_null_token_emb
        if transformer_null_token_emb == "none":
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        elif transformer_null_token_emb == "sequel":
            self.token_embedding = nn.Embedding(vocab_size + self.context_length - 1, transformer_width)
        else:
            raise ValueError(f"invalid null_token_embedding: {transformer_null_token_emb}")

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        if config_logit_scale is None:
            config_logit_scale = {}
        init_logit_scale = config_logit_scale.get("init", {"type": "scale", "value": 1 / 0.07})
        max_logit_scale = config_logit_scale.get("max", {"type": "raw_scale", "value": 4.6052})
        min_logit_scale = config_logit_scale.get("min", {"type": "raw_scale", "value": 0})
        self.logit_scale_act = config_logit_scale.get("activation", LogitScaleExp())
        self.init_logit_scale = self.raw_logit_scale(init_logit_scale)
        self.max_logit_scale = self.raw_logit_scale(max_logit_scale)
        self.min_logit_scale = self.raw_logit_scale(min_logit_scale)

        self.logit_scale = nn.Parameter(torch.ones([]) * self.init_logit_scale)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self) -> torch.Tensor:
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        x = self.visual(image)
        x = x @ self.image_projection

        return x
    
    def encode_image_all_patches(self, image: torch.Tensor) -> torch.Tensor:
        x = self.visual.forward_features(image)
        # forward_head() but keep all patches
        x = self.visual.fc_norm(x)
        x = self.visual.head(x)
        # last projection
        x = x @ self.image_projection

        return x

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def encode_text_all_tokens(self, text: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        
        # x.shape = [batch_size, n_ctx, transformer.width]
        # leave all features, unlike encode_text()
        x = x @ self.text_projection

        return x
    
    def preprocess_token_embedding(self, text: torch.Tensor) -> torch.Tensor:
        # text.shape: (batch, n_ctx)
        if self.transformer_null_token_emb == "none":
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        else:  # self.transformer_null_token_embed == "sequel"
            incr = torch.where(text==0, 1, 0)
            incr = torch.cumsum(incr, dim=1)
            mod_tokens = torch.where(text!=0, text, incr + (self.vocab_size-1))
            x = self.token_embedding(mod_tokens)
        return x
    
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> dict[str, torch.Tensor]:
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {"image_embed": image_embed,
                "text_embed": text_embed,
                "logit_scale": self.get_logit_scale()}
    
    def forward_all_tokens(self, image: torch.Tensor, text: torch.Tensor) -> dict[str, torch.Tensor]:
        image_embed = self.encode_image_all_patches(image)
        text_embed = self.encode_text_all_tokens(text)

        return {"image_embed": image_embed,
                "text_embed": text_embed,
                "logit_scale": self.get_logit_scale()}

    def raw_logit_scale(self, param: dict[str, Union[str, float]]) -> float:
        """
        param: dict
            keys: "type": "log_scale" or "scale"
                  "value": float
        """
        assert param["type"] in ["raw_scale", "scale"]
        if param["type"] == "raw_scale":
            return param["value"]
        else:
            return self.logit_scale_act.inverse(param["value"])

    def clamp_logit_scale(self):
        self.logit_scale.data.clamp_(self.min_logit_scale, self.max_logit_scale)

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale_act(self.logit_scale)


class ScaledTanh(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.tanh(x / self.alpha)
    
    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"

class ScaledSigmoid(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sigmoid(x / self.alpha)
    
    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"
 

class CLIP_WPSE_two_proj(CLIP):
    def __init__(self, *args, weight_act: Optional[nn.Module]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_weight_proj = nn.Parameter(torch.empty(self.vision_width, 1))
        self.text_weight_proj = nn.Parameter(torch.empty(self.transformer.width, 1))
        # initialization
        nn.init.normal_(self.image_weight_proj, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_weight_proj, std=self.transformer.width ** -0.5)

        if weight_act is not None:
            self.weight_act = weight_act
        else:
            self.weight_act = nn.Identity()


    def forward(self, image: torch.Tensor, text: torch.Tensor) -> dict[str, torch.Tensor]:
        # based on CLIP.forward_all_tokens(image, text)
        img_emb, img_w = self.encode_image(image)
        txt_emb, txt_w = self.encode_text(text)

        return {"image_embed": img_emb,
                "image_weight": img_w,
                "text_embed": txt_emb,
                "text_weight": txt_w,
                "logit_scale": self.get_logit_scale()}


    def encode_text(self, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        # based on CLIP.encode_text_all_tokens()
        x = self.preprocess_token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  ## NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  ## LND -> NLD
        x = self.ln_final(x)  # x.shape = [batch_size, n_ctx, transformer.width]

        # two projections for points and weights
        v = x @ self.text_projection
        w = x @ self.text_weight_proj
        w = self.weight_act(w)

        return v, w
    
    def encode_image(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        # based on CLIP.encode_image_all_patches()
        x = self.visual.forward_features(image)
        ## forward_head() but keep all patches
        x = self.visual.fc_norm(x)
        x = self.visual.head(x)

        ## last two projections for points and weights
        v = x @ self.image_projection
        w = x @ self.image_weight_proj
        w = self.weight_act(w)

        return v, w


def modify_layernorm_eps(mod: nn.Module, eps: float = 1e-6):
    if isinstance(mod, torch.nn.LayerNorm):
        mod.eps = eps


def CLIP_VITB16(ln_eps=None, **kwargs):
    vision_model = timm.create_model("vit_base_patch16_224", num_classes=0)
    model = CLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)
    if ln_eps is not None:
        modify_eps = partial(modify_layernorm_eps, eps=ln_eps)
        model.apply(modify_eps)
    return model


def CLIP_WPSE_VITB16(ln_eps=None, **kwargs):
    vision_model = timm.create_model("vit_base_patch16_224", num_classes=0)
    model = CLIP_WPSE_two_proj(
        embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)
    if ln_eps is not None:
        modify_eps = partial(modify_layernorm_eps, eps=ln_eps)
        model.apply(modify_eps)
    return model
