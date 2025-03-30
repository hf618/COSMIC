# for prompt learning 魔改版
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from clip import load_dinov2
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes, imagenet_templates, tip_imagenet_templates, simple_imagenet_template, \
    ID_to_prompts, ID_to_gptprompts_path
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *
import json
import ipdb

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT = '~/.cache/clip'


class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()

        self.cls_head = nn.Linear(embed_dim, n_class)

    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):  ### add image prompt here.
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end',
                 learned_cls=False, use_ema=False, use_v2=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls  # False
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]  ## 512
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:  ## a photo of a
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            # ipdb.set_trace()
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))  ## 4
            prompt = tokenize(ctx_init).to(self.device)  ## 1*77
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)  ## torch.Size([1, 77, 512])
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]  ## torch.Size([4, 512])
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None:
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  # (N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()  ## torch.Size([4, 512])
        self.use_ema = use_ema
        self.use_ema_v2 = use_v2
        if self.use_ema and self.use_ema_v2:
            # 扩展维度
            expanded_ctx_vectors = ctx_vectors.unsqueeze(0)  # 形状变为 (1, 4, 512)
            # 复制并展开
            final_ctx_vectors = expanded_ctx_vectors.repeat(n_cls, 1, 1)  # 形状变为 (n_cls, 4, 512)
            self.ctx_e = nn.Parameter(final_ctx_vectors)  # to be optimized
        else:
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.n_cls = n_cls  # 1000
        if self.use_ema and not self.use_ema_v2:
            if self.ctx.dim() == 2:
                self.ctx_e = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            elif not self.ctx.size()[0] == self.n_cls:
                self.ctx_e = self.ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        self.ctx_std = nn.Parameter(torch.ones(ctx_vectors.shape) * 1e-4)  # to be optimized
        # ipdb.set_trace()
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]  ## ['a photo of a agaric.', ]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)  # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)  ## torch.Size([1000, 77])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  ## torch.Size([1000, 77, 512])

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Size([1000, 77])
        self.name_lens = name_lens
        self.class_token_position = ctx_position  ## end

        self.n_ctx = n_ctx  ## 4
        self.classnames = classnames
        # ipdb.set_trace()  # 暂停执行

        # 制作 self.ctx_e

        # ipdb.set_trace()  # 暂停执行

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx = self.ctx.copy_(ctx_vectors)  # to be optimized torch.Size([4, 512])
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)  ## 200
        if self.use_ema:
            if self.use_ema_v2:
                expanded_ctx_vectors = self.ctx_init_state.unsqueeze(0)  # 形状变为 (1, 4, 512)
                final_ctx_vectors = expanded_ctx_vectors.repeat(self.n_cls, 1, 1)  # 形状变为 (n_cls, 4, 512)
                self.ctx_e = nn.Parameter(final_ctx_vectors)  # to be optimized
            else:
                self.ctx_e = self.ctx_e[:self.n_cls]
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]  ## 200
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim,
                                      dtype=self.dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)  ## torch.Size([200, 77])

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)  ## torch.Size([200, 77, 512])

        self.token_prefix = embedding[:, :1, :]  ## 200*1*512 前缀
        self.token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS ## torch.Size([200, 72, 512]) 后缀

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts  ## torch.Size([200, 77])
        self.classnames = classnames

    # FOR OURS
    def reset_ctx_e(self):

        if self.use_ema_v2:
            expanded_ctx_vectors = self.ctx_init_state.unsqueeze(0)  # 形状变为 (1, 4, 512)
            final_ctx_vectors = expanded_ctx_vectors.repeat(self.n_cls, 1, 1)  # 形状变为 (n_cls, 4, 512)
            self.ctx_e = nn.Parameter(final_ctx_vectors)
        else:
            if self.ctx.dim() == 2:
                self.ctx_e = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1).clone()
            elif not self.ctx.size()[0] == self.n_cls:
                self.ctx_e = self.ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1).clone()

    def update_prompts_ema2(self, pred, pro_cache, iter, idx=0, alpha=0.99):

        # 遍历 pro_cache 中的键和值
        self.ctx_e = self.ctx_e.clone()
        # for key, value in pro_cache.items():
        key = pred
        value = pro_cache[key]
        prompt = value[idx][0][0]

        # 替换 ctx 中第 key 行的张量
        # self.ctx_e[key] = value[0][0][0]

        with torch.no_grad():
            alpha_teacher = min(1 - 1 / (iter + 1), alpha)
            # alpha_teacher = alpha
            # print("key",key)
            # print('before',self.ctx_e[key])
            for param, new_value in zip(self.ctx_e[key], prompt):
                param.copy_(alpha_teacher * param + (1 - alpha_teacher) * new_value)

    def forward(self, init=None, with_std=True):
        # the init will be used when computing CLIP directional loss
        # ipdb.set_trace()
        if self.use_ema:
            ctx = self.ctx_e
        else:
            if init is not None:
                ctx = init
            else:
                ctx = self.ctx

            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            elif not ctx.size()[0] == self.n_cls:
                ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        # ipdb.set_trace()  # 暂停执行
        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None:
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        cls,  # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx  # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        return prompts


# class CrossAttention(nn.Module):
#     def __init__(self, dim_clip, dim_dino, hidden_dim):
#         super(CrossAttention, self).__init__()
#         self.query_proj = nn.Linear(dim_clip, hidden_dim)
#         self.key_proj = nn.Linear(dim_dino, hidden_dim)
#         self.value_proj = nn.Linear(dim_dino, hidden_dim)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def reset(self):
#         for name, param in self.named_parameters():
#             if 'weight' in name:
#                 nn.init.normal_(param, mean=0.0, std=0.02)  # 使用正态分布初始化
#                 print(f"Custom initialized {name} with mean 0.0 and std 0.02.")
#             elif 'bias' in name:
#                 nn.init.constant_(param, 0.0)  # 偏置初始化为 0
#                 print(f"Custom initialized {name} to zeros.")
#
#     def forward(self, clip_features, dino_features):
#         # 计算 query, key, value
#         query = self.query_proj(clip_features)  # [batch_size, 197, hidden_dim]
#         key = self.key_proj(dino_features)      # [batch_size, 197, hidden_dim]
#         value = self.value_proj(dino_features)  # [batch_size, 197, hidden_dim]
#
#         # 计算注意力权重
#         attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** 0.5)
#         attention_weights = self.softmax(attention_scores)
#
#         # 应用注意力，将 DINO 特征传递给 CLIP 特征
#         fused_features = torch.matmul(attention_weights, value)
#         return fused_features + clip_features  # 返回增强后的特征
class CrossAttention(nn.Module):
    def __init__(self, dim_clip, dim_dino, hidden_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(dim_clip, hidden_dim)
        self.key_proj = nn.Linear(dim_dino, hidden_dim)
        self.value_proj = nn.Linear(dim_dino, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.softmax = nn.Softmax(dim=-1)

    def reset(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)  # 使用正态分布初始化
                # print(f"Custom initialized {name} with mean 0.0 and std 0.02.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # 偏置初始化为 0
                # print(f"Custom initialized {name} to zeros.")

    def forward(self, clip_features, dino_features):
        query = self.query_proj(clip_features)
        key = self.key_proj(dino_features)
        value = self.value_proj(dino_features)
        # 加入relu
        # query = torch.relu(self.query_proj(clip_features))
        # key = torch.relu(self.key_proj(dino_features))
        # value = torch.relu(self.value_proj(dino_features))
        # 使用多头注意力
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output + clip_features


class CrossAttentionV2(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=1024, num_heads=8, dropout=0.1, resnet=False):
        super(CrossAttentionV2, self).__init__()
        self.resnet = resnet
        if self.resnet:
            self.name = "CrossAttentionV2_resnet"
        else:
            self.name = "CrossAttentionV2_no_resnet"
        # Multi-head Attention for Cross-Attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=dino_dim, num_heads=num_heads, dropout=dropout)
        # Linear layers to map the dimensions
        self.clip_to_dino_proj = nn.Linear(clip_dim, dino_dim)  # Project CLIP feature to DINOv2 space
        self.dino_to_clip_proj = nn.Linear(dino_dim, clip_dim)  # Project back to CLIP space
        # Layer norm to stabilize training
        self.layer_norm = nn.LayerNorm(clip_dim)

    def reset(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # nn.init.xavier_uniform_(param)
                nn.init.normal_(param, mean=0.0, std=0.02)  # 使用正态分布初始化
                # print(f"Custom initialized {name} with mean 0.0 and std 0.02.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # 偏置初始化为 0
                # print(f"Custom initialized {name} to zeros.")

    def forward(self, img_feats, dino_feats):
        """
        img_feats: CLIP features, shape: [bs, 197, 512] (class token + patches tokens)
        dino_feats: DINOv2 features, shape: [bs, 261, 1024] (combined DINOv2 tokens)
        """
        bs, num_clip_tokens, clip_dim = img_feats.shape
        _, num_dino_tokens, dino_dim = dino_feats.shape

        # Step 1: Project CLIP features to DINOv2 feature space for cross-attention
        img_feats_proj = self.clip_to_dino_proj(img_feats)  # [bs, 197, 1024]

        # Step 2: Reshape for multi-head attention (transpose for attention compatibility)
        # Cross-attention: Query (CLIP), Key and Value (DINOv2)
        img_feats_proj = img_feats_proj.transpose(0, 1)  # [197, bs, 1024]
        dino_feats = dino_feats.transpose(0, 1)  # [261, bs, 1024]

        # Step 3: Cross-Attention (CLIP's query attends to DINOv2's key/value)
        attn_output, attn_weights = self.cross_attention(query=img_feats_proj, key=dino_feats, value=dino_feats)

        # attn_output: [197, bs, 1024]
        # print("attn_weights:0",attn_weights[0])
        # Step 4: Project the result back to CLIP's original feature space
        attn_output = attn_output.transpose(0, 1)  # [bs, 197, 1024]
        attn_output_proj = self.dino_to_clip_proj(attn_output)  # [bs, 197, 512]
        # print("attn_output_proj:", attn_output_proj[0][0])
        # Step 5: Enhance CLIP features by adding the attention output (residual connection)
        if self.resnet:
            enhanced_feats = self.layer_norm(img_feats + attn_output_proj)  # [bs, 197, 512]
        else:
            enhanced_feats = self.layer_norm(attn_output_proj)

        # ipdb.set_trace()
        # Now enhanced_feats contains the enriched class token in [bs, 197, 512]
        # ipdb.set_trace()
        return enhanced_feats  # Also return attention weights for analysis


class LightweightCrossAttention(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=1024, intermediate_dim=256, dropout=0.1):
        super(LightweightCrossAttention, self).__init__()
        self.name = "LightweightCrossAttention"

        # 使用较小的中间维度投影空间来减少参数量
        self.clip_to_intermediate = nn.Linear(clip_dim, intermediate_dim, bias=False)
        self.dino_to_intermediate = nn.Linear(dino_dim, intermediate_dim, bias=False)

        # 单头注意力，用于增量特征学习
        self.attention = nn.MultiheadAttention(embed_dim=intermediate_dim, num_heads=1, dropout=dropout)

        # 投影回 CLIP 特征空间并生成增量
        self.intermediate_to_clip = nn.Linear(intermediate_dim, clip_dim, bias=False)

        # 层归一化
        self.layer_norm = nn.LayerNorm(clip_dim)

    def reset(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)  # 使用正态分布初始化
                # print(f"Custom initialized {name} with mean 0.0 and std 0.02.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # 偏置初始化为 0
                # print(f"Custom initialized {name} to zeros.")

    def forward(self, clip_feats, dino_feats):
        """
        clip_feats: CLIP特征, 形状: [bs, 197, 512]
        dino_feats: DINO特征, 形状: [bs, 261, 1024]
        """
        # 1. 将 CLIP 和 DINO 特征分别投影到较小的中间维度
        clip_proj = self.clip_to_intermediate(clip_feats)  # [bs, 197, 256]
        dino_proj = self.dino_to_intermediate(dino_feats)  # [bs, 261, 256]

        # 2. 转置维度以适应 attention 模块的输入要求
        clip_proj = clip_proj.transpose(0, 1)  # [197, bs, 256]
        dino_proj = dino_proj.transpose(0, 1)  # [261, bs, 256]

        # 3. 执行单头跨模态注意力，使得 CLIP 特征从 DINO 特征中学习增量
        attn_output, _ = self.attention(query=clip_proj, key=dino_proj, value=dino_proj)

        # 4. 转置回原始形状，并将增量投影回 CLIP 的特征空间
        attn_output = attn_output.transpose(0, 1)  # [bs, 197, 256]
        increment = self.intermediate_to_clip(attn_output)  # [bs, 197, 512]

        # 5. 通过残差连接将增量添加到原始 CLIP 特征中，并进行层归一化
        enhanced_feats = self.layer_norm(clip_feats + increment)  # [bs, 197, 512]

        return enhanced_feats  # 返回增强的 CLIP 特征


class CrossAttentionV2_sim(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=1024, hidden_dim=512):
        super(CrossAttentionV2_sim, self).__init__()

        # 将 DINOv2 的特征映射到 CLIP 的维度
        self.dino_to_clip = nn.Linear(dino_dim, clip_dim)

        # 用于融合后的隐层
        self.fc_hidden = nn.Linear(clip_dim, hidden_dim)

        # 注意力机制的 Query, Key, Value 线性变换
        self.query_proj = nn.Linear(clip_dim, hidden_dim)
        self.key_proj = nn.Linear(clip_dim, hidden_dim)
        self.value_proj = nn.Linear(clip_dim, hidden_dim)

        # 最终映射回 CLIP 的特征维度
        self.fc_out = nn.Linear(hidden_dim, clip_dim)

    def reset(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)  # 使用正态分布初始化
                # print(f"Custom initialized {name} with mean 0.0 and std 0.02.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # 偏置初始化为 0
                # print(f"Custom initialized {name} to zeros.")

    def forward(self, clip_feats, dino_feats):
        """
        clip_feats: [1, 512] CLIP 特征
        dino_feats: [1, 1024] DINOv2 特征
        """
        # Step 1: 将 DINO 特征映射到 CLIP 的维度 [1, 1024] -> [1, 512]
        dino_feats_mapped = self.dino_to_clip(dino_feats)  # [1, 512]

        # Step 2: 合并 CLIP 和 DINO 的特征
        combined_feats = clip_feats + dino_feats_mapped  # [1, 512]

        # Step 3: 使用交叉注意力机制
        query = self.query_proj(clip_feats)  # [1, hidden_dim]
        key = self.key_proj(dino_feats_mapped)  # [1, hidden_dim]
        value = self.value_proj(dino_feats_mapped)  # [1, hidden_dim]

        # 计算注意力权重并应用
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)), dim=-1)  # [1, 1]
        attention_output = torch.matmul(attention_weights, value)  # [1, hidden_dim]

        # Step 4: 通过线性层映射并加入残差连接
        attention_output = self.fc_out(attention_output)  # [1, 512]
        enhanced_feats = clip_feats + attention_output  # 残差连接

        return enhanced_feats  # 输出增强后的特征


class ConcatenationFusion(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=1024, output_dim=512):
        super(ConcatenationFusion, self).__init__()
        self.name = "ConcatenationFusion"
        # 线性层将拼接后的特征进行降维，生成增强特征
        self.fusion_layer = nn.Linear(clip_dim + dino_dim, output_dim)

    def reset(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)  # 使用正态分布初始化
                # print(f"Custom initialized {name} with mean 0.0 and std 0.02.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # 偏置初始化为 0
                # print(f"Custom initialized {name} to zeros.")

    def forward(self, clip_feats, dino_feats):
        """
        clip_feats: CLIP特征, 形状: [bs, 197, 512]
        dino_feats: DINO特征, 形状: [bs, 261, 1024]
        """
        # 将特征拼接，注意需要调整输入形状以适配特征数量
        dino_feats_reduced = dino_feats[:, :197, :]  # 保证形状一致 [bs, 197, 1024]
        fused_feats = torch.cat((clip_feats, dino_feats_reduced), dim=-1)  # [bs, 197, 1536]

        # 线性融合并生成增强特征
        enhanced_feats = self.fusion_layer(fused_feats)  # [bs, 197, 512]
        return enhanced_feats

class WeightedSumFusion(nn.Module):
    def __init__(self):
        super(WeightedSumFusion, self).__init__()
        self.name = 'WeightedSumFusion'
        # 两个可训练的权重参数
        self.clip_weight = nn.Parameter(torch.tensor(0.5))
        self.dino_weight = nn.Parameter(torch.tensor(0.5))

    def reset(self):
        self.clip_weight = nn.Parameter(torch.tensor(0.5))
        self.dino_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, clip_feats, dino_feats):
        """
        clip_feats: CLIP特征, 形状: [bs, 197, 512]
        dino_feats: DINO特征, 形状: [bs, 197, 512] （需要事先调整维度）
        """
        # 直接将 DINO 特征调整到与 CLIP 相同维度
        dino_feats_reduced = dino_feats[:, :197, :512]  # [bs, 197, 512]

        # 计算加权平均
        fused_feats = self.clip_weight * clip_feats + self.dino_weight * dino_feats_reduced  # [bs, 197, 512]
        return fused_feats
# 这个在DTD上还可以 会显著增加 mem模型的预测，但是原模型有所下降
# class LinearProjectionFusion(nn.Module):
#     def __init__(self, clip_dim=512, dino_dim=1024, output_dim=512):
#         super(LinearProjectionFusion, self).__init__()
#         self.name = "LinearProjectionFusion"
#         # 将 DINO 特征降维到 CLIP 特征的空间
#         self.dino_proj = nn.Linear(dino_dim, clip_dim)
#         # 融合后的线性层
#         self.fusion_layer = nn.Linear(clip_dim, output_dim)
#
#     def reset(self):
#         for name, param in self.named_parameters():
#             if 'weight' in name:
#                 nn.init.normal_(param, mean=0.0, std=0.02)  # 使用正态分布初始化
#                 # print(f"Custom initialized {name} with mean 0.0 and std 0.02.")
#             elif 'bias' in name:
#                 nn.init.constant_(param, 0.0)  # 偏置初始化为 0
#                 # print(f"Custom initialized {name} to zeros.")
#
#     def forward(self, clip_feats, dino_feats):
#         """
#         clip_feats: CLIP特征, 形状: [bs, 197, 512]
#         dino_feats: DINO特征, 形状: [bs, 261, 1024]
#         """
#         # 对 DINO 特征进行线性降维
#         dino_feats_reduced = self.dino_proj(dino_feats[:, :197, :])  # [bs, 197, 512]
#
#         # 直接相加融合两个特征
#         fused_feats = clip_feats + dino_feats_reduced  # [bs, 197, 512]
#
#         # 可选：再通过一个线性层进一步处理融合后的特征
#         enhanced_feats = self.fusion_layer(fused_feats)  # [bs, 197, 512]
#         return enhanced_feats


class LinearProjectionFusion(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=1024, output_dim=512):
        super(LinearProjectionFusion, self).__init__()
        self.name = "LinearProjectionFusion"
        # 将 DINO 特征降维到 CLIP 特征的空间
        self.dino_proj = nn.Linear(dino_dim, clip_dim)
        # 可学习的 alpha 参数，初始值为 1，表示开始时主要依赖 CLIP 特征
        self.alpha = nn.Parameter(torch.tensor(1.0))
        # 归一化层，确保 CLIP 和 DINO 特征处于相同的分布
        self.norm = nn.LayerNorm(clip_dim)
        # 融合后的线性层
        # self.fusion_layer = nn.Linear(clip_dim, output_dim)
        # # 最后的归一化层，作用于 enhanced_feats
        # self.final_norm = nn.LayerNorm(output_dim)
    def reset(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)  # 使用正态分布初始化
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # 偏置初始化为 0
    def forward(self, clip_feats, dino_feats):
        """
        clip_feats: CLIP特征, 形状: [bs, 197, 512]
        dino_feats: DINO特征, 形状: [bs, 261, 1024]
        """
        # 对 DINO 特征进行线性降维
        dino_feats_reduced = self.dino_proj(dino_feats[:, :197, :])  # [bs, 197, 512]
        # 对 CLIP 和 DINO 特征分别进行归一化
        clip_feats = self.norm(clip_feats)  # 对 CLIP 特征进行归一化
        dino_feats_reduced = self.norm(dino_feats_reduced)  # 对 DINO 降维后的特征进行归一化
        # 使用 alpha 进行加权融合，alpha 初始为 1，意味着初始主要依赖 CLIP 特征
        fused_feats = self.alpha * clip_feats + (1 - self.alpha) * dino_feats_reduced  # [bs, 197, 512]
        return fused_feats
        # # 线性层处理融合后的特征
        # enhanced_feats = self.fusion_layer(fused_feats)  # [bs, 197, 512]
        # # 对 enhanced_feats 进行归一化
        # enhanced_feats = self.final_norm(enhanced_feats)  # [bs, 197, 512]
        # return enhanced_feats
class SelfSupervisedFusion(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=1024, hidden_dim=256, output_dim=512):
        super(SelfSupervisedFusion, self).__init__()
        self.name = "SelfSupervisedFusion"
        # 自监督融合模块，带有一个轻量的网络
        self.fusion_network = nn.Sequential(
            nn.Linear(clip_dim + dino_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def reset(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)  # 使用正态分布初始化
                # print(f"Custom initialized {name} with mean 0.0 and std 0.02.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # 偏置初始化为 0
                # print(f"Custom initialized {name} to zeros.")

    def forward(self, clip_feats, dino_feats):
        # 将 DINO 特征降维并拼接
        dino_feats_reduced = dino_feats[:, :197, :]
        fused_feats = torch.cat((clip_feats, dino_feats_reduced), dim=-1)  # [bs, 197, 1536]

        # 通过轻量网络自动融合
        enhanced_feats = self.fusion_network(fused_feats)  # [bs, 197, 512]
        return enhanced_feats


class ClipFlex(nn.Module):
    def __init__(self, args, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, memory_size=10, text_prompt='tip'):
        super(ClipFlex, self).__init__()
        clip, _, transform = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.use_dino4mem = args.DINO and args.DINO4mem
        self.use_dino4cross = args.DINO and args.DINO4cross
        self.use_dino = args.DINO
        if args.DINO:
            dino, _ = load_dino()
            self.dino = dino
        self.use_dinov2 = args.DINOv2
        if args.DINOv2:
            dinov2 = load_dinov2(args.DINO_size)
            self.dinov2 = dinov2
            # ipdb.set_trace()
        print('clip transform', transform)
        self.clip = clip
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.first_flag = True
        if args.EMA4mem:
            self.memory_size = 1
        else:
            self.memory_size = memory_size
        self.return_local_feat = False
        self.text_prompt_type = text_prompt

        self.logit_scale = clip.logit_scale.data
        self.text_feat = None
        self.few_shot_mem = False
        # self.n_cls = len(classnames)  ## 200
        self.image_encoder = clip.visual

        # # ipdb.set_trace()
        self.text_encoder = TextEncoder(clip)
        # prompt tuning
        use_ema = args.EMA1
        v2 = args.v2

        self.simple_CA = args.simple_CA

        if self.use_dino4cross:
            self.cross_attention = CrossAttention(dim_clip=512, dim_dino=768, hidden_dim=512)
        elif self.use_dinov2:
            if args.simple_CA:
                self.cross_attention = CrossAttentionV2_sim()
            else:
                # self.cross_attention = CrossAttentionV2()
                # self.cross_attention = LinearProjectionFusion()
                # self.cross_attention = CrossAttentionV2()
                # self.cross_attention = LinearProjectionFusion()
                # self.cross_attention = CrossAttentionV2()
                self.cross_attention = WeightedSumFusion()

        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls,
                                            use_ema, v2)
        # self.criterion = criterion
        self.alpha3 = args.alpha3
        self.wei = args.wei
        self.new_up = args.new_up
        self.new_up2 = args.new_up2
        self.new_up_mix = args.new_up_mix

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, test_sets):
        self.n_cls = len(classnames)  ## 200
        self.classnames = [name.replace("_", " ") for name in classnames]
        print('class number:', self.n_cls)
        if self.text_prompt_type == 'simple':
            self.text_prompt = simple_imagenet_template  ## ['a photo of a {}.']
        elif self.text_prompt_type == 'tip':
            if len(test_sets) > 1:
                self.text_prompt = ID_to_prompts[test_sets.lower()]
            else:
                self.text_prompt = tip_imagenet_templates  ## seven text prompts
        elif self.text_prompt_type == 'tip_cupl':
            if len(test_sets) > 1:
                self.text_prompt = ID_to_prompts[test_sets.lower()]
                self.cupl_file = ID_to_gptprompts_path[test_sets.lower()]
            else:
                self.text_prompt = tip_imagenet_templates  ## seven text prompts
                self.cupl_file = "CuPL_prompts_imagenet.json"
            f = open('./data/gpt3_prompts/' + self.cupl_file)
            self.cupl_prompts = json.load(f)
        elif self.text_prompt_type == 'full':
            self.text_prompt = imagenet_templates
        else:
            raise NotImplementedError
        print('test sets, prompt', test_sets, self.text_prompt)
        # ipdb.set_trace()
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = [self.prompt_prefix + " " + name + "." for name in classnames] ## 200
        # tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)  ## torch.Size([200, 77])
        #
        # clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)
        #
        # with torch.no_grad():
        #     embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)  ## torch.Size([200, 77, 512])
        #
        # self.token_prefix = embedding[:, :1, :] ## 200*1*512 前缀
        # self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS ## torch.Size([200, 72, 512]) 后缀
        #
        # self.name_lens = name_lens
        # self.tokenized_prompts = tokenized_prompts  ## torch.Size([200, 77])
        # self.classnames = classnames
        self.first_flag = True

    def get_text_features(self):
        ## get the text feature only once, multiple class & multiple prompt
        # 魔改一手这个prompt learning
        text_feat = []
        text_label = []
        count = 0
        # 开一个Learnable prompt
        learnable_prompts = self.prompt_learner()  # torch.Size([n_cls, 77, 512])
        learnable_tokenized_prompts = self.prompt_learner.tokenized_prompts  # torch.Size([cls_n, 77])
        for name in self.classnames:
            # learnable_prompts_1 = learnable_prompts[count] # torch.Size([77, 512])
            learnable_tokenized_prompts_1 = learnable_tokenized_prompts[count].unsqueeze(0)  # torch.Size([1, 77])
            text_prompts = [template.format(name) for template in self.text_prompt]  # format with class
            if self.text_prompt_type == 'tip_cupl':
                text_prompts += self.cupl_prompts[name]
            # len(text_prompts) = 61
            texts = tokenize(
                text_prompts).cuda()  # tokenize torch.Size([61, 77]) [number of input strings, context_length]
            # 进行拼接
            texts = torch.cat((learnable_tokenized_prompts_1, texts), dim=0)  # torch.Size([62, 77])
            class_embeddings = self.clip.encode_text(texts)  # embed with text encoder torch.Size([61 or 62, 512])
            class_embeddings_full = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding_mean = class_embeddings_full.mean(dim=0)
            class_embedding_mean /= class_embedding_mean.norm()  # torch.Size([512])
            text_feat.append(class_embedding_mean)  ### 1024
            one_hot_target = torch.zeros(self.n_cls).to(class_embedding_mean.device)  # torch.Size([n_cls])
            one_hot_target[count] = 1
            text_label.append(one_hot_target)  ## 1 * d, turn it to one hot labels.
            count = count + 1
            # ipdb.set_trace()  # 暂停执行
        self.text_feat = torch.stack(text_feat, dim=0).cuda()  ## N*1024
        self.text_label = torch.stack(text_label, dim=0).cuda()  ## N*N

        self.text_feat_full = self.text_feat  ## not used.
        ######## 直接从这里找出 important text feat following APE. TO DO
        self.fixed_global_feat = self.text_feat.clone().unsqueeze(1)  ## N*1*C
        self.fixed_local_feat = self.text_feat.clone().unsqueeze(1)  ## N*1*C
        self.fixed_global_feat_vanilla = self.text_feat.clone().unsqueeze(1)  ## N*1*C
        self.fixed_local_feat_vanilla = self.text_feat.clone().unsqueeze(1)  ## N*1*C

        self.fixed_global_label = self.text_label.clone().unsqueeze(1)
        self.fixed_local_label = self.text_label.clone().unsqueeze(1)
        self.fixed_global_label_vanilla = self.text_label.clone().unsqueeze(1)
        self.fixed_local_label_vanilla = self.text_label.clone().unsqueeze(1)

        if self.first_flag:  ## initlize 初始化 mem
            if self.use_dino4mem:
                e_dim = self.dino.embed_dim
            else:
                e_dim = self.text_feat.shape[1]
            self.image_feature_memory = torch.zeros(self.n_cls, self.memory_size, e_dim).to(
                self.text_feat.device)  ## 如果满了，把entropy 最高的扔出去
            self.image_feature_memory_avg = torch.zeros(self.n_cls, e_dim).to(
                self.text_feat.device)
            self.image_feature_count_avg = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)
            self.image_prediction_mem = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(
                self.text_feat.device)  ## category prediction.
            self.image_entropy_mem = torch.zeros(self.n_cls, self.memory_size).to(
                self.text_feat.device)  ## category prediction.
            self.image_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)

            self.local_feature_memory = torch.zeros(self.n_cls, self.memory_size, e_dim).to(
                self.text_feat.device)
            self.local_prediction_mem = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(
                self.text_feat.device)  ## category prediction.
            self.local_entropy_mem = torch.zeros(self.n_cls, self.memory_size).to(
                self.text_feat.device)  ## category prediction.
            self.local_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)

            if self.new_up:
                self.image_feature_memory2 = torch.zeros(self.n_cls, self.memory_size, self.text_feat.shape[1]).to(
                    self.text_feat.device)  ## 如果满了，把entropy 最高的扔出去
                self.image_prediction_mem2 = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(
                    self.text_feat.device)  ## category prediction.
                self.image_dis_mem2 = torch.zeros(self.n_cls, self.memory_size).to(
                    self.text_feat.device)  ## category prediction.
                self.image_feature_count2 = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)
            if self.new_up2:
                self.image_feature_memory3 = torch.zeros(self.n_cls, self.memory_size, self.text_feat.shape[1]).to(
                    self.text_feat.device)  ## 如果满了，把entropy 最高的扔出去
                self.image_prediction_mem3 = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(
                    self.text_feat.device)  ## category prediction.
                self.image_dis_mem3 = torch.zeros(self.n_cls, self.memory_size).to(
                    self.text_feat.device)  ## category prediction.
                self.image_feature_count3 = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)
            if self.new_up_mix:
                self.image_feature_memory_mix = torch.zeros(self.n_cls, self.memory_size * 3,
                                                            self.text_feat.shape[1]).to(
                    self.text_feat.device)  ## 如果满了，把entropy 最高的扔出去
                self.image_prediction_mem_mix = torch.zeros(self.n_cls, self.memory_size * 3, self.n_cls).to(
                    self.text_feat.device)  ## category prediction.
            self.first_flag = False

        # torch.Size([n_cls, 512]) and torch.Size([n_cls, 512])
        # ipdb.set_trace()  # 暂停执行
        return self.text_feat, self.text_feat_full

        # text_features = []
        # prompts = self.prompt_learner(with_std=True)  ## torch.Size([1000, 77, 512])
        # tokenized_prompts = self.prompt_learner.tokenized_prompts
        # t_features = self.text_encoder(prompts, tokenized_prompts)  ## torch.Size([1000, 1024])
        # text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        # self.num_class = t_features.size(0)
        # text_features = torch.stack(text_features, dim=0)
        # # return text_features
        #
        # return torch.mean(text_features, dim=0)

    def update_text_feat(self, idx):
        ## get the text feature only once, multiple class & multiple prompt
        # 魔改一手这个prompt learning
        text_feat = []
        # 开一个Learnable prompt
        learnable_prompts = self.prompt_learner()  # torch.Size([n_cls, 77, 512])
        learnable_tokenized_prompts = self.prompt_learner.tokenized_prompts  # torch.Size([cls_n, 77])
        name = self.classnames[idx]
        # learnable_prompts_1 = learnable_prompts[count] # torch.Size([77, 512])
        learnable_tokenized_prompts_1 = learnable_tokenized_prompts[idx].unsqueeze(0)  # torch.Size([1, 77])
        text_prompts = [template.format(name) for template in self.text_prompt]  # format with class
        if self.text_prompt_type == 'tip_cupl':
            text_prompts += self.cupl_prompts[name]
        # len(text_prompts) = 61
        texts = tokenize(text_prompts).cuda()  # tokenize torch.Size([61, 77]) [number of input strings, context_length]
        # 进行拼接
        texts = torch.cat((learnable_tokenized_prompts_1, texts), dim=0)  # torch.Size([62, 77])
        class_embeddings = self.clip.encode_text(texts)  # embed with text encoder torch.Size([61 or 62, 512])
        class_embeddings_full = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        if self.wei > 1.0:
            # 定义权重
            weights = torch.ones(class_embeddings_full.shape[0]).cuda()  # 初始化权重为 1
            weights[0] = self.wei  # 给0行更大的权重
            # 计算加权平均
            class_embedding_mean = (class_embeddings_full.T @ weights) / weights.sum()
            # 归一化加权平均结果
            class_embedding_mean /= class_embedding_mean.norm()  # torch.Size([512])
        else:
            class_embedding_mean = class_embeddings_full.mean(dim=0)
            class_embedding_mean /= class_embedding_mean.norm()  # torch.Size([512])
        text_feat.append(class_embedding_mean)  ### 1024

        # print("self.text_feat 1",self.text_feat.shape)
        self.text_feat[idx] = torch.stack(text_feat, dim=0).cuda()  ## N*1024
        # print("self.text_feat 2", self.text_feat.shape)
        # ipdb.set_trace()  # 暂停执行

    # for TPT
    def get_text_features_0(self):
        text_features = []
        prompts = self.prompt_learner()  # grad在这
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)  # torch.Size([nls, 512])
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)
        # ipdb.set_trace()  # 暂停执行
        return torch.mean(text_features, dim=0)  # torch.Size([nls, 512])

    # 得到单个类别的复合型的text—_feats
    def get_text_features_one(self, idx):
        text_features_learn = self.get_text_features_0()[idx].unsqueeze(0)
        # text_features = []
        # prompts = self.prompt_learner()[idx, :, :].unsqueeze(0)
        # tokenized_prompts = self.prompt_learner.tokenized_prompts[idx, :].unsqueeze(0)
        # t_features = self.text_encoder(prompts, tokenized_prompts)
        # text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        # text_features = torch.stack(text_features, dim=0)
        # text_features_learn = torch.mean(text_features, dim=0)

        text_features_hand = self.text_feat[idx].unsqueeze(0)

        # 进行加权平均
        text_features = self.alpha3 * text_features_learn + (1 - self.alpha3) * text_features_hand

        return text_features  # 返回 torch.Size([1，512])

    def get_image_features(self, image):
        # image_features_vanilla = self.image_encoder(image.type(self.dtype))
        ## for Res50 128*1024 or 128*50*1024 [global feat; 7*7 local feature]
        ## for VIT,  128*512 or 128*197*512 [global feat; 14*14 local features]


        # # 检查模型是否有相关参数支持
        # # self.clip.visual.need_attention_weights = True  # 如果支持类似配置
        # # 用于存储每个 block 的注意力权重
        # attn_scores = [] # len = 12
        # # 定义钩子函数来捕获每个 block 的注意力权重
        # def get_attention_scores(module, input, output):
        #     attn_weights = output[1]  # output[1] 通常是注意力权重
        #     attn_scores.append(attn_weights.detach())  # 保存到 attention_scores 中
        #
        # # ipdb> attention_scores[-1].shape
        # # torch.Size([16, 197, 197])
        # # 注册钩子到每个 Transformer block 的 MultiheadAttention 层
        # for block in self.clip.visual.transformer.resblocks:
        #     block.attn.register_forward_hook(get_attention_scores)


        image_features = self.clip.encode_image(image)
        # 要在推理后面用钩子
        # self.attn_score_clip = attn_scores[0][:, 0, 1:].unsqueeze(1) # ([16, 1, 196])

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # torch.Size([32, 197, 512])
        self.image_features_gllo = image_features
        image_features_local = image_features[:, 1:, :]  ## B*L*C
        image_features_global = image_features[:, 0, :]  ## B*C

        self.image_features_local = image_features_local  # torch.Size([32, 196, 512]) 代表每个patches 的特征
        self.image_features_global = image_features_global  # torch.Size([32, 512])
        # ipdb.set_trace()  # 暂停执行
        # self.image_features_global, sim_matrix = compute_weighted_global_feature(image_features_local, image_features_global)

        return self.image_features_global, self.image_features_local

    def get_image_features_aux(self, image):
        # image torch.Size([bs, 3, 224, 224])
        # image_features_vanilla = self.image_encoder(image.type(self.dtype))
        ## for Res50 128*1024 or 128*50*1024 [global feat; 7*7 local feature]
        ## for VIT,  128*512 or 128*197*512 [global feat; 14*14 local features]
        if self.use_dino:
            image_features = self.dino.forward_features(image)  # torch.Size([32, 768])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # torch.Size([32, 197, 512])
            self.image_features_aux_gllo = image_features
            image_features_global = image_features[:, 0, :]  ## B*C
            image_features_local = image_features[:, 1:, :]  ## B*L*C
            self.image_features_global_aux = image_features_global  # torch.Size([32, 512])
            self.image_features_local_aux = image_features_local  # torch.Size([32, 196, 512]) 代表每个patches 的特征

            # ipdb.set_trace()  # 暂停执行
            # self.image_features_global, sim_matrix = compute_weighted_global_feature(image_features_local, image_features_global)

            return self.image_features_global_aux, self.image_features_local_aux
        elif self.use_dinov2:
            # self.attn_score_dinov2 = self.dinov2.get_last_self_attention(image) # torch.Size([bs, num_head, 261, 261])



            img_feats = self.dinov2(image)  # torch.Size([32, 768])
            x_norm_clstoken = img_feats['x_norm_clstoken']  # 1
            self.image_features_global_aux = x_norm_clstoken / x_norm_clstoken.norm(dim=-1, keepdim=True)   # torch.Size([32, 512])
            x_norm_regtokens = img_feats['x_norm_regtokens']  # 4
            x_norm_patchtokens = img_feats['x_norm_patchtokens']  # 256
            x_norm_clstoken = x_norm_clstoken.unsqueeze(1)  # 形状：[128, 1, 1024]
            image_features = torch.cat([x_norm_clstoken, x_norm_regtokens, x_norm_patchtokens], dim=1)

            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # torch.Size([32, 197, 512])
            # ipdb.set_trace()
            self.image_features_aux_gllo = image_features / image_features.norm(dim=-1, keepdim=True)

            self.image_features_local_aux = x_norm_patchtokens / x_norm_patchtokens.norm(dim=-1, keepdim=True)  # torch.Size([32, 196, 512]) 代表每个patches 的特征


            # self.image_features_global, sim_matrix = compute_weighted_global_feature(image_features_local, image_features_global)

            # ipdb.set_trace()
            return self.image_features_global_aux, self.image_features_local_aux
        return None, None

    def get_cross_image_features(self):
        # torch.Size([32, 197, 512]) torch.Size([32, 261, 1024])
        enhanced_clip_features = self.cross_attention(self.image_features_gllo, self.image_features_aux_gllo)
        enhanced_clip_features = enhanced_clip_features / enhanced_clip_features.norm(dim=-1, keepdim=True)
        # ipdb.set_trace()
        return enhanced_clip_features
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        # return logits
    def get_cross_image_features_sim(self):
        enhanced_clip_features = self.cross_attention(self.image_features_global, self.image_features_global_aux)
        enhanced_clip_features = enhanced_clip_features / enhanced_clip_features.norm(dim=-1, keepdim=True)
        return enhanced_clip_features

    # 进行改革开放
    def inference(self, image):
        if self.use_dino4cross:
            image_features_global, image_features_local = self.get_image_features(image.type(self.dtype))
            image_features_global_aux, image_features_local_aux = self.get_image_features_aux(image.type(self.dtype))
            # image_features_global = image_features_global.unsqueeze(1)
            # image_features_global = torch.cat((image_features_global, image_features_local), dim=1)
            #
            # image_features_global_aux = image_features_global_aux.unsqueeze(1)
            # image_features_global_aux = torch.cat((image_features_global_aux, image_features_local_aux), dim=1)
            # print("image_features_global 1", image_features_global[:, 0, :])
            enhanced_clip_features = self.get_cross_image_features()  # torch.Size([32, 197, 512])
            self.image_features_global = enhanced_clip_features[:, 0, :]
            image_features_global = enhanced_clip_features[:, 0, :]
            # print("image_features_global 2", image_features_global)
        elif self.use_dinov2:
            if self.simple_CA:
                image_features_global, image_features_local = self.get_image_features(image.type(self.dtype))
                image_features_global_aux, image_features_local_aux = self.get_image_features_aux(image.type(self.dtype))
                #ipdb.set_trace()
                enhanced_clip_features = self.get_cross_image_features_sim() # torch.Size([1, 512])
                self.image_features_global = enhanced_clip_features
                image_features_global = enhanced_clip_features
                #ipdb.set_trace()

            else:
                image_features_global, image_features_local = self.get_image_features(image.type(self.dtype))
                image_features_global_aux, image_features_local_aux = self.get_image_features_aux(image.type(self.dtype))
                enhanced_clip_features = self.get_cross_image_features()
                self.image_features_global = enhanced_clip_features[:, 0, :]
                image_features_global = enhanced_clip_features[:, 0, :]
        else:
            with torch.no_grad():
                image_features_global, image_features_local = self.get_image_features(image.type(self.dtype))
        # image_features = image_features_global[:1]  # torch.Size([1, 512])
        image_features = image_features_global  # torch.Size([32, 512])
        text_features_learn = self.get_text_features_0()  # torch.Size([n_cls, 512])
        # text_features_hand, _ = self.get_text_features()  # torch.Size([n_cls, 512]) # 超级慢
        text_features_hand = self.text_feat
        # 进行加权平均
        # ipdb.set_trace()  # 暂停执行
        text_features = self.alpha3 * text_features_learn + (1 - self.alpha3) * text_features_hand
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # torch.Size(bs, n_cls])
        # print("logits",logits.shape)
        # ipdb.set_trace()  # 暂停执行
        return logits, image_features_global

    def forward(self, input):
        # pass
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)


def get_flex_clip(args, clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False, memory_size=10, text_prompt='tip'):
    model = ClipFlex(args, device, classnames, None, arch=clip_arch, n_ctx=n_ctx, ctx_init=ctx_init,
                     learned_cls=learned_cls,
                     memory_size=memory_size, text_prompt=text_prompt)

    return model

