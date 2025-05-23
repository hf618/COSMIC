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


class ClipFix(nn.Module):
    def __init__(self, args, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, memory_size_clip=10, memory_size_dino=10, text_prompt='tip'):
        super(ClipFix, self).__init__()
        clip, _, transform = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        
        self.use_dinov2 = args.DINOv2
        if args.DINOv2:
            dinov2 = load_dinov2(args.DINO_size)
            self.dinov2 = dinov2


        # print('clip transform', transform)
        self.clip = clip
        self.classnames = [name.replace("_", " ") for name in classnames]

        self.return_local_feat = False
        self.text_prompt_type = text_prompt

        self.logit_scale = clip.logit_scale.data
        self.text_feat = None
        self.few_shot_mem = False
        # self.n_cls = len(classnames)  ## 200
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)

        self.clip_is_DMN = args.clip_is_DMN
        self.dino_is_DMN = args.dino_is_DMN
        self.first_flag = True
        self.first_flag2 = True
        self.memory_size_clip = memory_size_clip
        self.memory_size_dino = memory_size_dino
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype


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

        self.first_flag = True
        self.first_flag2 = True
    def get_text_features(self):
        ## get the text feature only once, multiple class & multiple prompt
        text_feat = []
        text_label = []
        count = 0
        for name in self.classnames:
            text_prompts = [template.format(name) for template in self.text_prompt]  # format with class
            if self.text_prompt_type =='tip_cupl':
                text_prompts += self.cupl_prompts[name]

            
            texts = tokenize(text_prompts).cuda()  # tokenize
            class_embeddings = self.clip.encode_text(texts)  # embed with text encoder

            
            class_embeddings_full = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding_mean = class_embeddings_full.mean(dim=0)
            class_embedding_mean /= class_embedding_mean.norm()
            text_feat.append(class_embedding_mean) ### 1024
            one_hot_target = torch.zeros(self.n_cls).to(class_embedding_mean.device)
            one_hot_target[count] = 1
            text_label.append(one_hot_target)  ## 1 * d, turn it to one hot labels.
            count = count + 1

        self.text_feat = torch.stack(text_feat, dim=0).cuda() ## N*1024
        self.text_label = torch.stack(text_label, dim=0).cuda()  ## N*N

        self.text_feat_full = self.text_feat ## not used.
        ######## 直接从这里找出 important text feat following APE. TO DO
        self.fixed_global_feat = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_local_feat = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_global_feat_vanilla = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_local_feat_vanilla = self.text_feat.clone().unsqueeze(1) ## N*1*C

        self.fixed_global_label = self.text_label.clone().unsqueeze(1)
        self.fixed_local_label = self.text_label.clone().unsqueeze(1)
        self.fixed_global_label_vanilla = self.text_label.clone().unsqueeze(1)
        self.fixed_local_label_vanilla = self.text_label.clone().unsqueeze(1)

        if self.first_flag and self.clip_is_DMN:  ## initlize
            self.image_feature_memory = torch.zeros(self.n_cls, self.memory_size_clip, self.text_feat.shape[1]).to(self.text_feat.device)       ## 如果满了，把entropy 最高的扔出去
            self.image_prediction_mem = torch.zeros(self.n_cls, self.memory_size_clip, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.image_entropy_mem = torch.zeros(self.n_cls, self.memory_size_clip).to(self.text_feat.device)   ## category prediction.
            self.image_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)

            self.local_feature_memory = torch.zeros(self.n_cls, self.memory_size_clip, self.text_feat.shape[1]).to(self.text_feat.device)
            self.local_prediction_mem = torch.zeros(self.n_cls, self.memory_size_clip, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.local_entropy_mem = torch.zeros(self.n_cls, self.memory_size_clip).to(self.text_feat.device)   ## category prediction.
            self.local_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)
            self.first_flag = False

        if self.first_flag2 and self.dino_is_DMN:
            self.image_feature_memory2 = torch.zeros(self.n_cls, self.memory_size_dino, self.dinov2.embed_dim).to(self.text_feat.device)       ## 如果满了，把entropy 最高的扔出去
            self.image_prediction_mem2 = torch.zeros(self.n_cls, self.memory_size_dino, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.image_entropy_mem2 = torch.zeros(self.n_cls, self.memory_size_dino).to(self.text_feat.device)   ## category prediction.
            self.image_feature_count2 = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)

            self.local_feature_memory2 = torch.zeros(self.n_cls, self.memory_size_dino, self.dinov2.embed_dim).to(self.text_feat.device)
            self.local_prediction_mem2 = torch.zeros(self.n_cls, self.memory_size_dino, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.local_entropy_mem2 = torch.zeros(self.n_cls, self.memory_size_dino).to(self.text_feat.device)   ## category prediction.
            self.local_feature_count2 = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)
            self.first_flag2 = False

        return self.text_feat, self.text_feat_full


    def get_image_features(self, image):

        image_features = self.clip.encode_image(image)
        

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # torch.Size([32, 197, 512])
        self.image_features_gllo = image_features
        image_features_local = image_features[:, 1:, :]  ## B*L*C
        image_features_global = image_features[:, 0, :]  ## B*C

        self.image_features_local = image_features_local  # torch.Size([32, 196, 512]) 代表每个patches 的特征
        self.image_features_global = image_features_global  # torch.Size([32, 512])

        return self.image_features_global, self.image_features_local

    def get_image_features_aux(self, image):

        img_feats = self.dinov2(image)  # torch.Size([32, 768])
        x_norm_clstoken = img_feats['x_norm_clstoken']  # 1
        self.image_features_global_aux = x_norm_clstoken / x_norm_clstoken.norm(dim=-1, keepdim=True)   # torch.Size([32, 512])
        x_norm_regtokens = img_feats['x_norm_regtokens']  # 4
        x_norm_patchtokens = img_feats['x_norm_patchtokens']  # 256
        x_norm_clstoken = x_norm_clstoken.unsqueeze(1)  # 形状：[128, 1, 1024]
        image_features = torch.cat([x_norm_clstoken, x_norm_regtokens, x_norm_patchtokens], dim=1)

        self.image_features_aux_gllo = image_features / image_features.norm(dim=-1, keepdim=True)

        self.image_features_local_aux = x_norm_patchtokens / x_norm_patchtokens.norm(dim=-1, keepdim=True)  # torch.Size([32, 196, 512]) 代表每个patches 的特征


        return self.image_features_global_aux, self.image_features_local_aux


    def forward(self, input):
        # pass
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)


def get_fixed_clip(args, clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False, memory_size_clip=10, memory_size_dino=10,
                  text_prompt='tip'):
    model = ClipFix(args, device, classnames, None, arch=clip_arch, n_ctx=n_ctx, ctx_init=ctx_init,
                     learned_cls=learned_cls,
                     memory_size_clip=memory_size_clip, memory_size_dino=memory_size_dino, text_prompt=text_prompt)

    return model

