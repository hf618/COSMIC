import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
import numpy as np
import ipdb
from clip import load_dinov2
from data.imagnet_prompts import (
    imagenet_classes, imagenet_templates, 
    tip_imagenet_templates, simple_imagenet_template,
    ID_to_prompts, ID_to_gptprompts_path
)
import json
from transformers import AlignProcessor, AlignTextModel, AlignVisionModel, AlignModel


class AlignModel_custom(nn.Module):
    """ALIGN 模型的基础实现"""
    def __init__(self, device, model_path="kakaobrain/align-base"):
        super().__init__()
        # 加载 ALIGN 模型
        self.processor = AlignProcessor.from_pretrained(model_path)
        self.text_encoder = AlignTextModel.from_pretrained(model_path).to(device)
        self.vision_encoder = AlignVisionModel.from_pretrained(model_path).to(device)
        self.device = device
        self.model = AlignModel.from_pretrained(model_path).to(device)
    
    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors='pt', do_rescale=False).to(self.device)
        image_embeds = self.model.get_image_features(
            pixel_values=inputs['pixel_values'],
        )
        return image_embeds
    
    def encode_text(self, text):
        inputs = self.processor(
            text=text, 
            padding=True,
            truncation=True,
            max_length=64,  # ALIGN 的最大文本长度
            return_tensors='pt'
        ).to(self.device)
        text_embeds = self.model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids'],
        )
        
        return text_embeds
    # def encode_image(self, image):
    #     """编码图像，返回特征序列"""

        
    #     inputs = self.processor(images=image, return_tensors='pt', do_rescale=False).to(self.device)
    #     outputs = self.vision_encoder(**inputs)
        
    #     return outputs.last_hidden_state  # 返回完整的特征序列
        
    # def encode_text(self, text):
    #     if isinstance(text, str):
    #         text = [text]
            
    #     # 添加 padding 和 truncation
    #     inputs = self.processor(
    #         text=text, 
    #         padding=True,
    #         truncation=True,
    #         max_length=64,  # ALIGN 的最大文本长度
    #         return_tensors='pt'
    #     ).to(self.device)

    #     outputs = self.text_encoder(**inputs)
    #     return outputs.last_hidden_state[:, 0]  # 使用 [CLS] token 的特征

class AlignFlex(nn.Module):
    """保持与原 ClipFlex 相同的接口"""
    def __init__(self, args, device, classnames, batch_size, memory_size=10, text_prompt='tip', model_path="kakaobrain/align-base"):
        super().__init__()

        self.use_dinov2 = args.DINOv2
        if args.DINOv2:
            dinov2 = load_dinov2(args.DINO_size)
            self.dinov2 = dinov2
            # ipdb.set_trace()


        self.device = device
        self.model = AlignModel_custom(device, model_path=model_path)
        
        # 初始化属性
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.first_flag = True
        self.memory_size = memory_size
        self.text_prompt_type = text_prompt
        
        # 初始化 logit scale 参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.text_feat = None
        self.few_shot_mem = False

        self.image_encoder = self.model.encode_image
        self.text_encoder = self.model.encode_text

    @property
    def dtype(self):
        """返回模型权重的数据类型"""
        return next(self.model.vision_encoder.parameters()).dtype
        
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
            
    def get_text_features(self):
        ## get the text feature only once, multiple class & multiple prompt
        text_feat = []
        text_label = []
        count = 0
        for name in self.classnames:
            text_prompts = [template.format(name) for template in self.text_prompt]  # format with class
            if self.text_prompt_type =='tip_cupl':
                text_prompts += self.cupl_prompts[name]

            
            texts = text_prompts

            class_embeddings = self.text_encoder(texts)  # embed with text encoder

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

        if self.first_flag:  ## initlize
            self.image_feature_memory = torch.zeros(self.n_cls, self.memory_size, self.text_feat.shape[1]).to(self.text_feat.device)       ## 如果满了，把entropy 最高的扔出去
            self.image_prediction_mem = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.image_entropy_mem = torch.zeros(self.n_cls, self.memory_size).to(self.text_feat.device)   ## category prediction.
            self.image_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)

            self.local_feature_memory = torch.zeros(self.n_cls, self.memory_size, self.text_feat.shape[1]).to(self.text_feat.device)
            self.local_prediction_mem = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.local_entropy_mem = torch.zeros(self.n_cls, self.memory_size).to(self.text_feat.device)   ## category prediction.
            self.local_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)
            self.first_flag = False

        
        return self.text_feat, self.text_feat_full
        
    def get_image_features(self, image):
        """获取图像特征"""
        if image.dtype != torch.float32:
            image = image.float()
        image = torch.clamp(image, 0, 1)
        
        # 获取图像特征
        image_features = self.model.encode_image(image)
        
        # 规范化特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # torch.Size([32, 197, 512])
        self.image_features_gllo = image_features
        
        # align只输出全局特征

        self.image_features_global = image_features
        self.image_features_local = image_features

        return self.image_features_global, self.image_features_local
    

    def get_image_features_aux(self, image):
        # image torch.Size([bs, 3, 224, 224])
        # image_features_vanilla = self.image_encoder(image.type(self.dtype))
        ## for Res50 128*1024 or 128*50*1024 [global feat; 7*7 local feature]
        ## for VIT,  128*512 or 128*197*512 [global feat; 14*14 local features]

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
    

    def forward(self, input):
        # pass
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)

def get_fixed_align(args, classnames, device, memory_size=10, text_prompt='tip', model_path="kakaobrain/align-base"):
    """获取 ALIGN 模型实例"""
    model = AlignFlex(args, device, classnames, None, 
                    memory_size=memory_size, 
                    text_prompt=text_prompt,
                    model_path=model_path)
    
    
    return model
