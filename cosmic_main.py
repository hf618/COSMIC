import argparse
import time

from copy import deepcopy

from PIL import Image
import numpy as np
import yaml
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
import torch.nn as nn
import os
import operator
import logging
import datetime
import sys
import networkx as nx
import matplotlib.pyplot as plt
import gc
from mpl_toolkits.mplot3d import Axes3D

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.fix_clip import get_fixed_clip
from clip.flex_clip import get_flex_clip
from clip.custom_clip import get_coop

from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, AugMemAugmenter, StrongAugmenter
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import ipdb
from torchviz import make_dot
from typing import Callable

from scipy.stats import multivariate_normal
from scipy.special import gammaln
from scipy.linalg import det, inv

from sklearn.manifold import TSNE

def print_logger(
        old_print: Callable,
        file_name: str,
) -> Callable:
    """Returns a function which calls `old_print` twice, specifying a `file=` on the second call.

    Arguments:
        old_print: The `print` function to call twice.
        file_name: The name to give the log file.
    """

    def log_print(*args, **kwargs):
        old_print(*args, **kwargs)
        with open(file_name, "a") as log_file:
            old_print(*args, file=log_file, **kwargs)

    return log_print


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
                       (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()


## following APE.
def important_channel_indice(args, model, only_use_txt=True):
    if only_use_txt or args.shot == 0:
        feats = model.text_feat.unsqueeze(1)  ## C * 1 * D
    else:
        feats = model.fixed_global_feat_vanilla  ## C * L * D, including text feat & few shot image feat.
    cate_num, samp_num, feat_dim = feats.shape

    sim_sum = torch.zeros((feat_dim)).to(feats.device)
    count = 0
    # ipdb.set_trace()
    for i in range(cate_num):
        for j in range(cate_num):
            for m in range(samp_num):
                for n in range(samp_num):
                    if i != j:
                        sim_sum += feats[i, m, :] * feats[j, n, :]
                        count += 1
    sim = sim_sum / count
    # ipdb.set_trace()
    criterion = (-1) * args.lambda_ape * sim + (1 - args.lambda_ape) * torch.var(model.text_feat, dim=0)
    _, indices = torch.topk(criterion, k=args.num_important_channel)
    return indices


def select_confident_samples_0(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


def select_confident_samples(prob, top):
    # ipdb.set_trace()
    batch_entropy = -(prob * torch.log(prob + 1e-6)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]  ## pick the min entropy
    idx_confused = torch.argsort(batch_entropy, descending=False)[
                   int(batch_entropy.size()[0] * top):]  ## pick the max entropy
    return prob[idx], idx, prob[idx_confused], idx_confused


def avg_entropy(outputs):
    ## N*Class
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]



    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


## the main component.
class DualMem(nn.Module):
    def __init__(self, args=None, beta=5.5, feat_dim=1024, feat_dim_aux=None, class_num=1000, mapping='bias'):
        super(DualMem, self).__init__()
        self.args = args
        self.indice = args.indice  ## indice of important channels.
        self.beta = beta
        self.rank = 4
        self.init_pred = 0
        self.init_pred = 0
        if args.DINO and args.DINO4mem:
            feat_dim_img = feat_dim_aux
            self.Proj = nn.Linear(feat_dim_aux, feat_dim)
        else:
            feat_dim_img = feat_dim
        if args.shared_param:
            self.global_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.
            self.global_bias_key = self.global_bias
            self.global_bias_value = self.global_bias

            self.global_ffn_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_ffn_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.
            self.text_affine = self.global_ffn_affine
            self.text_bias = self.global_ffn_bias
        else:
            self.global_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_bias = nn.Parameter(torch.zeros((class_num, feat_dim_img)))  ## unknown use the category mean.
            self.global_bias_key = nn.Parameter(
                torch.zeros((class_num, feat_dim_img)))  ## unknown use the category mean.
            self.global_bias_value = nn.Parameter(
                torch.zeros((class_num, feat_dim_img)))  ## unknown use the category mean.

            self.global_ffn_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_ffn_bias = nn.Parameter(
                torch.zeros((class_num, feat_dim_img)))  ## unknown use the category mean.
            self.text_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.text_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))
        self.learnable_mapping = args.mapping  ### bias | affine | all

    def update_memory_bank(self, model, target, use_dino=False, selected_idx=None, stand2=True):
        # updating
        mean_prob = self.init_pred[0]
        value, indice = mean_prob.max(0)
        pseudo_label = indice.item()
        # print(value, indice, target)
        text_features = model.text_feat[pseudo_label]  ## 512
        if use_dino:
            if selected_idx is not None:
                selected_image_features_global = model.image_features_global_aux[selected_idx].mean(0).unsqueeze(0)
            else:
                selected_image_features_global = model.image_features_global_aux[:1]
        else:
            if selected_idx is not None:
                selected_image_features_global = model.image_features_global[selected_idx].mean(0).unsqueeze(0)
            else:
                selected_image_features_global = model.image_features_global[:1]

            #current_instance_entropy = self.loss_total_now

            current_instance_entropy = -(mean_prob * (torch.log(mean_prob + 1e-8))).sum()

        if stand2:
            # ipdb.set_trace()
            if model.image_feature_count[pseudo_label] == model.memory_size:
                ###### if the new one is low entropy, find the sample with the max entropy, and replace it with the new one
                if (current_instance_entropy < model.image_entropy_mem[pseudo_label]).sum() == 0:
                    pass  ## the entropy of current test image is very large.
                else:
                    _, indice = torch.sort(model.image_entropy_mem[pseudo_label])
                    to_replace_indice = indice[-1]  ## with max entropy, ascending.
                    model.image_feature_memory[pseudo_label][to_replace_indice] = selected_image_features_global
                    model.image_prediction_mem[pseudo_label][to_replace_indice] = mean_prob[0]
                    model.image_entropy_mem[pseudo_label][to_replace_indice] = current_instance_entropy
            else:
                model.image_feature_memory[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = selected_image_features_global  # torch.Size([1, 512])
                model.image_prediction_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = mean_prob[0]  # torch.Size([])
                model.image_entropy_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = current_instance_entropy  # torch.Size([])
                model.image_feature_count[pseudo_label] += 1
            # ipdb.set_trace()
            if model.__class__.__name__ == "ClipFixed":
                pass
            else:
                if model.image_feature_count_avg[pseudo_label] == 1:
                    model.image_feature_memory_avg[pseudo_label] = (model.image_feature_memory_avg[pseudo_label] + selected_image_features_global) / 2
                else:
                    model.image_feature_memory_avg[pseudo_label] = selected_image_features_global
                    model.image_feature_count_avg[pseudo_label] += 1
        # ipdb.set_trace()

    # 多元化指标
    def update_memory_bank_2(self, model, target, dis_img_feat, selected_idx=None):
        # updating
        mean_prob = self.init_pred[0]
        value, indice = mean_prob.max(0)
        pseudo_label = indice.item()
        # print(value, indice, target)
        text_features = model.text_feat[pseudo_label]  ## 512
        if selected_idx is not None:
            selected_image_features_global = model.image_features_global[selected_idx].mean(0).unsqueeze(0)
        else:
            selected_image_features_global = model.image_features_global[:1]
        # current_instance_entropy = -(mean_prob * (torch.log(mean_prob + 1e-8))).sum()
        if model.image_feature_count2[pseudo_label] == model.memory_size:
            ###### if the new one is low entropy, find the sample with the max entropy, and replace it with the new one
            if (dis_img_feat > model.image_dis_mem2[pseudo_label]).sum() == 0:
                pass  ## the dis_img_feat  of current test image is very small.
            else:
                _, indice = torch.sort(model.image_dis_mem2[pseudo_label], descending=True)
                to_replace_indice = indice[-1]  ## with dis, decending.
                model.image_feature_memory2[pseudo_label][to_replace_indice] = selected_image_features_global
                model.image_prediction_mem2[pseudo_label][to_replace_indice] = mean_prob[0]
                model.image_dis_mem2[pseudo_label][to_replace_indice] = dis_img_feat

        else:
            model.image_feature_memory2[pseudo_label][model.image_feature_count2[
                pseudo_label, 0].item()] = selected_image_features_global  # torch.Size([1, 512])
            model.image_prediction_mem2[pseudo_label][model.image_feature_count2[pseudo_label, 0].item()] = mean_prob[
                0]  # torch.Size([])
            model.image_dis_mem2[pseudo_label][model.image_feature_count2[pseudo_label, 0].item()] = dis_img_feat
            model.image_feature_count2[pseudo_label] += 1
        # ipdb.set_trace()

    # 多元化指标
    def update_memory_bank_3(self, model, target, selected_idx=None):

        # updating
        mean_prob = self.init_pred[0]
        value, indice = mean_prob.max(0)
        pseudo_label = indice.item()
        # print(value, indice, target)
        text_features = model.text_feat[pseudo_label]  ## 512
        if selected_idx is not None:
            selected_image_features_global = model.image_features_global[selected_idx].mean(0).unsqueeze(0)
        else:
            selected_image_features_global = model.image_features_global[:1]
        # current_instance_entropy = -(mean_prob * (torch.log(mean_prob + 1e-8))).sum()
        anchor, dis_img_feat_anchor = calculate_anchor_and_distance(model, pseudo_label, selected_image_features_global)
        if model.image_feature_count3[pseudo_label] == model.memory_size:
            ###### if the new one is low entropy, find the sample with the max entropy, and replace it with the new one
            if (dis_img_feat_anchor < model.image_dis_mem3[pseudo_label]).sum() == 0:
                pass  ## the dis_img_feat  of current test image is very small.
            else:
                _, indice = torch.sort(model.image_dis_mem3[pseudo_label])
                to_replace_indice = indice[-1]  ## with dis, decending.
                model.image_feature_memory3[pseudo_label][to_replace_indice] = selected_image_features_global
                model.image_prediction_mem3[pseudo_label][to_replace_indice] = mean_prob[0]
                model.image_dis_mem3[pseudo_label][to_replace_indice] = dis_img_feat_anchor

        else:
            model.image_feature_memory3[pseudo_label][model.image_feature_count3[
                pseudo_label, 0].item()] = selected_image_features_global  # torch.Size([1, 512])
            model.image_prediction_mem3[pseudo_label][model.image_feature_count3[pseudo_label, 0].item()] = mean_prob[
                0]  # torch.Size([])
            model.image_dis_mem3[pseudo_label][model.image_feature_count3[pseudo_label, 0].item()] = dis_img_feat_anchor
            model.image_feature_count3[pseudo_label] += 1
        # ipdb.set_trace()

    # 附属预测
    def get_image_pred(self, model, new_up=0, return_full=False, return_logit=False):
        ## prediction with dynamic memory.
        img_feat = model.image_features_global[:1]  # 1*1024
        if new_up == "dino":
            img_feat = model.image_features_global_aux[:1]  # 1*1024
            count_image_feat = model.image_feature_count.clone()
            num_class = model.image_feature_memory.shape[0]
            # ipdb.set_trace()
            memorized_image_feat = model.image_feature_memory
            # ipdb.set_trace()
        elif new_up == '1':
            count_image_feat = model.image_feature_count2.clone()
            num_class = model.image_feature_memory2.shape[0]
            memorized_image_feat = torch.cat((model.image_feature_memory2, model.fixed_global_feat_vanilla),
                                             dim=1)  ## 200*11*1024
        elif new_up == "2":
            count_image_feat = model.image_feature_count3.clone()
            num_class = model.image_feature_memory3.shape[0]
            memorized_image_feat = torch.cat((model.image_feature_memory3, model.fixed_global_feat_vanilla),
                                             dim=1)  ## 200*11*1024
        elif new_up == "mix":
            count_image_feat = model.image_feature_count.clone() + model.image_feature_count2.clone() + model.image_feature_count3.clone()
            num_class = model.image_feature_memory3.shape[0]
            # 遍历每个类别
            for class_idx in range(num_class):
                feature_list = []
                # 处理 model.image_feature_memory
                current_count = model.image_feature_count[class_idx].item()
                if current_count > 0:
                    features = model.image_feature_memory[class_idx, :current_count, :]
                    feature_list.append(features)
                # 处理 model.image_feature_memory2
                current_count2 = model.image_feature_count2[class_idx].item()
                if current_count2 > 0:
                    features2 = model.image_feature_memory2[class_idx, :current_count2, :]
                    feature_list.append(features2)
                # 处理 model.image_feature_memory3
                current_count3 = model.image_feature_count3[class_idx].item()
                if current_count3 > 0:
                    features3 = model.image_feature_memory3[class_idx, :current_count3, :]
                    feature_list.append(features3)
                # 拼接非零特征向量并保存到 memory_mix 中
                if feature_list:
                    combined_features = torch.cat(feature_list, dim=0)  # 在维度0上拼接特征向量
                    model.image_feature_memory_mix[class_idx, :combined_features.shape[0], :] = combined_features

            memorized_image_feat = torch.cat((model.image_feature_memory_mix, model.fixed_global_feat_vanilla),
                                             dim=1)  ## 200*11*1024
        else:
            count_image_feat = model.image_feature_count.clone()
            num_class = model.image_feature_memory.shape[0]
            memorized_image_feat = torch.cat((model.image_feature_memory, model.fixed_global_feat_vanilla),
                                             dim=1)  ## 200*11*1024
            # memorized_image_feat = model.image_feature_memory

        image_classifier = 'similarity_weighted'  ## category_center | entropy_weighted | similarity_weighted
        ### similarity_weighted achieves the best results.

        if image_classifier == 'entropy_weighted':
            ############## weighted combine the memorized feature as the final classifier.
            merged_entropy = torch.cat(
                (model.image_entropy_mem, torch.zeros(num_class, 1).to(merged_image_feat.device)), dim=1)  ## 200*11
            filled_image_feat = (merged_image_feat * (- merged_entropy - math.log(1. / num_class)).unsqueeze(-1)).sum(
                1)  ## weighting with entropy.
            filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_feat @ filled_image_feat.t()
            return logits.softmax(dim=1)
        elif image_classifier == 'category_center':
            ############### assign each feature with equal weights.
            filled_image_feat = memorized_image_feat.sum(1) / (count_image_feat + 1)  ### no zero. 200*1024
            filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_feat @ filled_image_feat.t()
            return logits.softmax(dim=1)
        elif image_classifier == 'similarity_weighted':  ## this is an instance adaptative method.
            ## calculate the cos similarity betweeen image feature and memory feature, and then weighted the memorized features according to similarity.
            ###################### 有一些memory 是空的，现在却往里面塞了一个self.global_bias， 这不合理，还要把它继续置空。
            img_feat_mappling = img_feat  # 1*1024
            memorized_image_feat_K = memorized_image_feat  # 200*11*1024
            memorized_image_feat_V = memorized_image_feat  # 200*11*1024
            # ipdb.set_trace()
            with torch.no_grad():
                if self.args.position == 'query':
                    img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
                elif self.args.position == 'key':
                    memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(
                        1)  ## class*shot*1024
                elif self.args.position == 'value':
                    memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(
                        1)  ## class*shot*1024
                elif self.args.position == 'qkv' or self.args.position == 'all':
                    img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
                    memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(
                        1)  ## class*shot*1024
                    memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(
                        1)  ## class*shot*1024
                else:
                    pass
                memorized_image_feat_K = memorized_image_feat_K / memorized_image_feat_K.norm(dim=-1, keepdim=True)
                ## some memorized_image_feat slots are empty before mapping, reseting them to empty.
                memorized_image_feat_K[memorized_image_feat.sum(-1) == 0] = 0
                memorized_image_feat_V = memorized_image_feat_V / memorized_image_feat_V.norm(dim=-1, keepdim=True)
                memorized_image_feat_V[memorized_image_feat.sum(-1) == 0] = 0
                img_feat_mappling = img_feat_mappling / img_feat_mappling.norm(dim=-1, keepdim=True)

            similarity_matrix = (img_feat_mappling * memorized_image_feat_K).sum(
                -1)  ## 200*11  idealy [-1,1], practically [0.1, 0.2]
            # print("1", similarity_matrix)
            similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))
            # print("2",similarity_matrix)
            ### weighting memoried features with similarity weights.
            adaptive_image_feat = (memorized_image_feat_V * similarity_matrix.unsqueeze(-1)).sum(1)  # n_cls * e_dim
            # print("1",adaptive_image_feat)
            ## torch.Size([1, class, dim])
            if new_up == 'dino':
                # 计算范数并避免除以零
                norms = adaptive_image_feat.norm(dim=-1, keepdim=True)
                adaptive_image_feat = adaptive_image_feat / (norms + 1e-8)  # 加上一个小常数 1e-8 防止除零
            else:
                adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            # print("2", adaptive_image_feat)
            if self.args.position == 'output' or self.args.position == 'all':
                adaptive_image_feat = adaptive_image_feat + self.global_ffn_bias.unsqueeze(0)  ## class*shot*1024
            # print("3", adaptive_image_feat)
            if new_up == 'dino':
                # 计算范数并避免除以零
                norms = adaptive_image_feat.norm(dim=-1, keepdim=True)
                adaptive_image_feat = adaptive_image_feat / (norms + 1e-8)  # 加上一个小常数 1e-8 防止除零
            else:
                adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            # adaptive_image_feat: torch.Size([1, 102, 1024])
            # img_feat: torch.Size([1, 1024])
            logits = logit_scale * adaptive_image_feat @ img_feat.unsqueeze(-1)  ## used feat is not update.
            logits = logits[:, :, 0]

            # has_zero = (logits == 0).any()
            #
            # #print("logits", logits)
            # #ipdb.set_trace()
            # if has_zero:
            #     return torch.zeros(1, num_class).to(logits.device)
            self.adaptive_image_feat = adaptive_image_feat

            return logits.softmax(dim=1)
        else:
            raise NotImplementedError

    def get_image_pred_fewshot_global(self, model, return_full=False, return_logit=False):
        ## prediction with static memory.
        if return_full:
            img_feat = model.image_features_global  # 1*1024
        else:
            img_feat = model.image_features_global[:1, :]  # 1*1024
        num_class = model.image_feature_memory.shape[0]
        memorized_image_feat = model.fixed_global_feat  ## 200*11*1024, few shot samples and text features.
        img_feat_mappling = img_feat
        memorized_image_feat_K = memorized_image_feat
        memorized_image_feat_V = memorized_image_feat

        if self.args.position == 'query':
            img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
        elif self.args.position == 'key':
            memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(1)  ## class*shot*1024
        elif self.args.position == 'value':
            memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(1)  ## class*shot*1024
        elif self.args.position == 'qkv' or self.args.position == 'all':
            img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
            memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(1)  ## class*shot*1024
            memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(1)  ## class*shot*1024

        memorized_image_feat_K = memorized_image_feat_K / memorized_image_feat_K.norm(dim=-1, keepdim=True)
        memorized_image_feat_V = memorized_image_feat_V / memorized_image_feat_V.norm(dim=-1, keepdim=True)
        img_feat_mappling = img_feat_mappling / img_feat_mappling.norm(dim=-1, keepdim=True)
        ## calculate the cos similarity betweeen image feature and memory feature, and then weighted the memorized probability.
        ##  200*11*200；
        similarity_matrix = memorized_image_feat_K @ img_feat_mappling.T  ## class*shot*Batch
        similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))
        adaptive_image_feat = memorized_image_feat_V.transpose(1,
                                                               2) @ similarity_matrix  ## class * D * batch, 102*1024*204
        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        adaptive_image_feat = adaptive_image_feat.transpose(0, 2).transpose(1, 2)  ## 204*102*1024
        if self.args.position == 'output' or self.args.position == 'all':
            adaptive_image_feat = adaptive_image_feat + self.global_ffn_bias.unsqueeze(0)  ## class*shot*1024

        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
        # ipdb.set_trace()
        # adaptive_image_feat: 1*102*1024
        # img_feat: 1*1024
        logits = logit_scale * adaptive_image_feat[..., self.args.indice] @ img_feat[..., self.args.indice].unsqueeze(
            -1)  ## memoried features are not updated.
        if return_logit:
            return logits[:, :, 0]
        else:
            return logits[:, :, 0].softmax(dim=1)

    # 正统预测
    def get_text_prediction(self, model, return_full=True, return_logit=False):
        logit_scale = model.logit_scale.exp()
        if self.args.position == 'output' or self.args.position == 'all':
            text_feat = model.text_feat + self.text_bias
        else:
            text_feat = model.text_feat
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)  ## already filtered with indice.
        img_text_logit = logit_scale * model.image_features_global @ text_feat.t()  ## 128*200
        if return_full:
            pass
        else:
            img_text_logit = img_text_logit[:1]
        if return_logit:
            return img_text_logit
        else:
            return img_text_logit.softmax(-1)

    # with patehx 197
    def get_text_prediction_gllo(self, model, return_full=True, return_logit=False):
        logit_scale = model.logit_scale.exp()
        if self.args.position == 'output' or self.args.position == 'all':
            text_feat = model.text_feat + self.text_bias
        else:
            text_feat = model.text_feat
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)  ## already filtered with indice.
        img_text_logit = logit_scale * model.image_features_gllo @ text_feat.t()  ## 128*200
        if return_full:
            pass
        else:
            img_text_logit = img_text_logit[:1]
        if return_logit:
            return img_text_logit
        else:
            return img_text_logit.softmax(-1)


# 针对img的 使用 多存版gra_cache
def get_affinity2(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []  # for feats
        cache_values = []  # for label
        # 函数遍历 cache 字典，该字典按类别索引排序，并存储每个类别的缓存项。
        for class_index in sorted(cache.keys()):
            # 对于每个缓存项：
            for item in cache[class_index]:
                # print("item[0]",item[0])
                cache_keys.append(item[2])  # 将图像特征（键）添加到 cache_keys 列表
                if neg_mask_thresholds:  # 根据是否提供了 neg_mask_thresholds，将相应的类别索引或负掩码阈值添加到 cache_values 列表。
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)
        # 将 cache_keys 列表中的图像特征堆叠成一个张量，并进行排列，以便进行矩阵乘法。
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        # 如果提供了 neg_mask_thresholds，则将 cache_values 转换为一个张量，并根据阈值进行处理，然后将其转换为半精度浮点数并移动到GPU上。
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(
                torch.int8)).cuda().half()
        # 如果没有提供 neg_mask_thresholds，则使用 torch.one_hot 创建一个独热编码张量，表示类别索引，然后将其转换为半精度浮点数并移动到GPU上。
        else:
            cache_values = (
                F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()
        image_features = image_features.to(cache_keys.dtype)
        affinity = image_features @ cache_keys
        affinity = affinity.to(cache_values.dtype)
        return beta * affinity


def compute_similarity_loss(gra_0, gra_cache, affinity=None):
    # 拉平 gra_0
    gra_0_flat = gra_0.view(1, -1).clone()  # 变为 1*2048
    # 初始化损失
    total_loss = 0.0
    # 遍历缓存中的梯度
    # 函数遍历 cache 字典，该字典按类别索引排序，并存储每个类别的缓存项。
    num = 0
    loss_list = []
    for class_index in sorted(gra_cache.keys()):
        # 对于每个缓存项：
        for item in gra_cache[class_index]:
            # print("item",item)
            # 解释一下：gra_cache中item[0]是一个1*2048，pro_cache中item[0]是一个list，其中item[0][0]是4*512
            gra_flat = item[0][0].view(1, -1)  # 拉平缓存中的梯度 or prompt
            # 计算余弦相似度
            # print("gra_0_flat",gra_0_flat.shape)
            # print("gra_flat",gra_flat.shape)
            cosine_similarity = torch.nn.functional.cosine_similarity(gra_0_flat, gra_flat)
            # 计算损失 (1 - cosine_similarity)
            loss = 1 - cosine_similarity
            # print("cosine_similarity", cosine_similarity)
            # 累加损失
            # total_loss += loss[0]
            loss_list.append(loss[0])
            num += 1
            # ipdb.set_trace()
    # print("loss", loss_list)
    # print("affinity", affinity)
    if affinity is not None:
        # 1. 归一化 affinity
        affinity_norm = affinity / torch.sum(affinity)
        # 2. 转换 loss 为张量并确保形状兼容
        loss_tensor = torch.stack(loss_list, dim=0)  # 变为 (3,)
        # 2. 统一数据类型
        affinity_norm = affinity_norm.to(loss_tensor.dtype)
        # 3. 计算加权平均
        weighted_loss = torch.dot(affinity_norm.flatten(), loss_tensor)
        # print("weighted_loss", weighted_loss)
        # print("\n")
        return weighted_loss if gra_cache else total_loss
    else:
        # 返回平均损失
        return torch.mean(loss_list) if gra_cache else total_loss

def info_nce_loss(feature1, non_anchor_features, temperature=0.1):
    """
    feature1: anchor特征，shape: [512]
    non_anchor_features: 其他增强特征，shape: [bs-1, 512]
    temperature: 控制对比学习的强度
    """
    # 归一化特征
    feature1 = F.normalize(feature1, dim=-1)
    non_anchor_features = F.normalize(non_anchor_features, dim=-1)

    # 计算相似度
    logits = torch.matmul(non_anchor_features, feature1) / temperature  # [bs-1]

    # 构造标签，假设所有增强图像都是正样本
    labels = torch.ones(non_anchor_features.size(0), device=feature1.device)

    # 计算二元交叉熵损失
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    return loss

def distillation_loss(output, img_global_pred, temperature=1.0):
    """
    output: 学生模型的预测分布, shape: [bs, n_cls]
    img_global_pred: 教师模型的预测分布, shape: [1, n_cls]
    temperature: 温度缩放系数, 控制分布的平滑程度
    """
    # 对学生模型的输出进行log-softmax操作, 然后除以温度系数
    student_log_probs = F.log_softmax(output / temperature, dim=-1)  # [bs, n_cls]

    # 对教师模型的输出进行softmax操作, 然后除以温度系数
    teacher_probs = F.softmax(img_global_pred / temperature, dim=-1)  # [1, n_cls]

    # 计算KL散度损失, 使用teacher_probs作为目标
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    return kl_loss

# 进行prompt tuning
def test_time_tuning(cfg, model, inputs, optimizer, scaler, args, gra_cache, pro_cache, Train=False, img_global_pred=None):
    # inputs torch.Size([32, 3, 224, 224])
    # 说明是cocoop模型，需要特殊处理。将 pgen_ctx 设置为需要梯度，并创建一个新的优化器仅针对 pgen_ctx。
    # selected_idx 用于存储选择的置信度高的样本的索引，初始为 None。
    selected_idx = None
    # 循环 args.tta_steps 次，每次都会执行前向传播，选择置信度高的样本，并执行优化步骤。
    for j in range(args.tta_steps):
        # print("tta_steps:", j+1)
        with torch.cuda.amp.autocast():
            # with torch.autograd.set_detect_anomaly(True):
            output, _ = model(inputs)  # torch.Size([bs, n_cls])
            # make_dot(output, params=dict(model.named_parameters())).render("model_graph", format="png")
            # print("output", output.shape)
            # 如果之前已经选择了样本，则只考虑这些样本的输出；
            # ipdb.set_trace()  # 暂停执行

            # if selected_idx is not None:
            #     output = output[selected_idx]
            #     img_feats = model.image_features_global[selected_idx]
            # # 否则，使用 select_confident_samples 函数基于输出的熵选择置信度高的样本。
            # else:
            #     output, selected_idx = select_confident_samples_0(output, args.selection_p)
            #     img_feats = model.image_features_global[selected_idx]

            # # 计算 loss 1
            loss1 = avg_entropy(output)
            #print("loss1", loss1)
            # # 计算 loss 2
            feature1 = model.image_features_global[0]  # [512]
            non_anchor_features = model.image_features_global[1:]  # [bs-1, 512]
            # distances = torch.norm(non_anchor_features - feature1, p=2, dim=1)  # [bs-1]
            # average_distance = distances.mean()
            loss2 = info_nce_loss(feature1, non_anchor_features)
            #print("loss2", loss2)
            # 计算 loss 3
            # n_cls = model.image_feature_count_avg.shape[0]
            # non_zero_mask = model.image_feature_count_avg.squeeze() == 1  # [n_cls]，0-1张量
            # valid_memory_avg = model.image_feature_memory_avg[non_zero_mask]  # 筛选非全零类 [有效类数, e_dim]
            # min_distance_idx = None
            # if valid_memory_avg.size(0) > 0:
            #     # 3. 计算 feats 到每个非零类的 anchor 特征的欧氏距离
            #     distances = torch.norm(valid_memory_avg - feature1, p=2, dim=1)  # [有效类数]
            #     # 4. 找到距离最短的非零类特征索引
            #     min_distance_idx = torch.argmin(distances)  # 最短距离特征的索引
            #     # 5. 计算最短距离作为 loss3
            #     loss3 = distances[min_distance_idx]
            #     # 6. 获取伪标签，即对应原始 self.image_feature_memory_avg 的类索引
            #     # 通过 non_zero_mask 获取与有效特征对应的类索引
            #     valid_classes = torch.arange(n_cls, device=non_zero_mask.device)[non_zero_mask]  # 将 arange 移动到同一设备
            #     pseudo_label = valid_classes[min_distance_idx]  # 伪标签为最短距离对应的类
            # else:
            #     # 如果所有类特征都为全零，则 loss3 设为 0
            #     loss3 = torch.tensor(0.0, device=feature1.device)
            #     pseudo_label = torch.tensor(-1, device=feature1.device)  # 表示没有有效的伪标签
            # # print("loss3", loss3)
            # # loss1 = loss1 + average_distance + loss3
            # # # 计算 loss 4
            # if pseudo_label > 0:
            #     # 方法1：采用 idx=0 的 logits
            #     final_logits = output[0, :]  # 选择第0个batch的logits
            #     # 方法2：采用 logits 的平均值
            #     # final_logits = output.mean(dim=0)  # 对整个 batch 的 logits 求平均
            #
            #     loss4 = F.cross_entropy(final_logits.unsqueeze(0), pseudo_label.unsqueeze(0))  # 计算交叉熵损失
            # else:
            #     loss4 = torch.tensor(0.0, device=feature1.device)
            #print("loss4:", loss4)
            #loss1 = loss1 + average_distance
            #loss1 += loss2 + loss4
            # print("loss total", loss1)
            output_so = output.softmax(1)
            # 制作 loss 5
            loss5 = distillation_loss(output_so, img_global_pred)
            #print("loss5:", loss5)
            loss1 += loss2 + loss5
            #print("loss total", loss1)
            #ipdb.set_trace()
            # 尝试用偷换
        if not Train:
            return loss1

        if not args.loss_grad and not args.loss_prop:
            # ipdb.set_trace()
            optimizer.zero_grad()
            # compute gradient and do SGD step
            # for param in model.parameters():
            #     if param.grad is not None:
            #         print("1 ",param.grad)
            if not args.v2:
                scaler.scale(loss1).backward()
                # for param in model.parameters():
                #     if param.grad is not None:
                #         param.grad = param.grad.clone() * scaler.get_scale()
                #         print("2 ",param.grad)
                # ipdb.set_trace()
                # Unscales the gradients of optimizer's assigned params in-place
                # 查看 cross_attention.query_proj.weight 的梯度
                # print("cross_attention.clip_to_dino_proj.weight 的梯度:")
                # print(model.cross_attention.clip_to_dino_proj.weight.grad)
                #
                # # 查看 model.prompt_learner.ctx 的梯度
                # print("model.prompt_learner.ctx 的梯度:")
                # print(model.prompt_learner.ctx.grad)
                # ipdb.set_trace()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss1.backward()  # 不再使用 scaler
                # 进行优化器的更新步骤
                optimizer.step()
        else:
            optimizer.zero_grad()
            scaler.scale(loss1).backward(retain_graph=True)
            if args.loss_grad and gra_cache:
                prop_list = []
                grad_list = []
                for name, param in model.named_parameters():
                    if param.grad is not None and "prompt_learner" in name:
                        prop_list.append(param)
                        grad_list.append(param.grad.clone() / scaler.get_scale())
                img_feats = img_feats.mean(0).unsqueeze(0)
                grad = grad_list[0]
                affinity = get_affinity2(img_feats, gra_cache, cfg['positive']['alpha'], cfg['positive']['beta'],
                                         model.text_feat, neg_mask_thresholds=None)
                # ipdb.set_trace()
                loss2 = compute_similarity_loss(grad, gra_cache, affinity).detach()
                loss = loss1 + loss2
                if args.loss_prop and pro_cache:
                    prop = prop_list[0]
                    affinity = get_affinity2(img_feats, pro_cache, cfg['positive']['alpha'], cfg['positive']['beta'],
                                             model.text_feat, neg_mask_thresholds=None)
                    loss3 = compute_similarity_loss(prop, pro_cache, affinity).detach()
                    loss += loss3
            else:
                loss = loss1

            # ipdb.set_trace()
            if not args.v2:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            # ipdb.set_trace()
    return loss1

def test_time_tuning_sim(cfg, model, inputs, optimizer, scaler, args, gra_cache, pro_cache, Train=False, img_global_pred=None):
    # inputs torch.Size([32, 3, 224, 224])
    # 说明是cocoop模型，需要特殊处理。将 pgen_ctx 设置为需要梯度，并创建一个新的优化器仅针对 pgen_ctx。
    # selected_idx 用于存储选择的置信度高的样本的索引，初始为 None。
    selected_idx = None
    # 循环 args.tta_steps 次，每次都会执行前向传播，选择置信度高的样本，并执行优化步骤。
    for j in range(args.tta_steps):
        # print("tta_steps:", j+1)
        with torch.cuda.amp.autocast():
            # with torch.autograd.set_detect_anomaly(True):
            output, _ = model(inputs)  # torch.Size([bs, n_cls])
            # # 计算 loss 1
            loss1 = avg_entropy(output)
            #print("loss1", loss1)
            # # 计算 loss 2
            feature1 = model.image_features_global[0]  # [512]
            non_anchor_features = model.image_features_global[1:]  # [bs-1, 512]
            loss2 = info_nce_loss(feature1, non_anchor_features)
            #print("loss2", loss2)
            # print("loss total", loss1)
            output_so = output.softmax(1)
            # 制作 loss 5
            loss5 = distillation_loss(output_so, img_global_pred)
            #print("loss5:", loss5)
            loss1 += loss5
            #print("loss total", loss1)
            #ipdb.set_trace()
            # 尝试用偷换
    if not Train:
        return loss1
    else:
            optimizer.zero_grad()
            scaler.scale(loss1).backward()
            scaler.step(optimizer)
            scaler.update()
            # ipdb.set_trace()
    return loss1
def fine_tune_CA(cfg, model, inputs, optimizer, scaler, args, pred):
    anchor = model.fixed_global_feat_vanilla[pred]  # shape: [1, 512]
    _, cross_attention_feat = model(inputs)
    cosine_sim = F.cosine_similarity(cross_attention_feat, anchor, dim=-1)
    loss = 1 - cosine_sim  # 或者使用 -cosine_sim 以最大化相似度
    loss = loss.mean()

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return
    # ipdb.set_trace()


def get_searched_param(set_id, n_shot, ft):
    if ft:
        if set_id == 'I':
            return [0], [0.3], [0.00001], [100]
        elif set_id == 'Flower102':
            return [0], [0.3], [0.001], [100]
        elif set_id == 'DTD':
            return [0], [0.3], [0.0001], [100]
        elif set_id == 'Pets':
            return [0], [0.3], [0.0001], [20]
        elif set_id == 'Cars':
            return [0], [0.3], [0.0001], [100]
        elif set_id == 'UCF101':
            return [0], [0.3], [0.0001], [100]
        elif set_id == 'Caltech101':
            return [0], [0.3], [0.0001], [20]
        elif set_id == 'Food101':
            if n_shot >= 8:
                return [0], [0.3], [0.0001], [100]
            else:
                return [0], [0.3], [0.0001], [20]
        elif set_id == 'SUN397':
            return [0], [0.3], [0.0001], [20]
        elif set_id == 'Aircraft':
            return [0], [0.3], [0.0001], [100]
        elif set_id == 'eurosat':
            if n_shot >= 8:
                return [0], [0.3], [0.001], [100]
            else:
                return [0], [0.3], [0.0001], [100]
        else:
            raise NotImplementedError
    else:
        return [0], [0.3], [0.1], [20]  ## not used.


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "K"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"

    config_file = os.path.join(config_path, config_name)

    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# 注意修改
use_log = 1


def main():
    args = parser.parse_args()
    if use_log:
        # 获取当前时间并格式化为年月日时分的格式
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        dev_num = args.gpu
        #ipdb.set_trace()
        bs = args.batch_size
        log_filename = f"./logs/dev_{dev_num}/bs{bs}_{current_time}.log"
        sys.stdout = Logger(log_filename)
    # 遍历并打印 args 中的所有参数和值
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    args.log = args.log + '_' + str(args.gpu)
    set_random_seed(args.seed)
    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    # ipdb.set_trace()
    # 选择调度哪种model
    # 选择是否 fix
    if args.fix:
        model = get_fixed_clip(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init, memory_size=args.memory_size,
                               text_prompt=args.text_prompt)

        for name, param in model.named_parameters():
            param.requires_grad_(False)
    else:
        model = get_flex_clip(args, args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init,
                              memory_size=args.memory_size, text_prompt=args.text_prompt)
        # model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init, memory_size=args.memory_size)
        for name, param in model.named_parameters():
            if args.DINO4cross:
                if "prompt_learner" not in name and (("cross_attention" not in name) and args.DINO and args.DINO4cross):
                    param.requires_grad_(False)
                else:
                    print("Learn able: ", name)
            elif args.DINOv2:
                if "prompt_learner" not in name and (("cross_attention" not in name) and args.DINOv2):
                    param.requires_grad_(False)
                else:
                    print("Learn able: ", name)
            else:
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
                else:
                    print("Learn able: ", name)
    # 计算模型中参数的总数
    if args.DINOv2:
        total_params = sum(p.numel() for p in model.cross_attention.parameters())
        print(f"CA NAME:{model.cross_attention.name}")
        print(f"CA Param NUM: {total_params}")
    # # 打印所有可训练的参数名称
    # for name, param in model.named_parameters():
    #     if param.requires_grad:  # 只打印可训练的参数
    #         print(f"可学习: {name}")
    # ipdb.set_trace()
    if not args.fix:

        if (args.DINO and args.DINO4cross) or (args.DINOv2):
            trainable_param1 = list(model.prompt_learner.parameters())
            trainable_param2 = list(model.cross_attention.parameters())
            trainable_param = trainable_param1 + trainable_param2
        else:
            trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr_tpt)
        # 检查优化器中的参数
        # for param_group in optimizer.param_groups:
        #     for param in param_group['params']:
        #         if param.requires_grad:
        #             print(f"Parameter {param.shape} requires gradient.")
        #         else:
        #             print(f"Parameter {param.shape} does NOT require gradient.")
        optim_state = deepcopy(optimizer.state_dict())
        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    else:
        optimizer = None
        optim_state = None
        scaler = None
    model_state = None
    # ipdb.set_trace()
    print("=> Model created: visual backbone {}".format(args.arch))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    num_important_channel_list = args.num_important_channel.split("/")
    lambda_ape_list = args.lambda_ape.split("/")
    lr_list = args.lr.split("/")
    epoch_list = args.epoch.split("/")
    results = {}
    print_log = print_logger(print, os.path.join(args.log + '.txt'))
    # 来一个config
    config_path = args.config
    i = 0
    make_cp_datasets = args.select_ids.split("/")
    for set_id in datasets:
        if args.DINOv2 and not args.DINOv2_mem:
            if args.Choose_cp: # 手动选取制作好的cp
                is_train = False
                #N = "LinearProjectionFusion_ucf101_bs128.pth"
                N = "LinearProjectionFusion_flower102_bs64.pth"
                checkpoint_path = f"./Checkpoints/dev_{gpu}/{N}"
                checkpoint = torch.load(checkpoint_path)
                model.cross_attention.load_state_dict(checkpoint['cross_attention'])
                print(f"Datasets {set_id}, load from '{N}")

                for name, param in model.prompt_learner.named_parameters():
                    param.requires_grad_(True)
                for name, param in model.cross_attention.named_parameters():
                    param.requires_grad_(False)

            else:
                select_id = make_cp_datasets[i]
                #if set_id == select_id:
                checkpoint_path = f"./Checkpoints/dev_{gpu}/{model.cross_attention.name}_{select_id}_bs{args.batch_size}.pth"
                is_train = not os.path.exists(checkpoint_path)
                if is_train and set_id == select_id:
                    # 当前数据集是"I"，重置model.cross_attention
                    model.cross_attention.reset()
                    print(f"Datasets {set_id}, reset CA Param")
                    for name, param in model.prompt_learner.named_parameters():
                        param.requires_grad_(False)
                    for name, param in model.cross_attention.named_parameters():
                        param.requires_grad_(True)
                else:
                    # 当前数据集不是"I"，从checkpoint加载cross_attention参数
                    checkpoint = torch.load(checkpoint_path)
                    model.cross_attention.load_state_dict(checkpoint['cross_attention'])
                    print(f"Datasets {set_id}, load from '{select_id}' CA Param")

                    for name, param in model.prompt_learner.named_parameters():

                        param.requires_grad_(True)
                    for name, param in model.cross_attention.named_parameters():
                        param.requires_grad_(False)
        else:
            is_train = False


        if is_train:
            print("Training CA Yes " * 10)
        else:
            print("Training CA No " * 10)


        if args.use_searched_param:
            num_important_channel_list, lambda_ape_list, lr_list, epoch_list = get_searched_param(set_id, args.n_shot,
                                                                                                  args.ft)
        best_acc = 0
        print("*" * 80)
        print_log("processing the dataset {} \n".format(set_id), end="	")

        cfg = get_config_file(config_path, set_id)  # 获取数据指定config
        print("\nRunning dataset configurations:")
        print(cfg, "\n")

        for num_important_channel in num_important_channel_list:
            for lambda_ape in lambda_ape_list:
                for lr in lr_list:
                    for epoch in epoch_list:
                        print(
                            'adopt num_important_channel {}, lambda_ape: {}'.format(num_important_channel, lambda_ape))
                        args.lr = float(lr)
                        args.epoch = int(epoch)
                        args.num_important_channel = int(num_important_channel)
                        args.lambda_ape = float(lambda_ape)
                        base_transform = transforms.Compose([
                            transforms.Resize(args.resolution, interpolation=BICUBIC),
                            transforms.CenterCrop(args.resolution)])
                        preprocess = transforms.Compose([
                            transforms.ToTensor(),
                            normalize])
                        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1,
                                                             augmix=len(set_id) > 1, severity=50) ### aug mix not used for ImageNet test set.

                        # data_transform = AugMixAugmenter(base_transform,  preprocess, n_views=args.batch_size - 1, augmix=False)
                        # ipdb.set_trace()

                        test_transform = transforms.Compose([
                            transforms.Resize(args.resolution, interpolation=BICUBIC),
                            transforms.CenterCrop(args.resolution), transforms.ToTensor(), normalize])
                        batchsize = 1

                        print("evaluating: {}".format(set_id))
                        # reset the model
                        # Reset classnames of custom CLIP model
                        if len(set_id) > 1:
                            # fine-grained classification datasets
                            classnames = eval("{}_classes".format(set_id.lower()))
                        else:
                            assert set_id in ['A', 'R', 'K', 'V', 'I']
                            classnames_all = imagenet_classes
                            classnames = []
                            if set_id in ['A', 'R', 'V']:
                                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                                if set_id == 'R':
                                    for i, m in enumerate(label_mask):
                                        if m:
                                            classnames.append(classnames_all[i])
                                else:
                                    classnames = [classnames_all[i] for i in label_mask]
                            else:
                                classnames = classnames_all
                        # if coop:
                        #     model.reset_classnames(classnames, args.arch)
                        # else:
                        model.reset_classnames(classnames, set_id)
                        if not args.fix:
                            model.prompt_learner.reset_classnames(classnames, args.arch)
                            if args.EMA1:
                                model.prompt_learner.reset_ctx_e()
                            if args.DINO and args.DINO4cross:
                                model.cross_attention.reset()
                        # ipdb.set_trace()  # 暂停执行
                        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
                        print("number of test samples: {}".format(len(val_dataset)))
                        val_loader = torch.utils.data.DataLoader(
                            val_dataset,
                            batch_size=batchsize, shuffle=True,  ## the input has been shuffled.
                            num_workers=args.workers, pin_memory=True)
                        args.set_id = set_id
                        model.eval()
                        with torch.no_grad():
                            text_feat, text_feat_full = model.get_text_features()
                        if args.n_shot:
                            if args.n_augview == 0:
                                train_dataset_mem = build_dataset(set_id, test_transform, args.data, mode='train',
                                                                  n_shot=args.n_shot)
                                print("number of training samples: {}".format(len(train_dataset_mem)))
                                train_loader_mem = torch.utils.data.DataLoader(
                                    train_dataset_mem,
                                    batch_size=1, shuffle=False,  ## the input has been shuffled.
                                    num_workers=args.workers, pin_memory=True)
                                init_image_memory(train_loader_mem, model, args)
                                del train_dataset_mem, train_loader_mem
                            else:
                                ######### generate num_aug_view augmented views for each samples; APE adopt ten...
                                assert args.n_augview % args.n_shot == 0
                                num_aug_view = int(args.n_augview / args.n_shot)
                                data_transform_aug = AugMemAugmenter(base_transform, preprocess,
                                                                     n_views=num_aug_view - 1,
                                                                     augmix=len(
                                                                         set_id) > 1)  ### aug mix not used for ImageNet test set.
                                train_dataset_mem = build_dataset(set_id, data_transform_aug, args.data, mode='train',
                                                                  n_shot=args.n_shot)
                                print("number of training samples: {}, number of augview: {}".format(
                                    len(train_dataset_mem), args.n_augview))
                                train_loader_mem = torch.utils.data.DataLoader(
                                    train_dataset_mem,
                                    batch_size=1, shuffle=False,  ## the input has been shuffled.
                                    num_workers=args.workers, pin_memory=True)
                                init_image_memory(train_loader_mem, model, args)
                                del train_dataset_mem, train_loader_mem
                        ########## extract the importance channels via APE.
                        if args.num_important_channel != 0:
                            important_indice = important_channel_indice(args, model)  ##
                            args.indice = important_indice
                        else:
                            important_indice = torch.arange(model.text_feat.shape[1]).to(
                                model.text_feat.device)  ## use all channels.
                            args.indice = important_indice

                        # 重点关注

                        is_train = False
                        # 重点关注
                        results_temp = direct_inference(cfg, val_loader, model, model_state, optimizer, optim_state,
                                                        scaler, args, set_id, Train=is_train)
                        print_log(
                            "lr: {}, epoch:{}, num_important_channel{}, lambda_ape: {}, best acc{:.2f} \n".format(lr,
                                                                                                                  epoch,
                                                                                                                  num_important_channel,
                                                                                                                  lambda_ape,
                                                                                                                  results_temp[
                                                                                                                      3]),
                            end="	")
                        if results_temp[9] > best_acc:
                            results[set_id] = results_temp
                            best_acc = results_temp[9]
                        # results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args)
                        del val_dataset, val_loader

                        try:
                            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0],
                                                                                 results[set_id][1]))
                        except:
                            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))
                        length = len(results[set_id])


        #if set_id == select_id:
        if args.DINOv2 and is_train and set_id == select_id:
            # 假设训练完数据集"I"之后保存
            # path = f"./Checkpoints/dev_{gpu}/{model.cross_attention.name}_{select_id}_bs{args.batch_size}.pth"
            torch.save({
                'cross_attention': model.cross_attention.state_dict(),
            }, checkpoint_path)
            print(f"Cross-attention has been saved to {checkpoint_path}")
            i += 1
    args.indice = 0
    log = open(os.path.join(args.log + '.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()
    print_log("======== Result Summary ========")
    print_log("params: bs	lr	selection_p")
    print_log("params: {}	{}	{}".format(args.batch_size, args.lr, args.selection_p))
    print_log(
        "\t\t [set_id] \t\t Top-1 acc. \t\t Top-1 local acc, \t\t Top-1 global acc \t\t Searched acc \t\t beta \t\t gama.")
    for id in results.keys():
        print_log("{}".format(id), end="	")
    print_log('mean', end="	")
    print_log("\n")
    for i in range(length):
        cul_acc = 0
        cul_count = 0
        for id in results.keys():
            print_log("{:.3f}".format(results[id][i]), end="	")
            cul_acc += float(results[id][i])
            cul_count += 1
        print_log("{:.3f}".format(cul_acc), end="	")
        print_log("\n")


def entropy(outputs):
    # prob: 1*200, logit.
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0, keepdim=True) - np.log(
        logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]; log(mean_prob)
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    confidence_entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    return confidence_entropy


def init_image_memory(train_loader, model, args):
    model.eval()
    if model.first_flag:
        with torch.no_grad():
            text_feat, text_feat_full = model.get_text_features()
    else:
        print('the text feat has already initilized, pass it here.')
    memorized_image_global_feat = []  ## N*[shot*aug]*C
    memorized_image_local_feat = []  ## N*[shot*aug]*C
    memorized_image_global_feat_vanilla = []  ## N*shot*C
    memorized_image_local_feat_vanilla = []  ## N*shot*C
    memorized_labels = []

    for i in range(model.n_cls):
        memorized_image_global_feat.append([])
        memorized_image_local_feat.append([])
        memorized_image_global_feat_vanilla.append([])
        memorized_image_local_feat_vanilla.append([])
        memorized_labels.append([])

    for i, (images, target) in enumerate(train_loader):
        assert args.gpu is not None
        if isinstance(images, list):  ### augmix return, list
            images = torch.cat(images, dim=0)
            images = images.cuda(args.gpu, non_blocking=True)
        else:  ## standard return, Tensor
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            image_features_global, image_features_local = model.get_image_features(images)  ## 4*1024; 4*49*1024.
        text_features = model.text_feat[target]  ## 512
        ## only use the original ?? we should use all; however, only use the vanilla one in the dynamic memory.
        selected_image_features_local = model.image_features_local
        cos_sim = (selected_image_features_local * text_features).sum(-1)  ## between 0.2-0.3, very close.
        weight_prob = (cos_sim * 100).softmax(-1)  ## 1*197, following clip temperature.
        ########
        attented_feat = (weight_prob.unsqueeze(-1) * selected_image_features_local).sum(1)  ## 1*512
        attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 1*512
        memorized_image_global_feat[target].append(image_features_global)  ## aug*C
        memorized_image_local_feat[target].append(attented_feat)  # aug * C
        memorized_image_global_feat_vanilla[target].append(image_features_global[:1])  ## aug*C
        memorized_image_local_feat_vanilla[target].append(attented_feat[:1])  # aug * C
        one_hot_target = torch.zeros(1, model.n_cls).to(target.device)
        one_hot_target[0, target] = 1
        memorized_labels[target].append(one_hot_target)  ## 1 * C, turn it to one hot labels.

    for i in range(model.n_cls):
        memorized_image_global_feat[i] = torch.cat(memorized_image_global_feat[i], dim=0).unsqueeze(0)  ## 1*augshot*C
        memorized_image_local_feat[i] = torch.cat(memorized_image_local_feat[i], dim=0).unsqueeze(0)
        memorized_image_global_feat_vanilla[i] = torch.cat(memorized_image_global_feat_vanilla[i], dim=0).unsqueeze(
            0)  ## 1*shot*C
        memorized_image_local_feat_vanilla[i] = torch.cat(memorized_image_local_feat_vanilla[i], dim=0).unsqueeze(0)
        memorized_labels[i] = torch.cat(memorized_labels[i], dim=0).unsqueeze(0)

    memorized_image_global_feat = torch.cat(memorized_image_global_feat, dim=0)  ## n*shot*c
    memorized_image_local_feat = torch.cat(memorized_image_local_feat, dim=0)
    memorized_image_global_feat_vanilla = torch.cat(memorized_image_global_feat_vanilla, dim=0)  ## n*shot*c
    memorized_image_local_feat_vanilla = torch.cat(memorized_image_local_feat_vanilla, dim=0)
    memorized_labels = torch.cat(memorized_labels, dim=0)

    ######## memorized few shot features and labels.
    model.fewshot_image_global_feat = memorized_image_global_feat  ## class*augshot*c
    model.fewshot_image_local_feat = memorized_image_local_feat
    model.fewshot_image_global_feat_vanilla = memorized_image_global_feat_vanilla  ## class*shot*c
    model.fewshot_image_local_feat_vanilla = memorized_image_local_feat_vanilla
    model.fewshot_label = memorized_labels  ## class*shot*c, one hot labels

    ############# add features of labeled data to the dynamic memory. This is important when there are more labeled data.
    model.fixed_global_feat_vanilla = torch.cat((model.fixed_global_feat, memorized_image_global_feat_vanilla),
                                                dim=1)  ## N*1*C
    model.fixed_local_feat_vanilla = torch.cat((model.fixed_local_feat, memorized_image_local_feat_vanilla),
                                               dim=1)  ## N*1*C

    ###################### for static memory, with text feature and augmented image feat
    model.fixed_global_feat = torch.cat((model.fixed_global_feat, memorized_image_global_feat), dim=1)  ## N*1*C
    model.fixed_local_feat = torch.cat((model.fixed_local_feat, memorized_image_local_feat), dim=1)  ## N*1*C
    # ipdb.set_trace()
    print('appending the few shot image feature to fixed image memories.')


# for 杂交版 2.0 新增一个维护gradient 返回：是否加入 & 新元素占某类的第几个
def update_cache_p_new_uip(cache, pred, features_loss, shot_capacity, include_prob_map=False, dis=None):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    f = 0  # 是否有新东西加入
    new_item_index = 0  # Index of the new or updated item
    with torch.no_grad():
        # print("features_loss",type(features_loss))
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        # 每一类 进行一个存储
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
                f = 1
            elif features_loss[1] < cache[pred][-1][1] and dis is None:
                cache[pred][-1] = item
                f = 1
            elif features_loss[3] < cache[pred][-1][3] and features_loss[1] < cache[pred][-1][1] and dis:
                cache[pred][-1] = item
                f = 1
            # cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
            # print(cache)
            new_item_index = len(cache[pred]) - 1
        else:
            cache[pred] = [item]
            f = 1
    return f, new_item_index


# for 杂交版 2.0 新增一个维护gradient 返回：是否加入 & 新元素占某类的第几个
def update_cache_p(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    f = 0  # 是否有新东西加入
    new_item_index = 0  # Index of the new or updated item
    with torch.no_grad():
        # print("features_loss",type(features_loss))
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        # 每一类 进行一个存储
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
                f = 1
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
                f = 1
            # cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
            # print(cache)
            new_item_index = len(cache[pred]) - 1
        else:
            cache[pred] = [item]
            f = 1
    return f, new_item_index


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


# for 杂交版 EMA 动态维护clip_weights
def update_weights(model, idx, iter, alpha=0.99):
    with torch.no_grad():
        alpha_teacher = max(1 - 1 / (iter + 1), alpha)
        text_features = model.get_text_features_one(idx)  # 某一类
        model.text_feat[idx] = model.text_feat[idx].clone() * alpha_teacher + (1 - alpha_teacher) * text_features


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        # print("features_loss",type(features_loss))
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        # 每一类 进行一个存储
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)

            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item

            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]

def update_cache_ema(cache, pred, features_loss, alpha=0.2):
    """Update cache with new features and loss, maintaining the maximum shot capacity
    by using an Exponential Moving Average (EMA) of the features and loss."""
    with torch.no_grad():
        # Unpack features and loss from features_loss
        current_feature = features_loss[0]
        current_loss = features_loss[1]

        # Check if cache for the specific class (pred) already exists
        if pred in cache:
            # Retrieve the current cached feature and loss
            cached_feature, cached_loss = cache[pred][0]
            
            if current_loss < cached_loss: # 加一个 非无脑条件

                # Update the features and loss using EMA
                updated_feature = alpha * current_feature + (1 - alpha) * cached_feature
                updated_loss = alpha * current_loss + (1 - alpha) * cached_loss
    
                # Store the updated feature and loss back in the cache
                cache[pred][0] = [updated_feature, updated_loss]

        else:
            # Initialize cache with the current feature and loss if class is not yet present
            cache[pred] = [[current_feature, current_loss]]

def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []  # for feats
        cache_values = []  # for label
        # 函数遍历 cache 字典，该字典按类别索引排序，并存储每个类别的缓存项。
        for class_index in sorted(cache.keys()):
            # 对于每个缓存项：
            for item in cache[class_index]:
                cache_keys.append(item[0])  # 将图像特征（键）添加到 cache_keys 列表
                if neg_mask_thresholds:  # 根据是否提供了 neg_mask_thresholds，将相应的类别索引或负掩码阈值添加到 cache_values 列表。
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)
        # 将 cache_keys 列表中的图像特征堆叠成一个张量，并进行排列，以便进行矩阵乘法。
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        #print("first")
        #print("cache_keys", cache_keys.shape)
        # 如果提供了 neg_mask_thresholds，则将 cache_values 转换为一个张量，并根据阈值进行处理，然后将其转换为半精度浮点数并移动到GPU上。
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(
                torch.int8)).cuda().half()
        # 如果没有提供 neg_mask_thresholds，则使用 torch.one_hot 创建一个独热编码张量，表示类别索引，然后将其转换为半精度浮点数并移动到GPU上。
        else:
            #print("MEI YOU")
            cache_values = (
                F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(0))).cuda().half()
        #print("cache_values", cache_values.shape)
        # 使用矩阵乘法计算图像特征和缓存键之间的亲和性（affinity）。
        #ipdb.set_trace()
        affinity = image_features @ cache_keys
        affinity = affinity.to(cache_values.dtype)
        #print("affinity", affinity.shape)
        # print("affinity", affinity)
        # 使用亲和性和缓存值计算缓存logits。这里使用了指数函数和alpha缩放因子。
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # print("cache_logits", cache_logits)
        # ipdb.set_trace()
        return alpha * cache_logits


def average_distance_feats(image_features):
    """
    计算每个变换后的视觉特征与anchor之间的平均距离.

    Args:
    - image_features_global (torch.Tensor): 经过多次变换后输入视觉编码器得到的视觉特征,
                                            shape: (batch_size, feature_dim)

    Returns:
    - avg_distance (float): 所有变换特征与anchor特征之间距离的平均值.
    """
    # Step 1: 选择第一个特征作为anchor
    anchor_feature = image_features[0]

    # Step 2: 计算其他变换特征与anchor之间的距离
    distances = []
    for i in range(1, image_features.size(0)):  # 跳过anchor自己
        current_feature = image_features[i]
        distance = F.pairwise_distance(anchor_feature.unsqueeze(0), current_feature.unsqueeze(0))
        distances.append(distance)

    # Step 3: 计算平均距离
    avg_distance = torch.mean(torch.stack(distances)).item()

    return avg_distance


def calculate_anchor_and_distance(model, pseudo_label, selected_image_features_global):
    """
    根据伪标签计算该类的锚点并计算与选定图像特征的距离.
    Args:
    - model: 包含图像特征的模型，具有 image_feature_memory 属性.
    - pseudo_label (int): 伪标签标量，表示类别.
    - selected_image_features_global (torch.Tensor): 选定的图像特征, shape: (1, 512)
    Returns:
    - anchor (torch.Tensor): 该类的锚点 (512,)
    - distance (float): 选定图像特征与锚点之间的距离
    """
    # Step 1: 找到对应伪标签的特征
    class_features = model.image_feature_memory[pseudo_label]  # (n_shot, 512)
    # Step 2: 筛选非零特征并计算锚点
    non_zero_features = class_features[class_features.sum(dim=1) != 0]  # 过滤掉全零的特征
    if non_zero_features.size(0) > 0:
        anchor = torch.mean(non_zero_features, dim=0)  # (512,)
    else:
        raise ValueError("No non-zero features found for the given pseudo label.")
    # Step 3: 计算距离
    distance = F.pairwise_distance(anchor.unsqueeze(0), selected_image_features_global).item()  # L2 距离
    # ipdb.set_trace()
    return anchor, distance


def compute_weighted_global_feature(image_features_global, image_features_local, iters=None, sim_matrix=None):
    """
    计算全局特征和局部特征的相似度，并加权局部特征得到新的全局特征。
    参数：
    image_features_local (torch.Tensor): 局部 patch 特征，形状为 [batch_size, 196, 512]
    image_features_global (torch.Tensor): 全局特征，形状为 [batch_size, 512]
    返回：
    new_global_features (torch.Tensor): 加权后的新的全局特征，形状为 [batch_size, 512]
    sim_matrix (torch.Tensor): 每个全局特征与局部特征的相似度矩阵，形状为 [batch_size, 196]
    """
    # 计算全局特征和局部特征的相似度矩阵，使用余弦相似度
    # image_features_local: [batch_size, 196, 512], image_features_global: [batch_size, 512]
    # 对局部特征进行 L2 归一化
    image_features_local_norm = F.normalize(image_features_local, dim=-1)  # [batch_size, 196, 512]
    # 对全局特征进行 L2 归一化
    image_features_global_norm = F.normalize(image_features_global, dim=-1)  # [batch_size, 512]
    # 扩展 image_features_global_norm 的维度 [batch_size, 512] -> [batch_size, 1, 512]
    image_features_global_expanded = image_features_global_norm.unsqueeze(1)
    # 在第二个维度上拼接 [batch_size, 196, 512] + [batch_size, 1, 512] -> [batch_size, 197, 512]
    image_features_combined = torch.cat([image_features_global_expanded, image_features_local_norm], dim=1)
    if sim_matrix is None:
        # 计算余弦相似度: [batch_size, 197, 512] 和 [batch_size, 512] 的相似度
        sim_matrix = torch.einsum('bid,bd->bi', image_features_combined,
                                  image_features_global_norm)  # [batch_size, 197]
        # 对相似度进行 softmax 归一化，确保权重和为 1
        # ipdb.set_trace()
        # sim_matrix = F.softmax(sim_matrix, dim=-1)  # [batch_size, 197]
        # 假设 sim_matrix 是 [batch_size, 197] 形状的 tensor
        k = int(sim_matrix.shape[1] * 0.1)  # 每个向量保留 10% 的元素
        # 对每个向量排序，得到前 k 个最大值的位置
        topk_values, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
        # 构建一个与 sim_matrix 相同大小的全零 tensor
        mask = torch.zeros_like(sim_matrix)
        # 在 mask 中，将前 k 个最大值的位置标记为 1
        mask.scatter_(1, topk_indices, 1)
        # 使用 mask 将非最大值的位置置为 0
        sim_matrix = sim_matrix * mask
        # sum_sim_matrix = sim_matrix.sum(dim=1, keepdim=True)
        # sim_matrix = sim_matrix / sum_sim_matrix
        # 初始化参数
        T_0 = 0.1  # 初始温度
        k = 0.01  # 衰减率
        if iters is not None:
            # T = T_0 * np.exp(-k * iters)
            T = T_0 * np.exp(-k * iters)
        else:
            T = T_0
        sim_matrix = F.softmax(sim_matrix / T, dim=-1)  # [batch_size, 197]
    # 根据相似度对局部特征加权平均
    # sim_matrix: [batch_size, 197], image_features_local: [batch_size, 196, 512]
    new_global_features = torch.einsum('bi,bid->bd', sim_matrix, image_features_combined)  # [batch_size, 512]

    new_global_features = F.normalize(new_global_features, dim=-1)  # [batch_size, 512]
    return new_global_features, sim_matrix


def linear_decay(i, max_iters):
    """
    i: 当前的训练轮数
    max_iters: 最大训练轮数，决定alp何时减小到0
    """
    return max(0, 1 - i / max_iters)

def cosine_similarity(a, b):
    # ipdb.set_trace()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)


def batch_cosine_similarity(a, b, batch_size=1000):
    # 计算范数
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)

    # 初始化结果矩阵
    n, m = a.size(0), b.size(0)
    similarity_matrix = torch.zeros((n, m), dtype=torch.float32)

    # 分块计算
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        for j in range(0, m, batch_size):
            end_j = min(j + batch_size, m)
            # 计算分块的相似度
            similarity_matrix[i:end_i, j:end_j] = torch.matmul(a_norm[i:end_i], b_norm[j:end_j].T)

    return similarity_matrix


def compute_text_anchor_logits(image_features, anchor_feat, target_feat, max_cliques, R=0.2, softmax=False):
    beta = 5.5
    alpha = 1.0
    affinity = image_features @ anchor_feat.T # 1 * hyper_class_num
    # 找出 affinity 中最大值的索引
    # max_idx = torch.argmax(affinity, dim=-1)
    # Step 1: 获取当前最相关团的类别索引
    # relevant_indices = max_cliques[max_idx]  # 例如: [1, 2, 3]
    top_R_count = max(1, int(len(max_cliques) * R))  # 确保至少取一个团
    topk_values, topk_indices = torch.topk(affinity, k=top_R_count , dim=-1)  # 找到 Top-K 的值和索引
    max_idx = torch.unique(topk_indices, sorted=True)  # 去除重复的索引，并保持顺序 这是团索引
    relevant_indices = []  # 初始化一个空列表，存放所有的相关类别索引
    for idx in max_idx:
        relevant_indices.extend(max_cliques[idx.item()])  # 将每个团的索引添加到列表中
    relevant_indices = torch.unique(torch.tensor(relevant_indices)).tolist()  # 转为张量再去重，最后转为列表
    # Step 2: 从 target_feat 中提取相应类别特征
    # target_feat 的形状为 (1, n_cls)，所以我们可以直接用索引取值
    relevant_target_features = target_feat[relevant_indices, :]  # (1, len(relevant_indices))
    # Step 3: 计算 img_feats 与这些类别特征的相似度（点积或余弦相似度）
    # 这里假设使用点积相似度
    logits = torch.matmul(image_features, relevant_target_features.T)  # (1, len(relevant_indices))
    # if softmax:
    #     logits = logits.softmax(-1)
    # 创建一个全零的 logits 张量，形状为 (1, n_cls)
    n_cls = target_feat.shape[0]  # 获取类别数量
    full_logits = torch.zeros((1, n_cls), device=image_features.device)  # 确保设备一致性
    # 将对应的值放入 full_logits 中
    full_logits[0, relevant_indices] = logits.squeeze()  # 将相关的 logits 填入对应位置


    return full_logits, relevant_indices, torch.max(affinity)

def make_max_cliques(text_feat, threshold, reduce=False):
    # print("Start to make Graph and Cliques ......")
    # 计算相似度矩阵
    if reduce:
        similarity_matrix = batch_cosine_similarity(text_feat, text_feat, batch_size=1000)
    else:
        similarity_matrix = cosine_similarity(text_feat, text_feat)  # 形状为 [n_cls, n_cls]

    # if reduce:
    #     # 仅存储上三角部分（包括对角线）
    #     n = similarity_matrix.size(0)
    #     triu_indices = torch.triu_indices(n, n)
    #     upper_tri_values = similarity_matrix[triu_indices[0], triu_indices[1]]


    # 构造邻接矩阵 (一阶图)
    adjacency_matrix = (similarity_matrix > threshold).float()  # 形状为 [n_cls, n_cls]
    adjacency_matrix_0 = adjacency_matrix
    # 是否二阶图
    adjacency_matrix1 = torch.matmul(adjacency_matrix, adjacency_matrix)
    adjacency_matrix2 = adjacency_matrix * adjacency_matrix1
    adjacency_matrix = (adjacency_matrix2 > 1).float()
    # 构造图
    G = nx.from_numpy_matrix(adjacency_matrix.cpu().numpy())
    # ipdb.set_trace()
    # 非唯一版
    # # 找到所有的极大团
    cliques = list(nx.find_cliques(G))
    # 筛选出极大团（每个 node 都需属于一个极大团）
    max_cliques = []
    for clique in cliques:
        if len(clique) == len(set(clique)):  # 确保节点唯一
            max_cliques.append(clique)
    # 检查每个节点是否在任何一个极大团中
    all_nodes = set(range(text_feat.size(0)))
    nodes_in_cliques = set(node for clique in max_cliques for node in clique)
    # 找出不在任何极大团中的节点
    missing_nodes = all_nodes - nodes_in_cliques
    # ipdb.set_trace()
    # 处理缺失的节点（可以选择加入一个已有团，或创建新的团）
    for node in missing_nodes:
        # 将其添加到某个现有的团中或创建新的团
        # 这里简单地将其添加到第一个极大团中
        if max_cliques:
            max_cliques[0].append(node)
        else:
            max_cliques.append([node])  # max_cliques 包含所有找到的极大团
    # print("Finish Cliques ! the num of Cliques is {}".format(len(max_cliques)))
    # 开始造 text anchor
    text_feat_anchor = []
    for clique in max_cliques:
        text_anchor = text_feat[clique]
        text_anchor = torch.mean(text_anchor, dim=0)
        text_feat_anchor.append(text_anchor.unsqueeze(0))
    text_feat_anchor = torch.cat(text_feat_anchor, dim=0)  # cliques_num * feats_num


    if reduce:
        # 删除不再需要的变量
        del similarity_matrix, adjacency_matrix_0, adjacency_matrix1, adjacency_matrix2, G, cliques, all_nodes, nodes_in_cliques, missing_nodes
        gc.collect()
        torch.cuda.empty_cache()

    return max_cliques, text_feat_anchor


def mask_logits(dinov2_logits, relevant_indices):
    logits = dinov2_logits
    # 初始化一个与 dinov2_logits 相同形状的掩码值张量，掩码值设为 0.0
    mask = torch.full_like(logits, 0.0)
    # 将 relevant_indices 对应的值保留
    mask[0, relevant_indices] = logits[0, relevant_indices]
    dinov2_logits = mask
    return dinov2_logits


def make_prototypeAttn_features(dinov2_cache, img_feat, beta, n_cls, n_shot, n_dim):
    # 初始化 memorized_image_feat
    memorized_image_feat = torch.zeros((n_cls, n_shot, n_dim)).to(img_feat.device)
    for label, features_list in dinov2_cache.items():
        if 0 <= label < n_cls:
            for i, (feature, _) in enumerate(features_list):
                if i < n_shot:
                    memorized_image_feat[label, i] = feature.squeeze()
                else:
                    break

    img_feat_mappling = img_feat  # 1*1024
    memorized_image_feat_K = memorized_image_feat.clone()  # 200*11*1024
    memorized_image_feat_V = memorized_image_feat.clone()  # 200*11*1024

    with torch.no_grad():

        memorized_image_feat_K = memorized_image_feat_K / memorized_image_feat_K.norm(dim=-1, keepdim=True)
        memorized_image_feat_K[memorized_image_feat.sum(-1) == 0] = 0
        memorized_image_feat_V = memorized_image_feat_V / memorized_image_feat_V.norm(dim=-1, keepdim=True)
        memorized_image_feat_V[memorized_image_feat.sum(-1) == 0] = 0
        img_feat_mappling = img_feat_mappling / img_feat_mappling.norm(dim=-1, keepdim=True)

    similarity_matrix = (img_feat_mappling * memorized_image_feat_K).sum(-1)  # 200*11
    similarity_matrix = torch.exp(-beta * (-similarity_matrix + 1))

    adaptive_image_feat = (memorized_image_feat_V * similarity_matrix.unsqueeze(-1)).sum(1)  # n_cls * e_dim

    return adaptive_image_feat


def make_prototype_features(dinov2_cache, n_class, n_dim):
    # Step 1: Initialize a tensor of zeros with shape (n_class, 1024)
    prototype_features = torch.zeros((n_class, n_dim))
    # print("dinov2_cache",dinov2_cache)

    # Step 2: Iterate through the dictionary
    for class_id, features in dinov2_cache.items():
        if features:  # Ensure there are features to process
            # Extract the 1x1024 tensors and stack them into a single tensor
            tensors = torch.stack([feature[0] for feature in features])
            # Compute the mean of these tensors along the 0th dimension
            mean_tensor = tensors.mean(dim=0)
            # Step 3: Assign the mean tensor to the corresponding class in prototype_features
            prototype_features[class_id] = mean_tensor

    return prototype_features

def find_idx_in_sublists(max_cliques5, target_idx):
    result = []
    for sublist in max_cliques5:
        if target_idx in sublist:
            result.append(sublist)
    return result

def reweight_logits(sublists_with_idx, dinov2_logits, n_class, repeat=True):
    # Step 1: 记录类别索引及其出现次数
    class_counts = torch.zeros(n_class, dtype=torch.int).to(dinov2_logits.device)

    for sublist in sublists_with_idx:
        for idx in sublist:
            if idx > 0:  # 忽略0
                if repeat:
                    class_counts[idx - 1] += 1  # 类别索引从1开始，所以减1
                else:
                    class_counts[idx - 1] = 1

    # Step 2: 重新加权 dinov2_logits
    reweighted_logits = dinov2_logits * class_counts.float()

    return reweighted_logits

# 对both空间的idx进行转化
def process_cliques(cliques_with_idx4, n_class):
    processed_cliques = []
    for sublist in cliques_with_idx4:
        # 对每个子列表进行处理
        processed_sublist = [(idx % n_class if idx != 0 else 0) for idx in sublist]
        processed_cliques.append(processed_sublist)
    return processed_cliques



def linear_growth(t0, k, i):
    return min(1, t0 + k * i)

def exponential_growth(t0, k, i):
    return min(1, t0 * (1 - math.exp(-k * i)))

def sigmoid_growth(k, i0, i):
    return 1 / (1 + math.exp(-k * (i - i0)))

def logarithmic_growth(t0, k, i):
    return min(1, t0 + k * math.log(i + 1))

def sqrt_growth(t0, k, i):
    return min(1, t0 + k * math.sqrt(i))

def compute_pixel_attention_from_patches(attn_score_patches):
    batch_size, num_heads, num_patches = attn_score_patches.shape
    num_patches_per_dim = int(num_patches ** 0.5)  # 16, since 16x16 patches = 256
    # Step 1: 平均每个 head 的注意力分数，得到 (batch_size, 256) 的维度
    avg_attention_scores = attn_score_patches.mean(dim=1)  # shape: (batch_size, 256)
    # Reshape avg_attention_scores to (batch_size, 1, 16, 16) to represent 16x16 patches
    attention_scores_patches_reshaped = avg_attention_scores.view(batch_size, 1, num_patches_per_dim, num_patches_per_dim)
    # Step 2: 使用双线性插值将 patch 分数扩展到 (batch_size, 1, 224, 224)
    attention_scores_pixel = F.interpolate(attention_scores_patches_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
    # Remove the channel dimension (batch_size, 224, 224)
    attention_scores_pixel = attention_scores_pixel.squeeze(1)
    return attention_scores_pixel

def visualize_image(original_image, attn_map1, attn_map2, path):

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 把original_image从 [3, 224, 224] 转换为 [224, 224, 3] 并转换为 numpy 格式
    img = original_image.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]，方便显示


    attn1 = attn_map1.squeeze().cpu().numpy()
    attn2 = attn_map2.squeeze().cpu().numpy()
    # 创建一个图形，包含两个子图
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    # 左边显示原始图像
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    ax[1].imshow(attn1, cmap='plasma')
    ax[1].axis('off')
    ax[1].set_title('CLIP Attention Map')

    ax[2].imshow(attn2, cmap='plasma')
    ax[2].axis('off')
    ax[2].set_title('DINOv2 Attention Map')

    plt.savefig(path, dpi=300, bbox_inches='tight')  # bbox_inches='tight' 可以去除多余的边距
    plt.close(fig)  # 关闭图形，释放内存

def visualize_original_image(original_image, path):
    # Check if the directory exists, create it if not
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Convert original_image from [3, 224, 224] to [224, 224, 3] and normalize it
    img = original_image.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1] for display

    # Create a figure with one subplot for the original image
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis('off')  # Turn off axis
    # ax.set_title('Original Image')

    # Save the figure
    plt.savefig(path, dpi=300, bbox_inches='tight')  # bbox_inches='tight' removes unnecessary margin
    plt.close(fig)  # Close the figure to free up memory


def tsne_cache(model, dinov2_cache, save_path, i):

    # 提取 dinov2_cache 中的特征，记录和标签对应的关系
    dinov2_features = []
    dinov2_labels = []
    for label, features_list in dinov2_cache.items():
        for feature, loss in features_list:
            dinov2_features.append(feature.cpu().numpy().squeeze())
            dinov2_labels.append(label)

    dinov2_features = np.array(dinov2_features)

    # 提取 model.image_feature_memory 中的特征并生成标签
    image_feature_memory = model.image_feature_memory

    # 暂时展平特征并创建相应的标签
    clip_features = image_feature_memory.view(-1, image_feature_memory.shape[-1])
    clip_labels = np.repeat(np.arange(image_feature_memory.size(0)), image_feature_memory.size(1))

    # 去除全零向量和对应标签
    valid_indices = ~torch.all(clip_features == 0, dim=1)
    clip_features = clip_features[valid_indices].cpu().numpy()  # 将张量转换为 NumPy 数组
    clip_labels = clip_labels[valid_indices.cpu().numpy()]  # 先将 valid_indices 移至 CPU，再转换为 NumPy

    # t-SNE 可视化功能
    def plot_tsne(ax, features, labels, title):
        num_samples = features.shape[0]
        perplexity = min(30, max(5, num_samples - 1))

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_2d = tsne.fit_transform(features)

        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.colorbar(scatter, ax=ax)

    # 创建一个 1x2 的图形布局
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 对 clip_features 特征进行 t-SNE 可视化
    plot_tsne(axes[0], clip_features, clip_labels, 't-SNE of CLIP Cached Features')

    # 对 dinov2_cache 特征进行 t-SNE 可视化
    plot_tsne(axes[1], dinov2_features, dinov2_labels, 't-SNE of DINOv2 Cached Features')

    # 保存图像
    plt.tight_layout()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"{save_path}cache_{i}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def tsne_class(model_text_feat, text_feat_anchor4, text_feat_anchor5, image_features_global, image_features_global_aux,
               max_cliques4, max_cliques5, target, save_path, i):
    def plot_tsne_subplot(ax, features, query_feature, target_feature=None, clique_features=None, title='',
                          plot_cliques=True, plot_target=True):
        # 计算样本总数
        n_samples = len(features) + 1  # +1 for query_feature
        if plot_target and target_feature is not None:
            n_samples += 1
        if plot_cliques and clique_features is not None:
            n_samples += len(clique_features)

        # 根据样本数量动态调整 perplexity
        perplexity = min(30, n_samples - 1)  # perplexity 必须小于 n_samples

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)

        # 组合特征
        combined_features = [features, query_feature]
        if plot_target and target_feature is not None:
            combined_features.append(target_feature)
        if plot_cliques and clique_features is not None:
            combined_features.append(clique_features)

        combined_features = np.vstack(combined_features)

        features_2d = tsne.fit_transform(combined_features)

        # 分离出各种特征的降维结果
        orig_features_2d = features_2d[:len(features)]
        query_feature_2d = features_2d[len(features)]

        ax.scatter(query_feature_2d[0], query_feature_2d[1], color='red', marker='*', s=200, label='Query')
        # 绘制原始特征
        if plot_target and target_feature is not None:
            ax.scatter(orig_features_2d[:, 0], orig_features_2d[:, 1], alpha=0.6, c='blue', s=30, label='Class Centers')
        else:
            ax.scatter(orig_features_2d[:, 0], orig_features_2d[:, 1], alpha=0.6, c='blue', s=30, label='Hyper-class Centers')


        # 如果需要，绘制目标特征
        if plot_target and target_feature is not None:
            target_feature_2d = features_2d[len(features) + 1]
            ax.scatter(target_feature_2d[0], target_feature_2d[1], color='green', marker='o', s=100, label='Target Centers')

        # 如果需要，绘制包含目标的团
        if plot_cliques and clique_features is not None:
            clique_features_2d = features_2d[-len(clique_features):]
            ax.scatter(clique_features_2d[:, 0], clique_features_2d[:, 1], color='orange', marker='s', s=50,
                       label='Hyper-class with Target')

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend()

    # 准备查询特征
    query_512 = image_features_global[:1].cpu().numpy()
    query_1024 = image_features_global_aux[:1].cpu().numpy()

    # 获取目标类别的特征
    target_idx = target.item()
    target_feature_512 = model_text_feat[target_idx].cpu().numpy().reshape(1, -1)

    # 找到包含目标的团
    n_cls = model_text_feat.shape[0]
    cliques_with_target4 = [clique for clique in max_cliques4 if target_idx in clique or (target_idx + n_cls) in clique]
    cliques_with_target5 = [clique for clique in max_cliques5 if target_idx in clique]

    # 获取包含目标的团的特征
    clique_features4 = text_feat_anchor4[[max_cliques4.index(clique) for clique in cliques_with_target4]].cpu().numpy()
    clique_features5 = text_feat_anchor5[[max_cliques5.index(clique) for clique in cliques_with_target5]].cpu().numpy()


    # 创建一个1x3的图形布局
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 对 model.text_feat 展示 class features, query 和 target
    plot_tsne_subplot(axes[0], model_text_feat.cpu().numpy(), query_512, target_feature_512, None, 't-SNE of Class',
                      plot_cliques=False)

    # 对 text_feat_anchor4 和 text_feat_anchor5 只展示 class features, query 和 cliques with target
    plot_tsne_subplot(axes[1], text_feat_anchor4.cpu().numpy(), query_512, clique_features=clique_features4,
                      title='t-SNE of CSS Hyper-class', plot_target=False)
    plot_tsne_subplot(axes[2], text_feat_anchor5.cpu().numpy(), query_1024, clique_features=clique_features5,
                      title='t-SNE of AFV Hyper-class', plot_target=False)

    # 确保保存的路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{save_path}img_{i}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图形以释放内存


def compute_class_centroids(image_feature_memory):
    n_cls, shot, n_dim = image_feature_memory.shape

    # 初始化质心张量
    centroids = torch.zeros(n_cls, n_dim, device=image_feature_memory.device)

    for cls in range(n_cls):
        # 获取当前类别的所有特征
        class_features = image_feature_memory[cls]

        # 计算非零特征的掩码
        non_zero_mask = torch.any(class_features != 0, dim=1)

        if torch.any(non_zero_mask):
            # 如果存在非零特征，计算它们的平均值
            valid_features = class_features[non_zero_mask]
            centroids[cls] = valid_features.mean(dim=0)
        # 如果全是零特征，centroids[cls] 保持为零向量

    return centroids


def direct_inference(cfg, val_loader, model, model_state, optimizer, optim_state, scaler, args, set_id, Train=False):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top1_global = AverageMeter('AccImg@1', ':6.2f', Summary.AVERAGE)
    top1_global_fewshot = AverageMeter('AccGF@1', ':6.2f', Summary.AVERAGE)
    top1_text_vote = AverageMeter('AccVote1@1', ':6.2f', Summary.AVERAGE)
    top1_global_fewshot_vote = AverageMeter('AccVoteG@1', ':6.2f', Summary.AVERAGE)
    # top1_neg = AverageMeter('AccNegG@1', ':6.2f', Summary.AVERAGE)
    top1_dinov2_mem = AverageMeter('AccDinov2Mem@1', ':6.2f', Summary.AVERAGE)
    top1_mac = AverageMeter('AccMAC@1', ':6.2f', Summary.AVERAGE)
    top1_mac2 = AverageMeter('AccMAC2@1', ':6.2f', Summary.AVERAGE)

    top1_ab1 = AverageMeter('AccAB1@1', ':6.2f', Summary.AVERAGE)
    top1_ab2 = AverageMeter('AccAB2@1', ':6.2f', Summary.AVERAGE)
    top1_ab3 = AverageMeter('AccAB3@1', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top1_global, top1_dinov2_mem, top1_mac, top1_mac2, top1_ab1, top1_ab2, top1_ab3, top1_global_fewshot, top1_text_vote, top1_global_fewshot_vote],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()

    if model.first_flag:
        with torch.no_grad():
            text_feat, text_feat_full = model.get_text_features()
    else:
        print('the text feat has already initilized, pass it here.')
    ## text_feat: 200*1024
    ## text_feat_full:  200 * 7 * 1024
    class_num, feat_dim = model.text_feat.shape[0], model.text_feat.shape[1]
    if args.DINO:
        feat_dim_aux = model.dino.embed_dim
    else:
        feat_dim_aux = None
    pred_vanilla = []
    pred_global = []
    pred_global2 = []
    pred_global3 = []
    pred_local = []
    pred_fewshot_global = []
    pred_fewshot_local = []
    labels = []
    pred_neg = []
    pred_dinov2 = []
    pred_mac = []
    pred_mac2 = []

    pred_ablation_1 = []
    pred_ablation_2 = []
    pred_ablation_3 = []
    dmnet = DualMem(args=args, beta=args.beta, feat_dim=feat_dim, feat_dim_aux=feat_dim_aux, class_num=class_num,
                    mapping=args.mapping).cuda()
    ################################ fine tune clip adapter with few labeled training data.
    if args.n_shot and args.ft:
        epoch = args.epoch
        training_size = model.text_feat.shape[0] * args.n_shot
        #### construct the data loader,
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        data_transform_aug = StrongAugmenter(base_transform, preprocess,
                                             augmix=len(args.set_id) > 1)  ### aug mix not used for ImageNet test set.
        train_dataset_mem = build_dataset(args.set_id, data_transform_aug, args.data, mode='train', n_shot=args.n_shot)
        print("number of training samples: {}, number of augview: {}".format(len(train_dataset_mem), args.n_augview))
        train_loader_ft = torch.utils.data.DataLoader(
            train_dataset_mem,
            batch_size=128 if training_size > 128 else training_size, shuffle=True,  ## the input has been shuffled.
            num_workers=2, pin_memory=True)
        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(dmnet.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.wd)  #
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(dmnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)  #
        else:
            raise NotImplementedError
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch * len(train_loader_ft), eta_min=1e-7)
        Loss = SmoothCrossEntropy()
        timestamp = time.time()
        time_parts = time.gmtime(timestamp)
        hours = time.strftime("%H", time_parts)
        minutes = time.strftime("%M", time_parts)
        seconds = time.strftime("%S", time_parts)
        print("train start Time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))
        for train_idx in range(epoch):  ## for each epoch.
            dmnet.train()
            correct_samples, all_samples = 0, 0
            correct_samples_global, correct_samples_local = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, epoch))
            for i, (images, target) in enumerate(train_loader_ft):
                # print(dmnet.lora_b_FFN[0]) ## checked, learned parameters are udpated.
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features_global, image_features_local = model.get_image_features(images)  ##B*D, B*L*D
                fewshot_global_logit = dmnet.get_image_pred_fewshot_global(model, return_full=True,
                                                                           return_logit=True)  ## N*class, probability
                # fewshot_local_logit= dmnet.get_image_pred_fewshot_local(model, return_full=True, return_logit=True)  ### to do, get the prediction with local features.
                loss = Loss(fewshot_global_logit, target)
                # loss += Loss(fewshot_local_logit, target)
                if args.position == 'output' or args.position == 'all':
                    text_logit = dmnet.get_text_prediction(model, return_full=True, return_logit=True)
                    dmnet.init_pred = text_logit  ## use it for local few shot.
                    loss += Loss(text_logit, target)
                else:
                    with torch.no_grad():
                        text_logit = dmnet.get_text_prediction(model, return_full=True, return_logit=True)
                        dmnet.init_pred = text_logit  ## use it for local few shot.

                acc = cls_acc(text_logit, target)
                correct_samples += acc / 100 * len(text_logit)
                all_samples += len(text_logit)
                acc_global = cls_acc(fewshot_global_logit, target)
                correct_samples_global += acc_global / 100 * len(fewshot_global_logit)
                # acc_local = cls_acc(fewshot_local_logit, target)
                # correct_samples_local += acc_local / 100 * len(fewshot_local_logit)

                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, text:{:}, global: {:}, local: {:} All:{:}), Loss: {:.4f}'.format(current_lr,
                                                                                                correct_samples,
                                                                                                correct_samples_global,
                                                                                                correct_samples_local,
                                                                                                all_samples,
                                                                                                sum(loss_list) / len(
                                                                                                    loss_list)))

    dmnet.eval()
    end = time.time()
    timestamp = time.time()
    time_parts = time.gmtime(timestamp)
    hours = time.strftime("%H", time_parts)
    minutes = time.strftime("%M", time_parts)
    seconds = time.strftime("%S", time_parts)
    print("test start Time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))

    # if args.use_MAC:
    #
    #     # 开始造团
    #     max_cliques, text_feat_anchor, _ ,_ = make_max_cliques(model.text_feat, cfg["Mac"]["lambda"])
    #     # ipdb.set_trace()
    #     # 唯一版 平均版
    #     # # 1. 找到所有极大团
    #     # cliques = list(nx.find_cliques(G))
    #     # # 2. 按团大小升序排列，以尽量分散节点
    #     # cliques = sorted(cliques, key=lambda x: len(x))
    #     # # 3. 创建一个集合来跟踪已经分配的节点
    #     # assigned_nodes = set()
    #     # # 4. 创建一个新的列表，用来存储最终选择的极大团
    #     # max_cliques = []
    #     # # 5. 遍历排序后的极大团列表
    #     # for clique in cliques:
    #     #     # 当前团中的未被分配的节点
    #     #     unassigned_clique = [node for node in clique if node not in assigned_nodes]
    #     #     # 如果有未分配的节点，将该团加入最终列表
    #     #     if unassigned_clique:
    #     #         max_cliques.append(unassigned_clique)
    #     #         # 更新已分配节点的集合
    #     #         assigned_nodes.update(unassigned_clique)
    #     # # 如果需要，进一步均衡团的大小，可以尝试对 balanced_cliques 进行微调
    #
    #     # 开始计算 anchor - target relation
    #     # text_feat_anchor = text_feat
    #     anchor_target_relation = cosine_similarity(text_feat_anchor, model.text_feat) # cliques_num * n_class
    #
    #     # ipdb.set_trace()
    #     # 计算每个团的高斯分布参数（均值和协方差矩阵）
    #     # gaussians = calculate_gaussian_parameters(text_feat, max_cliques)


    # 开始制造pro_cache
    pro_cache = {}
    gra_cache = {}
    neg_cache = {}
    dinov2_cache = {}
    # max_i = len(val_loader)
    max_iters = len(val_loader)
    # 重点关注
    for i, (images, target) in enumerate(val_loader):
        #ipdb.set_trace()
        # print(i)
        # if i > 20:
        #     break
        assert args.gpu is not None
        if isinstance(images, list):  ### augmix return, list
            images = torch.cat(images, dim=0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images[:1]
        else:  ## standard return, Tensor
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images[:1]
        target = target.cuda(args.gpu, non_blocking=True)  # torch.Size([1])

        with torch.no_grad():
            # CLIP img_feats
            #if not Train:
            image_features_global, image_features_local = model.get_image_features(images)

        # image_features_global: torch.Size([128, 1024])
        # image_features_local: torch.Size([128, 49, 1024])
        # images torch.Size([32, 3, 224, 224])
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                img_text = dmnet.get_text_prediction(model)
                img_text_pred = img_text[:1]  ## current prediction. torch.Size([1, n_cls])

        confidence_prediction, selected_idx, confused_weak_output, confused_idx = select_confident_samples(img_text, args.selection_p)

        dmnet.init_pred = confidence_prediction.mean(0, keepdim=True)  # torch.Size([1, n_cls])
        # dmnet.loss_total_now = loss_total
        acc1, _ = accuracy(dmnet.init_pred, target, topk=(1, 5))
        top1_text_vote.update(acc1[0], image.size(0))
        # ipdb.set_trace()  # 暂停执行
        if args.n_shot:
            with torch.no_grad():
                with torch.no_grad():
                    fewshot_global_pred_fullview = dmnet.get_image_pred_fewshot_global(model)  ## N*class, probability
                fewshot_global_pred = fewshot_global_pred_fullview[:1]  ## 1*class
                confidence_prediction_fewshot_global, _, _, _ = select_confident_samples(fewshot_global_pred_fullview,1.0)
                acc1, _ = accuracy(confidence_prediction_fewshot_global.mean(0, keepdim=True), target, topk=(1, 5))
                top1_global_fewshot_vote.update(acc1[0], image.size(0))
        stand = True
        # ipdb.set_trace()
        if args.DINOv2 and args.DINOv2_mem:
            dmnet.update_memory_bank(model, target, stand2=stand)

        if args.DINO and args.DINO4mem:
            dmnet.update_memory_bank(model, target, True)
        elif not (args.DINOv2 and args.DINOv2_mem):
            dmnet.update_memory_bank(model, target)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.DINO and args.DINO4mem:
                    img_global_pred = dmnet.get_image_pred(model, 'dino')  ## with updated local
                else:
                    img_global_pred = dmnet.get_image_pred(model)  ## with updated local

        final_logits = img_text_pred + img_global_pred * 1.0 # 都素 softmaxed

        loss1 = avg_entropy(img_text)

        if args.DINOv2 and args.DINOv2_mem:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_features_global_aux, image_features_local_aux = model.get_image_features_aux(images)

        loss = loss1
        _, pred_indices = final_logits.topk(1, 1, True, True)
        pred = int(pred_indices.squeeze().item())
        # img_feats = model.image_features_global[selected_idx].mean(0).unsqueeze(0)

        if args.DINOv2 and args.DINOv2_mem:
            #prop_entropy = get_entropy(loss, model.text_feat)
            # with torch.no_grad():
            #     with torch.cuda.amp.autocast():
            #         image_features_global_aux, image_features_local_aux = model.get_image_features_aux(images)
            img_feats_dinov2 = image_features_global_aux[:1]

            if args.center_type == 'ema':
                update_cache_ema(dinov2_cache, pred, [img_feats_dinov2, loss], 0.1)
            else:
                update_cache(dinov2_cache, pred, [img_feats_dinov2, loss], args.DINO_shot)




            if dinov2_cache:
                dinov2_logits = compute_cache_logits(img_feats_dinov2, dinov2_cache, cfg["positive"]["alpha"], cfg["positive"]["beta"], model.text_feat)
            else:
                dinov2_logits = torch.zeros_like(final_logits)

            pred_dinov2.append(dinov2_logits)
            #dinov2_logits = img_global_pred

        if args.Logits_wei_ablation:
            use_soft = True
        else:
            use_soft = False

        if args.inrease_t:
            # 指定一下阈值
            t0 = cfg["Mac"]["lambda1"]  # 初始阈值
            k = 0.00001  # 增长速率
            Th_clip = linear_growth(t0, k, i)

            t0_2 = cfg["Mac"]["lambda2"]  # 初始阈值
            k_2 = 0.00001  # 增长速率
            Th_dino = linear_growth(t0_2, k_2, i)
        else:
            Th_clip = cfg["Mac"]["lambda1"]  # 初始阈值

            Th_dino = cfg["Mac"]["lambda2"]  # 初始阈值


        if args.use_MAC_logits:

            # css_class_centers = compute_class_centroids(model.image_feature_memory)
            # clip_both_feats_space = torch.cat((model.text_feat, css_class_centers), dim=0)

            clip_both_feats_space = torch.cat((model.text_feat, dmnet.adaptive_image_feat.squeeze(0)), dim=0)


            max_cliques4, text_feat_anchor4 = make_max_cliques(clip_both_feats_space, Th_clip, True)

            MAC_logits4, _, max_aff4 = compute_text_anchor_logits(image_features_global[:1], text_feat_anchor4, clip_both_feats_space, max_cliques4, args.R, use_soft)
            first_half = MAC_logits4[:, :MAC_logits4.shape[1] // 2]
            second_half = MAC_logits4[:, MAC_logits4.shape[1] // 2:]
            MAC_logits4 = (first_half + second_half) / 2

            pred_mac.append(MAC_logits4)

        if args.use_MAC_logits2:
            # 这儿是牛干版
            if args.center_type == 'default' or args.center_type == 'ema':
                dinov2_prototypes = make_prototype_features(dinov2_cache, model.text_feat.shape[0], image_features_global_aux[:1].shape[1])  # n_cls * 1024
            elif args.center_type == 'attn':
                dinov2_prototypes = make_prototypeAttn_features(dinov2_cache, image_features_global_aux[:1], args.beta,
                                                                model.text_feat.shape[0], args.DINO_shot, image_features_global_aux[:1].shape[1]) # n_cls * 1024


            dinov2_prototypes = dinov2_prototypes.to(image_features_global_aux.device)
            dionv2_feats_space = dinov2_prototypes
            max_cliques5, text_feat_anchor5 = make_max_cliques(dionv2_feats_space, Th_dino, True)
            # cliques_with_idx5 = find_idx_in_sublists(max_cliques5, 0)
            # MAC_logits5 = reweight_logits(cliques_with_idx5, dinov2_logits, model.text_feat.shape[0], repeat=False)
            MAC_logits5, _, max_aff5 = compute_text_anchor_logits(image_features_global_aux[:1], text_feat_anchor5, dionv2_feats_space, max_cliques5, args.R, use_soft)

            pred_mac2.append(MAC_logits5)
        # if i % 1000 == 0:
        # ipdb.set_trace()

        if args.Logits_wei_ablation:
            # 纯相加
            ablation_1_pred = img_text_pred + MAC_logits4 + MAC_logits5
            # sim 1
            w1 = torch.mm(img_text_pred, MAC_logits4.T)
            w2 = torch.mm(img_text_pred, MAC_logits5.T)
            w = torch.cat([w1, w2], dim=1)
            softmax_w = F.softmax(w, dim=1) # torch.Size([1, 2])
            ablation_2_pred = img_text_pred + MAC_logits4 * softmax_w[0][0] + MAC_logits5 * softmax_w[0][1]
            # sim 2

            w = torch.cat([max_aff4.view(1, 1), max_aff5.view(1, 1)], dim=1)
            softmax_w = F.softmax(w, dim=1) # torch.Size([1, 2])
            ablation_3_pred = img_text_pred + MAC_logits4 * softmax_w[0][0] + MAC_logits5 * softmax_w[0][1]

            pred_ablation_1.append(ablation_1_pred)
            pred_ablation_2.append(ablation_2_pred)
            pred_ablation_3.append(ablation_3_pred)

        if args.get_tsne:

            folder = 'TSNE'
            save_path = f"./Pics/{folder}/{set_id}_{args.DINO_shot}/"
            # 假设 dinov2_cache 和 model.image_feature_memory 已存在
            # dinov2_cache 是一个包含特征列表的字典，比如 {0: [[feature_tensor1, loss1], ...]}
            # model.image_feature_memory 是一个张量，形如 torch.Size([n_cls, shot, 512])
            if (i % args.print_freq == 0 and i > 0) or (i == max_iters - 1):
                tsne_cache(model, dinov2_cache, save_path, i)
            # tsne_class(model.text_feat, text_feat_anchor4, text_feat_anchor5, save_path, i)
            tsne_class(model.text_feat, text_feat_anchor4, text_feat_anchor5,
                       image_features_global[:1], image_features_global_aux[:1], max_cliques4, max_cliques5, target, save_path, i)

        if args.get_samples:
            folder = 'original'
            k = 5

            CSS_pred = (MAC_logits4 / 0.3).softmax(-1)
            AFV_pred = (MAC_logits5 / 0.1).softmax(-1)
            # CSS_pred = MAC_logits4 * 10
            # AFV_pred = MAC_logits5 * 30
            final_pred = img_text_pred + img_global_pred + CSS_pred + AFV_pred
            final_pred = (final_pred / 2.0).softmax(-1)
            # 获取 top k 的 logits 值和对应的标签索引

            topk_logits, topk_indices = final_pred.topk(k, 1, True, True)
            topk_logits = [round(val, 4) for val in topk_logits.squeeze().tolist()]  # 保留四位小数
            topk_indices = topk_indices.squeeze().tolist()  # top 5 标签索引

            topk_logits_s, topk_indices_s = CSS_pred.topk(k, 1, True, True)
            topk_logits_s = [round(val, 4) for val in topk_logits_s.squeeze().tolist()]  # 保留四位小数
            topk_indices_s = topk_indices_s.squeeze().tolist()  # top 5 标签索引

            topk_logits_a, topk_indices_a = AFV_pred.topk(k, 1, True, True)
            topk_logits_a = [round(val, 4) for val in topk_logits_a.squeeze().tolist()]  # 保留四位小数
            topk_indices_a = topk_indices_a.squeeze().tolist()  # top 5 标签索引

            topk_logits_ca, topk_indices_ca = img_global_pred.topk(k, 1, True, True)
            topk_logits_ca = [round(val, 4) for val in topk_logits_ca.squeeze().tolist()]  # 保留四位小数
            topk_indices_ca = topk_indices_ca.squeeze().tolist()  # top 5 标签索引

            topk_logits_zs, topk_indices_zs = img_text_pred.topk(k, 1, True, True)
            topk_logits_zs = [round(val, 4) for val in topk_logits_zs.squeeze().tolist()]  # 保留四位小数
            topk_indices_zs = topk_indices_zs.squeeze().tolist()  # top 5 标签索引

            # 判断正确性
            target_idx = target.item()  # 获取 target 索引值
            final_top1_logits, final_top1_indices = final_pred.topk(1, 1, True, True)
            img_text_top1_logits, img_text_top1_indices = img_text_pred.topk(1, 1, True, True)
            _, CSS_top1 = CSS_pred.topk(1, 1, True, True)
            _, AFV_top1 = AFV_pred.topk(1, 1, True, True)
            _, CA_top1 = img_global_pred.topk(1, 1, True, True)

            final_top1_idx = final_top1_indices.squeeze().item()
            img_text_top1_idx = img_text_top1_indices.squeeze().item()
            CSS_top1_idx = CSS_top1.squeeze().item()
            AFV_top1_idx = AFV_top1.squeeze().item()
            CA_top1_idx = CA_top1.squeeze().item()


            final_pred_correct = (final_top1_idx == target_idx)
            img_text_pred_correct = (img_text_top1_idx == target_idx)
            CSS_correct = (CSS_top1_idx == target_idx)
            AFV_correct = (AFV_top1_idx == target_idx)
            CA_correct = (CA_top1_idx == target_idx)

            # 判断
            tell1 = (CSS_correct and (not CA_correct))
            tell2 = (AFV_correct and (not CA_correct))

            classnames_list = model.classnames

            if tell1 or tell2:
                print("*"*80)

                print(f"CSS work: {tell1}")
                print(f"AFV work: {tell2}\n")

                print(f"The sample num is:{i}\n")
                # 输出预测正确性
                print(f"Real Target class: {classnames_list[target]}")
                print(f"final_pred is right: {final_pred_correct}")
                print(f"img_text_pred right: {img_text_pred_correct}")

                final_pred_classes = [(classnames_list[idx], logit) for idx, logit in zip(topk_indices, topk_logits)]
                img_text_pred_classes = [(classnames_list[idx], logit) for idx, logit in zip(topk_indices_zs, topk_logits_zs)]

                CSS_pred_classes = [(classnames_list[idx], logit) for idx, logit in zip(topk_indices_s, topk_logits_s)]
                AFV_pred_classes = [(classnames_list[idx], logit) for idx, logit in zip(topk_indices_a, topk_logits_a)]

                CA_pred_classes = [(classnames_list[idx], logit) for idx, logit in zip(topk_indices_ca, topk_logits_ca)]



                # 输出 final_pred 的 top k 类别名字及 logits
                print("\nfinal_pred  top k  logits :")
                for class_name, logit in final_pred_classes:
                    print(f"class: {class_name}, logits: {logit}")
                print("\nCSS_pred  top k  logits :")
                for class_name, logit in CSS_pred_classes:
                    print(f"class: {class_name}, logits: {logit}")
                print("\nAFV_pred  top k  logits :")
                for class_name, logit in AFV_pred_classes:
                    print(f"class: {class_name}, logits: {logit}")
                print("\nimg_global_pred top k  logits :")
                for class_name, logit in CA_pred_classes:
                    print(f"class: {class_name}, logits: {logit}")
                # 输出 img_text_pred 的 top k 类别名字及 logits
                print("\nimg_text_pred top k logits :")
                for class_name, logit in img_text_pred_classes:
                    print(f"class: {class_name}, logits: {logit}")

                visualize_original_image(images[0],f"./Pics/{folder}/{set_id}/img_{i}.png")
                # # visualize_image(images, attn_map1, attn_map2, path)
                # attn_score_dinov2_patches = model.attn_score_dinov2  # bs * n_heads * 261 * 261
                # attn_score_dinov2_patches = attn_score_dinov2_patches[:, :, 0, 5:] # bs * n_heads * 256
                # attn_score_dinov2_map = compute_pixel_attention_from_patches(attn_score_dinov2_patches) # torch.Size([16, 224, 224]) 这是新版 2D 插值
                #
                # attn_score_clip_patches = model.attn_score_clip # torch.Size([bs, head, 196])
                # attn_score_clip_map = compute_pixel_attention_from_patches(attn_score_clip_patches) #
                #
                # visualize_image(images[0], attn_score_clip_map[:1], attn_score_dinov2_map[:1], f"./Pics/{folder}/{set_id}/img_{i}.png")


        pred_vanilla.append(img_text_pred)
        pred_global.append(img_global_pred)

        if args.n_shot:
            pred_fewshot_global.append(fewshot_global_pred)

        labels.append(target)

        # # measure accuracy and record loss
        acc1, _ = accuracy(img_text_pred, target, topk=(1, 5))
        acc1_global, _ = accuracy(img_global_pred, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top1_global.update(acc1_global[0], image.size(0))

        if args.n_shot:
            acc1_global_fs, _ = accuracy(fewshot_global_pred, target, topk=(1, 5))

        if args.n_shot:
            top1_global_fewshot.update(acc1_global_fs[0], image.size(0))

        if args.DINOv2 and args.DINOv2_mem:
            acc1_dinov2mem, _ = accuracy(dinov2_logits, target, topk=(1, 5))
            top1_dinov2_mem.update(acc1_dinov2mem[0], image.size(0))

        if args.use_MAC_logits:
            acc1_mac, _ = accuracy(MAC_logits4, target, topk=(1, 5))
            top1_mac.update(acc1_mac[0], image.size(0))

        if args.use_MAC_logits2:
            acc1_mac2, _ = accuracy(MAC_logits5, target, topk=(1, 5))
            top1_mac2.update(acc1_mac2[0], image.size(0))

        if args.Logits_wei_ablation:
            acc1_ab1, _ = accuracy(ablation_1_pred, target, topk=(1, 5))
            top1_ab1.update(acc1_ab1[0], image.size(0))

            acc1_ab2, _ = accuracy(ablation_2_pred, target, topk=(1, 5))
            top1_ab2.update(acc1_ab2[0], image.size(0))

            acc1_ab3, _ = accuracy(ablation_3_pred, target, topk=(1, 5))
            top1_ab3.update(acc1_ab3[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

        # # 进行一些删除 释放显存
        # del clip_both_feats_space
        # del max_cliques4, text_feat_anchor4, adjacency_matrix4, similarity_matrix4
        # del dionv2_feats_space
        # del max_cliques5, text_feat_anchor5, adjacency_matrix5, similarity_matrix5
        # # 手动调用垃圾回收
        # gc.collect()
        #
        # # 释放 PyTorch 的 GPU 缓存
        # torch.cuda.empty_cache()


    timestamp = time.time()
    time_parts = time.gmtime(timestamp)
    hours = time.strftime("%H", time_parts)
    minutes = time.strftime("%M", time_parts)
    seconds = time.strftime("%S", time_parts)
    print("end Time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))

    progress.display_summary()
    pred_vanilla = torch.cat(pred_vanilla, dim=0)
    pred_global = torch.cat(pred_global, dim=0)

    if args.DINOv2 and args.DINOv2_mem:
        pred_dinov2 = torch.cat(pred_dinov2, dim=0)
    else:
        pred_dinov2= pred_vanilla

    if args.use_MAC_logits:
        pred_mac = torch.cat(pred_mac, dim=0)
    else:
        pred_mac = pred_vanilla

    if args.use_MAC_logits2:
        pred_mac2 = torch.cat(pred_mac2, dim=0)
    else:
        pred_mac2 = pred_vanilla


    if args.use_neg_cache:
        pred_neg = torch.cat(pred_neg, dim=0)
    else:
        pred_neg = pred_vanilla

    # pred_local = torch.cat(pred_local, dim=0)
    if args.n_shot:
        pred_fewshot_global = torch.cat(pred_fewshot_global, dim=0)
        # pred_fewshot_local = torch.cat(pred_fewshot_local, dim=0)
    else:
        pred_fewshot_global = pred_vanilla
        # pred_fewshot_local = pred_vanilla
    labels = torch.cat(labels, dim=0)
    ########## put the hyper parameter search here.
    ## final prediction = image_text_pred + alpha * global + beta * local
    weight_search = True
    search_step = 10
    if weight_search:
        # [1.0]
        beta1_list = [1.0]

        if args.Only_MACs:
            beta2_list = [0]
        else:
            beta2_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]

        if args.n_shot:
            beta3_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        else:
            beta3_list = [0]

        if args.DINOv2 and args.DINOv2_mem and (not args.Only_MACs):
            beta4_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
        else:
            beta4_list = [0]

        if args.use_MAC_logits:
            beta5_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        else:
            beta5_list = [0]

        if args.use_MAC_logits2:
            beta6_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        else:
            beta6_list = [0]

        # beta1_list = [i * (10 - 0.001) / search_step + 0.001 for i in range(search_step)]  ## 0.001 - 10
        # beta2_list = [i * (100 - 0.001) / search_step + 0.001 for i in range(search_step)]
        # beta2_list = [i * 0.001 for i in range(1000001)]  # 1000001 是因为要包含 1000
        # beta4_list = [i * 0.001 for i in range(1000001)]  # 1000001 是因为要包含 1000
        print('-' * 20)
        print('Starting searching...')
        print('     beta1 searching range: [0.001, ' + str(10) + ']')
        print('     beta2 searching range: [0.001, ' + str(10) + ']')
        print('     beta3 searching range: [0.001, ' + str(10) + ']')
        print('-' * 20)

        best_acc = 0.
        best_beta1 = 0.
        best_beta2 = 0.
        best_beta3 = 0.
        best_beta4 = 0.
        best_beta5 = 0.
        best_beta6 = 0.
        # ipdb.set_trace()
        # 在三只松鼠基础上进行  only MAC 1 搜索
        for beta1 in beta1_list:
            for beta2 in beta2_list:
                for beta3 in beta3_list:
                    for beta4 in beta4_list:
                        for beta5 in beta5_list:
                            for beta6 in [0]:
                                logits = (pred_vanilla * beta1 + pred_global * beta2 + pred_fewshot_global * beta3
                                            + pred_dinov2 * beta4 + pred_mac * beta5 + pred_mac2 * beta6)
                                acc, _ = accuracy(logits, labels, topk=(1, 5))
                                acc = acc.item()
                                if acc > best_acc:
                                    print(
                                        'New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; beta4: {:.4f}; beta5: {:.4f}; beta6: {:.4f}; Acc: {:.2f}'.format(
                                            beta1, beta2, beta3, beta4, beta5, beta6, acc))
                                    best_acc = acc
                                    best_beta1 = beta1
                                    best_beta2 = beta2
                                    best_beta3 = beta3
                                    best_beta4 = beta4
                                    best_beta5 = beta5
                                    best_beta6 = beta6

        print(f"Based MAC 1 Searched Acc: {best_acc:.2f} with beta1 {best_beta1:.3f}, dynamic {best_beta2:.3f}, static {best_beta3:.3f}, DINOv2 {best_beta4:.3f}, MAC1 {best_beta5:.3f}, MAC2 {best_beta6:.3f}")
        print("\n")
        best_acc = 0.
        best_beta1 = 0.
        best_beta2 = 0.
        best_beta3 = 0.
        best_beta4 = 0.
        best_beta5 = 0.
        best_beta6 = 0.
        # 在三只松鼠基础上进行  only MAC 2 搜索
        for beta1 in beta1_list:
            for beta2 in beta2_list:
                for beta3 in beta3_list:
                    for beta4 in beta4_list:
                        for beta5 in [0]:
                            for beta6 in beta6_list:
                                logits = (pred_vanilla * beta1 + pred_global * beta2 + pred_fewshot_global * beta3
                                            + pred_dinov2 * beta4 + pred_mac * beta5 + pred_mac2 * beta6)
                                acc, _ = accuracy(logits, labels, topk=(1, 5))
                                acc = acc.item()
                                if acc > best_acc:
                                    print(
                                        'New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; beta4: {:.4f}; beta5: {:.4f}; beta6: {:.4f}; Acc: {:.2f}'.format(
                                            beta1, beta2, beta3, beta4, beta5, beta6, acc))
                                    best_acc = acc
                                    best_beta1 = beta1
                                    best_beta2 = beta2
                                    best_beta3 = beta3
                                    best_beta4 = beta4
                                    best_beta5 = beta5
                                    best_beta6 = beta6

        print(f"Based MAC 2 Searched Acc: {best_acc:.2f} with beta1 {best_beta1:.3f}, dynamic {best_beta2:.3f}, static {best_beta3:.3f}, DINOv2 {best_beta4:.3f}, MAC1 {best_beta5:.3f}, MAC2 {best_beta6:.3f}")
        print("\n")
        best_acc = 0.
        best_beta1 = 0.
        best_beta2 = 0.
        best_beta3 = 0.
        best_beta4 = 0.
        best_beta5 = 0.
        best_beta6 = 0.
        # 只用两个 MAC 搜索
        results = []
        if args.Logits_wei_chart:
            new_list = np.arange(0, 10 + 0.05, 0.05).tolist()
            beta5_list_real = new_list
            beta6_list_real = new_list
        else:
            beta5_list_real = beta5_list
            beta6_list_real = beta6_list
        for beta1 in beta1_list:
            for beta2 in [0]:
                for beta3 in [0]:
                    for beta4 in [0]:
                        for beta5 in beta5_list_real:
                            for beta6 in beta6_list_real:
                                logits = (pred_vanilla * beta1 + pred_global * beta2 + pred_fewshot_global * beta3
                                            + pred_dinov2 * beta4 + pred_mac * beta5 + pred_mac2 * beta6)
                                acc, _ = accuracy(logits, labels, topk=(1, 5))
                                acc = acc.item()

                                results.append([beta5, beta6, acc])

                                if acc > best_acc:
                                    print(
                                        'New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; beta4: {:.4f}; beta5: {:.4f}; beta6: {:.4f}; Acc: {:.2f}'.format(
                                            beta1, beta2, beta3, beta4, beta5, beta6, acc))
                                    best_acc = acc
                                    best_beta1 = beta1
                                    best_beta2 = beta2
                                    best_beta3 = beta3
                                    best_beta4 = beta4
                                    best_beta5 = beta5
                                    best_beta6 = beta6
        print(f"Only MACs Searched Acc: {best_acc:.2f} with beta1 {best_beta1:.3f}, dynamic {best_beta2:.3f}, static {best_beta3:.3f}, DINOv2 {best_beta4:.3f}, MAC1 {best_beta5:.3f}, MAC2 {best_beta6:.3f}")
        print("\n")
        if args.Logits_wei_chart:
            # 转换为 NumPy 数组以便于绘图
            results = np.array(results)
            beta5_vals = results[:, 0]
            beta6_vals = results[:, 1]
            acc_vals = results[:, 2]

            # 绘制 3D 图
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(beta5_vals, beta6_vals, acc_vals, cmap='viridis')

            # 设置标签
            ax.set_xlabel(r'$\beta_2$')
            ax.set_ylabel(r'$\beta_3$')
            ax.set_zlabel('Accuracy (Top-1)')


            # 保存图像
            save_path = "./Pics/Logits_wei_chart/" + set_id + ".png"
            plt.savefig(save_path, format='png', dpi=300)  # 使用 300 DPI 提高图片质量




        best_acc = 0.
        best_beta1 = 0.
        best_beta2 = 0.
        best_beta3 = 0.
        best_beta4 = 0.
        best_beta5 = 0.
        best_beta6 = 0.
        # 只用 MAC1 搜索
        for beta1 in beta1_list:
            for beta2 in [0]:
                for beta3 in [0]:
                    for beta4 in [0]:
                        for beta5 in beta5_list:
                            for beta6 in [0]:
                                logits = (pred_vanilla * beta1 + pred_global * beta2 + pred_fewshot_global * beta3
                                            + pred_dinov2 * beta4 + pred_mac * beta5 + pred_mac2 * beta6)
                                acc, _ = accuracy(logits, labels, topk=(1, 5))
                                acc = acc.item()
                                if acc > best_acc:
                                    print(
                                        'New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; beta4: {:.4f}; beta5: {:.4f}; beta6: {:.4f}; Acc: {:.2f}'.format(
                                            beta1, beta2, beta3, beta4, beta5, beta6, acc))
                                    best_acc = acc
                                    best_beta1 = beta1
                                    best_beta2 = beta2
                                    best_beta3 = beta3
                                    best_beta4 = beta4
                                    best_beta5 = beta5
                                    best_beta6 = beta6

        print(f"Only MAC 1 Searched Acc: {best_acc:.2f} with beta1 {best_beta1:.3f}, dynamic {best_beta2:.3f}, static {best_beta3:.3f}, DINOv2 {best_beta4:.3f}, MAC1 {best_beta5:.3f}, MAC2 {best_beta6:.3f}")
        print("\n")
        best_acc = 0.
        best_beta1 = 0.
        best_beta2 = 0.
        best_beta3 = 0.
        best_beta4 = 0.
        best_beta5 = 0.
        best_beta6 = 0.
        # 只用 MAC2 搜索
        for beta1 in beta1_list:
            for beta2 in [0]:
                for beta3 in [0]:
                    for beta4 in [0]:
                        for beta5 in [0]:
                            for beta6 in beta6_list:
                                logits = (pred_vanilla * beta1 + pred_global * beta2 + pred_fewshot_global * beta3
                                            + pred_dinov2 * beta4 + pred_mac * beta5 + pred_mac2 * beta6)
                                acc, _ = accuracy(logits, labels, topk=(1, 5))
                                acc = acc.item()
                                if acc > best_acc:
                                    print(
                                        'New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; beta4: {:.4f}; beta5: {:.4f}; beta6: {:.4f}; Acc: {:.2f}'.format(
                                            beta1, beta2, beta3, beta4, beta5, beta6, acc))
                                    best_acc = acc
                                    best_beta1 = beta1
                                    best_beta2 = beta2
                                    best_beta3 = beta3
                                    best_beta4 = beta4
                                    best_beta5 = beta5
                                    best_beta6 = beta6

        print(f"Only MAC 2 Searched Acc: {best_acc:.2f} with beta1 {best_beta1:.3f}, dynamic {best_beta2:.3f}, static {best_beta3:.3f}, DINOv2 {best_beta4:.3f}, MAC1 {best_beta5:.3f}, MAC2 {best_beta6:.3f}")
        print("\n")
        best_acc = 0.
        best_beta1 = 0.
        best_beta2 = 0.
        best_beta3 = 0.
        best_beta4 = 0.
        best_beta5 = 0.
        best_beta6 = 0.
        # 全局搜索
        for beta1 in beta1_list:
            for beta2 in beta2_list:
                for beta3 in beta3_list:
                    for beta4 in beta4_list:
                        for beta5 in beta5_list:
                            for beta6 in beta6_list:
                                logits = (pred_vanilla * beta1 + pred_global * beta2 + pred_fewshot_global * beta3
                                            + pred_dinov2 * beta4 + pred_mac * beta5 + pred_mac2 * beta6)
                                acc, _ = accuracy(logits, labels, topk=(1, 5))
                                acc = acc.item()
                                if acc > best_acc:
                                    print(
                                        'New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; beta4: {:.4f}; beta5: {:.4f}; beta6: {:.4f}; Acc: {:.2f}'.format(
                                            beta1, beta2, beta3, beta4, beta5, beta6, acc))
                                    best_acc = acc
                                    best_beta1 = beta1
                                    best_beta2 = beta2
                                    best_beta3 = beta3
                                    best_beta4 = beta4
                                    best_beta5 = beta5
                                    best_beta6 = beta6

        print(f"Global Searched Acc: {best_acc:.2f} with beta1 {best_beta1:.3f}, dynamic {best_beta2:.3f}, static {best_beta3:.3f}, DINOv2 {best_beta4:.3f}, MAC1 {best_beta5:.3f}, MAC2 {best_beta6:.3f}")
        print("\n")
        # ipdb.set_trace()
    del pro_cache
    del gra_cache
    del neg_cache
    del dinov2_cache
    return [top1.avg, top1_global.avg, top1_dinov2_mem.avg, top1_mac.avg, top1_mac2.avg, top1_ab1.avg, top1_ab2.avg, top1_ab3.avg, top1_global_fewshot.avg,
            best_acc, best_beta1, best_beta2, best_beta3, best_beta4]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I',
                        help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')

    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')

    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_shot', type=int, default=None)
    parser.add_argument('--n_augview', type=int, default=0, help='use augmented few shot samples')
    parser.add_argument('--ft', action='store_true', default=False,
                        help="fine tuning the attention weight with few labeled data.")
    parser.add_argument('--use_searched_param', action='store_true', default=False,
                        help="using searched param for each dataset")

    parser.add_argument('--beta', default=5.5, type=float, help='loss weight')
    parser.add_argument('--mapping', type=str, default='bias', help='bias | affine | all')
    parser.add_argument('--position', type=str, default='all', help='query | key | value | qkv | output | all')
    parser.add_argument('--optimizer', type=str, default='adamw', help='adamw | sgd')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps, default 1e-8')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--lr', default='0.0001', type=str, help='learning rate')
    parser.add_argument('--lr_tpt', default=5e-3, type=float, help='initial learning rate')
    parser.add_argument('--epoch', type=str, default='20')
    parser.add_argument('--shared_param', action='store_true', default=False,
                        help="shared parameters acorss local | global | text.")
    parser.add_argument('--num_important_channel', type=str,
                        default='0')  ## if 0, use all channels; otherwise, selecting the ape_channel_num
    parser.add_argument('--lambda_ape', default='0.7', type=str, help='following ape.')
    parser.add_argument('--memory_size', type=int, default=50)
    parser.add_argument('--text_prompt', type=str, default='tip_cupl', help='simple | tip | full | tip_cupl')
    parser.add_argument('--log', type=str, default='loga', help='some places to write note')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--fix', action='store_true', default=False,
                        help="fix the backbone or not")
    parser.add_argument('--EMA1', action='store_true', default=False,
                        help="ema1 or not")
    parser.add_argument('--v2', action='store_true', default=False,
                        help="version 2 of ema1 or not")
    parser.add_argument('--EMA2', action='store_true', default=False,
                        help="ema2 or not")
    parser.add_argument('--loss_grad', action='store_true', default=False,
                        help="loss grad or not")
    parser.add_argument('--loss_prop', action='store_true', default=False,
                        help="loss prop or not")
    parser.add_argument('--alpha1', default=0.99, type=float, help='alpha for EMA 1')
    parser.add_argument('--alpha2', default=0.99, type=float, help='alpha for EMA 2')
    parser.add_argument('--alpha3', default=0.5, type=float, help='alpha for averaging hand and learn prompt')
    parser.add_argument('--wei', default=1.0, type=float,
                        help='Weight for averaging hand and learn prompt in update_text_feat')
    parser.add_argument('--use_neg_cache', action='store_true', default=False,
                        help="neg_cache grad or not")
    parser.add_argument('--use_log', action='store_true', default=False,
                        help="use log or not")
    parser.add_argument('--config', dest='config', required=True,
                        help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--new_up', action='store_true', default=False,
                        help="new update or not")
    parser.add_argument('--new_up2', action='store_true', default=False,
                        help="new update 2 or not")
    parser.add_argument('--EMA4mem', action='store_true', default=False,
                        help="EMA for mem or not")
    parser.add_argument('--new_up_mix', action='store_true', default=False,
                        help="new update mix or not")
    parser.add_argument('--DINO', action='store_true', default=False,
                        help="DINO or not")
    parser.add_argument('--DINOv2', action='store_true', default=False,
                        help="DINOv2 or not")
    parser.add_argument('--DINOv2_mem', action='store_true', default=False,
                        help="DINOv2_mem or not")
    parser.add_argument('--DINO4mem', action='store_true', default=False,
                        help="DINO for mem or not")
    parser.add_argument('--DINO4cross', action='store_true', default=False,
                        help="DINO for cross or not")
    parser.add_argument('--DINO_loss', action='store_true', default=False,
                        help="DINO_loss or not")
    parser.add_argument('--simple_CA', action='store_true', default=False,
                        help="simple_CA or not")
    parser.add_argument('--reset_CA', action='store_true', default=False,
                        help="reset for CA or not")
    parser.add_argument('--select_ids', default='I', type=str, help='selected dataset to be checkpoint')
    parser.add_argument('--Choose_cp', action='store_true', default=False, help='Choose checkpoint or not')
    parser.add_argument('--pred_local', action='store_true', default=False, help='pred_local or not')
    # 使用 MAC 相关技术
    parser.add_argument('--use_MAC', action='store_true', default=False, help='Choose MAC or not')
    parser.add_argument('--use_MAC_logits', action='store_true', default=False, help='Use MAC logits or not')
    parser.add_argument('--use_MAC_logits2', action='store_true', default=False, help='Use MAC logits 2 or not')
    parser.add_argument('--lam', default=0.7, type=float, help='Lambda For MAC')
    parser.add_argument('--MAC_freq', default=1, type=int, help='Frequency For MAC search')
    parser.add_argument('--Only_MACs', action='store_true', default=False, help='Only MACs or not')
    parser.add_argument('--Only_MAC1', action='store_true', default=False, help='Only MAC1 or not')
    parser.add_argument('--Only_MAC2', action='store_true', default=False, help='Only MAC2 or not')
    parser.add_argument('--Logits_wei_ablation', action='store_true', default=False, help='Logits_wei_ablation or not')
    parser.add_argument('--Logits_wei_chart', action='store_true', default=False, help='Logits_wei_ablation or not')
    parser.add_argument('--R', default=0.1, type=float, help='Ratio for Selected Cliques')
    parser.add_argument('--get_samples', action='store_true', default=False, help='visual_attn or not')
    parser.add_argument('--inrease_t', action='store_true', default=False, help='inrease_t or not')
    parser.add_argument('--DINO_shot', default=6, type=int, help='DINO shot or not')
    parser.add_argument('--DINO_size',
                        default='l',
                        type=str,
                        choices=['l', 'b', 's'],
                        help='DINO size: l (large), b (base), or s (small)')

    parser.add_argument('--center_type', default='default', type=str, help='center type')
    parser.add_argument('--get_tsne', action='store_true', default=False, help='get t-sne or not')
    main()
if use_log:
    # 程序结束时关闭日志文件
    sys.stdout.log.close()
