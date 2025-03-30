import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import ipdb
## the main component.
class DualMem(nn.Module):
    def __init__(self, args=None, beta=5.5, feat_dim=1024, class_num=1000):
        super(DualMem, self).__init__()
        self.args = args
        self.beta = beta
        self.rank = 4
        self.init_pred = 0

        feat_dim_img = feat_dim
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


    def update_memory_bank(self, model, selected_idx=None):
        # Initialize the updated flag
        updated = False
        
        # Updating
        mean_prob = self.init_pred[0]
        value, indice = mean_prob.max(0)
        pseudo_label = indice.item()
        text_features = model.text_feat[pseudo_label]  ## 512

        if selected_idx is not None:
            selected_image_features_global = model.image_features_global[selected_idx].mean(0).unsqueeze(0)
        else:
            selected_image_features_global = model.image_features_global[:1]
        current_instance_entropy = -(mean_prob * (torch.log(mean_prob + 1e-8))).sum()

        
        if model.image_feature_count[pseudo_label] == model.memory_size_clip:
            ###### if the new one is low entropy, find the sample with the max entropy, and replace it with the new one
            if (current_instance_entropy < model.image_entropy_mem[pseudo_label]).sum() == 0:
                pass  ## the entropy of current test image is very large.
            else:
                _, indice = torch.sort(model.image_entropy_mem[pseudo_label])
                to_replace_indice = indice[-1]  ## with max entropy, ascending.
                model.image_feature_memory[pseudo_label][to_replace_indice] = selected_image_features_global
                model.image_prediction_mem[pseudo_label][to_replace_indice] = mean_prob[0]
                model.image_entropy_mem[pseudo_label][to_replace_indice] = current_instance_entropy
                updated = True  # Mark as updated
        else:
            model.image_feature_memory[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = selected_image_features_global  # torch.Size([1, 512])
            model.image_prediction_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = mean_prob[0]  # torch.Size([])
            model.image_entropy_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = current_instance_entropy  # torch.Size([])
            model.image_feature_count[pseudo_label] += 1
            updated = True  # Mark as updated

        # Optionally print or return the updated flag
        # print("Memory bank updated:", updated)
        # print("Memory bank updated:", model.image_feature_memory)
        return updated  # Return the updated flag if needed
    


    # 附属预测
    def get_image_pred(self, model):
        ## prediction with dynamic memory.
        img_feat = model.image_features_global[:1]  # 1*1024

        count_image_feat = model.image_feature_count.clone()
        num_class = model.image_feature_memory.shape[0]
        memorized_image_feat = torch.cat((model.image_feature_memory, model.fixed_global_feat_vanilla), dim=1)  ## 200*11*1024
        # memorized_image_feat = torch.cat((model.fixed_global_feat_vanilla, model.image_feature_memory), dim=1)  ## 200*11*1024
            # memorized_image_feat = model.image_feature_memory

        if self.args.center_type_clip == 'default':
            ############### assign each feature with equal weights.
            # memorized_image_feat = memorized_image_feat[:, 1:, :]
            filled_image_feat = memorized_image_feat.sum(1) / (count_image_feat + 1)  ### no zero. 200*1024
            filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)


            self.adaptive_image_feat = filled_image_feat

            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_feat @ filled_image_feat.t()


            return logits.softmax(dim=1)
        elif self.args.center_type_clip == 'attn':  ## this is an instance adaptative method.
            ## calculate the cos similarity betweeen image feature and memory feature, and then weighted the memorized features according to similarity.
            ###################### 有一些memory 是空的，现在却往里面塞了一个self.global_bias， 这不合理，还要把它继续置空。
            img_feat_mappling = img_feat  # 1*1024
            memorized_image_feat_K = memorized_image_feat[:, 1:, :]   # 200*11*1024
            memorized_image_feat_V = memorized_image_feat[:, 1:, :]   # 200*11*1024
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

            adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            # print("2", adaptive_image_feat)
            if self.args.position == 'output' or self.args.position == 'all':
                adaptive_image_feat = adaptive_image_feat + self.global_ffn_bias.unsqueeze(0)  ## class*shot*1024
            # print("3", adaptive_image_feat)

            adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            # adaptive_image_feat: torch.Size([1, 102, 1024])
            # img_feat: torch.Size([1, 1024])
            logits = logit_scale * adaptive_image_feat @ img_feat.unsqueeze(-1)  ## used feat is not update.
            logits = logits[:, :, 0]


            self.adaptive_image_feat = adaptive_image_feat

            return logits.softmax(dim=1)
        else:
            raise NotImplementedError

    # 正统预测
    def get_text_prediction(self, model, return_full=True, return_logit=False):
        logit_scale = model.logit_scale.exp()

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
        