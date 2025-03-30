from PIL import Image
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import math
import operator
import networkx as nx
import gc

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from utils.tools import accuracy


def select_confident_samples(prob, top):
    batch_entropy = -(prob * torch.log(prob + 1e-6)).sum(1)
    n_select = max(1, int(batch_entropy.size()[0] * top))  # at least chose one
    idx = torch.argsort(batch_entropy, descending=False)[:n_select]
    idx_confused = torch.argsort(batch_entropy, descending=False)[n_select:]
    return prob[idx], idx, prob[idx_confused], idx_confused

def avg_entropy(outputs):
    ## N*Class
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]

    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    updated = False  # Initialize the updated flag
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        
        # 每一类进行一个存储
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
                updated = True  # Mark as updated

            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
                updated = True  # Mark as updated

            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]
            updated = True  # Mark as updated

    # Optionally print or return the updated flag
    # print("Cache updated:", updated)
    return updated  # Return the updated flag if needed


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

            if current_loss < cached_loss:  # 加一个 非无脑条件

                # Update the features and loss using EMA
                updated_feature = alpha * current_feature + (1 - alpha) * cached_feature
                updated_loss = alpha * current_loss + (1 - alpha) * cached_loss

                # Store the updated feature and loss back in the cache
                cache[pred][0] = [updated_feature, updated_loss]

        else:
            # Initialize cache with the current feature and loss if class is not yet present
            cache[pred] = [[current_feature, current_loss]]



def cosine_similarity(a, b):
    # ipdb.set_trace()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)



def batch_cosine_similarity(a, b, batch_size=20, eps=1e-8):

    a_norm = a / (a.norm(dim=1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=1, keepdim=True) + eps)


    device = a.device

    n, m = a.size(0), b.size(0)
    similarity_matrix = torch.zeros((n, m), dtype=torch.float32, device=device)


    with torch.cuda.amp.autocast():
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            for j in range(0, m, batch_size):
                end_j = min(j + batch_size, m)
                similarity_matrix[i:end_i, j:end_j] = torch.mm(a_norm[i:end_i], b_norm[j:end_j].T)

    return similarity_matrix



def compute_clique_logits(image_features, anchor_feat, target_feat, max_cliques, R=0.2, alpha=1.0, beta=1.0, softmax=False):

    affinity_0 = torch.mm(image_features, anchor_feat.T)  

    top_R_count = max(1, int(len(max_cliques) * R))  
    topk_values, topk_indices = torch.topk(affinity_0, k=top_R_count, dim=-1)  
    mac_idxs = torch.unique(topk_indices, sorted=True) 

    relevant_indices = torch.unique(torch.cat([torch.tensor(max_cliques[idx.item()]) for idx in mac_idxs])).tolist()


    relevant_target_features = torch.index_select(target_feat, 0, torch.tensor(relevant_indices, device=target_feat.device))  


    affinity = torch.mm(image_features, relevant_target_features.T)  


    cache_values = F.one_hot(torch.tensor(relevant_indices), num_classes=target_feat.size(0)).float().to(image_features.device)
    logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    logits = alpha * logits

    n_cls = target_feat.shape[0]  
    full_logits = torch.zeros((1, n_cls), device=image_features.device)  
    

    full_logits[0, relevant_indices] = logits.squeeze()[relevant_indices]


    if softmax:
        full_logits = full_logits.softmax(dim=1)


    return full_logits, relevant_indices, torch.max(affinity)

def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, softmax=False):
    """Compute logits using positive cache."""
    with torch.no_grad():
        cache_keys = []  # for feats
        cache_values = []  # for label
        
        for class_index in sorted(cache.keys()):
           
            for item in cache[class_index]:
                cache_keys.append(item[0])  

                cache_values.append(class_index)
        
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)

        # print("MEI YOU")
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(0))).cuda().half()

       
        affinity = image_features @ cache_keys
        affinity = affinity.to(cache_values.dtype)
     
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # print("cache_logits", cache_logits)

        if softmax:
            return (alpha * cache_logits).softmax(dim=1)
        else:
            return alpha * cache_logits



def build_graph_by_threshold(similarity_matrix, threshold):

    adj_matrix = (similarity_matrix > threshold).float()

    adj_matrix.fill_diagonal_(0)
    
    return adj_matrix


def build_graph_by_degree(similarity_matrix, target_avg_degree, tol=1e-2, max_iter=100):

    n = similarity_matrix.shape[0]
    low, high = similarity_matrix.min().item(), similarity_matrix.max().item()
    
    for _ in range(max_iter):
        mid = (low + high) / 2
        adj_matrix = (similarity_matrix > mid).float() 
        adj_matrix.fill_diagonal_(0)  
        avg_degree = adj_matrix.sum() / n  
        
        if abs(avg_degree - target_avg_degree) < tol:
            break
        elif avg_degree < target_avg_degree:
            high = mid 
        else:
            low = mid 
    # print("degree", avg_degree)
    return adj_matrix, avg_degree


def build_graph_by_degeneracy(similarity_matrix, target_degeneracy, tol=1e-2, max_iter=100):

    n = similarity_matrix.shape[0]
    low, high = similarity_matrix.min().item(), similarity_matrix.max().item()
    adj_matrix = None  

    for _ in range(max_iter):
        mid = (low + high) / 2
        adj_matrix = (similarity_matrix > mid).float() 
        adj_matrix.fill_diagonal_(0)  

       
        G = nx.from_numpy_matrix(adj_matrix.cpu().numpy())


        degeneracy = nx.algorithms.core.core_number(G)
        current_degeneracy = max(degeneracy.values()) if degeneracy else 0


        if abs(current_degeneracy - target_degeneracy) < tol:
            break
        elif current_degeneracy < target_degeneracy:
            low = mid  # 退化度太低，增加边
        else:
            high = mid  # 退化度太高，减少边

        if high - low < tol:
            break

    return adj_matrix, current_degeneracy



def make_max_cliques(text_feat, reduce=False, MAC_name=None, control=None, is_SOG=True):


    if reduce:
        similarity_matrix = batch_cosine_similarity(text_feat, text_feat, batch_size=10)
    else:
        similarity_matrix = torch.mm(text_feat, text_feat.T)  
    # 图构建
    control_method, control_value = control
    if control_method == "degeneracy":
        adjacency_matrix, value = build_graph_by_degeneracy(similarity_matrix, control_value, tol=1e-2, max_iter=100)
    elif control_method == "degree":
        adjacency_matrix, value = build_graph_by_degree(similarity_matrix, control_value, tol=1e-2, max_iter=500)
        
    elif control_method == "threshold":
        adjacency_matrix = build_graph_by_threshold(similarity_matrix, control_value)

    if is_SOG:
        adjacency_matrix1 = torch.mm(adjacency_matrix, adjacency_matrix)
        adjacency_matrix2 = adjacency_matrix * adjacency_matrix1
        adjacency_matrix = (adjacency_matrix2 > 1).float()


    G = nx.from_numpy_matrix(adjacency_matrix.cpu().numpy())
    cliques = list(nx.find_cliques(G))


    max_cliques = []
    for clique in cliques:
        if len(clique) == len(set(clique)):
            max_cliques.append(clique)


    all_nodes = set(range(text_feat.size(0)))
    nodes_in_cliques = set(node for clique in max_cliques for node in clique)
    missing_nodes = all_nodes - nodes_in_cliques

    for node in missing_nodes:
        if max_cliques:
            max_cliques[0].append(node)
        else:
            max_cliques.append([node])

    cliques_feat_anchor = torch.stack([torch.mean(text_feat[clique], dim=0) for clique in max_cliques])

    if reduce:
        del similarity_matrix, adjacency_matrix, adjacency_matrix1, adjacency_matrix2, G, cliques, all_nodes, nodes_in_cliques, missing_nodes
        gc.collect()
        torch.cuda.empty_cache()
    
    if MAC_name == "MAC1":
        print("MAC1*********")
        print("max_cliques", max_cliques)
        
    elif MAC_name == "MAC2":
        print("MAC2*********")
        print("max_cliques", max_cliques)

    return max_cliques, cliques_feat_anchor, None


def make_prototypeAttn_features(dinov2_cache, img_feat, beta, n_cls, n_shot, n_dim):

    memorized_image_feat = torch.zeros((n_cls, n_shot, n_dim)).to(img_feat.device)
    for label, features_list in dinov2_cache.items():
        if 0 <= label < n_cls:
            for i, (feature, _) in enumerate(features_list):
                if i < n_shot:
                    memorized_image_feat[label, i] = feature.squeeze()
                else:
                    break


    with torch.no_grad():
        memorized_image_feat_K = memorized_image_feat / memorized_image_feat.norm(dim=-1, keepdim=True)
        memorized_image_feat_K[memorized_image_feat.sum(-1) == 0] = 0
        memorized_image_feat_V = memorized_image_feat / memorized_image_feat.norm(dim=-1, keepdim=True)
        memorized_image_feat_V[memorized_image_feat.sum(-1) == 0] = 0
        img_feat_mappling = img_feat / img_feat.norm(dim=-1, keepdim=True)

    similarity_matrix = torch.matmul(img_feat_mappling, memorized_image_feat_K.transpose(-1, -2))  # 1 x n_cls x n_shot
    similarity_matrix = torch.exp(-beta * (-similarity_matrix + 1))

    adaptive_image_feat = torch.matmul(similarity_matrix.unsqueeze(1), memorized_image_feat_V).squeeze(1)  # n_cls x n_dim

    return adaptive_image_feat



def make_prototype_features(dinov2_cache, n_class, n_dim):
    # Step 1: Initialize a tensor of zeros with shape (n_class, n_dim)
    prototype_features = torch.zeros((n_class, n_dim)).to(next(iter(dinov2_cache.values()))[0][0].device)

    # Step 2: Extract features and compute means
    for class_id, features in dinov2_cache.items():
        if features:  # Ensure there are features to process
            # Stack all features into a single tensor
            tensors = torch.stack([feature[0] for feature in features])
            # Compute the mean of these tensors along the 0th dimension
            prototype_features[class_id] = tensors.mean(dim=0)

    return prototype_features

def linear_growth(t0, k, i):
    return min(1, t0 + k * i)


def exponential_growth(t0, k, i):
    return min(1, t0 * (1 - math.exp(-k * i)))


def sigmoid_growth(k, i0, i):
    return 1 / (1 + math.exp(-k * (i - i0)))


def logarithmic_growth(t0, k, i):
    return min(1, t0 + k * math.log(i + 1))


def exponential_decay(t0, min_v, k, i, max_iters):

    return max(min_v, t0 * math.exp(-k * i / max_iters))

def search_best_weights(pred_vanilla, pred_global, pred_dinov2, pred_mac, pred_mac2, labels, 
                       beta1_list, beta2_list, beta3_list, beta4_list, beta5_list, search_name=""):

    best_acc = [0.0, 0.0]
    
    best_betas = {'beta1': 0., 'beta2': 0., 'beta3': 0., 'beta4': 0., 'beta5': 0.}
    
    for beta1 in beta1_list:
        for beta2 in beta2_list:
            for beta3 in beta3_list:
                for beta4 in beta4_list:
                    for beta5 in beta5_list:
                        logits = (pred_vanilla * beta1 + pred_global * beta2 
                                + pred_dinov2 * beta3 + pred_mac * beta4 + pred_mac2 * beta5)
                        acc, acc5 = accuracy(logits, labels, topk=(1, 5))
                        acc = acc.item()
                        acc5 = acc5.item()
                        
                        if acc > best_acc[0]:
                            # print('New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; beta4: {:.4f}; beta5: {:.4f}; Acc: {:.2f}'.format(beta1, beta2, beta3, beta4, beta5, acc))
                            
                            best_acc[0] = acc
                            best_acc[1] = acc5

                            best_betas.update({
                                'beta1': beta1, 'beta2': beta2, 'beta3': beta3,
                                'beta4': beta4, 'beta5': beta5
                            })
    
    print(f"{search_name} Searched Acc: {best_acc[0]:.2f} with beta1 {best_betas['beta1']:.3f}, "
          f"Clip cache {best_betas['beta2']:.3f}, DINOv2 cache {best_betas['beta3']:.3f}, "
          f"MAC1 {best_betas['beta4']:.3f}, MAC2 {best_betas['beta5']:.3f}")
    print("\n")
    
    return best_acc, best_betas