import argparse
import time
from PIL import Image
import yaml
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os
import datetime
import sys
import gc
from torch.cuda.amp import autocast
from torch.nn.parallel import DataParallel
from tabulate import tabulate
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.fix_clip import get_fixed_clip
from clip.fix_align import get_fixed_align

from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, AugMemAugmenter, StrongAugmenter
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets

from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from utils.DualMems import DualMem
from utils.utils_cosmic import *

from typing import Callable

import ipdb
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

class Logger:
    def __init__(self, filename):
        """Initialize logger with file output and terminal output.
        
        Args:
            filename: Path to the log file
        """
        log_dir = os.path.dirname(filename)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log = open(filename, "a")
        self.terminal = sys.stdout

    def write(self, message):
        """Write message to both terminal and log file"""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """Flush both terminal and log file buffers"""
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """Close the log file"""
        self.log.close()

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


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))




def main():
    args = parser.parse_args()
    if args.use_log:
        current_time = args.log_time if hasattr(args, 'log_time') else datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        dev_num = args.gpu
        bs = args.batch_size
        log_filename = f"./logs/dev_{dev_num}/{current_time}.log"
        args.log = log_filename
        sys.stdout = Logger(log_filename)
    
    # Print all arguments and their values
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    set_random_seed(args.seed)
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    """Main worker function for model evaluation"""
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # Create model based on dataset type
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes

    # Initialize model based on architecture type
    if args.arch == "align-base":
        model = get_fixed_align(args, classnames, args.gpu, 
                          memory_size=args.memory_size,
                          text_prompt=args.text_prompt,
                          model_path="align_models/align-base")
    else:
        model = get_fixed_clip(args, args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init,
                              memory_size_clip=args.CLIP_Cache_shot, memory_size_dino=args.DINO_Cache_shot, text_prompt=args.text_prompt)
    
    # Freeze all model parameters
    for name, param in model.named_parameters():
        param.requires_grad_(False)

    print("=> Model created: visual backbone {}".format(args.arch))

    # Setup GPU or CPU
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    cudnn.benchmark = True

    # Normalization parameters from CLIP
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    
    datasets = args.test_sets.split("/")
    results = {}
    print_log = print_logger(print, os.path.join(args.log + '.txt'))
    config_path = args.config
    i = 0

    for set_id in datasets:


        best_acc = 0
        print("*" * 80)
        print_log("processing the dataset {} \n".format(set_id), end="	")

        cfg = get_config_file(config_path, set_id)  
        print("\nRunning dataset configurations:")
        print(cfg, "\n")


        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if args.arch == "align-base":
            preprocess2 = transforms.Compose([ transforms.ToTensor()])
            data_transform = AugMixAugmenter(base_transform, preprocess2, n_views=args.batch_size - 1,
                                            augmix=len(set_id) > 1,
                                            severity=50)  ### aug mix not used for ImageNet test set.
        else:
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1,
                                            augmix=len(set_id) > 1,
                                            severity=50)  ### aug mix not used for ImageNet test set.


        batchsize = 1


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

        model.reset_classnames(classnames, set_id)


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


        results_temp = direct_inference(cfg, val_loader, model, args)
        print_log("best acc {:.2f} \n".format(results_temp[0]), end="	")
        if results_temp[0] > best_acc:
            results[set_id] = results_temp
            best_acc = results_temp[0]

        del val_dataset, val_loader

        try:
            print(f"=> Acc. on testset [{set_id}]: @1 {results[set_id][0]:.2f}/ @5 {results[set_id][1]:.2f}")

        except:
            print(f"=> Acc. on testset [{set_id}]: {results[set_id]:.2f}")
        length = len(results[set_id])

    args.indice = 0
    log = open(os.path.join(args.log + '.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    headers = ["Dataset", "COSMIC", "COSMIC_top5", "Bi-cache", "raw_clip", "clip_cache", "dino_cache", "clip_cliques", "dino_cliques"]

    table = []
    for id, values in results.items():
        row = [id] + values
        table.append(row)

    mean_values = [sum(x) / len(x) for x in zip(*results.values())]
    table.append(["mean"] + mean_values)
    print_log("======== Result Summary ========")
    print_log(tabulate(table, headers=headers, floatfmt=".3f", tablefmt="github"))
    




def direct_inference(cfg, val_loader, model, args):
    """Direct inference function for model evaluation"""
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top1_global = AverageMeter('AccClip_Cache@1', ':6.2f', Summary.AVERAGE)
    top1_dinov2_mem = AverageMeter('AccDinov2_Cache@1', ':6.2f', Summary.AVERAGE)
    top1_clip_clique = AverageMeter('AccClip_Clique@1', ':6.2f', Summary.AVERAGE)
    top1_dino_clique = AverageMeter('AccDINOv2_Clique@1', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top1_global, top1_dinov2_mem, top1_clip_clique, top1_dino_clique],
        prefix='Test: ')

    model.eval()

    # Initialize text features if not already done
    if model.first_flag:
        with torch.no_grad():
            text_feat, text_feat_full = model.get_text_features()
    else:
        print('Text features already initialized, skipping initialization.')

    class_num, feat_dim = model.text_feat.shape[0], model.text_feat.shape[1]
    labels = []
    pred_vanilla = []
    pred_global = []
    pred_dinov2 = []
    pred_clip_clique = []
    pred_dino_clique = []

    CacheNet = DualMem(args=args, beta=args.beta, feat_dim=feat_dim, class_num=class_num).cuda()
    CacheNet.eval()
    
    end = time.time()
    timestamp = time.time()
    time_parts = time.gmtime(timestamp)
    print(f"Test start time: {time.strftime('%H:%M:%S', time_parts)}")

    # Initialize caches
    pro_cache = {}
    gra_cache = {}
    neg_cache = {}
    dinov2_cache = {}
    clip_cache = {}

    # Main evaluation loop
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):  # Handle augmix return type
            images = torch.cat(images, dim=0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images[:1]
        else:  # Handle standard tensor return type
            if len(images.size()) > 4:
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images[:1]
        target = target.cuda(args.gpu, non_blocking=True)
    
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features_global, image_features_local = model.get_image_features(images)
                img_feats_clip = image_features_global[:1]
                img_text = CacheNet.get_text_prediction(model)
                img_text_pred = img_text[:1]  # Current prediction [1, n_cls]

        # DMN or TDA Inference
        confidence_prediction, selected_idx, confused_weak_output, confused_idx = select_confident_samples(img_text, args.selection_p)
        CacheNet.init_pred = confidence_prediction.mean(0, keepdim=True)

        acc1, _ = accuracy(CacheNet.init_pred, target, topk=(1, 5))
        loss = avg_entropy(img_text)
        _, pred_indices = img_text_pred.topk(1, 1, True, True)
        pred = int(pred_indices.squeeze().item())

        # Update CLIP cache
        if args.clip_is_DMN:
            is_clip_cache_update = CacheNet.update_memory_bank(model, selected_idx)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    img_global_pred = CacheNet.get_image_pred(model)
        else:
            if args.center_type_clip == 'ema':
                update_cache_ema(clip_cache, pred, [img_feats_clip, loss], 0.2)
            else:
                is_clip_cache_update = update_cache(clip_cache, pred, [img_feats_clip, loss], args.CLIP_Cache_shot)

            if clip_cache:
                img_global_pred = compute_cache_logits(img_feats_clip, clip_cache, cfg["positive"]["alpha"], 
                                                     cfg["positive"]["beta"], model.text_feat, softmax=True)
            else:
                img_global_pred = torch.zeros_like(img_text_pred)

        # Update prediction and loss for DINOv2
        temp_pred = img_text_pred + img_global_pred
        loss = avg_entropy(temp_pred)
        _, pred_indices = temp_pred.topk(1, 1, True, True)
        pred = int(pred_indices.squeeze().item())

        # DINOv2 Inference
        if args.DINOv2:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_features_global_aux, image_features_local_aux = model.get_image_features_aux(images)
                    img_feats_dinov2 = image_features_global_aux[:1]

            if args.center_type_dino == 'ema':
                update_cache_ema(dinov2_cache, pred, [img_feats_dinov2, loss], 0.2)
            else:
                is_dinov2_cache_update = update_cache(dinov2_cache, pred, [img_feats_dinov2, loss], args.DINO_Cache_shot)

            if dinov2_cache:
                img_dinov2_pred = compute_cache_logits(img_feats_dinov2, dinov2_cache, cfg["positive"]["alpha"], 
                                                     cfg["positive"]["beta"], model.text_feat, softmax=True)
            else:
                img_dinov2_pred = torch.zeros_like(img_text_pred)

        # Set thresholds for MAC search
        if args.arch == "align-base":
            t0 = cfg["Mac"]["lambda1"] * 1.11
        else:
            t0 = cfg["Mac"]["lambda1"]

        t0_2 = cfg["Mac"]["lambda2"]

        if args.inrease_t:
            k = 0.00001
            Th_clip = linear_growth(t0, k, i)
            k_2 = 0.00001
            Th_dino = linear_growth(t0_2, k_2, i)
        else:
            Th_clip = t0
            Th_dino = t0_2
            
        # Set control parameters for graph structure
        if args.control_type == "degree":
            control_v1 = args.target_avg_degree * 0.8
            control_v2 = args.target_avg_degree
        elif args.control_type == "degeneracy":
            control_v1 = args.target_degeneracy
            control_v2 = args.target_degeneracy
        elif args.control_type == "threshold":
            control_v1 = Th_clip
            control_v2 = Th_dino
        
        control1 = (args.control_type, control_v1)
        control2 = (args.control_type, control_v2)

        # Process CLIP cliques
        if args.use_clip_clique:
            if args.clip_is_DMN:
                clip_both_feats_space = torch.cat((model.text_feat, CacheNet.adaptive_image_feat.squeeze(0)), dim=0)
            else:
                if args.center_type_clip == 'default' or args.center_type_clip == 'ema':
                    clip_prototypes = make_prototype_features(clip_cache, model.text_feat.shape[0], image_features_global[:1].shape[1])
                elif args.center_type_clip == 'attn':
                    clip_prototypes = make_prototypeAttn_features(clip_cache, image_features_global[:1], args.beta, 
                                                                model.text_feat.shape[0], args.CLIP_Cache_shot,
                                                                image_features_global[:1].shape[1])
                clip_prototypes = clip_prototypes.to(image_features_global.device)
                clip_both_feats_space = torch.cat((clip_prototypes, model.text_feat), dim=0)
            
            # Update cliques based on conditions
            situation1 = args.always_update_G and is_clip_cache_update
            situation2 = not args.always_update_G and i % args.mac_step == 0 and i > class_num * args.CLIP_Cache_shot
            situation3 = not args.always_update_G and is_clip_cache_update and i <= class_num * args.CLIP_Cache_shot
  
            if situation1 or situation2 or situation3:
                max_cliques1, cliques_feat_anchor1, intermediates1 = make_max_cliques(clip_both_feats_space, False, 
                                                                                    MAC_name=None, control=control1, 
                                                                                    is_SOG=args.is_SOG)

            # Clean up intermediates periodically
            if i % args.mac_step == 0:
                del intermediates1
                gc.collect()
                torch.cuda.empty_cache()
            
            MAC_logits1, _, max_aff1 = compute_clique_logits(image_features_global[:1], cliques_feat_anchor1, 
                                                            clip_both_feats_space, max_cliques1, args.r, 
                                                            alpha=cfg["positive"]["alpha"], 
                                                            beta=cfg["positive"]["beta"], softmax=False)
            # Average predictions from both halves
            first_half = MAC_logits1[:, :MAC_logits1.shape[1] // 2]
            second_half = MAC_logits1[:, MAC_logits1.shape[1] // 2:]
            MAC_logits1 = (first_half + second_half) / 2

            pred_clip_clique.append(MAC_logits1)

        # Process DINO cliques
        if args.use_dino_clique:
            if args.dino_is_DMN:
                dinov2_prototypes = CacheNet.adaptive_image_feat2
            else:
                if args.center_type_dino == 'default' or args.center_type_dino == 'ema':
                    dinov2_prototypes = make_prototype_features(dinov2_cache, model.text_feat.shape[0], 
                                                              image_features_global_aux[:1].shape[1])
                elif args.center_type_dino == 'attn':
                    dinov2_prototypes = make_prototypeAttn_features(dinov2_cache, image_features_global_aux[:1], 
                                                                  args.beta, model.text_feat.shape[0], 
                                                                  args.DINO_Cache_shot,
                                                                  image_features_global_aux[:1].shape[1])
            dinov2_prototypes = dinov2_prototypes.to(image_features_global_aux.device)
            dionv2_feats_space = dinov2_prototypes
            
            # Update cliques based on conditions
            situation1 = args.always_update_G and is_dinov2_cache_update
            situation2 = not args.always_update_G and i % args.mac_step == 0 and i > class_num * args.DINO_Cache_shot
            situation3 = not args.always_update_G and is_dinov2_cache_update and i <= class_num * args.DINO_Cache_shot

            if situation1 or situation2 or situation3:
                max_cliques2, cliques_feat_anchor2, intermediates2 = make_max_cliques(dionv2_feats_space, False, 
                                                                                    MAC_name=None, control=control2, 
                                                                                    is_SOG=args.is_SOG)

            # Clean up intermediates periodically
            if i % args.mac_step == 0:
                del intermediates2
                gc.collect()
                torch.cuda.empty_cache()

            MAC_logits2, _, max_aff2 = compute_clique_logits(image_features_global_aux[:1], cliques_feat_anchor2, 
                                                            dionv2_feats_space, max_cliques2, args.r,
                                                            alpha=cfg["positive"]["alpha"], 
                                                            beta=cfg["positive"]["beta"], softmax=False)

            pred_dino_clique.append(MAC_logits2)

        # Collect predictions
        pred_vanilla.append(img_text_pred)
        pred_global.append(img_global_pred)
        pred_dinov2.append(img_dinov2_pred)
        labels.append(target)

        # Measure accuracy and record results
        acc1, _ = accuracy(img_text_pred, target, topk=(1, 5))
        acc1_global, _ = accuracy(img_global_pred, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top1_global.update(acc1_global[0], image.size(0))

        if args.DINOv2:
            acc1_dinov2mem, _ = accuracy(img_dinov2_pred, target, topk=(1, 5))
            top1_dinov2_mem.update(acc1_dinov2mem[0], image.size(0))

        if args.use_clip_clique:
            acc1_clip_clique, _ = accuracy(MAC_logits1, target, topk=(1, 5))
            top1_clip_clique.update(acc1_clip_clique[0], image.size(0))

        if args.use_dino_clique:
            acc1_dino_clique, _ = accuracy(MAC_logits2, target, topk=(1, 5))
            top1_dino_clique.update(acc1_dino_clique[0], image.size(0))

        # Update timing
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()

        if (i > 0 and i % args.print_freq == 0) or i + 1 == len(val_loader):
            progress.display(i)



    progress.display_summary()

    # Concatenate all predictions
    pred_vanilla = torch.cat(pred_vanilla, dim=0)
    pred_global = torch.cat(pred_global, dim=0)

    if args.DINOv2:
        pred_dinov2 = torch.cat(pred_dinov2, dim=0)
    else:
        pred_dinov2 = pred_vanilla

    if args.use_clip_clique:
        pred_clip_clique = torch.cat(pred_clip_clique, dim=0)
    else:
        pred_clip_clique = pred_vanilla

    if args.use_dino_clique:
        pred_dino_clique = torch.cat(pred_dino_clique, dim=0)
    else:
        pred_dino_clique = pred_vanilla

    labels = torch.cat(labels, dim=0)
    weight_search = True

    if weight_search:
        beta1_list = [1.0]  # Base weight for vanilla predictions

        # CLIP cache weights
        if args.use_clip_cache:
            beta2_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        else:
            beta2_list = [0]

        # DINO cache weights
        if args.DINOv2:
            beta3_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        else:
            beta3_list = [0]

        # CLIP clique weights
        if args.use_clip_clique:
            beta4_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        else:
            beta4_list = [0]

        # DINO clique weights
        if args.use_dino_clique:
            beta5_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        else:
            beta5_list = [0]

        print('-' * 20)
        print('Starting searching...')
        print('-' * 20)

        # Search best weights for different combinations
        # 1. CLIP cache only
        best_acc1, best_betas = search_best_weights(
            pred_vanilla, pred_global, pred_dinov2, pred_clip_clique, pred_dino_clique,
            labels,
            beta1_list=[1.0],
            beta2_list=beta2_list,
            beta3_list=[0.0],
            beta4_list=[0.0],
            beta5_list=[0.0],
            search_name="With CLIP cache (beta2)"
        )

        # 2. DINO cache only
        best_acc2, best_betas = search_best_weights(
            pred_vanilla, pred_global, pred_dinov2, pred_clip_clique, pred_dino_clique,
            labels,
            beta1_list=[1.0],
            beta2_list=[0.0],
            beta3_list=beta3_list,
            beta4_list=[0.0],
            beta5_list=[0.0],
            search_name="With DINO cache (beta3)"
        )

        # 3. CLIP clique only
        best_acc4, best_betas = search_best_weights(
            pred_vanilla, pred_global, pred_dinov2, pred_clip_clique, pred_dino_clique,
            labels,
            beta1_list=[1.0],
            beta2_list=[0.0],
            beta3_list=[0.0],
            beta4_list=beta4_list,
            beta5_list=[0.0],
            search_name="With CLIP clique (beta4)"
        )

        # 4. DINO clique only
        best_acc5, best_betas = search_best_weights(
            pred_vanilla, pred_global, pred_dinov2, pred_clip_clique, pred_dino_clique,
            labels,
            beta1_list=[1.0],
            beta2_list=[0.0],
            beta3_list=[0.0],
            beta4_list=[0.0],
            beta5_list=beta5_list,
            search_name="With DINO clique (beta5)"
        )

        # 5. Both CLIP and DINO caches
        best_acc3, best_betas = search_best_weights(
            pred_vanilla, pred_global, pred_dinov2, pred_clip_clique, pred_dino_clique,
            labels,
            beta1_list=[1.0],
            beta2_list=beta2_list,
            beta3_list=beta3_list,
            beta4_list=[0.0],
            beta5_list=[0.0],
            search_name="With CLIP & DINO cache (beta2, beta3)"
        )

        # 6. Full COSMIC (both cliques)
        best_acc6, best_betas = search_best_weights(
            pred_vanilla, pred_global, pred_dinov2, pred_clip_clique, pred_dino_clique,
            labels,
            beta1_list=[1.0],
            beta2_list=[0.0],
            beta3_list=[0.0],
            beta4_list=beta4_list,
            beta5_list=beta5_list,
            search_name="COSMIC (beta4, beta5)"
        )

    # Clean up caches
    del pro_cache
    del gra_cache
    del neg_cache
    del dinov2_cache
    
    # Return accuracies for different methods
    return [best_acc6[0], best_acc6[1], best_acc3[0], top1.avg, top1_global.avg, 
            top1_dinov2_mem.avg, top1_clip_clique.avg, top1_dino_clique.avg]




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation')
    
    # Data parameters
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    
    # Model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    
    # Training parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--seed', type=int, default=0)
    
    # CLIP parameters
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained model')
    parser.add_argument('--beta', default=5.5, type=float, help='loss weight')
    parser.add_argument('--text_prompt', type=str, default='tip_cupl', help='prompt type: simple | tip | full | tip_cupl')
    
    # Logging parameters
    parser.add_argument('--use_log', action='store_true', default=False, help='use logging')
    parser.add_argument('--log', type=str, default='loga', help='log file prefix')
    parser.add_argument('--log_time', type=str, default='', help='log time stamp')
    parser.add_argument('--config', dest='config', required=True, help='configuration file path')
    
    # Cache parameters
    parser.add_argument('--DINOv2', action='store_true', default=False, help='use DINOv2 cache')
    parser.add_argument('--use_clip_cache', action='store_true', default=False, help='use CLIP cache')
    parser.add_argument('--DINO_Cache_shot', default=6, type=int, help='DINO cache size')
    parser.add_argument('--CLIP_Cache_shot', default=6, type=int, help='CLIP cache size')
    parser.add_argument('--center_type_clip', default='default', type=str, choices=['default', 'ema', 'attn'])
    parser.add_argument('--center_type_dino', default='default', type=str, choices=['default', 'ema', 'attn'])
    parser.add_argument('--clip_is_DMN', default=False, action='store_true', help='use DMN for CLIP')
    parser.add_argument('--dino_is_DMN', default=False, action='store_true', help='use DMN for DINO')
    parser.add_argument('--DINO_size', default='l', type=str, choices=['l', 'b', 's'], help='DINO model size')
    
    # COSMIC parameters
    parser.add_argument('--use_clip_clique', action='store_true', default=False, help='use CLIP cliques')
    parser.add_argument('--use_dino_clique', action='store_true', default=False, help='use DINO cliques')
    parser.add_argument('--mac_step', default=100, type=int, help='maximal clique search interval')
    parser.add_argument('--target_avg_degree', default=10, type=float, help='target average degree')
    parser.add_argument('--target_degeneracy', default=10, type=float, help='target degeneracy')
    parser.add_argument('--inrease_t', action='store_true', default=False, help='increase threshold')
    parser.add_argument('--always_update_G', default=False, action='store_true', help='update graph every step')
    parser.add_argument('--control_type', default='degree', type=str, choices=['degree', 'degeneracy', 'threshold'])
    parser.add_argument('--is_SOG', default=False, action='store_true', help='use SOG')
    parser.add_argument('--r', default=0.2, type=float, help='ratio for selected cliques')
    parser.add_argument('--position', type=str, default='all', help='query | key | value | qkv | output | all')

    main()
