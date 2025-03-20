import os
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import timm

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def load_dino():
    # 加载 DINO 的预训练权重路径
    dino_weights_path = "/home/hfd24/TPTs/pretrained/DINO/dino_vitbase16_pretrain.pth"

    # 加载 ViT 模型（与 DINO 权重匹配）
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)

    # 加载 DINO 的预训练权重
    state_dict = torch.load(dino_weights_path)
    model.load_state_dict(state_dict, strict=False)

    # Define the input resolution for the model and preprocessing steps
    input_resolution = 224  # Example resolution, depends on the specific DINO model


    return model, _transform(input_resolution)
