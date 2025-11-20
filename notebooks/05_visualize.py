"""
Explainability 시각화 스크립트 (Grad-CAM, Attention Map)
"""

import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import load_model
from utils.visualization import visualize_gradcam, visualize_vit_attention
from notebooks.data_pipeline import create_dataloaders


def visualize_explainability(config_path="configs/config.yaml"):
    """Explainability 시각화"""
    # 설정 로드
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 로더 생성
    _, test_loader, class_names = create_dataloaders(
        data_dir=config["data"]["test_dir"] if os.path.exists(config["data"]["test_dir"]) else config["data"]["train_dir"],
        batch_size=config["visualization"]["num_samples"],
        img_size=config["data"]["img_size"]
    )
    
    # 샘플 데이터 가져오기
    images, labels = next(iter(test_loader))
    images = images[:config["visualization"]["num_samples"]]
    labels = labels[:config["visualization"]["num_samples"]]
    
    save_dir = config["visualization"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    # CNN Grad-CAM 시각화
    cnn_model_path = os.path.join(config["training"]["save_dir"],
                                 f"cnn_{config['models']['cnn']['name']}_best.pth")
    if os.path.exists(cnn_model_path):
        print("\nGenerating Grad-CAM for CNN...")
        
        cnn_model = load_model(
            cnn_model_path,
            model_type="cnn",
            model_name=config["models"]["cnn"]["name"],
            num_classes=config["models"]["cnn"]["num_classes"],
            device=device
        )
        
        # ResNet의 경우 layer4 사용
        if config["models"]["cnn"]["name"].startswith("resnet"):
            target_layers = [cnn_model.layer4]
        elif config["models"]["cnn"]["name"].startswith("efficientnet"):
            # EfficientNet의 경우 마지막 블록 사용
            target_layers = [cnn_model.blocks[-1]]
        else:
            print(f"Warning: Unknown CNN architecture, using default layer")
            target_layers = [list(cnn_model.children())[-2]]
        
        gradcam_images = visualize_gradcam(
            cnn_model, images, labels, target_layers, class_names, device,
            save_path=os.path.join(save_dir, "cnn_gradcam.png")
        )
        print(f"Grad-CAM saved to {save_dir}/cnn_gradcam.png")
    
    # ViT Attention Map 시각화
    vit_model_path = os.path.join(config["training"]["save_dir"],
                                 f"vit_{config['models']['vit']['name']}_best.pth")
    if os.path.exists(vit_model_path):
        print("\nGenerating Attention Map for ViT...")
        
        vit_model = load_model(
            vit_model_path,
            model_type="vit",
            model_name=config["models"]["vit"]["name"],
            num_classes=config["models"]["vit"]["num_classes"],
            device=device
        )
        
        attention_images = visualize_vit_attention(
            vit_model, images, class_names, device,
            save_path=os.path.join(save_dir, "vit_attention.png")
        )
        print(f"Attention Map saved to {save_dir}/vit_attention.png")
    
    print("\nVisualization completed!")


if __name__ == "__main__":
    visualize_explainability()
