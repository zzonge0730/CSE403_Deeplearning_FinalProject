"""
시각화 유틸리티 (Grad-CAM, Attention Map 등)
"""

import os
import warnings
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - matplotlib optional in Kaggle runtime
    plt = None  # type: ignore
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None


def visualize_gradcam(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor,
                     target_layers: List[torch.nn.Module], class_names: List[str] = ["real", "fake"],
                     device: str = "cuda", save_path: Optional[str] = None) -> np.ndarray:
    """
    Grad-CAM 시각화
    
    Args:
        model: 모델
        images: 입력 이미지 텐서 (B, C, H, W)
        labels: 실제 라벨
        target_layers: Grad-CAM을 적용할 레이어 리스트
        class_names: 클래스 이름
        device: 디바이스
        save_path: 저장 경로 (선택사항)
    
    Returns:
        시각화된 이미지 배열
    """
    model.eval()
    images = images.to(device)

    # pytorch-grad-cam은 device 인자를 직접 받지 않고 use_cuda로 제어한다.
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # Grad-CAM 생성
    gradcam_images = []
    cam_instance = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    with cam_instance as cam_ctx:
        for i in range(images.shape[0]):
            # 예측 클래스에 대한 Grad-CAM
            with torch.no_grad():
                outputs = model(images[i:i+1])
                _, pred = torch.max(outputs, 1)
                pred_class = pred[0].item()
            
            targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = cam_ctx(input_tensor=images[i:i+1], targets=targets)[0]
            
            # 이미지 전처리 (정규화 해제)
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # 0-1 정규화
            
            # RGB로 변환 (필요한 경우)
            if img.shape[2] == 3:
                visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            else:
                visualization = show_cam_on_image(img[:, :, 0], grayscale_cam, use_rgb=False)
            
            gradcam_images.append(visualization)
    
    gradcam_images = np.array(gradcam_images)
    
    if save_path:
        save_gradcam_images(gradcam_images, labels, class_names, save_path)
    
    return gradcam_images


def visualize_vit_attention(model: torch.nn.Module, images: torch.Tensor, 
                           class_names: List[str] = ["real", "fake"],
                           device: str = "cuda", save_path: Optional[str] = None) -> np.ndarray:
    """
    ViT Attention Map 시각화
    
    Args:
        model: ViT 모델
        images: 입력 이미지 텐서 (B, C, H, W)
        class_names: 클래스 이름
        device: 디바이스
        save_path: 저장 경로 (선택사항)
    
    Returns:
        시각화된 이미지 배열
    """
    model.eval()
    images = images.to(device)
    
    attention_images = []
    
    with torch.no_grad():
        # ViT의 attention 가중치 추출
        # timm의 ViT 모델은 forward_features 메서드를 제공
        if hasattr(model, "forward_features"):
            features = model.forward_features(images)
        else:
            # 직접 attention 추출
            x = model.patch_embed(images)
            x = model._pos_embed(x)
            x = model.norm_pre(x)
            
            # Attention 블록들을 통과하며 attention 가중치 수집
            attentions = []
            for block in model.blocks:
                x, attn = block.attn(x, return_attention=True)
                attentions.append(attn)
            
            # 마지막 attention map 사용
            if len(attentions) > 0:
                attn_map = attentions[-1].mean(dim=1)[:, 0, 1:]  # CLS 토큰 제외
            else:
                # 간단한 대체 방법
                outputs = model(images)
                attn_map = None
        
        # Attention map을 이미지 크기로 리샘플링
        for i in range(images.shape[0]):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            
            if attn_map is not None:
                # Patch 크기 계산 (일반적으로 16x16)
                patch_size = 16
                h, w = img.shape[:2]
                num_patches_h = h // patch_size
                num_patches_w = w // patch_size
                
                # Attention map 리샘플링
                attn = attn_map[i].cpu().numpy()
                attn = attn.reshape(num_patches_h, num_patches_w)
                attn = cv2.resize(attn, (w, h))
                attn = (attn - attn.min()) / (attn.max() - attn.min())
                
                # 히트맵 생성
                heatmap = cv2.applyColorMap(np.uint8(255 * attn), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                visualization = heatmap * 0.4 + img * 0.6
            else:
                visualization = img
            
            attention_images.append(visualization)
    
    attention_images = np.array(attention_images)
    
    if save_path:
        save_attention_images(attention_images, class_names, save_path)
    
    return attention_images


def _ensure_matplotlib(feature: str) -> bool:
    """Check whether matplotlib is available before plotting."""
    if plt is not None:
        return True
    warnings.warn(
        f"Matplotlib is unavailable ({_MATPLOTLIB_IMPORT_ERROR}); "
        f"skipping {feature}.",
        RuntimeWarning,
    )
    return False


def save_gradcam_images(images: np.ndarray, labels: torch.Tensor,
                        class_names: List[str], save_path: str):
    """Grad-CAM 이미지 저장"""
    if not _ensure_matplotlib("Grad-CAM visualization"):
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, len(images) // 2 + len(images) % 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (img, label) in enumerate(zip(images, labels)):
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {class_names[label.item()]}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_attention_images(images: np.ndarray, class_names: List[str], save_path: str):
    """Attention 이미지 저장"""
    if not _ensure_matplotlib("attention visualization"):
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, len(images) // 2 + len(images) % 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(f"Attention Map {i+1}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: List[str] = ["real", "fake"],
                         save_path: Optional[str] = None):
    """Confusion Matrix 시각화"""
    if not _ensure_matplotlib("confusion matrix plotting"):
        return
    from sklearn.metrics import confusion_matrix

    try:
        import seaborn as sns
    except Exception as exc:  # pragma: no cover - seaborn optional
        warnings.warn(
            f"Seaborn is unavailable ({exc}); skipping confusion matrix plot.",
            RuntimeWarning,
        )
        return

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """학습 히스토리 시각화"""
    if not _ensure_matplotlib("training history plotting"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close()
