"""
Robustness 테스트 유틸리티 (노이즈, JPEG 압축 등)
"""

import torch
import numpy as np
from PIL import Image
import io
from typing import Dict, List
import torchvision.transforms as transforms


def add_gaussian_noise(image: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """
    가우시안 노이즈 추가
    
    Args:
        image: 입력 이미지 텐서 (C, H, W)
        noise_level: 노이즈 레벨 (표준편차)
    
    Returns:
        노이즈가 추가된 이미지
    """
    noise = torch.randn_like(image) * noise_level
    noisy_image = torch.clamp(image + noise, 0, 1)
    return noisy_image


def apply_jpeg_compression(image: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    JPEG 압축 적용
    
    Args:
        image: 입력 이미지 텐서 (C, H, W)
        quality: JPEG 품질 (1-100)
    
    Returns:
        압축된 이미지
    """
    # 텐서를 PIL Image로 변환
    img_np = image.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np)
    
    # JPEG 압축
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    
    # 다시 텐서로 변환
    compressed_np = np.array(compressed_image).astype(np.float32) / 255.0
    compressed_tensor = torch.from_numpy(compressed_np).permute(2, 0, 1)
    
    return compressed_tensor


def test_noise_robustness(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                         noise_levels: List[float], device: str = "cuda") -> Dict[str, Dict[str, float]]:
    """
    노이즈에 대한 강건성 테스트
    
    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        noise_levels: 테스트할 노이즈 레벨 리스트
        device: 디바이스
    
    Returns:
        노이즈 레벨별 메트릭 딕셔너리
    """
    from utils.metrics import evaluate_model
    
    model.eval()
    results = {}
    
    # 원본 성능 측정
    original_metrics, _, _ = evaluate_model(model, dataloader, device)
    results["original"] = original_metrics
    
    # 각 노이즈 레벨에 대해 테스트
    for noise_level in noise_levels:
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                noisy_images = []
                for img in images:
                    noisy_img = add_gaussian_noise(img, noise_level)
                    noisy_images.append(noisy_img)
                
                noisy_images = torch.stack(noisy_images).to(device)
                labels = labels.to(device)
                
                outputs = model(noisy_images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        from utils.metrics import calculate_metrics
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        results[f"noise_{noise_level}"] = metrics
    
    return results


def test_jpeg_robustness(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                        jpeg_qualities: List[int], device: str = "cuda") -> Dict[str, Dict[str, float]]:
    """
    JPEG 압축에 대한 강건성 테스트
    
    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        jpeg_qualities: 테스트할 JPEG 품질 리스트
        device: 디바이스
    
    Returns:
        JPEG 품질별 메트릭 딕셔너리
    """
    model.eval()
    results = {}
    
    # 원본 성능 측정
    from utils.metrics import evaluate_model
    original_metrics, _, _ = evaluate_model(model, dataloader, device)
    results["original"] = original_metrics
    
    # 각 JPEG 품질에 대해 테스트
    for quality in jpeg_qualities:
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                compressed_images = []
                for img in images:
                    compressed_img = apply_jpeg_compression(img, quality)
                    compressed_images.append(compressed_img)
                
                compressed_images = torch.stack(compressed_images).to(device)
                labels = labels.to(device)
                
                outputs = model(compressed_images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        from utils.metrics import calculate_metrics
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        results[f"jpeg_{quality}"] = metrics
    
    return results
