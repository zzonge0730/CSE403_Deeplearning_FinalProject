"""
GPU 메모리 최적화 유틸리티
제한된 GPU에서도 큰 모델을 사용할 수 있도록 도와주는 스크립트
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


def enable_mixed_precision_training():
    """
    Mixed Precision Training (FP16) 활성화
    메모리 사용량을 절반으로 줄이고 속도를 1.5-2배 향상
    """
    scaler = GradScaler()
    return scaler


def train_with_mixed_precision(model, images, labels, criterion, optimizer, scaler, device):
    """
    Mixed Precision으로 학습
    """
    model.train()
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()


def gradient_accumulation_step(model, images, labels, criterion, optimizer, 
                              accumulation_steps=4, device="cuda"):
    """
    Gradient Accumulation
    작은 배치로 큰 배치 효과를 내는 방법
    """
    model.train()
    
    # 여러 배치에 걸쳐 gradient 누적
    loss_sum = 0
    for i in range(accumulation_steps):
        outputs = model(images[i::accumulation_steps])
        labels_batch = labels[i::accumulation_steps]
        loss = criterion(outputs, labels_batch) / accumulation_steps
        loss.backward()
        loss_sum += loss.item()
    
    optimizer.step()
    optimizer.zero_grad()
    
    return loss_sum


def estimate_memory_usage(model_name, batch_size, img_size=224):
    """
    모델별 메모리 사용량 추정
    """
    # 대략적인 추정치 (실제 사용량은 약간 다를 수 있음)
    base_memory = {
        "resnet18": 1.0,
        "resnet50": 2.0,
        "efficientnet_b0": 0.8,
        "efficientnet_b3": 1.5,
        "vit_small_patch16_224": 1.5,
        "vit_base_patch16_224": 3.5,
    }
    
    # 배치 크기에 따른 메모리 증가 (선형적)
    model_memory = base_memory.get(model_name, 2.0)
    batch_memory = model_memory * (batch_size / 16)  # 16 기준
    
    # 추가 오버헤드 (약 20%)
    total_memory = batch_memory * 1.2
    
    return {
        "model": model_name,
        "batch_size": batch_size,
        "estimated_vram_gb": total_memory,
        "recommended_gpu": "6GB+" if total_memory > 3 else "4GB+"
    }


def optimize_batch_size(model, dataloader, device, max_memory_gb=4):
    """
    GPU 메모리에 맞는 최적 배치 크기 찾기
    """
    if device.type != "cuda":
        return 4  # CPU는 작은 배치
    
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    
    batch_sizes = [64, 32, 16, 8, 4]
    
    for batch_size in batch_sizes:
        try:
            # 테스트 배치 생성
            images = torch.randn(batch_size, 3, 224, 224).to(device)
            labels = torch.randint(0, 2, (batch_size,)).to(device)
            
            # Forward pass
            with torch.no_grad():
                _ = model(images)
            
            used_memory = (torch.cuda.memory_allocated() - initial_memory) / 1024**3
            
            if used_memory < max_memory_gb * 0.8:  # 80% 이하 사용
                torch.cuda.empty_cache()
                return batch_size
            
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return 4  # 최소값


if __name__ == "__main__":
    print("GPU 메모리 최적화 유틸리티")
    print("="*50)
    
    # 모델별 메모리 추정
    models = ["resnet18", "resnet50", "efficientnet_b0", "vit_small_patch16_224", "vit_base_patch16_224"]
    batch_sizes = [16, 32]
    
    print("\n모델별 메모리 사용량 추정:")
    print("-"*50)
    for model in models:
        for bs in batch_sizes:
            info = estimate_memory_usage(model, bs)
            print(f"{model:30s} | Batch {bs:2d} | ~{info['estimated_vram_gb']:.1f}GB | {info['recommended_gpu']}")
    
    print("\n최적화 팁:")
    print("-"*50)
    print("1. Mixed Precision (FP16): 메모리 50% 절약, 속도 1.5-2배 향상")
    print("2. Gradient Accumulation: 작은 배치로 큰 배치 효과")
    print("3. 배치 크기 조정: GPU 메모리에 맞게 조절")
    print("4. 모델 경량화: ResNet18, EfficientNet-B0 사용")
