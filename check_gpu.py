"""
GPU 및 메모리 확인 스크립트
"""

import torch
import sys

def check_environment():
    """환경 확인"""
    print("="*50)
    print("환경 확인")
    print("="*50)
    
    # CUDA 사용 가능 여부
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # 현재 사용 중인 메모리
            torch.cuda.reset_peak_memory_stats(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  할당된 메모리: {allocated:.2f} GB")
            print(f"  예약된 메모리: {reserved:.2f} GB")
    else:
        print("\n⚠️  CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        print("   학습 시간이 매우 오래 걸릴 수 있습니다.")
    
    # 모델 크기 추정
    print("\n" + "="*50)
    print("모델 크기 추정")
    print("="*50)
    
    model_sizes = {
        "resnet18": {"params": "11M", "size_mb": 45, "vram_gb": 1.5},
        "resnet50": {"params": "25M", "size_mb": 98, "vram_gb": 2.5},
        "efficientnet_b0": {"params": "5M", "size_mb": 20, "vram_gb": 1.0},
        "vit_small_patch16_224": {"params": "22M", "size_mb": 88, "vram_gb": 2.0},
        "vit_base_patch16_224": {"params": "86M", "size_mb": 330, "vram_gb": 4.0},
    }
    
    print("\n권장 모델 (WSL/제한된 GPU 메모리):")
    print("-" * 50)
    for model_name, info in [("resnet18", model_sizes["resnet18"]), 
                             ("efficientnet_b0", model_sizes["efficientnet_b0"]),
                             ("vit_small_patch16_224", model_sizes["vit_small_patch16_224"])]:
        print(f"{model_name:30s} | Params: {info['params']:8s} | Size: {info['size_mb']:4d}MB | VRAM: ~{info['vram_gb']:.1f}GB")
    
    print("\n큰 모델 (Colab/충분한 GPU 메모리 필요):")
    print("-" * 50)
    for model_name, info in [("resnet50", model_sizes["resnet50"]), 
                             ("vit_base_patch16_224", model_sizes["vit_base_patch16_224"])]:
        print(f"{model_name:30s} | Params: {info['params']:8s} | Size: {info['size_mb']:4d}MB | VRAM: ~{info['vram_gb']:.1f}GB")
    
    # 권장사항
    print("\n" + "="*50)
    print("권장사항")
    print("="*50)
    
    if cuda_available:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n현재 GPU 메모리: {gpu_memory_gb:.1f} GB")
        
        if gpu_memory_gb < 4:
            print("⚠️  GPU 메모리가 4GB 미만입니다.")
            print("   → config_wsl.yaml 사용 권장 (resnet18, vit_small)")
            print("   → 배치 크기를 8-16으로 설정")
        elif gpu_memory_gb < 8:
            print("✓ GPU 메모리가 4-8GB입니다.")
            print("   → config_wsl.yaml 사용 가능 (resnet18, vit_small)")
            print("   → 배치 크기를 16-32로 설정 가능")
        else:
            print("✓ GPU 메모리가 충분합니다 (8GB 이상).")
            print("   → 기본 config.yaml 사용 가능")
            print("   → 더 큰 모델도 사용 가능")
    else:
        print("\n⚠️  CPU 모드:")
        print("   → 매우 작은 모델만 사용 (resnet18)")
        print("   → 배치 크기를 4-8로 설정")
        print("   → Colab 사용을 강력히 권장합니다")
    
    print("\n" + "="*50)
    print("사용 방법")
    print("="*50)
    print("\nWSL에서 실행:")
    print("  python notebooks/02_train_cnn.py --config configs/config_wsl.yaml")
    print("\n또는 코드에서 직접:")
    print("  train_cnn(config_path='configs/config_wsl.yaml')")
    
    return cuda_available

if __name__ == "__main__":
    check_environment()
