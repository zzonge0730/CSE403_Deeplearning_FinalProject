"""
효율성 측정 스크립트 (Inference Time, FLOPs, VRAM)
"""

import os
import sys
import yaml
import torch
import time
import pandas as pd
import json
from thop import profile, clever_format

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import load_model
from notebooks.data_pipeline import create_dataloaders


def measure_inference_time(model, dataloader, device, num_runs=100, warmup_runs=10):
    """추론 시간 측정"""
    model.eval()
    
    # Warmup
    dummy_input = next(iter(dataloader))[0][:1].to(device)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # GPU 동기화
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # 측정
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(dummy_input)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ms로 변환
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return {
        "mean": avg_time,
        "std": std_time,
        "min": min(times),
        "max": max(times)
    }


def measure_flops(model, input_size=(1, 3, 224, 224), device="cuda"):
    """FLOPs 측정"""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    
    return {
        "flops": flops,
        "params": params
    }


def measure_vram_usage(model, device):
    """VRAM 사용량 측정"""
    if device.type != "cuda":
        return {"vram_mb": None, "message": "CUDA not available"}
    
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    
    # 모델 메모리
    model_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Peak 메모리
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    return {
        "model_memory_mb": model_memory,
        "peak_memory_mb": peak_memory
    }


def measure_efficiency(config_path="configs/config.yaml"):
    """효율성 측정 메인 함수"""
    # 설정 로드
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 로더 생성 (더미용)
    _, test_loader, _ = create_dataloaders(
        data_dir=config["data"]["test_dir"] if os.path.exists(config["data"]["test_dir"]) else config["data"]["train_dir"],
        batch_size=1,
        img_size=config["data"]["img_size"]
    )
    
    results = {}
    save_dir = config["efficiency"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    # CNN 모델 측정
    cnn_model_path = os.path.join(config["training"]["save_dir"],
                                  f"cnn_{config['models']['cnn']['name']}_best.pth")
    if os.path.exists(cnn_model_path):
        print("\n" + "="*50)
        print("Measuring CNN Efficiency")
        print("="*50)
        
        cnn_model = load_model(
            cnn_model_path,
            model_type="cnn",
            model_name=config["models"]["cnn"]["name"],
            num_classes=config["models"]["cnn"]["num_classes"],
            device=device
        )
        
        # Inference Time
        print("Measuring inference time...")
        inference_time = measure_inference_time(
            cnn_model, test_loader, device,
            num_runs=config["efficiency"]["num_runs"],
            warmup_runs=config["efficiency"]["warmup_runs"]
        )
        print(f"Mean inference time: {inference_time['mean']:.2f} ms")
        
        # FLOPs
        print("Measuring FLOPs...")
        flops_info = measure_flops(cnn_model, (1, 3, config["data"]["img_size"], config["data"]["img_size"]), device)
        print(f"FLOPs: {flops_info['flops']}, Params: {flops_info['params']}")
        
        # VRAM
        print("Measuring VRAM usage...")
        vram_info = measure_vram_usage(cnn_model, device)
        print(f"VRAM: {vram_info}")
        
        results["cnn"] = {
            "inference_time": inference_time,
            "flops": flops_info,
            "vram": vram_info
        }
    
    # ViT 모델 측정
    vit_model_path = os.path.join(config["training"]["save_dir"],
                                 f"vit_{config['models']['vit']['name']}_best.pth")
    if os.path.exists(vit_model_path):
        print("\n" + "="*50)
        print("Measuring ViT Efficiency")
        print("="*50)
        
        vit_model = load_model(
            vit_model_path,
            model_type="vit",
            model_name=config["models"]["vit"]["name"],
            num_classes=config["models"]["vit"]["num_classes"],
            device=device
        )
        
        # Inference Time
        print("Measuring inference time...")
        inference_time = measure_inference_time(
            vit_model, test_loader, device,
            num_runs=config["efficiency"]["num_runs"],
            warmup_runs=config["efficiency"]["warmup_runs"]
        )
        print(f"Mean inference time: {inference_time['mean']:.2f} ms")
        
        # FLOPs
        print("Measuring FLOPs...")
        flops_info = measure_flops(vit_model, (1, 3, config["data"]["img_size"], config["data"]["img_size"]), device)
        print(f"FLOPs: {flops_info['flops']}, Params: {flops_info['params']}")
        
        # VRAM
        print("Measuring VRAM usage...")
        vram_info = measure_vram_usage(vit_model, device)
        print(f"VRAM: {vram_info}")
        
        results["vit"] = {
            "inference_time": inference_time,
            "flops": flops_info,
            "vram": vram_info
        }
    
    # 결과 저장
    if len(results) > 0:
        # JSON 저장
        with open(os.path.join(save_dir, "efficiency_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # 비교 테이블 생성
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                "Model": model_name.upper(),
                "Inference Time (ms)": f"{result['inference_time']['mean']:.2f} ± {result['inference_time']['std']:.2f}",
                "FLOPs": result["flops"]["flops"],
                "Params": result["flops"]["params"],
                "VRAM (MB)": result["vram"].get("peak_memory_mb", "N/A")
            })
        
        df = pd.DataFrame(comparison_data)
        print("\n" + "="*50)
        print("Efficiency Comparison")
        print("="*50)
        print(df.to_string(index=False))
        
        df.to_csv(os.path.join(save_dir, "efficiency_comparison.csv"), index=False)
        print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    measure_efficiency()
