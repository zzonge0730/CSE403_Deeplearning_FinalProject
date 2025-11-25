"""
Robustness 테스트 스크립트 (노이즈, JPEG 압축)
"""

import os
import sys
import yaml
import torch
import pandas as pd
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import load_model
from utils.robustness import test_noise_robustness, test_jpeg_robustness
from notebooks.data_pipeline import create_dataloaders


def plot_robustness_results(results: dict, test_type: str, save_path: str):
    """Robustness 결과 시각화"""
    metrics = ["accuracy", "f1_score", "precision", "recall"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for model_name, model_results in results.items():
            labels, values = [], []

            for key, value in model_results.items():
                metric_dict = value.get("metrics", value) if isinstance(value, dict) else {}
                if metric not in metric_dict:
                    continue

                if key == "original":
                    labels.append("Original")
                elif key.startswith(test_type):
                    suffix = key.split("_", 1)[1]
                    labels.append(f"{test_type}_{suffix}")
                else:
                    continue
                values.append(metric_dict[metric])

            if labels and values:
                ax.plot(range(len(values)), values, marker="o", label=model_name.upper())
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_xlabel("Test Condition")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} vs {test_type.title()}")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def test_robustness(config_path="configs/config.yaml"):
    """Robustness 테스트 메인 함수"""
    # 설정 로드
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 로더 생성
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=config["data"]["test_dir"] if os.path.exists(config["data"]["test_dir"]) else config["data"]["train_dir"],
        batch_size=config["data"]["batch_size"],
        img_size=config["data"]["img_size"]
    )
    test_loader = val_loader
    
    results = {}
    save_dir = config["robustness"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    # CNN 모델 테스트
    cnn_model_path = os.path.join(config["training"]["save_dir"],
                                  f"cnn_{config['models']['cnn']['name']}_best.pth")
    if os.path.exists(cnn_model_path):
        print("\n" + "="*50)
        print("Testing CNN Robustness")
        print("="*50)
        
        cnn_model = load_model(
            cnn_model_path,
            model_type="cnn",
            model_name=config["models"]["cnn"]["name"],
            num_classes=config["models"]["cnn"]["num_classes"],
            device=device
        )
        
        # 노이즈 테스트
        print("\nTesting noise robustness...")
        noise_results = test_noise_robustness(
            cnn_model, test_loader, config["robustness"]["noise_levels"], device
        )
        results["cnn"] = {"noise": noise_results}
        
        # JPEG 압축 테스트
        print("\nTesting JPEG compression robustness...")
        jpeg_results = test_jpeg_robustness(
            cnn_model, test_loader, config["robustness"]["jpeg_qualities"], device
        )
        results["cnn"]["jpeg"] = jpeg_results
    
    # ViT 모델 테스트
    vit_model_path = os.path.join(config["training"]["save_dir"],
                                 f"vit_{config['models']['vit']['name']}_best.pth")
    if os.path.exists(vit_model_path):
        print("\n" + "="*50)
        print("Testing ViT Robustness")
        print("="*50)
        
        vit_model = load_model(
            vit_model_path,
            model_type="vit",
            model_name=config["models"]["vit"]["name"],
            num_classes=config["models"]["vit"]["num_classes"],
            device=device
        )
        
        # 노이즈 테스트
        print("\nTesting noise robustness...")
        noise_results = test_noise_robustness(
            vit_model, test_loader, config["robustness"]["noise_levels"], device
        )
        results["vit"] = {"noise": noise_results}
        
        # JPEG 압축 테스트
        print("\nTesting JPEG compression robustness...")
        jpeg_results = test_jpeg_robustness(
            vit_model, test_loader, config["robustness"]["jpeg_qualities"], device
        )
        results["vit"]["jpeg"] = jpeg_results
    
    # 결과 저장 및 시각화
    if len(results) > 0:
        # JSON 저장
        with open(os.path.join(save_dir, "robustness_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # 노이즈 결과 시각화
        noise_results_dict = {k: v["noise"] for k, v in results.items()}
        plot_robustness_results(noise_results_dict, "noise",
                               os.path.join(save_dir, "noise_robustness.png"))
        
        # JPEG 결과 시각화
        jpeg_results_dict = {k: v["jpeg"] for k, v in results.items()}
        plot_robustness_results(jpeg_results_dict, "jpeg",
                               os.path.join(save_dir, "jpeg_robustness.png"))
        
        print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    test_robustness()
