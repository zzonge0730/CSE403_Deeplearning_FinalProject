"""
모델 평가 스크립트
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import load_model
from utils.metrics import evaluate_model, calculate_confusion_matrix, calculate_class_wise_metrics
from utils.visualization import plot_confusion_matrix
from notebooks.data_pipeline import create_dataloaders


def evaluate_all_models(config_path="configs/config.yaml"):
    """모든 모델 평가"""
    # 설정 로드
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 로더 생성 (test_loader 사용 - 진짜 테스트셋!)
    # 주의: 학습 시와 동일한 시드(42)로 분할하므로 같은 test set이 생성됩니다
    _, _, test_loader, class_names = create_dataloaders(
        data_dir=config["data"]["test_dir"] if os.path.exists(config["data"]["test_dir"]) else config["data"]["train_dir"],
        batch_size=config["data"]["batch_size"],
        img_size=config["data"]["img_size"]
    )
    
    # 데이터 로더 None 체크
    if test_loader is None or class_names is None:
        print("❌ 테스트 데이터 로더 생성 실패. 프로그램을 종료합니다.")
        return
    
    print(f"\n🔍 Final Evaluation on {len(test_loader.dataset)} Test Images (Unseen Data)")
    
    results = {}
    
    # CNN 모델 평가
    cnn_model_path = os.path.join(config["training"]["save_dir"], 
                                  f"cnn_{config['models']['cnn']['name']}_best.pth")
    if os.path.exists(cnn_model_path):
        print("\n" + "="*50)
        print("Evaluating CNN Model")
        print("="*50)
        
        cnn_model = load_model(
            cnn_model_path,
            model_type="cnn",
            model_name=config["models"]["cnn"]["name"],
            num_classes=config["models"]["cnn"]["num_classes"],
            device=device
        )
        
        metrics, y_true, y_pred = evaluate_model(cnn_model, test_loader, device)
        cm = calculate_confusion_matrix(y_true, y_pred)
        class_metrics = calculate_class_wise_metrics(y_true, y_pred, class_names)
        
        results["cnn"] = {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "class_wise_metrics": class_metrics
        }
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        # Confusion Matrix 저장
        save_dir = config["evaluation"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        plot_confusion_matrix(y_true, y_pred, class_names, 
                            os.path.join(save_dir, "cnn_confusion_matrix.png"))
    else:
        print(f"CNN model not found: {cnn_model_path}")
    
    # ViT 모델 평가
    vit_model_path = os.path.join(config["training"]["save_dir"],
                                  f"vit_{config['models']['vit']['name']}_best.pth")
    if os.path.exists(vit_model_path):
        print("\n" + "="*50)
        print("Evaluating ViT Model")
        print("="*50)
        
        vit_model = load_model(
            vit_model_path,
            model_type="vit",
            model_name=config["models"]["vit"]["name"],
            num_classes=config["models"]["vit"]["num_classes"],
            device=device
        )
        
        metrics, y_true, y_pred = evaluate_model(vit_model, test_loader, device)
        cm = calculate_confusion_matrix(y_true, y_pred)
        class_metrics = calculate_class_wise_metrics(y_true, y_pred, class_names)
        
        results["vit"] = {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "class_wise_metrics": class_metrics
        }
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        # Confusion Matrix 저장
        save_dir = config["evaluation"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        plot_confusion_matrix(y_true, y_pred, class_names,
                            os.path.join(save_dir, "vit_confusion_matrix.png"))
    else:
        print(f"ViT model not found: {vit_model_path}")
    
    # 결과 비교 및 저장
    if len(results) > 0:
        print("\n" + "="*50)
        print("Model Comparison")
        print("="*50)
        
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                "Model": model_name.upper(),
                "Accuracy": result["metrics"]["accuracy"],
                "F1-Score": result["metrics"]["f1_score"],
                "Precision": result["metrics"]["precision"],
                "Recall": result["metrics"]["recall"]
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # 결과 저장
        save_dir = config["evaluation"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        
        # JSON 저장
        with open(os.path.join(save_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # CSV 저장
        df.to_csv(os.path.join(save_dir, "model_comparison.csv"), index=False)
        
        print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    evaluate_all_models()
