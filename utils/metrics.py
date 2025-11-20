"""
평가 메트릭 계산 유틸리티
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, List, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    분류 메트릭 계산
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
    
    Returns:
        메트릭 딕셔너리
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
    }
    
    return metrics


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Confusion Matrix 계산
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
    
    Returns:
        Confusion Matrix (2x2)
    """
    return confusion_matrix(y_true, y_pred)


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                   device: str = "cuda") -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    모델 평가
    
    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        device: 디바이스
    
    Returns:
        (메트릭 딕셔너리, 실제 라벨, 예측 라벨)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds)
    
    return metrics, all_labels, all_preds


def calculate_class_wise_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 class_names: List[str] = ["real", "fake"]) -> Dict[str, Dict[str, float]]:
    """
    클래스별 메트릭 계산
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        class_names: 클래스 이름 리스트
    
    Returns:
        클래스별 메트릭 딕셔너리
    """
    cm = confusion_matrix(y_true, y_pred)
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(tp + fn)
        }
    
    return class_metrics
