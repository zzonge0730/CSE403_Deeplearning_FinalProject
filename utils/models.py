"""
모델 정의 및 로딩 유틸리티
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


def get_cnn_model(model_name: str = "resnet50", num_classes: int = 2, pretrained: bool = True):
    """
    CNN 모델 로드 (ResNet 또는 EfficientNet)
    
    Args:
        model_name: 모델 이름 ('resnet50', 'efficientnet_b0')
        num_classes: 분류 클래스 수
        pretrained: 사전 학습된 가중치 사용 여부
    
    Returns:
        모델 인스턴스
    """
    if model_name.startswith("resnet"):
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    elif model_name.startswith("efficientnet"):
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"지원하지 않는 CNN 모델: {model_name}")
    
    return model


def get_vit_model(model_name: str = "vit_base_patch16_224", num_classes: int = 2, pretrained: bool = True):
    """
    Vision Transformer 모델 로드
    
    Args:
        model_name: 모델 이름 ('vit_base_patch16_224', 'swin_base_patch4_window7_224')
        num_classes: 분류 클래스 수
        pretrained: 사전 학습된 가중치 사용 여부
    
    Returns:
        모델 인스턴스
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def load_model(model_path: str, model_type: str = "cnn", model_name: str = "resnet50", 
               num_classes: int = 2, device: str = "cuda"):
    """
    저장된 모델 로드
    
    Args:
        model_path: 모델 파일 경로
        model_type: 모델 타입 ('cnn' 또는 'vit')
        model_name: 모델 이름
        num_classes: 분류 클래스 수
        device: 디바이스 ('cuda' 또는 'cpu')
    
    Returns:
        로드된 모델
    """
    if model_type == "cnn":
        model = get_cnn_model(model_name, num_classes, pretrained=False)
    elif model_type == "vit":
        model = get_vit_model(model_name, num_classes, pretrained=False)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    except Exception as e:
        raise RuntimeError(f"모델 로드 중 오류 발생: {e}")
    
    # 체크포인트 형식에 따라 가중치 로드
    try:
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        raise RuntimeError(f"모델 가중치 로드 실패: {e}. 모델 구조가 일치하는지 확인하세요.")
    
    model.to(device)
    model.eval()
    
    return model
