# AI 이미지 탐지 프로젝트 (CNN vs Vision Transformer)

## 프로젝트 개요
생성형 AI로 생성된 이미지를 탐지하는 Binary Classification 문제를 CNN과 Vision Transformer로 비교 분석합니다.

## 주요 특징
- **모델 비교**: ResNet/EfficientNet (CNN) vs ViT/Swin Transformer
- **Explainability**: Grad-CAM (CNN) 및 Attention Map (ViT) 시각화
- **Robustness 테스트**: 노이즈 및 JPEG 압축에 대한 강건성 평가
- **효율성 분석**: Inference Time, FLOPs, VRAM 사용량 측정

## 프로젝트 구조
```
deeplearning/
├── data/                    # 데이터셋 저장 폴더
├── notebooks/               # 실험 노트북
│   ├── 01_data_pipeline.py
│   ├── 02_train_cnn.py
│   ├── 03_train_vit.py
│   ├── 04_evaluate.py
│   ├── 05_visualize.py
│   ├── 06_robustness.py
│   └── 07_efficiency.py
├── models/                  # 학습된 모델 저장
├── results/                 # 실험 결과 저장
│   ├── metrics/
│   ├── visualizations/
│   └── logs/
├── configs/                 # 설정 파일
│   └── config.yaml
└── utils/                   # 유틸리티 함수
    ├── __init__.py
    ├── models.py
    ├── metrics.py
    ├── visualization.py
    └── robustness.py
```

## 설치 방법
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 준비
데이터셋을 `data/` 폴더에 다음 구조로 배치:
```
data/
├── train/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### 2. 모델 학습
```bash
# CNN 모델 학습
python notebooks/02_train_cnn.py

# ViT 모델 학습
python notebooks/03_train_vit.py
```

### 3. 평가 및 시각화
```bash
# 모델 평가
python notebooks/04_evaluate.py

# Explainability 시각화
python notebooks/05_visualize.py

# Robustness 테스트
python notebooks/06_robustness.py

# 효율성 분석
python notebooks/07_efficiency.py
```

## 실험 결과
실험 결과는 `results/` 폴더에 저장됩니다.
