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

### 기본 설치
```bash
pip install -r requirements.txt
```

### Kaggle 데이터셋 사용 시
```bash
pip install kaggle
# Kaggle API 토큰 설정 필요 (자세한 내용은 KAGGLE_SETUP.md 참고)
```

## 사용 방법

### 로컬 환경 (WSL/Linux)

#### 1. 데이터 준비

**방법 A: Kaggle에서 자동 다운로드 (권장)**
```bash
# Kaggle API 설정 후
python scripts/download_kaggle.py --dataset sattyam96/realifake
```

**방법 B: 기존 데이터 준비**
```bash
# Realifake 폴더가 있는 경우
python scripts/prepare_data.py --source /path/to/Realifake --target data/train --mode copy
```

**방법 C: 수동 준비**
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

자세한 내용은 `DATA_PREPARATION.md` 및 `KAGGLE_SETUP.md` 참고

#### 2. 모델 학습
```bash
# CNN 모델 학습
python notebooks/02_train_cnn.py

# ViT 모델 학습 (경량 설정 사용)
python notebooks/02_train_cnn.py --config configs/config_wsl.yaml
python notebooks/03_train_vit.py --config configs/config_wsl.yaml
```

#### 3. 평가 및 시각화
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

### Google Colab 환경

#### 빠른 시작
1. `colab_notebook.ipynb` 파일을 Colab에 업로드
2. Google Drive에 프로젝트 폴더 업로드
3. 노트북 실행

#### 상세 가이드
`COLAB_GUIDE.md` 파일을 참고하세요.

#### 주요 단계
```python
# 1. Drive 마운트
from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/deeplearning')

# 2. 패키지 설치
!pip install -r requirements.txt

# 3. 학습 실행
!python notebooks/02_train_cnn.py
!python notebooks/03_train_vit.py
```

## 실험 결과
실험 결과는 `results/` 폴더에 저장됩니다.
