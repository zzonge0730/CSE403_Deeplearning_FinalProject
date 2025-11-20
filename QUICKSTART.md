# 빠른 시작 가이드

## 1. 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

## 2. 데이터 준비

데이터셋을 `data/` 폴더에 다음 구조로 배치:

```
data/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── fake/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── test/  (선택사항)
    ├── real/
    └── fake/
```

## 3. 설정 파일 확인

`configs/config.yaml` 파일을 열어 데이터 경로와 모델 설정을 확인하세요.

## 4. 모델 학습

### CNN 모델 학습
```bash
python notebooks/02_train_cnn.py
```

### ViT 모델 학습
```bash
python notebooks/03_train_vit.py
```

학습된 모델은 `models/` 폴더에 저장됩니다.

## 5. 모델 평가

```bash
python notebooks/04_evaluate.py
```

결과는 `results/metrics/` 폴더에 저장됩니다.

## 6. Explainability 시각화

```bash
python notebooks/05_visualize.py
```

Grad-CAM 및 Attention Map이 `results/visualizations/` 폴더에 저장됩니다.

## 7. Robustness 테스트

```bash
python notebooks/06_robustness.py
```

노이즈 및 JPEG 압축에 대한 강건성 테스트 결과가 저장됩니다.

## 8. 효율성 측정

```bash
python notebooks/07_efficiency.py
```

Inference Time, FLOPs, VRAM 사용량이 측정됩니다.

## 주의사항

1. **GPU 메모리**: ViT 모델은 CNN보다 더 많은 VRAM을 사용할 수 있습니다. 배치 크기를 조절하세요.
2. **데이터 경로**: `config.yaml`의 데이터 경로가 실제 데이터 위치와 일치하는지 확인하세요.
3. **학습 시간**: 모델 학습은 GPU 환경에서 권장됩니다. CPU만 사용 가능한 경우 에폭 수를 줄이거나 모델 크기를 줄이세요.

## 문제 해결

### ImportError 발생 시
```bash
# 프로젝트 루트에서 실행하는지 확인
cd /root/deeplearning
python notebooks/02_train_cnn.py
```

### CUDA out of memory
- `config.yaml`에서 `batch_size`를 줄이세요 (예: 64 → 32)
- 모델 크기를 줄이세요 (예: resnet50 → resnet18)

### 데이터를 찾을 수 없음
- `config.yaml`의 `data.train_dir` 경로를 확인하세요
- 데이터 폴더 구조가 올바른지 확인하세요
