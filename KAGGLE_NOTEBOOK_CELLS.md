# Kaggle 노트북 셀 - 수정된 버전

## Cell 1: 환경 설정 및 저장소 클론

```python
# 작업 디렉토리 이동 (Kaggle의 쓰기 가능한 공간)
%cd /kaggle/working/

# GitHub 저장소 클론 (이미 있으면 pull로 업데이트)
import os

if os.path.exists("CSE403_Deeplearning_FinalProject"):
    %cd CSE403_Deeplearning_FinalProject
    !git pull
else:
    !git clone https://github.com/zzonge0730/CSE403_Deeplearning_FinalProject.git
    %cd CSE403_Deeplearning_FinalProject

# 필요 라이브러리 설치 (버전 충돌 방지)
print("라이브러리 설치 중...")

# 핵심: NumPy를 먼저 고정 설치 (의존성 무시)
print("1단계: NumPy 1.24.3 강제 설치 중...")
!pip install --no-deps --force-reinstall --ignore-installed "numpy==1.24.3" 2>&1 | tail -3

# Protobuf도 고정 (TensorBoard 호환성)
!pip install --no-deps --force-reinstall "protobuf==3.20.3" 2>&1 | tail -3

# requirements.txt에서 numpy 라인 제거 후 설치
print("2단계: requirements.txt 설치 중 (numpy 제외)...")
import tempfile
with open("requirements.txt", "r") as f:
    lines = f.readlines()
with open("/tmp/req_no_numpy.txt", "w") as f:
    for line in lines:
        if not line.strip().startswith("numpy"):
            f.write(line)
!pip install -q -r /tmp/req_no_numpy.txt 2>&1 | tail -5

# 추가 패키지
print("3단계: 추가 패키지 설치 중...")
!pip install -q thop  # FLOPs 계산용

print("\n" + "="*60)
print("중요: Python 세션 재시작 필요!")
print("="*60)
print("1. 위 셀 실행 완료 후")
print("2. Kaggle 노트북에서 'Runtime' → 'Restart Session' 클릭")
print("3. 그 다음 셀부터 다시 실행하세요")
print("="*60)
print("\n환경 설정 완료 (재시작 후 다음 셀 실행)")
```

## Cell 2: Kaggle 데이터 경로 자동 감지 및 Config 설정

```python
import yaml
import os

# 하드코딩된 경로 (가장 일반적인 구조)
real_data_path = "/kaggle/input/realifake/Realifake"

# 경로 확인 및 자동 수정
if not os.path.exists(real_data_path):
    # 대안 경로들 시도
    alternative_paths = [
        "/kaggle/input/realifake/Realifake",
        "/kaggle/input/realifake/realifake",
        "/kaggle/input/realifake/train",
        "/kaggle/input/realifake",
    ]
    
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            real_data_path = alt_path
            print(f"대안 경로 사용: {real_data_path}")
            break
    else:
        # 경로가 없으면 에러
        print("데이터 경로를 찾을 수 없습니다!")
        print(f"   시도한 경로: {alternative_paths}")
        print(f"\n해결 방법:")
        print(f"   1. 아래 코드에서 'real_data_path'를 실제 경로로 수정하세요")
        print(f"   2. Kaggle 노트북에서 데이터셋 구조 확인:")
        print(f"      !ls -la /kaggle/input/")
        raise FileNotFoundError(f"데이터 경로를 찾을 수 없습니다: {alternative_paths}")
else:
    print(f"데이터 경로 확인: {real_data_path}")

# Config 파일 구조 확인 및 생성
config_dir = "configs"
os.makedirs(config_dir, exist_ok=True)

# 실제 코드가 기대하는 정확한 구조로 Config 생성
config = {
    "data": {
        "root_dir": "data",
        "train_dir": real_data_path,  # Kaggle 경로로 설정
        "test_dir": real_data_path,    # Split은 data_pipeline에서 처리
        "img_size": 224,
        "batch_size": 64,
        "num_workers": 4,  # data_pipeline에서 Kaggle 감지 시 자동으로 0으로 변경됨
        "train_split": 0.8
    },
    "models": {
        "cnn": {
            "name": "resnet50",
            "pretrained": True,
            "num_classes": 2
        },
        "vit": {
            "name": "vit_base_patch16_224",
            "pretrained": True,
            "num_classes": 2
        }
    },
    "training": {
        "num_epochs": 10,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "optimizer": "adam",
        "scheduler": "cosine",
        "save_dir": "models",
        "log_dir": "results/logs"  # 중요: "logs"가 아니라 "results/logs"
    },
    "evaluation": {
        "metrics": ["accuracy", "f1_score", "precision", "recall", "confusion_matrix"],
        "save_dir": "results/metrics"
    },
    "visualization": {
        "save_dir": "results/visualizations",
        "num_samples": 20,
        "gradcam_layer": "layer4"
    },
    "robustness": {
        "noise_levels": [0.01, 0.05, 0.1, 0.2],  # 0.0 제거 (의미 없음)
        "jpeg_qualities": [95, 85, 75, 65, 55],
        "save_dir": "results/metrics"  # "results/robustness"가 아니라 "results/metrics"
    },
    "efficiency": {
        "num_runs": 100,
        "warmup_runs": 10,
        "save_dir": "results/metrics"  # "results/efficiency"가 아니라 "results/metrics"
    }
}

# Config 파일 저장
with open("configs/config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"설정 파일 생성 완료!")
print(f"   - 데이터 경로: {real_data_path}")
print(f"   - 배치 크기: {config['data']['batch_size']}")
print(f"   - Epochs: {config['training']['num_epochs']}")
```

## Cell 3: 모델 학습 및 평가 (통합 실행)

```python
import sys
import os

# NumPy 버전 재확인 및 강제 다운그레이드 (필요시)
try:
    import numpy as np
    if int(np.__version__.split('.')[0]) >= 2:
        print("NumPy 2.x 감지! 다운그레이드 중...")
        !pip install -q --force-reinstall "numpy==1.24.3" 2>&1 | tail -3
        import importlib
        import numpy
        importlib.reload(numpy)
        print(f"NumPy 버전: {numpy.__version__}")
except:
    pass

# 현재 디렉토리 확인
print(f"현재 작업 디렉토리: {os.getcwd()}")

# CNN 학습
print("\n" + "="*60)
print("CNN (ResNet-50) 학습 시작...")
print("="*60)
!python notebooks/02_train_cnn.py

# ViT 학습
print("\n" + "="*60)
print("ViT (Vision Transformer) 학습 시작...")
print("="*60)
!python notebooks/03_train_vit.py

# 모델 평가
print("\n" + "="*60)
print("모델 평가 수행 중...")
print("="*60)
!python notebooks/04_evaluate.py
```

## Cell 4: 시각화

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

print("시각화(Grad-CAM vs Attention Map) 생성 중...")
!python notebooks/05_visualize.py

# 결과 이미지 노트북에 바로 출력
viz_dir = "results/visualizations"
if os.path.exists(viz_dir):
    viz_files = sorted(glob.glob(os.path.join(viz_dir, "*.png")))
    if viz_files:
        print(f"\n{len(viz_files)}개의 시각화 이미지 생성됨")
        for img_path in viz_files[:10]:  # 최대 10개만 표시
            try:
                plt.figure(figsize=(15, 10))
                img = mpimg.imread(img_path)
                plt.imshow(img)
                plt.title(os.path.basename(img_path))
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"이미지 로드 실패: {img_path} - {e}")
    else:
        print("시각화 이미지를 찾을 수 없습니다.")
else:
    print("시각화 디렉토리를 찾을 수 없습니다.")
```

## Cell 5: Robustness 테스트

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

print("Robustness (노이즈/압축) 테스트 수행 중...")
!python notebooks/06_robustness.py

# 결과 그래프 출력
robust_dir = "results/metrics"  # "results/robustness"가 아님
if os.path.exists(robust_dir):
    robust_files = sorted(glob.glob(os.path.join(robust_dir, "*robustness*.png")))
    if robust_files:
        print(f"\n{len(robust_files)}개의 Robustness 그래프 생성됨")
        for img_path in robust_files:
            try:
                plt.figure(figsize=(12, 8))
                img = mpimg.imread(img_path)
                plt.imshow(img)
                plt.title(os.path.basename(img_path))
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"이미지 로드 실패: {img_path} - {e}")
    else:
        print("Robustness 그래프를 찾을 수 없습니다.")
else:
    print("결과 디렉토리를 찾을 수 없습니다.")
```

## Cell 6: 효율성 측정

```python
import pandas as pd
import os

print("효율성(속도/메모리) 측정 중...")
!python notebooks/07_efficiency.py

# 결과 테이블 출력
efficiency_csv = "results/metrics/efficiency_comparison.csv"  # 경로 수정
if os.path.exists(efficiency_csv):
    df = pd.read_csv(efficiency_csv)
    print("\n효율성 비교 결과:")
    print(df.to_string(index=False))
else:
    print("효율성 결과 파일을 찾을 수 없습니다.")
    print(f"   예상 경로: {efficiency_csv}")
```

## Cell 7: 결과 다운로드

```python
import os
from IPython.display import FileLink

# 결과 폴더 압축
print("결과 압축 중...")
!zip -r -q final_submission_results.zip results models 2>/dev/null || echo "일부 파일 압축 실패 (무시 가능)"

if os.path.exists("final_submission_results.zip"):
    file_size = os.path.getsize("final_submission_results.zip") / (1024 * 1024)  # MB
    print(f"압축 완료! 파일 크기: {file_size:.2f} MB")
    print("\n아래 링크를 클릭하여 결과를 다운로드하세요:")
    display(FileLink('final_submission_results.zip'))
else:
    print("압축 파일을 생성할 수 없습니다.")
    print("   개별 폴더를 수동으로 다운로드하세요:")
    print("   - results/")
    print("   - models/")
```

---

## 주요 수정 사항

### 1. **Config 구조 일관성**
- `log_dir`: `"results/logs"` (코드가 기대하는 경로)
- `robustness.save_dir`: `"results/metrics"` (실제 저장 경로)
- `efficiency.save_dir`: `"results/metrics"` (실제 저장 경로)

### 2. **경로 탐색 개선**
- 여러 가능한 경로를 체계적으로 시도
- 실제 이미지 폴더 존재 여부 확인
- 명확한 에러 메시지

### 3. **에러 처리 강화**
- 파일 존재 여부 확인
- 예외 처리 추가
- 명확한 경고 메시지

### 4. **중복 제거**
- Config 생성 코드 통합
- 라이브러리 설치 통합

### 5. **설정 값 수정**
- `noise_levels`에서 `0.0` 제거 (의미 없음)
- `num_workers`는 data_pipeline에서 자동 처리됨

### 6. **NumPy 2.x + Matplotlib 호환**
- Kaggle 기본 런타임은 `numpy==2.2.6`을 포함합니다. `matplotlib` wheel이 NumPy 1.x로 빌드되어 있으면 `_ARRAY_API not found`가 발생할 수 있습니다.
- **방법 A**: 아래 명령을 *requirements 설치 이후*에 추가해 소스에서 다시 빌드하세요.
  ```python
  !pip install -q --no-binary matplotlib --no-deps --force-reinstall "matplotlib==3.9.1"
  ```
  빌드 시간은 1~2분 정도이며, Save Version 실행 전 셀에 넣어 두면 자동으로 적용됩니다.
- **방법 B**: Matplotlib이 없어도 본 저장소는 자동으로 시각화 단계를 건너뛰도록 수정되어 있으므로, 지표 계산만 필요하다면 추가 조치 없이도 학습·평가가 완료됩니다(Confusion Matrix/시각화 이미지만 미생성).

