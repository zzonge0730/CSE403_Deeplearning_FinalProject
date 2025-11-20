# Google Colab 사용 가이드

## 🚀 빠른 시작

### 1단계: Colab 노트북 생성

1. [Google Colab](https://colab.research.google.com/) 접속
2. `파일` → `새 노트북` 클릭
3. GPU 활성화: `런타임` → `런타임 유형 변경` → `GPU` 선택

### 2단계: 프로젝트 업로드

#### 방법 A: Google Drive에 업로드 (권장)

```python
# Colab 노트북 첫 셀에 실행
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 폴더로 이동
import os
os.chdir('/content/drive/MyDrive/deeplearning')
```

**준비 작업:**
1. 프로젝트 폴더를 ZIP으로 압축
2. Google Drive에 업로드
3. 압축 해제 (Colab에서)

#### 방법 B: GitHub에서 클론

```python
# Colab 노트북 첫 셀에 실행
!git clone https://github.com/your-username/deeplearning.git
%cd deeplearning
```

#### 방법 C: 직접 업로드

```python
# Colab 노트북에서 파일 업로드
from google.colab import files
uploaded = files.upload()  # 프로젝트 ZIP 파일 업로드

# 압축 해제
!unzip -q deeplearning.zip
%cd deeplearning
```

### 3단계: 패키지 설치

```python
# Colab 노트북에서 실행
!pip install -r requirements.txt
```

### 4단계: 데이터 준비

#### 방법 A: Google Drive에 데이터 업로드

```python
# 데이터를 Google Drive에 업로드한 경우
# data/ 폴더를 Drive에 업로드하고 심볼릭 링크 생성
!ln -s /content/drive/MyDrive/data /content/deeplearning/data
```

#### 방법 B: Kaggle에서 직접 다운로드

```python
# Kaggle API 사용
!pip install kaggle

# Kaggle API 토큰 업로드 (kaggle.json)
from google.colab import files
files.upload()  # kaggle.json 업로드

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 데이터셋 다운로드
!kaggle datasets download -d your-dataset-name
!unzip -q your-dataset-name.zip -d data/
```

#### 방법 C: 직접 업로드

```python
# 작은 데이터셋의 경우 직접 업로드
from google.colab import files
# 여러 파일 업로드 (브라우저에서 선택)
```

### 5단계: 학습 실행

```python
# CNN 학습
!python notebooks/02_train_cnn.py

# ViT 학습
!python notebooks/03_train_vit.py
```

## 📝 완전한 Colab 노트북 예제

```python
# ============================================
# 셀 1: 환경 설정
# ============================================
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/deeplearning')

# GPU 확인
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============================================
# 셀 2: 패키지 설치
# ============================================
!pip install -r requirements.txt -q

# ============================================
# 셀 3: 데이터 준비
# ============================================
# 데이터가 이미 Drive에 있는 경우
# !ln -s /content/drive/MyDrive/data data

# 또는 Kaggle에서 다운로드
# !kaggle datasets download -d your-dataset
# !unzip -q your-dataset.zip -d data/

# ============================================
# 셀 4: CNN 학습
# ============================================
!python notebooks/02_train_cnn.py

# ============================================
# 셀 5: ViT 학습
# ============================================
!python notebooks/03_train_vit.py

# ============================================
# 셀 6: 평가
# ============================================
!python notebooks/04_evaluate.py

# ============================================
# 셀 7: 시각화
# ============================================
!python notebooks/05_visualize.py

# ============================================
# 셀 8: 결과 확인
# ============================================
from IPython.display import Image, display
import os

# Confusion Matrix 표시
if os.path.exists('results/metrics/cnn_confusion_matrix.png'):
    display(Image('results/metrics/cnn_confusion_matrix.png'))
```

## 🔧 Colab 특화 설정

### 세션 관리

```python
# 세션 시간 확인
import time
start_time = time.time()

# 학습 후 경과 시간 확인
elapsed = time.time() - start_time
print(f"Elapsed time: {elapsed/3600:.2f} hours")
```

### 파일 저장 (Drive에 자동 저장)

```python
# 학습 중간 결과를 Drive에 저장
import shutil

def save_to_drive(src, dst):
    """Drive에 파일 복사"""
    drive_dst = f'/content/drive/MyDrive/deeplearning/{dst}'
    os.makedirs(os.path.dirname(drive_dst), exist_ok=True)
    shutil.copy(src, drive_dst)
    print(f"Saved to Drive: {drive_dst}")

# 모델 저장 후
# save_to_drive('models/cnn_resnet50_best.pth', 'models/cnn_resnet50_best.pth')
```

### TensorBoard 사용

```python
# TensorBoard 실행
%load_ext tensorboard
%tensorboard --logdir results/logs
```

## ⚠️ 주의사항

### 1. 세션 시간 제한
- 무료 Colab: 약 12시간 (비활성 시 중단 가능)
- Pro: 더 긴 세션 시간
- **대응**: 중간 결과를 Drive에 저장

### 2. GPU 할당 불안정
- 무료 Colab: GPU 할당이 보장되지 않음
- **대응**: GPU 없이도 실행 가능하도록 코드 작성

### 3. 데이터 크기 제한
- Drive 무료: 15GB
- **대응**: 필요시 데이터 압축 또는 외부 저장소 사용

### 4. 파일 경로
- Colab 작업 디렉토리: `/content/`
- Drive 마운트: `/content/drive/MyDrive/`
- **주의**: 절대 경로 사용 권장

## 🎯 최적화 팁

### 1. 빠른 시작을 위한 설정

```python
# config.yaml 수정 (Colab용)
# - 배치 크기 증가 (32-64)
# - 큰 모델 사용 가능 (ResNet50, ViT-Base)
# - 에폭 수 조정
```

### 2. 메모리 관리

```python
# GPU 메모리 정리
import gc
torch.cuda.empty_cache()
gc.collect()
```

### 3. 진행 상황 저장

```python
# 주기적으로 체크포인트 저장
# config.yaml에서 자동 저장 설정
```

## 📊 Colab vs WSL 비교

| 항목 | Colab | WSL |
|------|-------|-----|
| GPU | 무료 (T4/V100) | 구매 필요 |
| 메모리 | 15GB+ | GPU에 따라 다름 |
| 세션 시간 | 12시간 제한 | 무제한 |
| 인터넷 | 필요 | 불필요 |
| 데이터 업로드 | 필요 | 불필요 |
| 편의성 | 높음 (설정 간단) | 중간 (초기 설정 필요) |

## 🚨 문제 해결

### GPU가 할당되지 않는 경우

```python
# GPU 재할당 시도
# 런타임 → 런타임 다시 시작
# 또는 런타임 → 팩토리 런타임 재설정
```

### 메모리 부족 오류

```python
# 배치 크기 줄이기
# config.yaml에서 batch_size: 32 → 16
```

### 파일을 찾을 수 없는 경우

```python
# 현재 디렉토리 확인
import os
print(f"Current directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

# 경로 수정
os.chdir('/content/drive/MyDrive/deeplearning')
```

## 📁 권장 폴더 구조 (Colab)

```
/content/
├── drive/
│   └── MyDrive/
│       └── deeplearning/          # 프로젝트 폴더
│           ├── notebooks/
│           ├── utils/
│           ├── configs/
│           ├── data/               # 데이터 (Drive에 저장)
│           ├── models/             # 학습된 모델 (Drive에 저장)
│           └── results/            # 결과 (Drive에 저장)
```

## ✅ 체크리스트

Colab 사용 전 확인사항:

- [ ] Google Drive에 프로젝트 업로드 완료
- [ ] GPU 활성화 확인
- [ ] 데이터 경로 설정 확인
- [ ] requirements.txt 설치 확인
- [ ] config.yaml 경로 확인
- [ ] 중간 저장 설정 확인 (세션 끊김 대비)

## 🎓 실전 예제

완전한 Colab 노트북 예제는 `colab_notebook.ipynb` 파일을 참고하세요.
