# Colab에서 Kaggle 데이터셋 사용 가이드

## 🚀 빠른 시작 (Colab)

### 방법 1: 완전 자동화 노트북 사용 (가장 쉬움)

1. **노트북 열기**
   - `colab_kaggle_setup.ipynb` 파일을 Colab에 업로드
   - 또는 Colab에서 새 노트북 생성 후 아래 코드 복사

2. **셀 순서대로 실행**

### 방법 2: 단계별 실행

#### 셀 1: 환경 설정
```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 폴더로 이동 (Drive에 있는 경우)
import os
project_path = '/content/drive/MyDrive/deeplearning'
if os.path.exists(project_path):
    os.chdir(project_path)
else:
    # 프로젝트가 없으면 GitHub에서 클론
    !git clone https://github.com/your-username/deeplearning.git
    %cd deeplearning

# GPU 확인
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

#### 셀 2: 패키지 설치
```python
!pip install -q -r requirements.txt
!pip install -q kaggle
```

#### 셀 3: Kaggle API 설정
```python
# kaggle.json 파일 업로드
from google.colab import files
print("kaggle.json 파일을 업로드하세요:")
uploaded = files.upload()

# 토큰 배치
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

#### 셀 4: 데이터셋 다운로드 및 준비
```python
# 데이터셋 다운로드
!kaggle datasets download -d sattyam96/realifake -p data/

# 압축 해제
!unzip -q data/realifake.zip -d data/temp

# 데이터 준비 (FAKE → fake, REAL → real)
import shutil
from pathlib import Path

train_dir = Path('data/train')
train_dir.mkdir(parents=True, exist_ok=True)

# FAKE → fake
if Path('data/temp/FAKE').exists():
    shutil.copytree('data/temp/FAKE', 'data/train/fake', dirs_exist_ok=True)
    print(f"✓ FAKE 복사 완료")

# REAL → real
if Path('data/temp/REAL').exists():
    shutil.copytree('data/temp/REAL', 'data/train/real', dirs_exist_ok=True)
    print(f"✓ REAL 복사 완료")

# 임시 파일 정리
!rm -rf data/temp data/realifake.zip

print("✅ 데이터 준비 완료!")
```

#### 셀 5: 데이터 로더 테스트
```python
!python notebooks/data_pipeline.py
```

#### 셀 6: 학습 시작
```python
# CNN 학습
!python notebooks/02_train_cnn.py

# ViT 학습
!python notebooks/03_train_vit.py
```

## 📝 완전한 코드 (한 번에 복사)

```python
# ============================================
# Colab에서 Kaggle 데이터셋 다운로드 및 프로젝트 실행
# ============================================

# 1. 환경 설정
from google.colab import drive
drive.mount('/content/drive')

import os
project_path = '/content/drive/MyDrive/deeplearning'
if os.path.exists(project_path):
    os.chdir(project_path)
else:
    print("프로젝트를 Drive에 업로드하거나 GitHub에서 클론하세요")

# 2. 패키지 설치
!pip install -q -r requirements.txt
!pip install -q kaggle

# 3. Kaggle API 설정
from google.colab import files
print("kaggle.json 파일을 업로드하세요:")
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 4. 데이터셋 다운로드
!kaggle datasets download -d sattyam96/realifake -p data/

# 5. 압축 해제 및 준비
!unzip -q data/realifake.zip -d data/temp

import shutil
from pathlib import Path

Path('data/train').mkdir(parents=True, exist_ok=True)

if Path('data/temp/FAKE').exists():
    shutil.copytree('data/temp/FAKE', 'data/train/fake', dirs_exist_ok=True)
    print(f"✓ FAKE: {len(list(Path('data/train/fake').glob('*')))}개")

if Path('data/temp/REAL').exists():
    shutil.copytree('data/temp/REAL', 'data/train/real', dirs_exist_ok=True)
    print(f"✓ REAL: {len(list(Path('data/train/real').glob('*')))}개")

!rm -rf data/temp data/realifake.zip
print("✅ 데이터 준비 완료!")

# 6. 데이터 로더 테스트
!python notebooks/data_pipeline.py

# 7. 학습 시작
!python notebooks/02_train_cnn.py
```

## 🔑 Kaggle API 토큰 받기

1. **Kaggle 계정 로그인**
   - https://www.kaggle.com 접속

2. **API 토큰 다운로드**
   - https://www.kaggle.com/settings 접속
   - "Create New Token" 클릭
   - `kaggle.json` 파일이 자동 다운로드됨

3. **Colab에 업로드**
   - 위 코드의 "셀 3" 실행 시 파일 선택 창이 나타남
   - 다운로드한 `kaggle.json` 선택

## ⚠️ 주의사항

### 1. 세션 시간 제한
- 무료 Colab: 약 12시간
- **대응**: 중간 결과를 Drive에 저장

### 2. 데이터 크기
- Realifake 데이터셋은 약 수 GB 크기
- 다운로드에 시간이 걸릴 수 있음

### 3. 프로젝트 위치
- **옵션 A**: Google Drive에 프로젝트 업로드 (권장)
- **옵션 B**: GitHub에서 클론
- **옵션 C**: ZIP 파일로 업로드

## 🎯 권장 워크플로우

### 1. 프로젝트 준비
```python
# Drive에 프로젝트가 있는 경우
os.chdir('/content/drive/MyDrive/deeplearning')

# 또는 GitHub에서 클론
!git clone https://github.com/your-username/deeplearning.git
%cd deeplearning
```

### 2. 데이터 다운로드
```python
# 위의 "완전한 코드" 실행
```

### 3. 학습 및 결과 저장
```python
# 학습
!python notebooks/02_train_cnn.py

# 결과를 Drive에 저장
import shutil
shutil.copytree('models', '/content/drive/MyDrive/deeplearning_results/models', dirs_exist_ok=True)
shutil.copytree('results', '/content/drive/MyDrive/deeplearning_results/results', dirs_exist_ok=True)
```

## 🐛 문제 해결

### "kaggle: command not found"
```python
!pip install kaggle
```

### "403 - Forbidden"
```python
# kaggle.json 파일 확인
!cat ~/.kaggle/kaggle.json

# 권한 재설정
!chmod 600 ~/.kaggle/kaggle.json
```

### "Dataset not found"
- 데이터셋 이름 확인: `sattyam96/realifake`
- Kaggle에서 데이터셋이 공개되어 있는지 확인

### "디스크 공간 부족"
```python
# 불필요한 파일 삭제
!rm -rf data/temp
!rm -f data/*.zip
```

## 📊 데이터 확인

```python
# 데이터 구조 확인
!ls -la data/train/
!find data/train/fake -type f | wc -l
!find data/train/real -type f | wc -l

# 샘플 이미지 확인
from IPython.display import Image, display
from pathlib import Path

fake_samples = list(Path('data/train/fake').glob('*'))[:3]
for img_path in fake_samples:
    display(Image(str(img_path)))
```

## ✅ 체크리스트

- [ ] Colab에서 GPU 활성화
- [ ] 프로젝트 파일 준비 (Drive 또는 GitHub)
- [ ] Kaggle API 토큰 다운로드
- [ ] kaggle.json 업로드
- [ ] 데이터셋 다운로드 완료
- [ ] 데이터 준비 완료 (fake/, real/ 폴더)
- [ ] 데이터 로더 테스트 통과
- [ ] 학습 시작

## 🎓 완전한 예제 노트북

`colab_kaggle_setup.ipynb` 파일을 Colab에 업로드하여 사용하세요!

이 노트북은 모든 단계를 포함하고 있어 셀을 순서대로 실행하면 됩니다.
