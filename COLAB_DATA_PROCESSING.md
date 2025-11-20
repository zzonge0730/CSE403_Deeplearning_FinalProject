# Colab에서 Kaggle 데이터셋 다운로드 후 처리

## 🎯 상황
Kaggle에서 `sattyam96/realifake` 데이터셋을 다운로드했고, 이제 프로젝트에 맞게 처리해야 합니다.

## ✅ 빠른 해결책 (한 번에 실행)

Colab 노트북에 다음 코드를 복사해서 실행하세요:

```python
# ============================================
# Kaggle 데이터셋 다운로드 후 처리 (Colab)
# ============================================

# 1. Kaggle API 설정 (최초 1회만)
from google.colab import files
!pip install -q kaggle

print("kaggle.json 파일을 업로드하세요:")
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 2. 데이터셋 다운로드
!kaggle datasets download -d sattyam96/realifake -p data/

# 3. 압축 해제
!unzip -q data/realifake.zip -d data/temp

# 4. 데이터 준비 (FAKE → fake, REAL → real)
import shutil
from pathlib import Path

# 대상 폴더 생성
train_dir = Path('data/train')
train_dir.mkdir(parents=True, exist_ok=True)

# FAKE → fake 복사
if Path('data/temp/FAKE').exists():
    shutil.copytree('data/temp/FAKE', 'data/train/fake', dirs_exist_ok=True)
    fake_count = len(list(Path('data/train/fake').glob('*')))
    print(f"✓ FAKE → fake: {fake_count:,}개")

# REAL → real 복사
if Path('data/temp/REAL').exists():
    shutil.copytree('data/temp/REAL', 'data/train/real', dirs_exist_ok=True)
    real_count = len(list(Path('data/train/real').glob('*')))
    print(f"✓ REAL → real: {real_count:,}개")

# 임시 파일 정리
!rm -rf data/temp data/realifake.zip

print("\n✅ 데이터 준비 완료!")
print(f"위치: {train_dir.absolute()}")

# 5. 데이터 확인
!ls -la data/train/
print(f"\nFAKE 파일 수: {len(list(Path('data/train/fake').glob('*'))):,}")
print(f"REAL 파일 수: {len(list(Path('data/train/real').glob('*'))):,}")
```

## 📝 단계별 설명

### 이미 다운로드했다면?

만약 이미 `kaggle datasets download`를 실행했다면, 3번부터 시작하세요:

```python
# 압축 해제
!unzip -q data/realifake.zip -d data/temp

# 데이터 준비
import shutil
from pathlib import Path

Path('data/train').mkdir(parents=True, exist_ok=True)

# FAKE → fake
if Path('data/temp/FAKE').exists():
    shutil.copytree('data/temp/FAKE', 'data/train/fake', dirs_exist_ok=True)
    print(f"✓ FAKE 복사 완료")

# REAL → real
if Path('data/temp/REAL').exists():
    shutil.copytree('data/temp/REAL', 'data/train/real', dirs_exist_ok=True)
    print(f"✓ REAL 복사 완료")

# 정리
!rm -rf data/temp data/realifake.zip

print("✅ 완료!")
```

## 🔍 데이터 확인

```python
# 폴더 구조 확인
!ls -la data/train/

# 파일 수 확인
!find data/train/fake -type f | wc -l
!find data/train/real -type f | wc -l

# 샘플 이미지 확인
from IPython.display import Image, display
from pathlib import Path

fake_samples = list(Path('data/train/fake').glob('*'))[:3]
for img_path in fake_samples:
    display(Image(str(img_path)))
    print(img_path.name)
```

## ✅ 다음 단계

데이터 준비가 완료되면:

```python
# 데이터 로더 테스트
!python notebooks/data_pipeline.py

# 학습 시작
!python notebooks/02_train_cnn.py
```

## 💡 주의사항

1. **폴더 이름**: 프로젝트는 소문자(`fake`, `real`)를 기대합니다
2. **위치**: `data/train/` 폴더에 있어야 합니다
3. **이미지 전처리**: 불필요합니다! 학습 시 자동으로 처리됩니다

## 🐛 문제 해결

### "No such file or directory"
```python
# 현재 디렉토리 확인
import os
print(f"현재 위치: {os.getcwd()}")

# 프로젝트 폴더로 이동 (필요시)
# os.chdir('/content/drive/MyDrive/deeplearning')
```

### "Permission denied"
```python
# 권한 확인 및 수정
!ls -la data/
!chmod -R 755 data/
```

## 요약

**Colab에서 Kaggle 데이터 다운로드 후:**

1. ✅ 압축 해제: `!unzip -q data/realifake.zip -d data/temp`
2. ✅ 폴더 변환: `FAKE` → `fake`, `REAL` → `real`
3. ✅ 복사: `data/train/` 폴더에 배치
4. ✅ 정리: 임시 파일 삭제

**이미지 전처리는 불필요합니다!** 학습 시 자동으로 처리됩니다.
