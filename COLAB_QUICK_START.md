# Colab 빠른 시작 가이드

## 🚀 Colab에서 Kaggle 데이터셋 다운로드 후 처리

### 완전한 코드 (복사해서 실행)

```python
# ============================================
# Colab에서 Kaggle 데이터셋 다운로드 및 처리
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

Path('data/train').mkdir(parents=True, exist_ok=True)

# FAKE → fake
if Path('data/temp/FAKE').exists():
    shutil.copytree('data/temp/FAKE', 'data/train/fake', dirs_exist_ok=True)
    print(f"✓ FAKE: {len(list(Path('data/train/fake').glob('*'))):,}개")

# REAL → real
if Path('data/temp/REAL').exists():
    shutil.copytree('data/temp/REAL', 'data/train/real', dirs_exist_ok=True)
    print(f"✓ REAL: {len(list(Path('data/train/real').glob('*'))):,}개")

# 임시 파일 정리
!rm -rf data/temp data/realifake.zip

print("✅ 데이터 준비 완료!")

# 5. 데이터 확인
!ls -la data/train/
```

## 📝 이미 다운로드했다면?

다운로드는 이미 했고, 압축 해제와 폴더 변환만 필요하다면:

```python
# 압축 해제 (아직 안 했다면)
!unzip -q data/realifake.zip -d data/temp

# 데이터 준비
import shutil
from pathlib import Path

Path('data/train').mkdir(parents=True, exist_ok=True)

shutil.copytree('data/temp/FAKE', 'data/train/fake', dirs_exist_ok=True)
shutil.copytree('data/temp/REAL', 'data/train/real', dirs_exist_ok=True)

!rm -rf data/temp data/realifake.zip

print("✅ 완료!")
```

## ✅ 확인

```python
# 데이터 구조 확인
!ls -la data/train/
!find data/train/fake -type f | wc -l
!find data/train/real -type f | wc -l
```

## 🎯 다음 단계

```python
# 데이터 로더 테스트
!python notebooks/data_pipeline.py

# 학습 시작
!python notebooks/02_train_cnn.py
```

## 💡 핵심 포인트

1. **다운로드**: `!kaggle datasets download -d sattyam96/realifake -p data/`
2. **압축 해제**: `!unzip -q data/realifake.zip -d data/temp`
3. **폴더 변환**: `FAKE` → `fake`, `REAL` → `real`
4. **복사**: `data/train/` 폴더에 배치
5. **정리**: 임시 파일 삭제

**이미지 전처리는 불필요합니다!** 학습 시 자동으로 처리됩니다.
