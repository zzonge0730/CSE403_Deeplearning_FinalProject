# Kaggle NumPy 버전 충돌 해결 가이드

## 문제
Kaggle 환경에서 NumPy 2.x가 자동으로 설치되어 TensorBoard와 호환되지 않습니다.

## 해결 방법

### 방법 1: Python 세션 재시작 (권장)

```python
# 1단계: NumPy 1.x 설치
!pip install --no-deps --force-reinstall --ignore-installed "numpy==1.24.3"
!pip install --no-deps --force-reinstall "protobuf==3.20.3"

# 2단계: requirements.txt 설치 (numpy 제외)
import tempfile
with open("requirements.txt", "r") as f:
    lines = f.readlines()
with open("/tmp/req_no_numpy.txt", "w") as f:
    for line in lines:
        if not line.strip().startswith("numpy"):
            f.write(line)
!pip install -q -r /tmp/req_no_numpy.txt

# 3단계: Kaggle 노트북에서 "Runtime" → "Restart Session" 클릭

# 4단계: 이 셀 실행 (재시작 후)
import numpy as np
print(f"NumPy 버전: {np.__version__}")  # 1.24.3이어야 함

# 5단계: 학습 실행
!python notebooks/02_train_cnn.py
```

### 방법 2: TensorBoard 비활성화 (빠른 해결)

TensorBoard가 문제라면, 임시로 비활성화할 수 있습니다:

```python
# 학습 스크립트 수정 (임시)
import os
os.environ['DISABLE_TENSORBOARD'] = '1'

# 또는 직접 실행
!python -c "
import sys
sys.path.insert(0, '.')
code = open('notebooks/02_train_cnn.py').read()
code = code.replace('from torch.utils.tensorboard import SummaryWriter', 'SummaryWriter = None  # 비활성화')
code = code.replace('writer = SummaryWriter', 'writer = None  # 비활성화')
exec(code)
"
```

### 방법 3: 완전한 환경 재설정

```python
# 모든 NumPy 관련 패키지 제거
!pip uninstall -y numpy matplotlib tensorboard

# NumPy 1.x 설치
!pip install --no-deps "numpy==1.24.3"

# 필수 패키지만 설치
!pip install torch torchvision timm tqdm pyyaml pillow

# 학습 실행
!python notebooks/02_train_cnn.py
```

## 확인 방법

```python
import numpy as np
import sys

print(f"NumPy 버전: {np.__version__}")
print(f"NumPy 경로: {np.__file__}")

# NumPy 1.x인지 확인
if int(np.__version__.split('.')[0]) >= 2:
    print("❌ NumPy 2.x가 로드되었습니다!")
    print("   Python 세션을 재시작하세요.")
else:
    print("✅ NumPy 1.x 확인됨!")
```

