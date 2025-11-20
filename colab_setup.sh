#!/bin/bash
# Colab 환경 설정 스크립트
# Colab 노트북에서 실행: !bash colab_setup.sh

echo "=========================================="
echo "Colab 환경 설정 시작"
echo "=========================================="

# 1. 패키지 설치
echo "1. 패키지 설치 중..."
pip install -q -r requirements.txt

# 2. GPU 확인
echo ""
echo "2. GPU 확인 중..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 3. 디렉토리 구조 확인
echo ""
echo "3. 디렉토리 구조 확인 중..."
if [ -d "notebooks" ]; then
    echo "✓ notebooks 폴더 확인됨"
else
    echo "✗ notebooks 폴더를 찾을 수 없습니다"
fi

if [ -d "utils" ]; then
    echo "✓ utils 폴더 확인됨"
else
    echo "✗ utils 폴더를 찾을 수 없습니다"
fi

if [ -d "configs" ]; then
    echo "✓ configs 폴더 확인됨"
else
    echo "✗ configs 폴더를 찾을 수 없습니다"
fi

# 4. 데이터 확인
echo ""
echo "4. 데이터 확인 중..."
if [ -d "data" ]; then
    echo "✓ data 폴더 확인됨"
    echo "  파일 수: $(find data -type f | wc -l)"
else
    echo "⚠ data 폴더를 찾을 수 없습니다"
    echo "  데이터를 준비하세요:"
    echo "  - Google Drive에 업로드"
    echo "  - 또는 Kaggle에서 다운로드"
fi

echo ""
echo "=========================================="
echo "환경 설정 완료!"
echo "=========================================="
echo ""
echo "다음 단계:"
echo "1. 데이터 준비 확인"
echo "2. python notebooks/02_train_cnn.py 실행"
