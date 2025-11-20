# WSL 환경 가이드

## WSL에서 실행 가능합니다!

WSL에서도 충분히 실행 가능하지만, GPU 메모리에 따라 설정을 조정해야 합니다.

## 1. GPU 확인

먼저 GPU 상태를 확인하세요:

```bash
python check_gpu.py
```

이 스크립트는 다음을 확인합니다:
- CUDA 사용 가능 여부
- GPU 메모리 크기
- 권장 모델 및 설정

## 2. 모델 크기 비교

| 모델 | 파라미터 | 모델 크기 | VRAM 필요량 | 권장 환경 |
|------|---------|----------|------------|----------|
| ResNet18 | 11M | 45MB | ~1.5GB | WSL (4GB+ GPU) |
| EfficientNet-B0 | 5M | 20MB | ~1.0GB | WSL (4GB+ GPU) |
| ResNet50 | 25M | 98MB | ~2.5GB | WSL (6GB+ GPU) |
| ViT-Small | 22M | 88MB | ~2.0GB | WSL (4GB+ GPU) |
| ViT-Base | 86M | 330MB | ~4.0GB | Colab 권장 |

## 3. WSL용 설정 사용

WSL 환경에서는 경량 설정 파일을 사용하세요:

```bash
# 경량 모델로 학습
python notebooks/02_train_cnn.py --config configs/config_wsl.yaml
python notebooks/03_train_vit.py --config config_wsl.yaml
```

`config_wsl.yaml`의 주요 변경사항:
- **모델**: ResNet18, ViT-Small (더 작은 모델)
- **배치 크기**: 16 (메모리에 따라 8, 4로 조정 가능)
- **에폭 수**: 10 (빠른 테스트)

## 4. GPU 메모리 부족 시 대응

### 배치 크기 줄이기
`config_wsl.yaml`에서:
```yaml
data:
  batch_size: 8  # 또는 4
```

### Gradient Accumulation 사용
작은 배치 크기로도 효과적인 학습이 가능합니다.

### Mixed Precision Training
FP16을 사용하여 메모리 사용량을 절반으로 줄일 수 있습니다.

## 5. WSL vs Colab 비교

### WSL 장점
- ✅ 로컬 환경에서 실행 가능
- ✅ 데이터 업로드 없이 사용 가능
- ✅ 인터넷 연결 불필요 (학습 후)
- ✅ 무료

### WSL 단점
- ⚠️ GPU 메모리 제한 (보통 4-8GB)
- ⚠️ 큰 모델 사용 어려움
- ⚠️ CPU만 있는 경우 매우 느림

### Colab 장점
- ✅ 무료 GPU (T4, V100 등)
- ✅ 충분한 메모리 (15GB+)
- ✅ 큰 모델 사용 가능
- ✅ 빠른 학습

### Colab 단점
- ⚠️ 세션 시간 제한 (12시간)
- ⚠️ 데이터 업로드 필요
- ⚠️ 인터넷 연결 필요

## 6. 권장 워크플로우

### GPU 메모리 4GB 미만
```bash
# 1. 경량 설정 사용
python notebooks/02_train_cnn.py --config configs/config_wsl.yaml

# 2. 배치 크기 8로 줄이기
# config_wsl.yaml에서 batch_size: 8로 수정
```

### GPU 메모리 4-8GB
```bash
# 경량 설정 사용 가능
python notebooks/02_train_cnn.py --config configs/config_wsl.yaml

# 또는 기본 설정에서 배치 크기만 줄이기
# config.yaml에서 batch_size: 32로 수정
```

### GPU 메모리 8GB 이상
```bash
# 기본 설정 사용 가능
python notebooks/02_train_cnn.py
```

### CPU만 있는 경우
**Colab 사용을 강력히 권장합니다!**
- CPU 학습은 매우 느립니다 (에폭당 수 시간 소요)
- ResNet18도 CPU에서는 실용적이지 않습니다

## 7. Colab 사용 시

Colab을 사용하는 경우:

1. **노트북 업로드**: 프로젝트를 Google Drive에 업로드
2. **GPU 활성화**: Runtime → Change runtime type → GPU 선택
3. **설정 변경**: `config.yaml`에서 큰 모델 사용 가능
   ```yaml
   models:
     cnn:
       name: "resnet50"  # resnet18 대신
     vit:
       name: "vit_base_patch16_224"  # vit_small 대신
   ```

## 8. 실용적인 조언

**WSL에서 시작하는 것을 권장합니다:**
1. 작은 모델로 프로토타입 테스트 (ResNet18)
2. 코드가 정상 작동하는지 확인
3. 필요시 Colab으로 큰 모델 학습

이렇게 하면:
- 빠른 반복 개발 가능
- Colab 시간 절약
- 로컬에서 디버깅 용이

## 결론

**WSL에서도 충분히 가능합니다!** 
- GPU가 있다면: ResNet18/ViT-Small로 시작
- GPU가 없다면: Colab 사용 권장
- 큰 모델이 필요하면: Colab으로 전환
