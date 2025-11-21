# 버그 수정 내역

## 수정된 문제

### 1. TypeError: learning_rate가 문자열로 읽힘

**문제:**
```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

**원인:**
- YAML 파일에서 `learning_rate: 1e-4`가 문자열로 읽힘
- PyTorch 옵티마이저는 숫자(float)를 기대함

**해결:**
- 코드에서 `float()` 변환 추가
- YAML 파일에서 과학적 표기법 대신 소수점 사용

**수정 파일:**
- `notebooks/02_train_cnn.py`
- `notebooks/03_train_vit.py`
- `configs/config.yaml`
- `configs/config_wsl.yaml`

### 2. DataLoader num_workers 경고

**문제:**
```
UserWarning: This DataLoader will create 4 worker processes...
Our suggested max number of worker in current system is 2
```

**해결:**
- `data_pipeline.py`에서 시스템 CPU 수에 맞게 자동 조정

## 적용 방법

수정된 파일을 다시 실행하면 됩니다:

```bash
python notebooks/02_train_cnn.py
```

또는 Colab에서:
```python
!python notebooks/02_train_cnn.py
```

## 확인 사항

학습이 정상적으로 시작되면 다음과 같은 출력이 나타납니다:

```
Epoch 1/20
Training: 100%|########| 2181/2181 [XX:XX<XX:XX, X.XXit/s]
Train Loss: X.XXXX, Train Acc: XX.XX%
Validation: 100%|########| 546/546 [XX:XX<XX:XX, X.XXit/s]
Val Loss: X.XXXX, Val Acc: XX.XX%
```
