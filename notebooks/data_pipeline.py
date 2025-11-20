
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def create_dataloaders(data_dir, batch_size=32, img_size=224):
    """
    데이터셋 경로를 기반으로 학습 및 검증용 DataLoader를 생성합니다.

    Args:
        data_dir (str): 데이터셋의 루트 디렉터리. 
                        이 디렉터리 내에 'train' 또는 유사한 폴더가 있고,
                        그 안에 'real', 'fake' 등의 클래스 폴더가 있어야 합니다.
        batch_size (int): 한 번에 처리할 이미지의 수.
        img_size (int): 모델에 입력될 이미지의 크기.

    Returns:
        tuple: (train_loader, val_loader, class_names)
               학습용 DataLoader, 검증용 DataLoader, 클래스 이름 리스트.
    """
    # 1. 이미지 전처리 정책 정의
    #    - 모든 이미지를 224x224 크기로 조정
    #    - 이미지를 PyTorch 텐서로 변환 (0~1 값으로 정규화)
    #    - 이미지 채널을 평균 0.5, 표준편차 0.5로 정규화 (-1~1 값으로 변환)
    #      (ImageNet으로 사전 학습된 모델들은 보통 특정 평균/표준편차를 사용하며, 이는 추후 변경 가능)
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])

    # 2. ImageFolder를 사용하여 데이터셋 로드
    #    - data_dir 경로의 하위 폴더 이름을 클래스로 자동 인식합니다.
    #    - 예: data_dir/real, data_dir/fake
    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
        class_names = full_dataset.classes
        print(f"데이터셋 로드 성공. 클래스: {class_names}")
        print(f"전체 이미지 수: {len(full_dataset)}")
    except FileNotFoundError:
        print(f"오류: '{data_dir}' 경로를 찾을 수 없습니다.")
        print("Kaggle 데이터셋을 다운로드하여 'data' 폴더에 압축을 풀었는지 확인해주세요.")
        print("데이터셋 구조 예시: 'data/train/real', 'data/train/fake'")
        return None, None, None

    # 3. 데이터셋을 학습용과 검증용으로 분할 (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"학습 데이터셋 크기: {len(train_dataset)}")
    print(f"검증 데이터셋 크기: {len(val_dataset)}")

    # 4. DataLoader 생성
    #    - shuffle=True: 학습 시 데이터 순서를 섞어 모델이 순서에 의존하지 않도록 함
    #    - num_workers: 데이터를 미리 불러올 프로세스 수 (CPU 코어 수에 맞게 조절)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, class_names

if __name__ == '__main__':
    # --- 설정 ---
    # 데이터셋이 저장된 기본 경로를 지정하세요.
    # CIFAKE 데이터셋의 경우 'train' 폴더를 지정해야 할 수 있습니다.
    # 예: 'data/CIFAKE/train'
    DATA_PATH = "data/" 
    BATCH_SIZE = 64

    # --- 실행 ---
    train_dataloader, val_dataloader, classes = create_dataloaders(data_dir=DATA_PATH, batch_size=BATCH_SIZE)

    # --- 테스트 ---
    # DataLoader가 정상적으로 작동하는지, 데이터 한 배치를 뽑아서 확인
    if train_dataloader:
        print("\n--- DataLoader 테스트 ---")
        # next(iter(...))를 통해 데이터로더에서 첫 번째 배치를 가져옴
        images, labels = next(iter(train_dataloader))

        print(f"이미지 배치 형태: {images.shape}")
        print(f"라벨 배치 형태: {labels.shape}")
        print(f"클래스: {classes}")
        print(f"첫 번째 배치의 라벨: {labels}")
        # 라벨 0: classes[0], 라벨 1: classes[1] 에 해당

