import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def create_dataloaders(data_dir, batch_size=32, img_size=224, split_ratio=(0.8, 0.1, 0.1)):
    """
    데이터셋 경로를 기반으로 학습, 검증, 테스트용 DataLoader를 생성합니다.

    Args:
        data_dir (str): 데이터셋의 루트 디렉터리. 
                        이 디렉터리 내에 'train' 또는 유사한 폴더가 있고,
                        그 안에 'real', 'fake' 등의 클래스 폴더가 있어야 합니다.
                        Kaggle 노트북에서는 '/kaggle/input/...' 경로를 사용할 수 있습니다.
        batch_size (int): 한 번에 처리할 이미지의 수.
        img_size (int): 모델에 입력될 이미지의 크기.
        split_ratio (tuple): (train, val, test) 비율. 기본값 (0.8, 0.1, 0.1)

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
               학습용, 검증용, 테스트용 DataLoader, 클래스 이름 리스트.
    """
    # Kaggle 노트북 환경 감지
    is_kaggle = os.path.exists("/kaggle/input")
    if is_kaggle and not os.path.exists(data_dir):
        # Kaggle에서 자동으로 데이터 경로 찾기 시도
        kaggle_input = "/kaggle/input"
        possible_paths = [
            os.path.join(kaggle_input, "realifake", "Realifake"),
            os.path.join(kaggle_input, "realifake", "train"),
            os.path.join(kaggle_input, "realifake"),
            data_dir  # 원래 경로도 시도
        ]
        for path in possible_paths:
            if os.path.exists(path):
                print(f"🔍 Kaggle 노트북: 데이터 경로 자동 감지 -> {path}")
                data_dir = path
                break
    
    # 시드 고정 (재현성을 위해)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # 이미지 전처리
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])

    # 데이터셋 로드
    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
        class_names = full_dataset.classes
        print(f"데이터셋 로드 성공. 클래스: {class_names}")
        print(f"전체 이미지 수: {len(full_dataset)}")
        
        if len(full_dataset) == 0:
            print("❌ 경고: 데이터셋이 비어있습니다!")
            return None, None, None, None
    except FileNotFoundError as e:
        print(f"❌ 오류: '{data_dir}' 경로를 찾을 수 없습니다.")
        if is_kaggle:
            print("\n📌 Kaggle 노트북 사용 시:")
            print("   1. 데이터셋을 'Add Data' 버튼으로 추가했는지 확인하세요")
            print("   2. 데이터 경로가 '/kaggle/input/데이터셋이름/train' 형식인지 확인하세요")
            print("   3. config.yaml의 train_dir을 올바른 경로로 수정하세요")
        else:
            print("   Kaggle 데이터셋을 다운로드하여 'data' 폴더에 압축을 풀었는지 확인해주세요.")
        print("   데이터셋 구조 예시: 'data/train/real', 'data/train/fake'")
        return None, None, None, None
    except Exception as e:
        print(f"❌ 데이터셋 로드 중 예상치 못한 오류 발생: {e}")
        return None, None, None, None

    # 데이터셋을 학습/검증/테스트로 분할 (8:1:1)
    total_size = len(full_dataset)
    train_size = int(split_ratio[0] * total_size)
    val_size = int(split_ratio[1] * total_size)
    test_size = total_size - train_size - val_size  # 남은 거 전부
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    print(f"분할 완료 -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # DataLoader 생성
    # 🚨 [핵심 수정] num_workers를 2로 고정 (속도 최적화)
    max_workers = 2
    print(f"🚀 데이터 로더 워커 수: {max_workers} (속도 최적화)")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=max_workers, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=max_workers, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=max_workers, pin_memory=True if torch.cuda.is_available() else False)

    return train_loader, val_loader, test_loader, class_names

if __name__ == '__main__':
    # --- 설정 ---
    # 데이터셋이 저장된 기본 경로를 지정하세요.
    # CIFAKE 데이터셋의 경우 'train' 폴더를 지정해야 할 수 있습니다.
    # 예: 'data/CIFAKE/train'
    DATA_PATH = "data/" 
    BATCH_SIZE = 64

    # --- 실행 ---
    train_dataloader, val_dataloader, test_dataloader, classes = create_dataloaders(data_dir=DATA_PATH, batch_size=BATCH_SIZE)

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
