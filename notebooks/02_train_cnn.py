"""
CNN 모델 학습 스크립트 (ResNet/EfficientNet)
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import get_cnn_model
from notebooks.data_pipeline import create_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_cnn(config_path="configs/config.yaml"):
    """CNN 모델 학습 메인 함수"""
    # 설정 로드
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 데이터 로더 생성 (test_loader는 무시)
    train_loader, val_loader, _, class_names = create_dataloaders(
        data_dir=config["data"]["train_dir"],
        batch_size=config["data"]["batch_size"],
        img_size=config["data"]["img_size"]
    )
    
    # 데이터 로더 None 체크
    if train_loader is None or val_loader is None or class_names is None:
        print("데이터 로더 생성에 실패했습니다. 프로그램을 종료합니다.")
        return None, None
    
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # 모델 생성
    model = get_cnn_model(
        model_name=config["models"]["cnn"]["name"],
        num_classes=config["models"]["cnn"]["num_classes"],
        pretrained=config["models"]["cnn"]["pretrained"]
    )
    model = model.to(device)
    
    print(f"Model: {config['models']['cnn']['name']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    
    # 설정값을 숫자로 변환 (YAML에서 문자열로 읽힐 수 있음)
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    
    if config["training"]["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    
    # 스케줄러
    num_epochs = int(config["training"]["num_epochs"])
    if config["training"]["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    
    # TensorBoard 로거
    log_dir = os.path.join(config["training"]["log_dir"], "cnn")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 학습 히스토리
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = -1.0  # 초기값을 -1로 설정 (첫 epoch에서 무조건 저장되도록)
    patience = 3  # Early stopping patience
    patience_counter = 0
    
    # 학습 루프
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 학습
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 검증
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 히스토리 저장
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # TensorBoard 로깅
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 최고 성능 모델 저장 및 Early Stopping 체크
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # 성능 개선 시 카운터 리셋
            save_dir = config["training"]["save_dir"]
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"cnn_{config['models']['cnn']['name']}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "history": history
            }, model_path)
            print(f"Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                print(f"Best Val Accuracy: {best_val_acc:.2f}%")
                break
    
    writer.close()
    
    # 최종 모델 저장
    final_model_path = os.path.join(config["training"]["save_dir"], f"cnn_{config['models']['cnn']['name']}_final.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history
    }, final_model_path)
    
    # 히스토리 저장
    history_path = os.path.join(config["training"]["log_dir"], "cnn_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    
    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="설정 파일 경로")
    args = parser.parse_args()
    train_cnn(args.config)
