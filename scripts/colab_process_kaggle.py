"""
Colab에서 Kaggle 데이터셋 다운로드 후 처리 스크립트
이미 다운로드한 경우에도 사용 가능
"""

import shutil
from pathlib import Path
import subprocess
import os


def process_kaggle_data(zip_path="data/realifake.zip", target_dir="data/train"):
    """
    Kaggle에서 다운로드한 데이터 처리
    
    Args:
        zip_path: 다운로드한 ZIP 파일 경로
        target_dir: 최종 데이터 위치
    """
    print("="*50)
    print("Kaggle 데이터 처리")
    print("="*50)
    
    zip_file = Path(zip_path)
    temp_dir = Path("data/temp")
    target_path = Path(target_dir)
    
    # 1. 압축 해제
    if zip_file.exists():
        print(f"\n1. 압축 해제 중: {zip_file.name}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print("압축 해제 완료")
    elif temp_dir.exists():
        print("이미 압축 해제된 폴더 발견")
    else:
        print(f"ZIP 파일을 찾을 수 없습니다: {zip_path}")
        print("다운로드: !kaggle datasets download -d sattyam96/realifake -p data/")
        return False
    
    # 2. 데이터 구조 확인
    print("\n2. 데이터 구조 확인 중...")
    
    fake_source = temp_dir / "FAKE"
    real_source = temp_dir / "REAL"
    
    # 다른 가능한 구조 확인
    if not fake_source.exists():
        for possible in ["Fake", "fake", "FAKE", "train/FAKE"]:
            test_path = temp_dir / possible
            if test_path.exists():
                fake_source = test_path
                break
    
    if not real_source.exists():
        for possible in ["Real", "real", "REAL", "train/REAL"]:
            test_path = temp_dir / possible
            if test_path.exists():
                real_source = test_path
                break
    
    if not fake_source.exists() or not real_source.exists():
        print("표준 폴더 구조를 찾을 수 없습니다.")
        print("압축 해제된 폴더 구조:")
        for item in sorted(temp_dir.rglob("*"))[:20]:
            if item.is_dir():
                print(f"  {item.relative_to(temp_dir)}")
        return False
    
    print(f"FAKE 폴더: {fake_source}")
    print(f"REAL 폴더: {real_source}")
    
    # 3. 데이터 준비
    print("\n3. 데이터 준비 중...")
    target_path.mkdir(parents=True, exist_ok=True)
    
    fake_target = target_path / "fake"
    real_target = target_path / "real"
    
    # FAKE → fake
    if fake_source.exists():
        if fake_target.exists():
            shutil.rmtree(fake_target)
        shutil.copytree(fake_source, fake_target)
        fake_count = len(list(fake_target.glob("*")))
        print(f"FAKE → fake: {fake_count:,}개")
    
    # REAL → real
    if real_source.exists():
        if real_target.exists():
            shutil.rmtree(real_target)
        shutil.copytree(real_source, real_target)
        real_count = len(list(real_target.glob("*")))
        print(f"REAL → real: {real_count:,}개")
    
    # 4. 정리
    print("\n4. 임시 파일 정리 중...")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if zip_file.exists():
        zip_file.unlink()  # ZIP 파일 삭제 (선택사항)
    
    print("\n" + "="*50)
    print("데이터 준비 완료!")
    print("="*50)
    print(f"위치: {target_path.absolute()}")
    print(f"FAKE: {fake_count:,}개")
    print(f"REAL: {real_count:,}개")
    print(f"총: {fake_count + real_count:,}개")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str, default="data/realifake.zip",
                       help="ZIP 파일 경로")
    parser.add_argument("--target", type=str, default="data/train",
                       help="대상 폴더")
    
    args = parser.parse_args()
    
    success = process_kaggle_data(args.zip, args.target)
    
    if success:
        print("\n다음 단계:")
        print("  !python notebooks/data_pipeline.py  # 데이터 로더 테스트")
    else:
        exit(1)
