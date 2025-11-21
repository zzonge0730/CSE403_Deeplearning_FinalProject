"""
Kaggle 데이터셋 다운로드 및 자동 준비 스크립트
"""

import os
import zipfile
import shutil
from pathlib import Path
import subprocess
import json
from tqdm import tqdm


def check_kaggle_api():
    """Kaggle API 설정 확인"""
    print("="*50)
    print("Kaggle API 확인")
    print("="*50)
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("kaggle.json 파일을 찾을 수 없습니다.")
        print("\n설정 방법:")
        print("1. https://www.kaggle.com/settings 에서 API 토큰 다운로드")
        print("2. kaggle.json 파일을 ~/.kaggle/ 폴더에 배치")
        print("3. 권한 설정: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # 권한 확인
    stat = os.stat(kaggle_json)
    if stat.st_mode & 0o077 != 0:
        print("kaggle.json 권한이 너무 열려있습니다.")
        print("권한 설정: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("Kaggle API 설정 확인됨")
    return True


def setup_kaggle_api(kaggle_json_path=None):
    """Kaggle API 설정"""
    print("="*50)
    print("Kaggle API 설정")
    print("="*50)
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    if kaggle_json_path:
        # 제공된 경로에서 복사
        source = Path(kaggle_json_path)
        if source.exists():
            shutil.copy(source, kaggle_dir / "kaggle.json")
            os.chmod(kaggle_dir / "kaggle.json", 0o600)
            print(f"kaggle.json 복사 완료: {source} -> {kaggle_dir / 'kaggle.json'}")
            return True
        else:
            print(f"파일을 찾을 수 없습니다: {kaggle_json_path}")
            return False
    else:
        print("\nKaggle API 토큰이 필요합니다:")
        print("1. https://www.kaggle.com/settings 접속")
        print("2. 'Create New Token' 클릭하여 kaggle.json 다운로드")
        print("3. 다운로드한 파일 경로를 입력하거나")
        print("4. ~/.kaggle/kaggle.json 에 직접 배치")
        return False


def download_dataset(dataset_name, output_dir="data"):
    """
    Kaggle 데이터셋 다운로드
    
    Args:
        dataset_name: 데이터셋 이름 (예: 'sattyam96/realifake')
        output_dir: 다운로드할 디렉토리
    """
    print("\n" + "="*50)
    print(f"데이터셋 다운로드: {dataset_name}")
    print("="*50)
    
    # Kaggle API 확인
    if not check_kaggle_api():
        print("\nKaggle API가 설정되지 않았습니다.")
        print("설정 후 다시 시도하세요.")
        return False
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 다운로드 실행
    try:
        print(f"\n다운로드 중... (이 작업은 시간이 걸릴 수 있습니다)")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print("다운로드 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"다운로드 실패: {e.stderr}")
        return False
    except FileNotFoundError:
        print("kaggle 명령어를 찾을 수 없습니다.")
        print("설치: pip install kaggle")
        return False


def extract_and_prepare(dataset_zip, target_dir="data/train"):
    """
    압축 파일 해제 및 데이터 준비
    
    Args:
        dataset_zip: 다운로드한 ZIP 파일 경로
        target_dir: 대상 디렉토리
    """
    print("\n" + "="*50)
    print("압축 해제 및 데이터 준비")
    print("="*50)
    
    zip_path = Path(dataset_zip)
    if not zip_path.exists():
        print(f"파일을 찾을 수 없습니다: {dataset_zip}")
        return False
    
    # 임시 압축 해제 디렉토리
    temp_dir = zip_path.parent / "temp_extract"
    temp_dir.mkdir(exist_ok=True)
    
    # 압축 해제
    print(f"\n압축 해제 중: {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 진행 상황 표시
            file_list = zip_ref.namelist()
            for file in tqdm(file_list, desc="압축 해제"):
                zip_ref.extract(file, temp_dir)
        print("압축 해제 완료")
    except Exception as e:
        print(f"압축 해제 실패: {e}")
        return False
    
    # 데이터 구조 확인 및 변환
    print("\n데이터 구조 확인 중...")
    
    # 가능한 폴더 구조 확인
    possible_structures = [
        ("FAKE", "REAL"),
        ("fake", "real"),
        ("Fake", "Real"),
        ("train/FAKE", "train/REAL"),
        ("train/fake", "train/real"),
    ]
    
    fake_dir = None
    real_dir = None
    
    for fake_name, real_name in possible_structures:
        fake_path = temp_dir / fake_name
        real_path = temp_dir / real_name
        
        if fake_path.exists() and real_path.exists():
            fake_dir = fake_path
            real_dir = real_path
            print(f"데이터 구조 발견: {fake_name}/, {real_name}/")
            break
    
    if fake_dir is None or real_dir is None:
        # 폴더 구조 출력하여 사용자에게 확인 요청
        print("\n표준 폴더 구조를 찾을 수 없습니다.")
        print("압축 해제된 폴더 구조:")
        for item in sorted(temp_dir.rglob("*"))[:20]:
            if item.is_dir():
                print(f"  📁 {item.relative_to(temp_dir)}")
            elif item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                print(f"  🖼️  {item.relative_to(temp_dir)}")
        print("\n수동으로 데이터를 준비하거나 폴더 구조를 확인하세요.")
        return False
    
    # 대상 디렉토리 생성
    target_path = Path(target_dir)
    fake_target = target_path / "fake"
    real_target = target_path / "real"
    
    fake_target.mkdir(parents=True, exist_ok=True)
    real_target.mkdir(parents=True, exist_ok=True)
    
    # 파일 복사
    print(f"\n데이터 준비 중: {target_dir}")
    
    fake_files = list(fake_dir.glob("*"))
    real_files = list(real_dir.glob("*"))
    
    copied = {"fake": 0, "real": 0}
    
    for img_path in tqdm(fake_files, desc="FAKE 복사"):
        if img_path.is_file():
            target_file = fake_target / img_path.name
            if not target_file.exists():
                shutil.copy2(img_path, target_file)
                copied["fake"] += 1
    
    for img_path in tqdm(real_files, desc="REAL 복사"):
        if img_path.is_file():
            target_file = real_target / img_path.name
            if not target_file.exists():
                shutil.copy2(img_path, target_file)
                copied["real"] += 1
    
    # 임시 파일 정리
    print("\n임시 파일 정리 중...")
    shutil.rmtree(temp_dir)
    zip_path.unlink()  # ZIP 파일 삭제 (선택사항)
    
    print("\n" + "="*50)
    print("데이터 준비 완료!")
    print("="*50)
    print(f"FAKE: {copied['fake']:,}개")
    print(f"REAL: {copied['real']:,}개")
    print(f"총: {copied['fake'] + copied['real']:,}개")
    print(f"위치: {target_path.absolute()}")
    
    return True


def download_and_prepare(dataset_name="sattyam96/realifake", output_dir="data", target_dir="data/train"):
    """
    Kaggle 데이터셋 다운로드 및 자동 준비 (원스톱)
    
    Args:
        dataset_name: Kaggle 데이터셋 이름
        output_dir: 다운로드할 디렉토리
        target_dir: 최종 데이터 위치
    """
    print("="*50)
    print("Kaggle 데이터셋 다운로드 및 준비")
    print("="*50)
    print(f"데이터셋: {dataset_name}")
    print(f"다운로드 위치: {output_dir}")
    print(f"최종 위치: {target_dir}")
    print("="*50)
    
    # 1. Kaggle API 확인
    if not check_kaggle_api():
        print("\nKaggle API 설정이 필요합니다.")
        setup_choice = input("지금 설정하시겠습니까? (y/n): ")
        if setup_choice.lower() == 'y':
            json_path = input("kaggle.json 파일 경로를 입력하세요 (엔터로 건너뛰기): ")
            if json_path:
                if not setup_kaggle_api(json_path):
                    return False
            else:
                print("수동으로 설정하세요: https://www.kaggle.com/settings")
                return False
        else:
            return False
    
    # 2. 데이터셋 다운로드
    if not download_dataset(dataset_name, output_dir):
        return False
    
    # 3. 압축 해제 및 준비
    zip_file = Path(output_dir) / f"{dataset_name.split('/')[-1]}.zip"
    if not zip_file.exists():
        # 다른 가능한 이름 확인
        zip_files = list(Path(output_dir).glob("*.zip"))
        if zip_files:
            zip_file = zip_files[0]
        else:
            print(f"압축 파일을 찾을 수 없습니다: {output_dir}")
            return False
    
    if not extract_and_prepare(zip_file, target_dir):
        return False
    
    print("\n모든 작업 완료!")
    print(f"\n다음 단계:")
    print(f"  python notebooks/data_pipeline.py  # 데이터 로더 테스트")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kaggle 데이터셋 다운로드 및 준비")
    parser.add_argument("--dataset", type=str, default="sattyam96/realifake",
                       help="Kaggle 데이터셋 이름 (기본: sattyam96/realifake)")
    parser.add_argument("--output", type=str, default="data",
                       help="다운로드 위치 (기본: data)")
    parser.add_argument("--target", type=str, default="data/train",
                       help="최종 데이터 위치 (기본: data/train)")
    parser.add_argument("--setup-api", type=str, default=None,
                       help="Kaggle API 설정 (kaggle.json 경로)")
    
    args = parser.parse_args()
    
    # API 설정이 요청된 경우
    if args.setup_api:
        if setup_kaggle_api(args.setup_api):
            print("API 설정 완료")
        else:
            print("API 설정 실패")
            exit(1)
    
    # 다운로드 및 준비
    success = download_and_prepare(args.dataset, args.output, args.target)
    
    if not success:
        exit(1)
