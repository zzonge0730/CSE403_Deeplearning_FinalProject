"""
Colab에서 Kaggle 데이터셋을 쉽게 다운로드하는 헬퍼 함수
"""

def setup_kaggle_colab():
    """Colab에서 Kaggle API 설정"""
    from google.colab import files
    import os
    
    print("="*50)
    print("Kaggle API 설정")
    print("="*50)
    print("\n1. kaggle.json 파일을 업로드하세요")
    print("   (https://www.kaggle.com/settings 에서 다운로드)")
    
    uploaded = files.upload()
    
    if 'kaggle.json' not in uploaded:
        print("❌ kaggle.json 파일을 찾을 수 없습니다.")
        return False
    
    # 토큰 배치
    os.makedirs('/root/.kaggle', exist_ok=True)
    
    import shutil
    shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')
    os.chmod('/root/.kaggle/kaggle.json', 0o600)
    
    print("✓ Kaggle API 설정 완료")
    return True


def download_realifake_dataset():
    """Realifake 데이터셋 다운로드 및 준비"""
    import subprocess
    import shutil
    from pathlib import Path
    
    print("\n" + "="*50)
    print("Realifake 데이터셋 다운로드")
    print("="*50)
    
    # Kaggle 패키지 설치
    subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)
    
    # 데이터셋 다운로드
    print("\n다운로드 중... (시간이 걸릴 수 있습니다)")
    subprocess.run([
        'kaggle', 'datasets', 'download',
        '-d', 'sattyam96/realifake',
        '-p', 'data/'
    ], check=True)
    
    # 압축 해제
    print("\n압축 해제 중...")
    subprocess.run([
        'unzip', '-q', 'data/realifake.zip',
        '-d', 'data/temp'
    ], check=True)
    
    # 데이터 준비
    print("\n데이터 준비 중...")
    train_dir = Path('data/train')
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # FAKE → fake
    fake_source = Path('data/temp/FAKE')
    fake_target = train_dir / 'fake'
    if fake_source.exists():
        shutil.copytree(fake_source, fake_target, dirs_exist_ok=True)
        fake_count = len(list(fake_target.glob('*')))
        print(f"✓ FAKE → fake: {fake_count:,}개")
    
    # REAL → real
    real_source = Path('data/temp/REAL')
    real_target = train_dir / 'real'
    if real_source.exists():
        shutil.copytree(real_source, real_target, dirs_exist_ok=True)
        real_count = len(list(real_target.glob('*')))
        print(f"✓ REAL → real: {real_count:,}개")
    
    # 임시 파일 정리
    shutil.rmtree('data/temp', ignore_errors=True)
    Path('data/realifake.zip').unlink(missing_ok=True)
    
    print("\n✅ 데이터 준비 완료!")
    print(f"위치: {train_dir.absolute()}")
    
    return True


if __name__ == "__main__":
    # Colab에서 사용 예시
    print("Colab에서 사용하려면:")
    print("1. setup_kaggle_colab() 실행")
    print("2. download_realifake_dataset() 실행")
