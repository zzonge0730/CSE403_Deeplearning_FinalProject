"""
데이터 준비 및 전처리 스크립트
Realifake 폴더의 FAKE/REAL 구조를 프로젝트에 맞게 변환
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse


def check_data_structure(source_dir):
    """데이터 구조 확인"""
    print("="*50)
    print("데이터 구조 확인")
    print("="*50)
    
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {source_dir}")
        return None
    
    # FAKE, REAL 폴더 확인
    fake_dir = source_path / "FAKE"
    real_dir = source_path / "REAL"
    
    structure = {
        "source": str(source_path),
        "has_fake": fake_dir.exists(),
        "has_real": real_dir.exists(),
        "fake_count": 0,
        "real_count": 0,
        "fake_files": [],
        "real_files": []
    }
    
    if structure["has_fake"]:
        fake_files = list(fake_dir.glob("*"))
        structure["fake_count"] = len(fake_files)
        structure["fake_files"] = [f.name for f in fake_files[:5]]  # 샘플만
    
    if structure["has_real"]:
        real_files = list(real_dir.glob("*"))
        structure["real_count"] = len(real_files)
        structure["real_files"] = [f.name for f in real_files[:5]]  # 샘플만
    
    print(f"\n소스 디렉토리: {structure['source']}")
    print(f"FAKE 폴더 존재: {structure['has_fake']}")
    if structure["has_fake"]:
        print(f"  - 파일 수: {structure['fake_count']:,}")
        print(f"  - 샘플 파일: {structure['fake_files'][:3]}")
    
    print(f"REAL 폴더 존재: {structure['has_real']}")
    if structure["real_count"]:
        print(f"  - 파일 수: {structure['real_count']:,}")
        print(f"  - 샘플 파일: {structure['real_files'][:3]}")
    
    return structure


def check_image_format(image_path):
    """이미지 형식 및 크기 확인"""
    try:
        img = Image.open(image_path)
        return {
            "format": img.format,
            "size": img.size,
            "mode": img.mode,
            "valid": True
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }


def analyze_images(source_dir, sample_size=100):
    """이미지 샘플 분석"""
    print("\n" + "="*50)
    print("이미지 형식 분석 (샘플)")
    print("="*50)
    
    source_path = Path(source_dir)
    fake_dir = source_path / "FAKE"
    real_dir = source_path / "REAL"
    
    formats = {}
    sizes = []
    invalid_files = []
    
    # FAKE 이미지 샘플 분석
    if fake_dir.exists():
        fake_files = list(fake_dir.glob("*"))[:sample_size]
        for img_path in tqdm(fake_files, desc="FAKE 이미지 분석"):
            info = check_image_format(img_path)
            if info["valid"]:
                fmt = info["format"] or "Unknown"
                formats[fmt] = formats.get(fmt, 0) + 1
                sizes.append(info["size"])
            else:
                invalid_files.append((img_path.name, info.get("error", "Unknown")))
    
    # REAL 이미지 샘플 분석
    if real_dir.exists():
        real_files = list(real_dir.glob("*"))[:sample_size]
        for img_path in tqdm(real_files, desc="REAL 이미지 분석"):
            info = check_image_format(img_path)
            if info["valid"]:
                fmt = info["format"] or "Unknown"
                formats[fmt] = formats.get(fmt, 0) + 1
                sizes.append(info["size"])
            else:
                invalid_files.append((img_path.name, info.get("error", "Unknown")))
    
    print(f"\n이미지 형식 분포:")
    for fmt, count in formats.items():
        print(f"  {fmt}: {count}개")
    
    if sizes:
        avg_size = (sum(s[0] for s in sizes) / len(sizes), 
                   sum(s[1] for s in sizes) / len(sizes))
        print(f"\n평균 이미지 크기: {avg_size[0]:.0f}x{avg_size[1]:.0f}")
    
    if invalid_files:
        print(f"\n⚠️ 손상된 파일: {len(invalid_files)}개")
        for name, error in invalid_files[:5]:
            print(f"  - {name}: {error}")
    
    return {
        "formats": formats,
        "avg_size": avg_size if sizes else None,
        "invalid_count": len(invalid_files)
    }


def prepare_data(source_dir, target_dir="data/train", copy_mode=True, 
                lowercase=True, check_images=True):
    """
    데이터 준비 및 변환
    
    Args:
        source_dir: Realifake 폴더 경로 (FAKE, REAL 포함)
        target_dir: 대상 폴더 (data/train)
        copy_mode: True면 복사, False면 이동
        lowercase: 폴더 이름을 소문자로 변환 (FAKE -> fake, REAL -> real)
        check_images: 이미지 유효성 검사 수행
    """
    print("\n" + "="*50)
    print("데이터 준비 시작")
    print("="*50)
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 대상 폴더 생성
    fake_target = target_path / ("fake" if lowercase else "FAKE")
    real_target = target_path / ("real" if lowercase else "REAL")
    
    fake_target.mkdir(parents=True, exist_ok=True)
    real_target.mkdir(parents=True, exist_ok=True)
    
    # 소스 폴더 확인
    fake_source = source_path / "FAKE"
    real_source = source_path / "REAL"
    
    if not fake_source.exists() and not real_source.exists():
        print(f"❌ FAKE 또는 REAL 폴더를 찾을 수 없습니다: {source_dir}")
        return False
    
    copied_count = {"fake": 0, "real": 0}
    skipped_count = {"fake": 0, "real": 0}
    error_count = {"fake": 0, "real": 0}
    
    # FAKE 이미지 복사/이동
    if fake_source.exists():
        print(f"\nFAKE 이미지 처리 중...")
        fake_files = list(fake_source.glob("*"))
        
        for img_path in tqdm(fake_files, desc="FAKE"):
            try:
                # 이미지 유효성 검사
                if check_images:
                    info = check_image_format(img_path)
                    if not info["valid"]:
                        error_count["fake"] += 1
                        continue
                
                target_file = fake_target / img_path.name
                
                # 이미 존재하는 파일 건너뛰기
                if target_file.exists():
                    skipped_count["fake"] += 1
                    continue
                
                if copy_mode:
                    shutil.copy2(img_path, target_file)
                else:
                    shutil.move(str(img_path), str(target_file))
                
                copied_count["fake"] += 1
            except Exception as e:
                error_count["fake"] += 1
                print(f"\n⚠️ 오류 ({img_path.name}): {e}")
    
    # REAL 이미지 복사/이동
    if real_source.exists():
        print(f"\nREAL 이미지 처리 중...")
        real_files = list(real_source.glob("*"))
        
        for img_path in tqdm(real_files, desc="REAL"):
            try:
                # 이미지 유효성 검사
                if check_images:
                    info = check_image_format(img_path)
                    if not info["valid"]:
                        error_count["real"] += 1
                        continue
                
                target_file = real_target / img_path.name
                
                # 이미 존재하는 파일 건너뛰기
                if target_file.exists():
                    skipped_count["real"] += 1
                    continue
                
                if copy_mode:
                    shutil.copy2(img_path, target_file)
                else:
                    shutil.move(str(img_path), str(target_file))
                
                copied_count["real"] += 1
            except Exception as e:
                error_count["real"] += 1
                print(f"\n⚠️ 오류 ({img_path.name}): {e}")
    
    # 결과 출력
    print("\n" + "="*50)
    print("처리 완료")
    print("="*50)
    print(f"\nFAKE:")
    print(f"  복사/이동: {copied_count['fake']:,}개")
    print(f"  건너뛰기: {skipped_count['fake']:,}개")
    print(f"  오류: {error_count['fake']:,}개")
    
    print(f"\nREAL:")
    print(f"  복사/이동: {copied_count['real']:,}개")
    print(f"  건너뛰기: {skipped_count['real']:,}개")
    print(f"  오류: {error_count['real']:,}개")
    
    print(f"\n총 처리: {copied_count['fake'] + copied_count['real']:,}개")
    print(f"대상 폴더: {target_path.absolute()}")
    
    return True


def create_symlink(source_dir, target_dir="data/train"):
    """심볼릭 링크 생성 (복사 없이)"""
    print("\n" + "="*50)
    print("심볼릭 링크 생성")
    print("="*50)
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    fake_source = source_path / "FAKE"
    real_source = source_path / "REAL"
    
    fake_target = target_path / "fake"
    real_target = target_path / "real"
    
    try:
        if fake_source.exists():
            if fake_target.exists():
                fake_target.unlink()
            fake_target.symlink_to(fake_source)
            print(f"✓ FAKE 링크 생성: {fake_target} -> {fake_source}")
        
        if real_source.exists():
            if real_target.exists():
                real_target.unlink()
            real_target.symlink_to(real_source)
            print(f"✓ REAL 링크 생성: {real_target} -> {real_source}")
        
        return True
    except Exception as e:
        print(f"❌ 심볼릭 링크 생성 실패: {e}")
        print("   Windows에서는 관리자 권한이 필요할 수 있습니다.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터 준비 및 전처리")
    parser.add_argument("--source", type=str, required=True,
                       help="Realifake 폴더 경로 (FAKE, REAL 포함)")
    parser.add_argument("--target", type=str, default="data/train",
                       help="대상 폴더 (기본: data/train)")
    parser.add_argument("--mode", type=str, choices=["copy", "move", "link"],
                       default="copy", help="복사/이동/링크 모드")
    parser.add_argument("--no-check", action="store_true",
                       help="이미지 유효성 검사 건너뛰기")
    parser.add_argument("--analyze-only", action="store_true",
                       help="분석만 수행하고 변환하지 않음")
    
    args = parser.parse_args()
    
    # 데이터 구조 확인
    structure = check_data_structure(args.source)
    
    if structure is None:
        exit(1)
    
    # 이미지 분석
    if not args.analyze_only:
        analyze_images(args.source, sample_size=100)
    
    # 데이터 준비
    if not args.analyze_only:
        if args.mode == "link":
            create_symlink(args.source, args.target)
        else:
            prepare_data(
                args.source,
                args.target,
                copy_mode=(args.mode == "copy"),
                check_images=not args.no_check
            )
        
        print("\n✅ 데이터 준비 완료!")
        print(f"다음 단계: python notebooks/data_pipeline.py로 데이터 로더 테스트")
