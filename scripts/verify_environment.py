"""
FLX4-Net 개발 환경 검증 스크립트

실행: python scripts/verify_environment.py
"""

import sys
from pathlib import Path


def print_header(title: str):
    print(f"\n{'='*50}")
    print(f" {title}")
    print('='*50)


def print_status(name: str, status: bool, detail: str = ""):
    icon = "✓" if status else "✗"
    msg = f"  {icon} {name}"
    if detail:
        msg += f": {detail}"
    print(msg)


def check_python():
    print(f"\nPython: {sys.version}")
    version_ok = sys.version_info >= (3, 10)
    print_status("Python 3.10+", version_ok)
    return version_ok


def check_pytorch():
    print_header("PyTorch & CUDA")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print_status("CUDA available", cuda_available)
        
        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            
            props = torch.cuda.get_device_properties(0)
            print(f"  GPU Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            
            # GPU 연산 테스트
            try:
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.matmul(x, x)
                del x, y
                torch.cuda.empty_cache()
                print_status("GPU 연산 테스트", True)
            except Exception as e:
                print_status("GPU 연산 테스트", False, str(e))
                return False
        else:
            print("  ⚠ GPU를 사용할 수 없습니다!")
            print("  → 드라이버 확인: nvidia-smi")
            print("  → PyTorch 재설치: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        
        return cuda_available
        
    except ImportError as e:
        print_status("PyTorch import", False, str(e))
        return False


def check_audio_libs():
    print_header("Audio Libraries")
    
    all_ok = True
    
    # librosa
    try:
        import librosa
        print(f"librosa: {librosa.__version__}")
        
        # 테스트
        import numpy as np
        sr = 44100
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
        rms = librosa.feature.rms(y=audio)
        print_status("librosa 처리", True)
    except ImportError:
        print_status("librosa", False, "설치 필요")
        all_ok = False
    except Exception as e:
        print_status("librosa 테스트", False, str(e))
        all_ok = False
    
    # soundfile
    try:
        import soundfile
        print(f"soundfile: {soundfile.__version__}")
        print_status("soundfile", True)
    except ImportError:
        print_status("soundfile", False, "설치 필요")
        all_ok = False
    
    return all_ok


def check_beat_tracking():
    print_header("Beat Tracking")
    
    backend = None
    
    # Essentia (1순위)
    try:
        import essentia
        import essentia.standard as es
        print(f"essentia: {essentia.__version__}")
        print_status("Essentia", True, "1순위 backend")
        backend = "essentia"
    except ImportError:
        print_status("Essentia", False, "설치 안됨")
    
    # madmom (2순위)
    try:
        import madmom
        print(f"madmom: {madmom.__version__}")
        if backend is None:
            print_status("madmom", True, "fallback으로 사용")
            backend = "madmom"
        else:
            print_status("madmom", True, "백업으로 설치됨")
    except ImportError:
        print_status("madmom", False, "설치 안됨")
    
    # librosa beat (3순위)
    if backend is None:
        print("  → librosa.beat 사용 예정 (정확도 낮음)")
        backend = "librosa"
    
    return backend


def check_data_libs():
    print_header("Data Libraries")
    
    all_ok = True
    
    # numpy
    try:
        import numpy as np
        print(f"numpy: {np.__version__}")
    except ImportError:
        print_status("numpy", False)
        all_ok = False
    
    # pandas
    try:
        import pandas as pd
        print(f"pandas: {pd.__version__}")
    except ImportError:
        print_status("pandas", False)
        all_ok = False
    
    # pyarrow (parquet)
    try:
        import pyarrow
        print(f"pyarrow: {pyarrow.__version__}")
        
        # Parquet 테스트
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        test_path = Path("_test_parquet.parquet")
        df.to_parquet(test_path)
        df_loaded = pd.read_parquet(test_path)
        test_path.unlink()
        print_status("Parquet 읽기/쓰기", True)
    except ImportError:
        print_status("pyarrow", False, "설치 필요")
        all_ok = False
    except Exception as e:
        print_status("Parquet 테스트", False, str(e))
        all_ok = False
    
    # scipy
    try:
        import scipy
        print(f"scipy: {scipy.__version__}")
    except ImportError:
        print_status("scipy", False)
        all_ok = False
    
    return all_ok


def check_musdb():
    print_header("MUSDB18")
    
    try:
        import musdb
        print(f"musdb: {musdb.__version__}")
        print_status("musdb 패키지", True)
        print("  → 데이터셋 경로는 config에서 설정 필요")
        return True
    except ImportError:
        print_status("musdb", False, "pip install musdb")
        return False


def check_mir_eval():
    print_header("MIR Evaluation")
    
    try:
        import mir_eval
        import numpy as np
        
        # SDR 계산 테스트
        ref = np.random.randn(44100)
        est = ref + 0.1 * np.random.randn(44100)
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            ref.reshape(1, -1), est.reshape(1, -1)
        )
        print_status("mir_eval", True, f"SDR 테스트 OK ({sdr[0]:.1f} dB)")
        return True
    except ImportError:
        print_status("mir_eval", False, "pip install mir_eval")
        return False
    except Exception as e:
        print_status("mir_eval 테스트", False, str(e))
        return False


def check_mss_training():
    print_header("Music-Source-Separation-Training")
    
    # CMSS_v1 구조: 루트에 바로 있음
    mss_path = Path("Music-Source-Separation-Training")
    
    if mss_path.exists():
        print_status("MSS 레포지토리", True, str(mss_path))
        
        # inference.py 존재 확인
        if (mss_path / "inference.py").exists():
            print_status("inference.py", True)
        else:
            print_status("inference.py", False, "파일 없음")
        
        # configs 폴더 확인
        configs_path = mss_path / "configs"
        if configs_path.exists():
            config_files = list(configs_path.glob("*.yaml"))
            print_status(f"configs/", True, f"{len(config_files)} yaml files")
        
        return True
    else:
        print_status("MSS 레포지토리", False, "클론 필요")
        print("  → git clone https://github.com/ZFTurbo/Music-Source-Separation-Training")
        return False


def check_datasets():
    print_header("Datasets")
    
    # MUSDB18-HQ
    musdb_path = Path("Dataset/musdb18hq")
    if musdb_path.exists():
        test_path = musdb_path / "test"
        train_path = musdb_path / "train"
        test_count = len(list(test_path.iterdir())) if test_path.exists() else 0
        train_count = len(list(train_path.iterdir())) if train_path.exists() else 0
        print_status("MUSDB18-HQ", True, f"{test_count} test, {train_count} train tracks")
    else:
        print_status("MUSDB18-HQ", False, f"not found at {musdb_path}")
    
    # MedleyVox
    medley_path = Path("Dataset/MedleyVox")
    if medley_path.exists():
        subfolders = [d.name for d in medley_path.iterdir() if d.is_dir()]
        print_status("MedleyVox", True, f"subfolders: {subfolders}")
    else:
        print_status("MedleyVox", False, f"not found at {medley_path}")


def main():
    print_header("FLX4-Net 개발 환경 검증")
    
    results = {}
    
    results['python'] = check_python()
    results['pytorch'] = check_pytorch()
    results['audio'] = check_audio_libs()
    results['beat_backend'] = check_beat_tracking()
    results['data'] = check_data_libs()
    results['musdb'] = check_musdb()
    results['mir_eval'] = check_mir_eval()
    results['mss'] = check_mss_training()
    check_datasets()  # 데이터셋 존재 확인 추가
    
    # 요약
    print_header("검증 결과 요약")
    
    critical_ok = all([
        results['python'],
        results['pytorch'],
        results['audio'],
        results['beat_backend'] is not None,
        results['data']
    ])
    
    if critical_ok:
        print("✓ 핵심 환경 구성 완료!")
        print(f"  Beat tracking backend: {results['beat_backend']}")
    else:
        print("✗ 일부 핵심 구성 요소가 누락되었습니다.")
        print("  위의 오류 메시지를 확인하세요.")
    
    optional_ok = results['musdb'] and results['mir_eval'] and results['mss']
    if not optional_ok:
        print("\n⚠ 일부 선택적 구성 요소가 누락되었습니다.")
        print("  Track A 실행 전에 설치가 필요합니다.")
    
    print("\n" + "="*50)
    
    return 0 if critical_ok else 1


if __name__ == "__main__":
    sys.exit(main())
