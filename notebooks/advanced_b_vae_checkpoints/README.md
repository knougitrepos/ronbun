# Advanced Beta-VAE 체크포인트 관리

이 디렉토리는 고급 Beta-VAE 모델의 체크포인트를 체계적으로 관리하기 위한 공간입니다.

## 📁 파일 구조

### 체크포인트 파일 패턴
- `best_model_{실험ID}.pth`: 최적 성능을 달성한 모델
- `latest_model_{실험ID}.pth`: 가장 최근 훈련 상태 (훈련 재개용)
- `epoch_{에포크번호}_{실험ID}.pth`: 주기적 백업 체크포인트
- `config_{실험ID}.json`: 실험 설정 및 메타데이터

### 실험 ID 형식
- 타임스탬프 기반: `YYYYMMDD_HHMMSS` (예: `20250122_143015`)

## 🚀 체크포인트 사용법

### 1. 최적 모델 로딩 (배포/추론용)
```python
checkpoint = torch.load('advanced_b_vae_checkpoints/best_model_20250122_143015.pth')
model = AdvancedBetaVAE(
    input_dim=checkpoint['hyperparameters']['input_dim'],
    latent_dim=checkpoint['hyperparameters']['latent_dim'],
    beta=checkpoint['hyperparameters']['beta'],
    dropout_rate=checkpoint['hyperparameters']['dropout_rate']
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"최적 모델 로딩 완료 - 검증 손실: {checkpoint['best_val_loss']:.4f}")
```

### 2. 훈련 재개
```python
checkpoint = torch.load('advanced_b_vae_checkpoints/latest_model_20250122_143015.pth')

# 모델 재생성
model = AdvancedBetaVAE(...).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# 옵티마이저 복원
optimizer = torch.optim.AdamW(model.parameters(), lr=checkpoint['hyperparameters']['base_lr'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 훈련 기록 복원
start_epoch = checkpoint['epoch'] + 1
train_losses = checkpoint['train_losses']
val_losses = checkpoint['val_losses']

print(f"훈련 재개 - 에포크 {start_epoch}부터 시작")
```

### 3. 설정 확인
```python
import json
with open('advanced_b_vae_checkpoints/config_20250122_143015.json', 'r') as f:
    config = json.load(f)

print(f"실험 설정:")
print(f"  - 모델 파라미터: {config['total_params']:,}")
print(f"  - 압축률: {config['latent_dim'] / config['input_dim'] * 100:.1f}%")
print(f"  - 디바이스: {config['device']}")
```

## 📊 체크포인트 내용

각 체크포인트 파일에는 다음 정보가 포함됩니다:

### 모델 관련
- `model_state_dict`: 모델 가중치
- `optimizer_state_dict`: 옵티마이저 상태

### 훈련 기록
- `train_losses`: 에포크별 훈련 손실
- `val_losses`: 에포크별 검증 손실
- `learning_rates`: 에포크별 학습률
- `recon_losses`: 재구성 손실 기록
- `kld_losses`: KL Divergence 손실 기록

### 실험 정보
- `experiment_id`: 실험 고유 식별자
- `epoch`: 현재 에포크
- `best_val_loss`: 최적 검증 손실
- `best_epoch`: 최적 성능 달성 에포크
- `hyperparameters`: 모든 하이퍼파라미터
- `timestamp`: 저장 시점

### 추가 정보 (해당하는 경우)
- `early_stopped`: 조기 종료 여부
- `stopped_epoch`: 조기 종료 에포크
- `training_completed`: 훈련 완료 여부
- `compression_ratio`: 차원 압축률

## 💾 관리 권장사항

### 정기적 백업
```bash
# 중요한 체크포인트를 별도 위치에 백업
cp best_model_*.pth /backup/location/
```

### 저장공간 관리
```python
# 오래된 주기적 체크포인트 정리 (최근 5개만 유지)
import os
import glob

checkpoint_files = glob.glob('epoch_*_*.pth')
checkpoint_files.sort(key=os.path.getmtime, reverse=True)
for old_file in checkpoint_files[5:]:  # 최근 5개 제외하고 삭제
    os.remove(old_file)
    print(f"삭제됨: {old_file}")
```

### 실험 추적
- config.json 파일을 활용하여 실험 비교
- 실험 ID로 관련 파일들을 그룹화
- 성능 기록을 통한 모델 개선 추적

## 🔧 문제 해결

### 체크포인트 로딩 실패
1. 파일 경로 확인
2. 모델 구조 일치 확인 (input_dim, latent_dim 등)
3. PyTorch 버전 호환성 확인

### 메모리 부족
```python
# CPU에서 로딩 후 GPU로 이동
checkpoint = torch.load(file_path, map_location='cpu')
model = model.to(device)
```

### 부분적 로딩
```python
# 일부 정보만 필요한 경우
checkpoint = torch.load(file_path)
print(f"훈련 기록: {len(checkpoint['train_losses'])} 에포크")
print(f"최적 성능: {checkpoint['best_val_loss']:.4f}")
```

---

이 체크포인트 시스템을 통해 Beta-VAE 실험의 안정성과 재현성을 보장할 수 있습니다. 