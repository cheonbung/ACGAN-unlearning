# 🛡️ Discriminator-Guided Unlearning: A Framework for Selective Forgetting in Conditional GANs

[![Language](https://img.shields.io/badge/language-English-orange.svg)](./README.md)
[![Language](https://img.shields.io/badge/language-Korean-blue.svg)](./README_KR.md)

본 리포지토리는 **TRUST-AI (ECAI 2025)** 워크샵에 채택된 논문 **"Discriminator-Guided Unlearning: A Framework for Selective Forgetting in Conditional GANs"**의 공식 구현체 및 실험 코드를 포함함.

본 연구는 **ACGAN (Auxiliary Classifier GAN)** 모델에서 특정 클래스 데이터를 선택적으로 삭제(Unlearning)하기 위한 새로운 2단계 프레임워크를 제안함. 생성자(Generator)를 직접 수정하는 대신, 판별자(Discriminator)의 능력을 의도적으로 약화시키고 그 피드백을 통해 생성자를 유도하는 방식을 사용함. 이를 통해 **재학습(Retraining)** 수준의 망각 성능을 달성하면서도 **치명적 망각(Catastrophic Forgetting)** 문제를 효과적으로 완화함.

---

## 1. 주요 기여 (Key Contributions)

*   **판별자 유도 언러닝 (Discriminator-Guided Unlearning)**: 판별자에 혼란을 주어 생성자가 망각 대상 클래스 생성을 스스로 중단하도록 유도하는 메커니즘 제안.
*   **2단계 프레임워크 구조**:
    *   **1단계 (Soft Forgetting)**: 판별자가 망각 대상 클래스를 제대로 인식하지 못하도록 약화시킴.
    *   **2단계 (Fine-tuning)**: 약화된 판별자의 피드백을 사용하여 생성자를 미세 조정함.
*   **포괄적 검증**:
    *   MNIST, FashionMNIST, SVHN, CIFAR-10 데이터셋에 대한 정량적 평가 (FID, KID, IS).
    *   **멤버십 추론 공격(MIA)** 및 **모델 역전 공격(Inversion Attack)**을 통한 프라이버시 보호 성능 검증.
*   **실험 효율성**: 단일 GPU 및 다중 GPU (DDP) 환경을 완벽 지원하며, 재학습 대비 획기적인 시간 단축 달성.

---

## 2. 프로젝트 구조 (Project Structure)

```text
acgan_unlearning_project/
│
├── acgan/                     # [Package] 핵심 로직
│   ├── dataset.py             # 데이터 로드 및 전처리 (DDP 지원)
│   ├── model.py               # ACGAN 모델 및 Efficient GAN Loss (Hinge Loss)
│   ├── trainer.py             # 단계별 학습 함수 (Original, Baseline, Unlearning)
│   └── utils.py               # 로깅 및 시각화 유틸리티
│
├── train.py                   # [Main] 학습 실행 스크립트
├── evaluate_all.py            # [Eval] 생성 품질 및 망각 성능 평가 (FID, IS)
├── evaluate_mia.py            # [Eval] 멤버십 추론 공격 방어 성능 평가
├── evaluate_inversion.py      # [Eval] 모델 역전 공격 시각화 평가
├── config.json                # 실험 설정 및 하이퍼파라미터
├── requirements.txt           # 의존성 패키지
└── README_KR.md               # 프로젝트 설명서 (국문)
```

---

## 3. 실행 방법 (Usage)

### 3.1. 환경 설정 및 설치
```bash
git clone <repository_url>
cd acgan_unlearning_project
pip install -r requirements.txt
```

### 3.2. 설정 (`config.json`)
`config.json` 파일에서 망각 대상 클래스(`forget_label`) 및 학습 파라미터 수정 가능.

### 3.3. 학습 (`train.py`)
Single-GPU 및 Multi-GPU 환경을 자동 감지하여 실행함.
```bash
# 단일 GPU 실행
python train.py

# 다중 GPU 실행 (예: GPU 2개)
torchrun --standalone --nproc_per_node=2 train.py
```

### 3.4. 평가 및 분석
*   **성능 평가**: `python evaluate_all.py` (FID, IS 등 측정 후 CSV 저장)
*   **프라이버시 평가**: `python evaluate_mia.py` (MIA 공격 정확도 측정)
*   **시각적 검증**: `python evaluate_inversion.py` (모델 역전 공격을 통한 이미지 복원 시도)

---

## 4. 인용 (Citation)

본 코드를 연구에 활용할 경우, 아래 논문을 인용 바람.

```bibtex
@inproceedings{lee2025discriminator,
  title={Discriminator-Guided Unlearning: A Framework for Selective Forgetting in Conditional GANs},
  author={Lee, Byeongcheon and Kim, Sangmin and Park, Sungwoo and Rho, Seungmin and Lee, Mi Young},
  booktitle={TRUST-AI: The European Workshop on Trustworthy AI (ECAI 2025)},
  year={2025},
  organization={European Conference of Artificial Intelligence}
}
```

---

## 5. 특허 (Patent)

본 연구 결과물은 대한민국 특허청에 출원됨.

*   **발명의 명칭**: 판별기 기반 조건부 생성적 적대 신경망에서의 선택적 데이터 망각 방법 그 장치
    *   (METHOD FOR SELECTIVE DATA FORGETTING IN DISCRIMINATOR-BASED CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS)
*   **출원 번호**: 10-2025-0133282
*   **출원 일자**: 2025.09.17
*   **출원인**: 중앙대학교 산학협력단
*   **발명자**: 노승민, 김상민, 이미영, 이병천

---

## 6. 라이선스 (License)

이 프로젝트는 **Creative Commons Attribution 4.0 International License (CC BY 4.0)**에 따라 라이선스가 부여됨.

*   **저작권자**: 이병천, 김상민, 박성우, 노승민, 이미영
*   **출처**: TRUST-AI @ ECAI 2025