---

# ✨ ACGAN 기반의 선택적 데이터 언러닝(Unlearning)

본 프로젝트는 **ACGAN (Auxiliary Classifier Generative Adversarial Network)** 모델에 **머신 언러닝(Machine Unlearning)** 기술을 적용하여, 특정 클래스(레이블)의 데이터를 학습된 모델에서 선택적으로 제거하는 과정을 구현합니다.

## 1. 주요 특징 (Key Features)

*   **포괄적인 실험 설계**: 제안하는 언러닝 방법론뿐만 아니라, 성능 비교를 위한 두 가지 핵심 **베이스라인(Retrain, Finetune)**을 포함합니다.
*   **유연한 실행 환경**: Single-GPU (`python train.py`)와 Multi-GPU (`torchrun ...`) 환경을 모두 지원합니다.
*   **고성능 모델 아키텍처**: Self-Attention, Residual Block, Spectral Normalization, Hinge Loss 등 최신 GAN 기술을 적용했습니다.
*   **상세한 로깅 및 평가**:
    *   **[수정된 부분]** 모든 학습 단계(Original, Baselines, Unlearning)의 **실행 시간**을 자동으로 측정하고, 최종 결과를 `execution_times.csv` 파일로 종합하여 각 방법론의 효율성을 명확하게 비교할 수 있습니다.
    *   TensorBoard를 통해 모든 실험의 손실 및 이미지 생성을 실시간으로 모니터링합니다.
    *   `evaluate_all.py` 스크립트로 모든 실험의 **성능 지표(FID, IS 등)**를 종합하여 `evaluation_summary.csv` 파일로 자동 저장합니다.
    *   **[수정된 부분]** **멤버십 추론 공격 (MIA)** 및 **모델 역전 공격 (Model Inversion Attack)** 평가 스크립트를 포함하여 망각 성능을 다각도로 검증합니다.

## 2. 프로젝트 구조 (Project Structure)

**[수정된 부분]** 실제 코드의 출력 경로와 평가 스크립트 구성을 정확하게 반영했습니다.

```
acgan_unlearning_project/
│
├── outputs/
│   ├── ACGAN(self_attention+residual_block+hinge_loss)/
│   │   ├── [Dataset_Name]/
│   │   │   ├── Original/            # Stage 1-1 결과
│   │   │   ├── Baseline_Retrain/    # 베이스라인(재학습) 결과
│   │   │   ├── Baseline_Finetune/   # 베이스라인(미세조정) 결과
│   │   │   └── Unlearning_soft.../  # Stage 1-2, 2 결과
│
├── inversion_attack_results/      # [자동 생성] 모델 역전 공격 결과 이미지
│
├── acgan/
│   ├── trainer.py             # 모든 학습(Stage, Baseline) 로직
│   └── ...
│
├── train.py                   # 실험을 시작하는 메인 실행 스크립트
├── evaluate_all.py            # 모든 실험 결과를 종합 평가
├── evaluate_mia.py            # [수정] 타겟화된(Forget/Retain) MIA 평가
├── evaluate_inversion.py      # [추가] 모델 역전 공격 평가
├── execution_times.csv        # [자동 생성] 모든 학습 시간 요약 결과
├── evaluation_summary.csv     # [자동 생성] 모든 성능 평가 요약 결과
├── mia_evaluation_results_targeted.csv # [자동 생성] MIA 평가 결과
├── config.json                # 모든 하이퍼파라미터 관리
└── README.md                  # 프로젝트 설명서
```

## 3. 실행 환경 설정 (Setup)

### 3.1. 사전 요구사항

*   Python 3.8 이상
*   NVIDIA GPU 및 CUDA 11.x 이상

### 3.2. 설치 (Installation)

1.  **프로젝트 클론**
    ```bash
    git clone <저장소_URL>
    cd acgan_unlearning_project
    ```

2.  **필요 라이브러리 설치**
    PyTorch는 사용자의 CUDA 버전에 맞게 설치하는 것을 권장합니다. (공식 홈페이지: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))

    ```bash
    # 나머지 의존성 라이브러리 설치
    pip install -r requirements.txt
    ```

## 4. 모델 학습 및 언러닝 (Training and Unlearning)

### 4.1. `config.json` 설정하기

모든 실험의 하이퍼파라미터는 `config.json` 파일에서 관리됩니다. 학습을 시작하기 전에 이 파일을 열어 원하는 설정을 구성하세요.

*   `dataset_configs`: 학습에 사용할 데이터셋의 `enabled` 값을 `true`로 설정합니다.
*   `forget_label`: 모델에서 잊게 할 클래스(레이블)를 지정합니다.
*   `original_epochs`, `step1_epochs`, `step2_epochs`: 각 단계별 학습 에폭 수를 조절합니다.
*   `run_*_flag`: 특정 학습 단계(Stage, Baseline)의 실행 여부를 `true` 또는 `false`로 제어합니다.

### 4.2. 학습 실행

터미널에서 프로젝트의 최상위 폴더로 이동한 후, 아래 명령어 중 하나를 실행하여 학습을 시작합니다.

#### **Single-GPU 환경**

GPU가 하나인 환경에서는 간단하게 `python` 명령어를 사용합니다.

```bash
python train.py
```

#### **Multi-GPU 환경 (DDP)**

사용 가능한 GPU가 여러 개인 경우, `torchrun`을 사용하여 분산 학습(DDP, Distributed Data Parallel)을 실행하면 학습 속도를 크게 향상시킬 수 있습니다.

*   `--nproc_per_node`: 사용할 GPU의 개수를 지정합니다.

**예시 (GPU 2개 사용):**
```bash
torchrun --standalone --nproc_per_node=2 train.py
```

> **💡 참고**: 학습이 시작되면 `config.json`에 설정된 플래그(`run_*_flag`)에 따라 모든 단계가 순차적으로 자동 실행됩니다. 모든 학습이 완료되면 프로젝트 루트에 `execution_times.csv` 파일이 자동으로 생성됩니다.

### 4.3. 실험 단계별 설명 (Experiment Stage Details)

본 프로젝트는 제안하는 언러닝 방법론의 성능을 객관적으로 검증하기 위해, **두 가지 베이스라인(Baseline) 실험**과 **세 단계의 언러닝(Unlearning) 실험**으로 구성됩니다.

---

#### **비교 기준이 되는 베이스라인 모델 (Baseline Models)**

##### **1. 원본 모델 (Original Model)**
*   **목표**: 모든 비교의 출발점이 되는, 모든 클래스를 완벽하게 학습한 고성능 모델을 생성합니다.
*   **학습 데이터**: 전체 데이터셋
*   **핵심 동작**: 표준 ACGAN 학습을 `original_epochs` 동안 진행합니다.
*   **의미**: 이 모델의 성능이 언러닝 후에도 최대한 보존되어야 할 '기억'의 총량입니다.

##### **2. 재학습 모델 (Retrain from Scratch Baseline)**
*   **목표**: 언러닝 성능의 "이상적인 목표(Gold Standard)". 즉, 잊어야 할 데이터를 처음부터 아예 보지 않은 모델의 성능을 측정합니다.
*   **학습 데이터**: `forget_label`을 제외한 데이터셋
*   **핵심 동작**: 제외된 데이터셋으로 모델을 처음부터 `original_epochs` 동안 재학습시킵니다.
*   **의미**: 언러닝된 모델의 성능은 이 재학습 모델의 성능에 가까워야 하며, 언러닝 시간은 재학습 시간보다 훨씬 짧아야 합니다.

##### **3. 미세조정 모델 (Finetune Baseline)**
*   **목표**: 가장 간단하고 직관적인 언러닝 방법과의 성능을 비교합니다.
*   **학습 데이터**: `forget_label`을 제외한 데이터셋
*   **핵심 동작**: **원본 모델(Original Model)**을 불러와, 제외된 데이터셋으로 짧은 `finetune_epochs` 동안 추가 학습(미세조정)을 진행합니다.
*   **의미**: 이 방법은 `forget_label`을 빠르게 잊지만, 나머지 클래스에 대한 지식도 함께 손상(Catastrophic Forgetting)될 가능성이 높습니다. 제안하는 언러닝 방법이 이 미세조정 방법보다 나머지 클래스의 성능을 더 잘 보존함을 보여주는 것이 중요합니다.

---

#### **제안하는 2단계 언러닝 방법론 (Proposed 2-Stage Unlearning)**

##### **Stage 1-2: 판별자 소프트 망각 (Discriminator Soft Forgetting)**
*   **목표**: 생성자의 품질은 유지하면서, 판별자(Discriminator)가 '잊어야 할 클래스'(`forget_label`)에 대한 판단력을 의도적으로 흐리게 만듭니다.
*   **핵심 동작**:
    *   판별자는 `forget_label`의 진짜 이미지를 봤을 때, `soft_label_target` 값에 따라 이를 '진짜' 또는 '가짜'로 취급하도록 특수한 손실 함수로 학습됩니다.
    *   생성자는 이 단계에서 일반적인 ACGAN 손실 함수로 학습하여 전체적인 이미지 생성 품질을 유지하고 판별자 학습을 돕는 "스파링 파트너" 역할을 합니다.
*   **[수정된 부분] 결과물**: 특정 클래스를 잘 구분하지 못하게 된 `discriminator_step1.pth`와 학습 중간 결과물인 `generator_step1.pth`.

##### **Stage 2: 생성자/판별자 최종 미세 조정 (Final Fine-tuning)**
*   **목표**: '흐려진' 판별자의 가이드를 받아, 생성자(Generator)가 더 이상 `forget_label`의 이미지를 생성하지 않도록 최종적으로 학습을 완료합니다.
*   **핵심 동작**:
    *   **입력**: Stage 1-1의 **원본 생성자**와 Stage 1-2의 **'혼란스러운' 판별자**를 가져와 학습을 시작합니다.
    *   **생성자**: `forget_label`의 이미지를 생성하려고 시도할 때 추가적인 페널티(손실 가중치 `beta`)를 받습니다. 이는 생성자가 해당 클래스의 이미지 생성을 회피하도록 만듭니다.
    *   **판별자**: Stage 1-2와 동일한 '소프트 망각' 손실 함수를 계속 사용하며 학습을 이어갑니다.
*   **[수정된 부분] 결과물**: 최종적으로 언러닝이 완료된 `generator_final_soft_{...}.pth`와 `discriminator_final_soft_{...}.pth`.

## 5. 학습 과정 모니터링 (Monitoring with TensorBoard) 📈

학습 과정 중 생성되는 손실(Loss) 그래프와 이미지 샘플을 실시간으로 확인하려면 TensorBoard를 사용할 수 있습니다.

1.  새 터미널에서 프로젝트의 최상위 폴더로 이동한 후, 아래 명령어를 실행하여 TensorBoard 서버를 시작합니다.
    ```bash
    tensorboard --logdir outputs/ --port 6006
    ```
2.  웹 브라우저를 열고, `http://localhost:6006` 주소로 접속하면 실시간으로 업데이트되는 대시보드를 확인할 수 있습니다.

> #### 💡 **TensorBoard 활용 팁**
>
> 이제 **모든 학습 단계(Original, Baselines, Stage 1-2, Stage 2)의 손실과 생성 이미지가 TensorBoard에 기록**됩니다. 이를 통해 다음과 같은 다양한 비교 분석이 가능합니다.
>
> *   **학습 단계별 비교**: 특정 `soft_target` 값에 대한 Stage 1-2와 Stage 2의 학습 안정성 비교
> *   **언러닝 강도별 비교**: 서로 다른 `soft_target` 값(예: 0.1 vs 0.9)에 따른 Stage 2의 최종 성능 비교

## 6. 모델 평가 (Evaluating the Model) 📊

학습된 모델의 평가는 **성능(Performance)**, **효율성(Efficiency)**, **망각 성능(Forgetting)** 세 가지 측면에서 이루어집니다.

### 6.1. 단일 모델 성능 평가 (`evaluate.py`)

특정 실험 타입과 파라미터를 지정하여 성능(FID, IS 등)을 빠르게 확인하고 싶을 때 사용합니다.

**실행 명령어 형식:**
```bash
# 기본/베이스라인 모델 평가
python evaluate.py --dataset [데이터셋] --type [original|retrain|finetune]

# 언러닝 모델 평가 (Stage 2 최종 결과)
python evaluate.py --dataset [데이터셋] --type unlearning_final --soft_target [값]
```

### 6.2. 전체 모델 성능 종합 평가 (`evaluate_all.py`)

`outputs` 폴더에 저장된 **모든 실험 결과**를 자동으로 찾아 한 번에 **성능**을 평가하고, 결과를 분석하기 쉬운 `evaluation_summary.csv` 파일로 저장합니다.

**실행 명령어:**
```bash
python evaluate_all.py
```

### 6.3. [수정] 전체 학습 시간 종합 평가 (`train.py` 실행 시 자동 생성)

`train.py` 스크립트는 각 실험 단계가 완료될 때마다 소요 시간을 자동으로 측정합니다. 모든 학습이 정상적으로 종료되면, 프로젝트 최상위 폴더에 `execution_times.csv` 파일이 생성됩니다. 이 파일에는 각 실험에 걸린 시간이 초 단위로 기록되어 있어, 제안하는 언러닝 방법론이 'Retrain' 베이스라인에 비해 얼마나 **효율적**인지 정량적으로 비교할 수 있습니다.

**생성되는 `execution_times.csv` 파일 예시:**
| dataset | stage | soft_target | gpus | execution_time_seconds |
| :--- | :--- | :--- | :--- | :--- |
| SVHN | Original | N/A | 1 | 3621.45 |
| SVHN | Baseline_Retrain | N/A | 1 | 3580.12 |
| SVHN | Baseline_Finetune | N/A | 1 | 895.50 |
| SVHN | Unlearning_Stage1_2 | 0.0 | 1 | 3605.88 |
| SVHN | Unlearning_Stage2 | 0.0 | 1 | 901.23 |
| ... | ... | ... | ... | ... |

### 6.4. [수정] 타겟화된 멤버십 추론 공격(MIA) 망각 성능 평가 (`evaluate_mia.py`)

이 평가는 모델이 특정 데이터(Forget Set)를 얼마나 잘 "잊었는지"를 측정합니다. **Forget Set (잊어야 할 데이터 그룹)**과 **Retain Set (기억해야 할 데이터 그룹)** 각각에 대해 MIA 공격을 수행하여, 모델이 두 그룹의 데이터를 얼마나 다르게 취급하는지 정확도로 나타냅니다.

*   **White-Box(WB) MIA**: 모델의 판별자 점수를 직접 활용하여 공격합니다.
*   **Black-Box(BB) MIA**: 섀도우 모델링 기법을 사용하여 공격합니다.

**실행 명령어:**
스크립트를 실행하면 `config.json`에 활성화된 모든 데이터셋에 대해 모든 모델 타입의 망각 성능을 평가하고, `mia_evaluation_results_targeted.csv` 파일에 결과를 저장합니다.
```bash
python evaluate_mia.py
```

**결과 해석:**
공격 정확도(`wb_acc_forget`, `bb_acc_forget` 등)가 **0.5 (50%)에 가까울수록** 모델이 해당 데이터 그룹의 멤버십 정보를 노출하지 않아 망각 성능이 우수함을 의미합니다. `Retrain` 모델의 점수가 가장 이상적인 목표치입니다.

### 6.5. [추가] 모델 역전 공격(Model Inversion Attack) 평가 (`evaluate_inversion.py`)

모델 역전 공격은 학습된 모델(특히 판별자)을 이용하여 각 클래스를 대표하는 이미지를 복원하는 기법입니다. 이를 통해 모델이 특정 클래스에 대해 어떤 정보를 "기억"하고 있는지 시각적으로 확인할 수 있습니다. 언러닝이 성공적으로 수행되었다면, `forget_label`에 해당하는 복원 이미지는 품질이 낮거나 알아볼 수 없게 나타나야 합니다.

**실행 명령어:**
스크립트를 실행하면 모든 모델에 대해 각 클래스별 대표 이미지를 생성하고 `inversion_attack_results/[데이터셋]` 폴더에 그리드 이미지로 저장합니다.
```bash
python evaluate_inversion.py
```

**결과 확인:**
생성된 `inversion_attack_results/` 폴더 안의 이미지들을 열어 `Original` 모델과 `Unlearning_Final` 모델의 `forget_label`에 해당하는 이미지를 시각적으로 비교하여 망각 성능을 정성적으로 평가할 수 있습니다.