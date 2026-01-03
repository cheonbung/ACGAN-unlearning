# 🛡️ Discriminator-Guided Unlearning: A Framework for Selective Forgetting in Conditional GANs

[![Language](https://img.shields.io/badge/language-English-orange.svg)](./README.md)
[![Language](https://img.shields.io/badge/language-Korean-blue.svg)](./README_KR.md)

This repository contains the official implementation and experimental code for the paper **"Discriminator-Guided Unlearning: A Framework for Selective Forgetting in Conditional GANs."**

We propose a novel two-step framework for **ACGAN (Auxiliary Classifier GAN)** to selectively remove specific class data from a trained model. Instead of directly modifying the generator, our approach intentionally weakens the discriminator's ability to recognize the target class and uses this "confused" feedback to guide the generator. This effectively mitigates **catastrophic forgetting** while achieving unlearning performance comparable to retraining from scratch.

---

## 1. Key Contributions

*   **Discriminator-Guided Unlearning**: A novel mechanism that induces confusion in the discriminator to guide the generator, avoiding direct damage to the generator's weights.
*   **Two-Step Framework**:
    *   **Step 1 (Soft Forgetting)**: Weakens the discriminator's recognition of the target class.
    *   **Step 2 (Fine-tuning)**: Updates the generator using the feedback from the weakened discriminator.
*   **Comprehensive Evaluation**:
    *   **Quantitative**: Performance measured using **FID**, **KID**, and **IS** on MNIST, FashionMNIST, SVHN, and CIFAR-10.
    *   **Privacy & Robustness**: Verified through **Membership Inference Attacks (MIA)** and **Model Inversion Attacks**.
*   **Flexible Environment**: Supports both **Single-GPU** and **Multi-GPU (DDP)** training.

---

## 2. Project Structure

```text
acgan_unlearning_project/
│
├── acgan/                     # [Package] Core Logic
│   ├── __init__.py
│   ├── dataset.py             # Data loading & Preprocessing
│   ├── model.py               # ACGAN Generator, Discriminator, EfficientGANLoss
│   ├── trainer.py             # Training loops (Original, Baseline, Unlearning Steps)
│   └── utils.py               # Visualization & Logging utilities
│
├── train.py                   # [Main] Training Script (Single/Multi-GPU)
├── evaluate_all.py            # [Eval] Performance Metrics (FID, IS, KID)
├── evaluate_mia.py            # [Eval] Membership Inference Attack (Targeted)
├── evaluate_inversion.py      # [Eval] Model Inversion Attack (Visual Inspection)
├── config.json                # Hyperparameters & Experiment Settings
├── requirements.txt           # Dependencies
└── README.md                  # Documentation (English)
```

---

## 3. Setup

### 3.1. Prerequisites
*   Python 3.8+
*   NVIDIA GPU & CUDA 11.x+
*   PyTorch (CUDA support required)

### 3.2. Installation
```bash
git clone <repository_url>
cd acgan_unlearning_project
pip install -r requirements.txt
```

---

## 4. Usage

### 4.1. Configuration (`config.json`)
Modify `config.json` to set the target class (`forget_label`) and hyperparameters.
*   `soft_label_target_list`: List of soft target values for Step 1 (e.g., `[0.0, 0.1]`).
*   `run_*_flag`: Control execution of Original training, Baselines (Retrain/Finetune), and Unlearning steps.

### 4.2. Training (`train.py`)
The script automatically handles Single-GPU or Multi-GPU (DDP) based on the environment.

```bash
# Single GPU
python train.py

# Multi-GPU (e.g., 2 GPUs)
torchrun --standalone --nproc_per_node=2 train.py
```
*   **Execution Time Logging**: Training times for all stages are automatically saved to `execution_times.csv`.

### 4.3. Evaluation

#### Performance (FID, IS, KID)
Evaluates the generation quality and forgetting effectiveness.
```bash
python evaluate_all.py --output evaluation_summary.csv
```

#### Privacy Assessment (MIA)
Measures the defense capability against Membership Inference Attacks on Forget/Retain sets.
```bash
python evaluate_mia.py
```

#### Visual Inspection (Inversion Attack)
Performs Model Inversion Attack to visually verify if the target class has been unlearned.
```bash
python evaluate_inversion.py
```

---

## 5. Citation

If you use this code, please cite our paper:

```bibtex
@article{lee2025discriminator,
  title={Discriminator-Guided Unlearning: A Framework for Selective Forgetting in Conditional GANs},
  author={Lee, Byeongcheon and Kim, Sangmin and Park, Sungwoo and Rho, Seungmin and Lee, Mi Young},
  year={2025}
}
```

---

## 6. Patent

This technology is patent pending with the Korean Intellectual Property Office (KIPO).

*   **Title**: METHOD FOR SELECTIVE DATA FORGETTING IN DISCRIMINATOR-BASED CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS
*   **Application Number**: 10-2025-0133282
*   **Date of Application**: 2025.09.17
*   **Applicant**: Industry-University Cooperation Foundation, Chung-Ang University
*   **Inventors**: Seungmin Rho, Sangmin Kim, Mi Young Lee, Byeongcheon Lee

---

## 7. License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

*   **Copyright**: Byeongcheon Lee, Sangmin Kim, Sungwoo Park, Seungmin Rho, Mi Young Lee
*   **Source**: Discriminator-Guided Unlearning (2025)