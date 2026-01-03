# evaluate_mia_targeted.py
# (Measures Targeted MIA for Forget Set and Retain Set)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import os
import json
import argparse

# --- 하이퍼파라미터 ---
SHADOW_EPOCHS = 100

# --- 프로젝트 내부 모듈 ---
from acgan.model import ACGANGenerator, ACGANDiscriminator, EfficientGANLoss
from acgan.dataset import get_dataloader

# ===================================================================
#  모델 로딩 및 공통 유틸리티
# ===================================================================
def load_models(g_path, d_path, config, dataset_name, device):
    d_config = config["dataset_configs"][dataset_name]
    img_size, img_channels = d_config["img_size"], d_config["img_channels"]
    generator = ACGANGenerator(latent_dim=config["latent_dim"], num_classes=config["num_classes"], label_embedding_dim=config["label_embedding_dim_g"], img_size=img_size, img_channels=img_channels).to(device)
    if os.path.exists(g_path):
        g_state_dict = torch.load(g_path, map_location=device, weights_only=True)
        new_g_state_dict = {k.replace('module.', ''): v for k, v in g_state_dict.items()}
        generator.load_state_dict(new_g_state_dict)
    generator.eval()
    discriminator = ACGANDiscriminator(num_classes=config["num_classes"], img_size=img_size, img_channels=img_channels).to(device)
    if os.path.exists(d_path):
        d_state_dict = torch.load(d_path, map_location=device, weights_only=True)
        new_d_state_dict = {k.replace('module.', ''): v for k, v in d_state_dict.items()}
        discriminator.load_state_dict(new_d_state_dict)
    discriminator.eval()
    return generator, discriminator

# ===================================================================
#  MIA 공격 함수
# ===================================================================

# =================== 추가된 함수 ===================
@torch.no_grad()
def get_discriminator_scores(discriminator, dataloader, device):
    """화이트박스 공격을 위해 판별자의 신뢰도 점수를 계산합니다."""
    scores = []
    for images, _ in tqdm(dataloader, desc="[WB-DMIA] Getting Scores", leave=False):
        images = images.to(device)
        source_preds, _ = discriminator(images)
        scores.append(torch.sigmoid(source_preds).cpu().numpy())
    return np.concatenate(scores).flatten()
# ===============================================

def run_targeted_attack(scores, membership_labels, class_labels, forget_label):
    """주어진 데이터 그룹(전체, Forget, Retain)에 대해 MIA 정확도를 계산합니다."""
    forget_mask = (class_labels == forget_label)
    forget_scores = scores[forget_mask]
    forget_labels = membership_labels[forget_mask]
    
    retain_mask = (class_labels != forget_label)
    retain_scores = scores[retain_mask]
    retain_labels = membership_labels[retain_mask]
    
    def calculate_acc(sub_scores, sub_labels):
        n_members = int(np.sum(sub_labels))
        if n_members == 0 or len(sub_labels) == n_members: return 0.5
        sorted_indices = np.argsort(-sub_scores)
        predictions = np.zeros_like(sub_labels, dtype=int)
        predictions[sorted_indices[:n_members]] = 1
        return accuracy_score(sub_labels, predictions)

    acc_overall = calculate_acc(scores, membership_labels)
    acc_forget = calculate_acc(forget_scores, forget_labels)
    acc_retain = calculate_acc(retain_scores, retain_labels)
    
    return acc_overall, acc_forget, acc_retain

# ===================================================================
#  블랙박스 공격 (섀도우 모델링)
# ===================================================================
def train_shadow_model(shadow_train_loader, config, dataset_name, device):
    print(f"\n--- [Black-box] Training Shadow Model for {SHADOW_EPOCHS} epochs ---")
    d_config = config["dataset_configs"][dataset_name]; img_size, img_channels = d_config["img_size"], d_config["img_channels"]
    shadow_generator = ACGANGenerator(config["latent_dim"], config["num_classes"], config["label_embedding_dim_g"], img_size, img_channels).to(device)
    shadow_discriminator = ACGANDiscriminator(config["num_classes"], img_size, img_channels).to(device)
    gan_loss_fn = EfficientGANLoss().to(device); aux_criterion = nn.CrossEntropyLoss().to(device)
    g_optimizer = optim.Adam(shadow_generator.parameters(), lr=config["lr_g"], betas=(0.5, 0.999))
    d_optimizer = optim.Adam(shadow_discriminator.parameters(), lr=config["lr_d"], betas=(0.5, 0.999))
    epoch_pbar = tqdm(range(SHADOW_EPOCHS), desc="[BB] Shadow Training", leave=False)
    for epoch in epoch_pbar:
        for real_imgs, labels in shadow_train_loader:
            real_imgs, labels, batch_size = real_imgs.to(device), labels.to(device), real_imgs.size(0)
            d_optimizer.zero_grad()
            real_src_pred, real_cls_pred = shadow_discriminator(real_imgs)
            z_d = torch.randn(batch_size, config["latent_dim"], device=device); gen_labels_d = torch.randint(0, config["num_classes"], (batch_size,), device=device)
            with torch.no_grad(): fake_imgs = shadow_generator(z_d, gen_labels_d)
            fake_src_pred, fake_cls_pred = shadow_discriminator(fake_imgs)
            d_loss_gan = gan_loss_fn.discriminator_loss(real_src_pred, fake_src_pred); d_loss_aux = aux_criterion(real_cls_pred, labels) + aux_criterion(fake_cls_pred, gen_labels_d); d_loss = d_loss_gan + config["aux_loss_weight"] * d_loss_aux
            d_loss.backward(); d_optimizer.step()
            g_optimizer.zero_grad()
            z_g, gen_labels_g = torch.randn(batch_size, config["latent_dim"], device=device), torch.randint(0, config["num_classes"], (batch_size,), device=device)
            gen_imgs_g = shadow_generator(z_g, gen_labels_g); g_src_output, g_cls_output = shadow_discriminator(gen_imgs_g)
            g_loss_gan = gan_loss_fn.generator_loss(g_src_output); g_loss_aux = aux_criterion(g_cls_output, gen_labels_g); g_loss = g_loss_gan + config["aux_loss_weight"] * g_loss_aux
            g_loss.backward(); g_optimizer.step()
    shadow_discriminator.eval()
    return shadow_discriminator

@torch.no_grad()
def extract_attack_features_and_labels(discriminator, dataloader, device):
    all_features = []; all_class_labels = []
    for images, labels in tqdm(dataloader, desc="[BB] Extracting Features", leave=False):
        images = images.to(device)
        _, class_preds = discriminator(images)
        features = F.softmax(class_preds, dim=1).cpu().numpy()
        all_features.append(features)
        all_class_labels.append(labels.cpu().numpy())
    return np.concatenate(all_features), np.concatenate(all_class_labels)

def train_attack_model(shadow_discriminator, shadow_train_loader, shadow_test_loader, device):
    print("--- [Black-box] Training Attack Model ---")
    member_features, _ = extract_attack_features_and_labels(shadow_discriminator, shadow_train_loader, device)
    non_member_features, _ = extract_attack_features_and_labels(shadow_discriminator, shadow_test_loader, device)
    X_attack = np.concatenate([member_features, non_member_features])
    y_attack = np.concatenate([np.ones(len(member_features)), np.zeros(len(non_member_features))])
    attack_model = LogisticRegression(solver='liblinear', class_weight='balanced')
    attack_model.fit(X_attack, y_attack)
    print("--- [Black-box] Attack Model is Ready ---")
    return attack_model

# ===================================================================
#  메인 실행 로직
# ===================================================================
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    results = []
    enabled_datasets = {k: v for k, v in config["dataset_configs"].items() if v["enabled"]}
    
    for dataset_name, d_config in tqdm(enabled_datasets.items(), desc="All Datasets"):
        print(f"\n===== Targeted MIA Evaluation for Dataset: {dataset_name} =====")
        
        temp_config = config.copy(); temp_config['rank'] = -1
        full_train_ds = get_dataloader(dataset_name, train=True, **temp_config)[0].dataset
        full_test_ds = get_dataloader(dataset_name, train=False, **temp_config)[0].dataset
        
        target_train_indices, shadow_train_indices = random_split(range(len(full_train_ds)), [len(full_train_ds)//2, len(full_train_ds) - len(full_train_ds)//2])
        target_test_indices, shadow_test_indices = random_split(range(len(full_test_ds)), [len(full_test_ds)//2, len(full_test_ds) - len(full_test_ds)//2])
        
        target_train_set = Subset(full_train_ds, target_train_indices); target_test_set = Subset(full_test_ds, target_test_indices)
        shadow_train_set = Subset(full_train_ds, shadow_train_indices); shadow_test_set = Subset(full_test_ds, shadow_test_indices)
        target_train_loader = DataLoader(target_train_set, batch_size=config['batch_size']); target_test_loader = DataLoader(target_test_set, batch_size=config['batch_size'])
        shadow_train_loader = DataLoader(shadow_train_set, batch_size=config['batch_size']); shadow_test_loader = DataLoader(shadow_test_set, batch_size=config['batch_size'])
        
        shadow_discriminator = train_shadow_model(shadow_train_loader, config, dataset_name, device)
        bb_attack_model = train_attack_model(shadow_discriminator, shadow_train_loader, shadow_test_loader, device)

        model_types_to_evaluate = {
            "Original": "Original", "Retrain": "Baseline_Retrain", "Finetune": "Baseline_Finetune",
            **{f"Unlearn_s={s:.1f}": f"Unlearning_soft_target_{s:.1f}" for s in config["soft_label_target_list"]}
        }
        for model_name, folder_name in tqdm(model_types_to_evaluate.items(), desc=f"Attacking {dataset_name} Models"):
            d_path = f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}/{folder_name}/models/discriminator_final_soft_{float(model_name.split('=')[-1]):.1f}.pth" if "Unlearn" in model_name else f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}/{folder_name}/models/discriminator.pth"
            if not os.path.exists(d_path): continue
            
            _, discriminator = load_models("", d_path, config, dataset_name, device)
            
            member_scores = get_discriminator_scores(discriminator, target_train_loader, device)
            non_member_scores = get_discriminator_scores(discriminator, target_test_loader, device)
            wb_scores = np.concatenate([member_scores, non_member_scores])
            member_features, member_class_labels = extract_attack_features_and_labels(discriminator, target_train_loader, device)
            non_member_features, non_member_class_labels = extract_attack_features_and_labels(discriminator, target_test_loader, device)
            bb_features = np.concatenate([member_features, non_member_features])
            membership_labels = np.concatenate([np.ones_like(member_scores), np.zeros_like(non_member_scores)])
            class_labels = np.concatenate([member_class_labels, non_member_class_labels])
            
            _, wb_acc_forget, wb_acc_retain = run_targeted_attack(wb_scores, membership_labels, class_labels, config["forget_label"])
            
            bb_scores_from_preds = bb_attack_model.predict_proba(bb_features)[:, 1]
            _, bb_acc_forget, bb_acc_retain = run_targeted_attack(bb_scores_from_preds, membership_labels, class_labels, config["forget_label"])

            print(f"  - [{model_name}] WB-DMIA (Forget/Retain): {wb_acc_forget:.4f} / {wb_acc_retain:.4f} | BB-DMIA (Forget/Retain): {bb_acc_forget:.4f} / {bb_acc_retain:.4f}")
            results.append({
                "dataset": dataset_name, "model_type": model_name,
                "wb_acc_forget": wb_acc_forget, "wb_acc_retain": wb_acc_retain,
                "bb_acc_forget": bb_acc_forget, "bb_acc_retain": bb_acc_retain
            })
    
    if results:
        results_df = pd.DataFrame(results)
        print("\n\n===== Final Targeted MIA Results ====="); print(results_df.to_string())
        results_df.to_csv("mia_evaluation_results_targeted.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Targeted (Forget/Retain Set) MIA."); parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file.')
    args = parser.parse_args()
    if not os.path.exists(args.config): raise FileNotFoundError(f"Config file not found at: {args.config}")
    with open(args.config, 'r') as f: config = json.load(f)
    main(config)