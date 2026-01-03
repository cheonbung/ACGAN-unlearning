# evaluate.py

import torch
import json
import os
import argparse
from tqdm import tqdm
import numpy as np

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from acgan.model import ACGANGenerator
from acgan.dataset import get_dataloader

@torch.no_grad()
def calculate_metrics_for_scenario(generator, config, dataloader, scenario, device, num_images=5000):
    print(f"\n--- Scenario: {scenario.replace('_', ' ').title()} ---")
    inception = InceptionScore().to(device)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=min(1000, num_images // 5), feature=2048).to(device)
    
    # --- 수정된 부분 시작 ---
    d_config = config["dataset_configs"][config["current_dataset_name"]]
    img_channels = d_config["img_channels"]
    # --- 수정된 부분 끝 ---

    print("Processing real images for this scenario...")
    processed_real_images = 0
    for real_imgs, _ in tqdm(dataloader, desc="Real Images"):
        real_imgs_uint8 = ((real_imgs + 1) / 2 * 255).to(torch.uint8).to(device)
        
        # --- 수정된 부분 시작 ---
        # 1채널 이미지를 3채널로 복제
        if img_channels == 1:
            real_imgs_uint8 = real_imgs_uint8.repeat(1, 3, 1, 1)
        # --- 수정된 부분 끝 ---
            
        fid.update(real_imgs_uint8, real=True); kid.update(real_imgs_uint8, real=True)
        processed_real_images += real_imgs.size(0)
        if processed_real_images >= num_images: break
        
    latent_dim, num_classes, forget_label = config["latent_dim"], config["num_classes"], config["forget_label"]
    batch_size = 64
    labels_to_generate = []
    if scenario == 'all_labels': labels_to_generate = list(range(num_classes))
    elif scenario == 'exclude_target': labels_to_generate = [l for l in range(num_classes) if l != forget_label]
    elif scenario == 'target_only': labels_to_generate = [forget_label]
    if not labels_to_generate: return {}
    print(f"Generating {num_images} fake images for labels: {labels_to_generate}...")
    generated_count = 0
    pbar = tqdm(total=num_images, desc=f"Fake Images ({scenario})")
    while generated_count < num_images:
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_labels_np = np.random.choice(labels_to_generate, size=batch_size)
        gen_labels = torch.from_numpy(gen_labels_np).to(device)
        fake_imgs = generator(z, gen_labels)
        fake_imgs_uint8 = ((fake_imgs + 1) / 2 * 255).to(torch.uint8)
        
        # --- 수정된 부분 시작 ---
        # 생성된 가짜 이미지도 1채널이면 3채널로 복제
        if img_channels == 1:
            fake_imgs_uint8 = fake_imgs_uint8.repeat(1, 3, 1, 1)
        # --- 수정된 부분 끝 ---

        inception.update(fake_imgs_uint8); fid.update(fake_imgs_uint8, real=False); kid.update(fake_imgs_uint8, real=False)
        generated_count += batch_size
        pbar.update(batch_size)
    pbar.close()
    is_mean, is_std = inception.compute(); fid_score = fid.compute(); kid_mean, kid_std = kid.compute()
    print(f"Results for '{scenario}':")
    print(f"  Inception Score (IS): {is_mean:.4f} ± {is_std:.4f}")
    print(f"  Frechet Inception Distance (FID): {fid_score:.4f}")
    print(f"  Kernel Inception Distance (KID): {kid_mean:.4f} ± {kid_std:.4f}")

def main(args):
    with open(args.config, 'r') as f: config = json.load(f)
    dataset_name, exp_type = args.dataset, args.type
    
    # --- 수정된 부분 시작 ---
    config["current_dataset_name"] = dataset_name # 현재 데이터셋 이름을 config에 추가
    base_path = f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}"
    # --- 수정된 부분 끝 ---
    
    model_filename = "generator.pth"
    
    if exp_type == 'unlearning_final':
        if args.soft_target is None: raise ValueError("--soft_target is required for 'unlearning_final' type.")
        soft_target = args.soft_target
        exp_dir = os.path.join(base_path, f"Unlearning_soft_target_{soft_target:.1f}")
        model_filename = f"generator_final_soft_{soft_target:.1f}.pth"
    elif exp_type == 'unlearning_step1':
        if args.soft_target is None: raise ValueError("--soft_target is required for 'unlearning_step1' type.")
        soft_target = args.soft_target
        exp_dir = os.path.join(base_path, f"Unlearning_soft_target_{soft_target:.1f}")
        model_filename = "generator_step1.pth"
    elif exp_type == 'original': exp_dir = os.path.join(base_path, "Original")
    elif exp_type == 'retrain': exp_dir = os.path.join(base_path, "Baseline_Retrain")
    elif exp_type == 'finetune': exp_dir = os.path.join(base_path, "Baseline_Finetune")
    else: raise ValueError(f"Invalid experiment type: {exp_type}")

    model_path = os.path.join(exp_dir, "models", model_filename)
    if not os.path.exists(model_path): raise FileNotFoundError(f"Model not found at: {model_path}\nPlease check if the experiment was run.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model: {model_path}"); print(f"Using device: {device}")
    
    d_config = config["dataset_configs"][dataset_name]
    generator = ACGANGenerator(config["latent_dim"], config["num_classes"], config["label_embedding_dim_g"], img_size=d_config["img_size"], img_channels=d_config["img_channels"]).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    generator.eval()
    
    dataloader, _ = get_dataloader(dataset_name, d_config["img_size"], d_config["img_channels"], 64, 1, 0)
    
    for scenario in ['all_labels', 'exclude_target', 'target_only']:
        calculate_metrics_for_scenario(generator, config, dataloader, scenario, device, num_images=args.num_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a specific ACGAN model.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name to evaluate (e.g., SVHN).')
    parser.add_argument('--type', type=str, required=True, choices=['original', 'retrain', 'finetune', 'unlearning_step1', 'unlearning_final'], help='Type of the experiment to evaluate.')
    parser.add_argument('--soft_target', type=float, help="Soft target value, required for 'unlearning_step1' and 'unlearning_final' types.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file.')
    parser.add_argument('--num_images', type=int, default=5000, help='Number of images for evaluation.')
    
    args = parser.parse_args()
    main(args)