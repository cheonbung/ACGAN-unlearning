# evaluate_all.py

import torch
import json
import os
import argparse
import csv
from tqdm import tqdm
import numpy as np
import re

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from acgan.model import ACGANGenerator
from acgan.dataset import get_dataloader

@torch.no_grad()
def calculate_metrics_for_scenario(generator, config, img_channels, dataloader, scenario, device, num_images=5000):
    inception = InceptionScore().to(device)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=min(1000, num_images // 5), feature=2048).to(device)
    
    processed_real_images = 0
    for real_imgs, _ in dataloader:
        real_imgs_uint8 = ((real_imgs + 1) / 2 * 255).to(torch.uint8).to(device)
        
        if img_channels == 1:
            real_imgs_uint8 = real_imgs_uint8.repeat(1, 3, 1, 1)
            
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
    
    generated_count = 0
    while generated_count < num_images:
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_labels_np = np.random.choice(labels_to_generate, size=batch_size)
        gen_labels = torch.from_numpy(gen_labels_np).to(device)
        fake_imgs = generator(z, gen_labels)
        fake_imgs_uint8 = ((fake_imgs + 1) / 2 * 255).to(torch.uint8)
        
        if img_channels == 1:
            fake_imgs_uint8 = fake_imgs_uint8.repeat(1, 3, 1, 1)

        inception.update(fake_imgs_uint8); fid.update(fake_imgs_uint8, real=False); kid.update(fake_imgs_uint8, real=False)
        generated_count += batch_size
        
    is_mean, is_std = inception.compute(); fid_score = fid.compute(); kid_mean, kid_std = kid.compute()
    return {"is_mean": round(is_mean.item(), 4), "is_std": round(is_std.item(), 4), "fid": round(fid_score.item(), 4), "kid_mean": round(kid_mean.item(), 4), "kid_std": round(kid_std.item(), 4)}

def find_experiments(root_dir, config):
    experiments = []
    enabled_datasets = {k for k, v in config["dataset_configs"].items() if v.get("enabled", False)}
    for dataset_name in enabled_datasets:
        dataset_path = os.path.join(root_dir, f"ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}")
        if not os.path.isdir(dataset_path): continue
        for exp_dir_name in os.listdir(dataset_path):
            exp_path = os.path.join(dataset_path, exp_dir_name)
            model_dir = os.path.join(exp_path, "models")
            if not os.path.isdir(model_dir): continue
            
            base_models = {"Original": "Original", "Baseline_Retrain": "Retrain", "Baseline_Finetune": "Finetune"}
            if exp_dir_name in base_models:
                model_path = os.path.join(model_dir, "generator.pth")
                if os.path.exists(model_path):
                    experiments.append({"dataset": dataset_name, "type": base_models[exp_dir_name], "soft_target": "N/A", "model_path": model_path})

            elif exp_dir_name.startswith("Unlearning_soft_target_"):
                try:
                    # --- 수정된 부분 시작 ---
                    soft_target_match = re.search(r'(\d+\.\d+|\d+)', exp_dir_name)
                    if not soft_target_match: continue
                    # soft_target 변수에 값을 할당
                    soft_target = float(soft_target_match.group(1))
                    
                    step1_model_path = os.path.join(model_dir, "generator_step1.pth")
                    if os.path.exists(step1_model_path):
                        experiments.append({"dataset": dataset_name, "type": "Unlearning_Step1", "soft_target": soft_target, "model_path": step1_model_path})

                    # final_model_path를 만들 때 올바른 변수(soft_target) 사용
                    final_model_path = os.path.join(model_dir, f"generator_final_soft_{soft_target:.1f}.pth")
                    if os.path.exists(final_model_path):
                        experiments.append({"dataset": dataset_name, "type": "Unlearning_Final", "soft_target": soft_target, "model_path": final_model_path})
                    # --- 수정된 부분 끝 ---
                except (AttributeError, ValueError):
                    continue
    return experiments

def main(args):
    with open(args.config, 'r') as f: config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    experiments_to_run = find_experiments('outputs', config)
    if not experiments_to_run:
        print("No trained models found in 'outputs' directory for enabled datasets. Nothing to evaluate.")
        return
        
    all_results = []
    scenarios = ['all_labels', 'exclude_target', 'target_only']
    
    dataloaders = {}
    for exp in experiments_to_run:
        dataset_name = exp["dataset"]
        if dataset_name not in dataloaders:
            d_config = config["dataset_configs"][dataset_name]
            dataloader, _ = get_dataloader(dataset_name, d_config["img_size"], d_config["img_channels"], 64, 1, 0, exclude_label=None)
            dataloaders[dataset_name] = dataloader

    exp_pbar = tqdm(experiments_to_run, desc="Total Experiments")
    for exp in exp_pbar:
        dataset_name = exp["dataset"]
        exp_pbar.set_description(f"Evaluating [{dataset_name} - {exp['type']} - ST: {exp['soft_target']}]")
        
        d_config = config["dataset_configs"][dataset_name]
        dataloader = dataloaders[dataset_name]
        
        generator = ACGANGenerator(config["latent_dim"], config["num_classes"], config["label_embedding_dim_g"], img_size=d_config["img_size"], img_channels=d_config["img_channels"]).to(device)
        generator.load_state_dict(torch.load(exp["model_path"], map_location=device, weights_only=True))
        generator.eval()
        
        for scenario in scenarios:
            tqdm.write(f"  > Scenario: {scenario}")
            metrics = calculate_metrics_for_scenario(generator, config, d_config["img_channels"], dataloader, scenario, device, num_images=args.num_images)
            if metrics:
                all_results.append({"dataset": dataset_name, "type": exp['type'], "soft_target": exp['soft_target'], "scenario": scenario, **metrics})

    if not all_results:
        print("\nEvaluation completed, but no results were generated.")
        return
    try:
        fieldnames = sorted(list(set(k for res in all_results for k in res.keys())))
        
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✅ All evaluations complete. Results saved to '{args.output}'")
    except (IOError, IndexError) as e:
        print(f"Error: Could not write to file '{args.output}' or no results to write. Details: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate all trained models found in 'outputs' and save to CSV.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file.')
    parser.add_argument('--num_images', type=int, default=5000, help='Number of images for evaluation.')
    parser.add_argument('--output', type=str, default='evaluation_summary.csv', help='Path to save the output CSV file.')
    args = parser.parse_args()
    main(args)