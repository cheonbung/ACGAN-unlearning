# evaluate_inversion.py

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image, make_grid
from tqdm.auto import tqdm
import os
import json
import argparse

# 프로젝트 내부 모듈 임포트
from acgan.model import ACGANGenerator, ACGANDiscriminator

def load_models(g_path, d_path, config, dataset_name, device):
    """생성자와 판별자 모델을 모두 로드합니다."""
    d_config = config["dataset_configs"][dataset_name]
    img_size, img_channels = d_config["img_size"], d_config["img_channels"]

    # 생성자 로드
    generator = ACGANGenerator(
        latent_dim=config["latent_dim"],
        num_classes=config["num_classes"],
        label_embedding_dim=config["label_embedding_dim_g"],
        img_size=img_size,
        img_channels=img_channels
    ).to(device)
    # 보안 경고를 해결하기 위해 weights_only=True 추가
    g_state_dict = torch.load(g_path, map_location=device, weights_only=True)
    new_g_state_dict = {k.replace('module.', ''): v for k, v in g_state_dict.items()}
    generator.load_state_dict(new_g_state_dict)
    generator.eval()

    # 판별자 로드
    discriminator = ACGANDiscriminator(
        num_classes=config["num_classes"],
        img_size=img_size,
        img_channels=img_channels
    ).to(device)
    # 보안 경고를 해결하기 위해 weights_only=True 추가
    d_state_dict = torch.load(d_path, map_location=device, weights_only=True)
    new_d_state_dict = {k.replace('module.', ''): v for k, v in d_state_dict.items()}
    discriminator.load_state_dict(new_d_state_dict)
    discriminator.eval()
    
    return generator, discriminator

# 
# =============================================================
#  핵심 수정 사항: 아래 @torch.no_grad() 줄을 삭제했습니다.
# =============================================================
# @torch.no_grad() <-- 이 줄을 삭제하세요.
#
def run_inversion_attack(generator, discriminator, target_class, config, device,
                         steps=2000, lr=0.1, realness_weight=0.1):
    """
    최적화 기반 모델 역전 공격을 수행하여 한 클래스에 대한 대표 이미지를 생성합니다.
    """
    # 1. 최적화할 잠재 벡터 z 초기화
    # requires_grad=True를 통해 이 텐서에 대한 그래디언트를 계산하도록 설정
    z = torch.randn(1, config["latent_dim"], device=device, requires_grad=True)
    
    # z를 위한 옵티마이저 설정
    optimizer = optim.Adam([z], lr=lr)
    
    # 공격 대상 레이블
    target_label = torch.tensor([target_class], dtype=torch.long, device=device)

    # 2. 최적화 루프
    for _ in range(steps):
        optimizer.zero_grad()
        
        # z로부터 이미지 생성 (이 과정은 그래디언트 흐름이 유지됨)
        fake_image = generator(z, target_label)
        
        # 가이드 모델(판별자)로부터 점수 획득
        # 판별자는 업데이트하지 않으므로, torch.no_grad() 컨텍스트 내에서 실행해도 되지만,
        # 코드가 복잡해지므로 그냥 실행. 성능에 큰 영향 없음.
        source_pred, class_pred = discriminator(fake_image)
        
        # 3. 손실 함수 정의: 클래스 점수와 진짜다움 점수를 최대화
        class_scores = class_pred[:, target_class]
        class_loss = -class_scores.mean() 
        
        realness_loss = -source_pred.mean()
        
        total_loss = class_loss + realness_weight * realness_loss
        
        # 4. z에 대해 역전파 및 업데이트
        # total_loss가 z와 연결되어 있으므로 정상적으로 작동
        total_loss.backward()
        optimizer.step()

    # 5. 최종 최적화된 z로 이미지 생성 후 반환
    final_image = generator(z, target_label).detach()
    return final_image

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    output_base_dir = "inversion_attack_results"
    os.makedirs(output_base_dir, exist_ok=True)
    
    enabled_datasets = {k: v for k, v in config["dataset_configs"].items() if v["enabled"]}
    
    for dataset_name, d_config in tqdm(enabled_datasets.items(), desc="All Datasets"):
        print(f"\n===== Starting Model Inversion for Dataset: {dataset_name} =====")
        dataset_output_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        model_types_to_evaluate = {
            "Original": "Original",
            "Retrain": "Baseline_Retrain",
            "Finetune": "Baseline_Finetune",
            **{f"Unlearn_s={s:.1f}": f"Unlearning_soft_target_{s:.1f}" for s in config["soft_label_target_list"]}
        }
        
        for model_name, folder_name in tqdm(model_types_to_evaluate.items(), desc=f"Inverting {dataset_name} Models"):
            
            g_filename, d_filename = "generator.pth", "discriminator.pth"
            if "Unlearn" in model_name:
                soft_target = float(model_name.split('=')[-1])
                g_filename = f"generator_final_soft_{soft_target:.1f}.pth"
                d_filename = f"discriminator_final_soft_{soft_target:.1f}.pth"
            
            g_path = f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}/{folder_name}/models/{g_filename}"
            d_path = f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}/{folder_name}/models/{d_filename}"
            
            if not (os.path.exists(g_path) and os.path.exists(d_path)):
                continue

            generator, discriminator = load_models(g_path, d_path, config, dataset_name, device)
            
            inverted_images = []
            class_pbar = tqdm(range(config['num_classes']), desc=f"Inverting classes for {model_name}", leave=False)
            for target_class in class_pbar:
                inverted_image = run_inversion_attack(generator, discriminator, target_class, config, device)
                inverted_images.append(inverted_image)
            
            if inverted_images:
                grid = make_grid(torch.cat(inverted_images), nrow=config['num_classes'], normalize=True)
                save_path = os.path.join(dataset_output_dir, f"inversion_{model_name.replace('=', '_')}.png")
                save_image(grid, save_path)
                print(f"  ✅ Saved inverted images grid to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Model Inversion Attack on trained ACGAN generators.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file.')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")
        
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    main(config)