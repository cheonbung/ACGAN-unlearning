# train.py

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
import os
import json
import csv # 💡 CSV 저장을 위한 라이브러리 임포트

from acgan.dataset import get_dataloader
from acgan.trainer import run_stage1_1, run_stage1_2, run_stage2, run_baseline_retrain, run_baseline_finetune

def setup_ddp():
    """DDP를 위한 프로세스 그룹을 초기화합니다."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # 💡 DDP 버그가 해결되었으므로, 원래의 안정적인 file-based 초기화 방식을 사용합니다.
    sync_file_path = os.path.abspath("ddp_sync_file")
    path_for_uri = sync_file_path.replace('\\', '/')
    sync_file_uri = f"file:///{path_for_uri}"
    dist.init_process_group(
        backend="gloo", # Windows에서는 'gloo' 사용
        init_method=sync_file_uri,
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)
    return rank, world_size, sync_file_path

def cleanup_ddp(rank, sync_file_path):
    """DDP 프로세스 그룹을 정리하고 공유 파일을 삭제합니다."""
    if rank == 0 and os.path.exists(sync_file_path):
        os.remove(sync_file_path)
    dist.destroy_process_group()


def main_worker(rank, world_size, config):
    """(Single-GPU와 DDP 모두에서) 각 프로세스에서 실행될 메인 학습 함수입니다."""
    if world_size > 1:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        if world_size > 1:
            print(f"DDP-Worker: Running on {world_size} GPUs using 'gloo' backend.")
        else:
            print(f"Single-GPU Mode: Running on device: {device}")
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    
    # 💡 실행 시간 기록을 위한 리스트 초기화 (rank 0에서만 관리)
    timing_results = []

    enabled_datasets = {k: v for k, v in config["dataset_configs"].items() if v["enabled"]}
    dataset_pbar = tqdm(enabled_datasets.items(), desc="Overall Progress", position=0, disable=(rank != 0))

    for dataset_name, dataset_config in dataset_pbar:
        if rank == 0:
            dataset_pbar.set_description(f"Processing Dataset: {dataset_name}")
        
        config["current_dataset_name"] = dataset_name
        
        fixed_noise = torch.randn(config["num_classes"] * 10, config["latent_dim"], device=device)
        fixed_labels = torch.tensor([i for i in range(config["num_classes"]) for _ in range(10)], dtype=torch.long, device=device)
        
        # --- Stage 1-1 (Original Model) ---
        if config["run_stage1_1_flag"]:
            full_dataloader, full_sampler = get_dataloader(
                dataset_name=dataset_name, img_size=dataset_config["img_size"],
                img_channels=dataset_config["img_channels"], batch_size=config["batch_size"],
                world_size=world_size, rank=rank
            )
            # 💡 elapsed_time 변수로 소요 시간을 받음
            elapsed_time = run_stage1_1(config, full_dataloader, full_sampler, rank, world_size, device, fixed_noise, fixed_labels)
            
            # 💡 rank 0 프로세스만 시간 기록
            if rank == 0 and elapsed_time > 0:
                timing_results.append({
                    "dataset": dataset_name, "stage": "Original", "soft_target": "N/A", 
                    "gpus": world_size, "execution_time_seconds": round(elapsed_time, 2)
                })
        else:
            if rank == 0: print(f"\n--- Stage 1-1: {dataset_name} Original ACGAN 훈련 SKIPPED ---")

        # --- Baseline Experiments (Retrain & Finetune) ---
        if config.get("run_baseline_retrain_flag", False) or config.get("run_baseline_finetune_flag", False):
            if rank == 0: print("\n--- Baseline 실험 준비 중 ---")
            baseline_dataloader, baseline_sampler = get_dataloader(
                dataset_name=dataset_name, img_size=dataset_config["img_size"],
                img_channels=dataset_config["img_channels"], batch_size=config["batch_size"],
                world_size=world_size, rank=rank, exclude_label=config["forget_label"]
            )
            if config.get("run_baseline_retrain_flag", False):
                elapsed_time = run_baseline_retrain(config, baseline_dataloader, baseline_sampler, rank, world_size, device, fixed_noise, fixed_labels)
                if rank == 0 and elapsed_time > 0:
                    timing_results.append({
                        "dataset": dataset_name, "stage": "Baseline_Retrain", "soft_target": "N/A",
                        "gpus": world_size, "execution_time_seconds": round(elapsed_time, 2)
                    })
            
            if config.get("run_baseline_finetune_flag", False):
                elapsed_time = run_baseline_finetune(config, baseline_dataloader, baseline_sampler, rank, world_size, device, fixed_noise, fixed_labels)
                if rank == 0 and elapsed_time > 0:
                    timing_results.append({
                        "dataset": dataset_name, "stage": "Baseline_Finetune", "soft_target": "N/A",
                        "gpus": world_size, "execution_time_seconds": round(elapsed_time, 2)
                    })

        # --- Unlearning Experiments ---
        full_dataloader_unlearn, full_sampler_unlearn = get_dataloader(
            dataset_name=dataset_name, img_size=dataset_config["img_size"],
            img_channels=dataset_config["img_channels"], batch_size=config["batch_size"],
            world_size=world_size, rank=rank
        )
        
        soft_label_pbar = tqdm(
            config["soft_label_target_list"],
            desc=f"[{dataset_name}] Soft Label Sweep",
            position=1, leave=False, disable=(rank != 0)
        )
        for soft_label_target_val in soft_label_pbar:
            if rank == 0:
                soft_label_pbar.set_description(f"[{dataset_name}] Sweeping soft_target={soft_label_target_val:.2f}")

            if config["run_stage1_2_flag"]:
                elapsed_time = run_stage1_2(config, full_dataloader_unlearn, full_sampler_unlearn, soft_label_target_val, rank, world_size, device, fixed_noise, fixed_labels)
                if rank == 0 and elapsed_time > 0:
                    timing_results.append({
                        "dataset": dataset_name, "stage": "Unlearning_Stage1_2", "soft_target": soft_label_target_val,
                        "gpus": world_size, "execution_time_seconds": round(elapsed_time, 2)
                    })
            
            if config["run_stage2_flag"]:
                elapsed_time = run_stage2(config, full_dataloader_unlearn, full_sampler_unlearn, soft_label_target_val, rank, world_size, device, fixed_noise, fixed_labels)
                if rank == 0 and elapsed_time > 0:
                    timing_results.append({
                        "dataset": dataset_name, "stage": "Unlearning_Stage2", "soft_target": soft_label_target_val,
                        "gpus": world_size, "execution_time_seconds": round(elapsed_time, 2)
                    })
                
        if rank == 0: print(f"\n===== {dataset_name} 데이터셋 모든 실험 완료 =====")

    if rank == 0:
        print("\n===== 모든 데이터셋 실험 완료 =====")
        # 💡 모든 실험 완료 후, 기록된 소요 시간을 CSV 파일에 저장
        if timing_results:
            csv_file = 'execution_times.csv'
            file_exists = os.path.isfile(csv_file)
            try:
                with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                    fieldnames = ["dataset", "stage", "soft_target", "gpus", "execution_time_seconds"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                        
                    writer.writerows(timing_results)
                print(f"\n✅ Execution times appended to '{csv_file}'")
            except IOError as e:
                print(f"Error: Could not write execution times to '{csv_file}'. Details: {e}")


if __name__ == '__main__':
    config_path = 'config.json'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)

    is_ddp = 'WORLD_SIZE' in os.environ
    
    if is_ddp:
        rank, world_size, sync_file = setup_ddp()
        main_worker(rank, world_size, config)
        cleanup_ddp(rank, sync_file)
    else:
        main_worker(rank=0, world_size=1, config=config)