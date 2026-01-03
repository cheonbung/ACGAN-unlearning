# acgan/dataset.py

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

def get_dataloader(dataset_name, train, **config):
    """
    지정된 데이터셋 이름에 대한 DataLoader를 생성하고 반환합니다.
    **config를 통해 유연하게 파라미터를 받습니다.

    Args:
        dataset_name (str): 데이터셋 이름 (e.g., "CIFAR10").
        train (bool): True이면 학습 데이터셋, False이면 테스트 데이터셋을 로드.
        **config: batch_size, world_size, rank 등 추가 설정이 담긴 딕셔너리.
    """
    # config 딕셔너리에서 필요한 파라미터 추출
    d_config = config["dataset_configs"][dataset_name]
    img_size, img_channels = d_config["img_size"], d_config["img_channels"]
    batch_size = config["batch_size"]
    
    # DDP 관련 파라미터는 없으면 기본값으로 설정
    world_size = config.get("world_size", 1)
    rank = config.get("rank", 0)
    
    # 필터링 관련 파라미터
    exclude_label = config.get("exclude_label", None)

    if img_size not in [28, 32, 96]:
        raise ValueError(f"⚠️ 경고: {dataset_name} ({img_size}x{img_size})는 지원되지 않습니다. 모델은 28, 32, 96만 지원합니다.")

    # 데이터 변환 설정
    transform_list = [transforms.Resize(img_size), transforms.ToTensor()]
    if img_channels == 1:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    else: 
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)

    # 데이터셋 이름과 train 플래그에 따라 적절한 데이터셋 로드
    dataset_map = {
        "SVHN": (datasets.SVHN, {"split": "train" if train else "test"}),
        "CIFAR10": (datasets.CIFAR10, {"train": train}),
        "MNIST": (datasets.MNIST, {"train": train}),
        "FashionMNIST": (datasets.FashionMNIST, {"train": train}),
        "STL10": (datasets.STL10, {"split": "train" if train else "test"}),
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_class, kwargs = dataset_map[dataset_name]
    full_dataset = dataset_class(root="./data", download=True, transform=transform, **kwargs)
    
    if len(full_dataset) == 0:
        raise RuntimeError(f"ERROR: Dataset {dataset_name} (train={train}) is empty after loading.")
    
    # 레이블 제외 필터링 로직
    dataset_to_use = full_dataset
    if exclude_label is not None:
        # 데이터셋마다 레이블 속성 이름이 다름 ('targets' 또는 'labels')
        targets = getattr(full_dataset, 'targets', getattr(full_dataset, 'labels', None))
        if targets is None:
            raise AttributeError(f"Dataset {dataset_name} does not have 'targets' or 'labels' attribute for filtering.")

        original_len = len(full_dataset)
        indices = [i for i, label in enumerate(targets) if label != exclude_label]
        dataset_to_use = Subset(full_dataset, indices)
        
        if rank == 0:
            print(f"✅ Excluding label {exclude_label}. Original size: {original_len}, New size: {len(dataset_to_use)}")

    # DDP를 위한 Sampler 설정
    sampler, shuffle = None, True
    if world_size > 1:
        sampler = DistributedSampler(dataset_to_use, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False

    # DataLoader 생성
    dataloader = DataLoader(
        dataset_to_use, batch_size=batch_size, shuffle=shuffle, 
        num_workers=2, pin_memory=True, drop_last=True, sampler=sampler
    )
    
    if rank == 0:
        log_msg = f"✅ {dataset_name} (train={train})"
        if exclude_label is not None:
            log_msg += f" (excluding label {exclude_label})"
        log_msg += f" 데이터 로더 생성 완료. (총 {len(dataset_to_use)}개 이미지, DDP: {'Enabled' if sampler else 'Disabled'})"
        print(log_msg)
        
    return dataloader, sampler