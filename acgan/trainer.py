# acgan/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
import os
import json
import time # 💡 소요 시간 측정을 위해 time 라이브러리 임포트

from .model import ACGANGenerator, ACGANDiscriminator, initialize_weights, EfficientGANLoss
from .utils import create_training_gif, save_loss_plot

# 💡 모든 run_* 함수가 소요 시간을 반환하도록 수정되었습니다.

def run_stage1_1(config, dataloader, sampler, rank, world_size, device, fixed_noise, fixed_labels):
    start_time = time.time() # 💡 시작 시간 측정
    dataset_name = config["current_dataset_name"]
    d_config = config["dataset_configs"][dataset_name]
    img_size, img_channels = d_config["img_size"], d_config["img_channels"]

    base_dir = f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}/Original"
    model_save_dir = os.path.join(base_dir, "models")

    skip_flag = torch.tensor([0], device=device)
    if rank == 0:
        if os.path.exists(os.path.join(model_save_dir, "generator.pth")) and os.path.exists(os.path.join(model_save_dir, "discriminator.pth")):
            tqdm.write(f"\n✅ Stage 1-1 SKIPPED: Original models already exist.")
            skip_flag.fill_(1)
    
    if world_size > 1: torch.distributed.broadcast(skip_flag, src=0)
    if skip_flag.item() == 1:
        if world_size > 1: torch.distributed.barrier()
        return 0.0 # 💡 스킵된 경우 0.0 반환

    # ... (중간 학습 로직은 동일) ...
    writer, loss_log = None, None
    if rank == 0:
        print(f"\n--- Stage 1-1: {dataset_name} Original ACGAN 훈련 ---")
        img_dir, gif_dir = [os.path.join(base_dir, d) for d in ["images", "gifs"]]
        os.makedirs(model_save_dir, exist_ok=True); os.makedirs(img_dir, exist_ok=True); os.makedirs(gif_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(base_dir, "runs", "Original_ACGAN"))
        loss_log = {"epoch": [], "g_loss": [], "d_loss": [], "g_loss_gan": [], "g_loss_aux": [], "d_loss_gan": [], "d_loss_aux_real": [], "d_loss_aux_fake": []}

    generator = ACGANGenerator(config["latent_dim"], config["num_classes"], config["label_embedding_dim_g"], img_size, img_channels).to(device)
    discriminator = ACGANDiscriminator(config["num_classes"], img_size, img_channels).to(device)
    initialize_weights(generator); initialize_weights(discriminator)

    if world_size > 1:
        generator = DDP(generator, device_ids=[rank], find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[rank], find_unused_parameters=True)

    gan_loss_fn, aux_criterion = EfficientGANLoss().to(device), nn.CrossEntropyLoss().to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=config["lr_g"], betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=(0.5, 0.999))
    
    epochs = config["original_epochs"]
    epoch_pbar = tqdm(range(epochs), desc="Stage 1-1 Epochs", position=2, leave=False, disable=(rank != 0))
    for epoch in epoch_pbar:
        if sampler is not None: sampler.set_epoch(epoch)
        
        epoch_g_loss, epoch_d_loss, epoch_g_loss_gan, epoch_g_loss_aux = 0.0, 0.0, 0.0, 0.0
        epoch_d_loss_gan, epoch_d_loss_aux_real, epoch_d_loss_aux_fake = 0.0, 0.0, 0.0

        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=3, leave=False, disable=(rank != 0))
        for real_imgs, labels in batch_pbar:
            real_imgs, labels, batch_size = real_imgs.to(device), labels.to(device), real_imgs.size(0)
            
            d_optimizer.zero_grad()
            real_src_pred, real_cls_pred = discriminator(real_imgs)
            z_d = torch.randn(batch_size, config["latent_dim"], device=device)
            gen_labels_d = torch.randint(0, config["num_classes"], (batch_size,), device=device)
            with torch.no_grad(): fake_imgs = generator(z_d, gen_labels_d)
            fake_src_pred, fake_cls_pred = discriminator(fake_imgs)
            d_loss_gan = gan_loss_fn.discriminator_loss(real_src_pred, fake_src_pred)
            d_loss_aux_real = aux_criterion(real_cls_pred, labels)
            d_loss_aux_fake = aux_criterion(fake_cls_pred, gen_labels_d)
            d_loss = d_loss_gan + config["aux_loss_weight"] * (d_loss_aux_real + d_loss_aux_fake)
            d_loss.backward(); d_optimizer.step()

            g_optimizer.zero_grad()
            z_g, gen_labels_g = torch.randn(batch_size, config["latent_dim"], device=device), torch.randint(0, config["num_classes"], (batch_size,), device=device)
            gen_imgs_g = generator(z_g, gen_labels_g)
            g_src_output, g_cls_output = discriminator(gen_imgs_g)
            g_loss_gan = gan_loss_fn.generator_loss(g_src_output)
            g_loss_aux = aux_criterion(g_cls_output, gen_labels_g)
            g_loss = g_loss_gan + config["aux_loss_weight"] * g_loss_aux
            g_loss.backward(); g_optimizer.step()
            
            epoch_d_loss += d_loss.item(); epoch_g_loss += g_loss.item()
            epoch_d_loss_gan += d_loss_gan.item(); epoch_g_loss_gan += g_loss_gan.item()
            epoch_d_loss_aux_real += d_loss_aux_real.item(); epoch_d_loss_aux_fake += d_loss_aux_fake.item()
            epoch_g_loss_aux += g_loss_aux.item()

        if rank == 0:
            num_batches = len(dataloader)
            avg_g_loss, avg_d_loss = epoch_g_loss / num_batches, epoch_d_loss / num_batches
            avg_g_gan, avg_g_aux = epoch_g_loss_gan / num_batches, epoch_g_loss_aux / num_batches
            avg_d_gan, avg_d_aux_r, avg_d_aux_f = epoch_d_loss_gan / num_batches, epoch_d_loss_aux_real / num_batches, epoch_d_loss_aux_fake / num_batches

            loss_log["epoch"].append(epoch + 1); loss_log["g_loss"].append(avg_g_loss); loss_log["d_loss"].append(avg_d_loss)
            loss_log["g_loss_gan"].append(avg_g_gan); loss_log["g_loss_aux"].append(avg_g_aux); loss_log["d_loss_gan"].append(avg_d_gan)
            loss_log["d_loss_aux_real"].append(avg_d_aux_r); loss_log["d_loss_aux_fake"].append(avg_d_aux_f)
            
            writer.add_scalar('Stage1-1/Loss/G_Total', avg_g_loss, epoch+1); writer.add_scalar('Stage1-1/Loss/G_GAN', avg_g_gan, epoch+1); writer.add_scalar('Stage1-1/Loss/G_Aux', avg_g_aux, epoch+1)
            writer.add_scalar('Stage1-1/Loss/D_Total', avg_d_loss, epoch+1); writer.add_scalar('Stage1-1/Loss/D_GAN', avg_d_gan, epoch+1)
            writer.add_scalar('Stage1-1/Loss/D_Aux_Real', avg_d_aux_r, epoch+1); writer.add_scalar('Stage1-1/Loss/D_Aux_Fake', avg_d_aux_f, epoch+1)
            
            if (epoch+1) % config["save_every"] == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    g_eval = generator.module if world_size > 1 else generator
                    g_eval.eval()
                    samples = g_eval(fixed_noise, fixed_labels).detach().cpu()
                    vutils.save_image(samples, os.path.join(img_dir, f"epoch_{epoch+1:03d}.png"), nrow=10, normalize=True)
                    writer.add_image('Stage1-1/Generated_Images', vutils.make_grid(samples, nrow=10, normalize=True), epoch+1)
                    g_eval.train()

    elapsed_time = time.time() - start_time # 💡 종료 시간 측정 및 계산
    if rank == 0:
        if writer is not None: writer.close()
        create_training_gif(img_dir, gif_dir, "Original_ACGAN")
        loss_log_path, plot_path = os.path.join(base_dir, "loss_log.json"), os.path.join(base_dir, "loss_plot.png")
        with open(loss_log_path, 'w') as f: json.dump(loss_log, f, indent=4)
        save_loss_plot(loss_log_path, plot_path, "Original ACGAN")
        g_to_save, d_to_save = (m.module if world_size > 1 else m for m in [generator, discriminator])
        torch.save(g_to_save.state_dict(), os.path.join(model_save_dir, "generator.pth")); torch.save(d_to_save.state_dict(), os.path.join(model_save_dir, "discriminator.pth"))
        tqdm.write(f"✅ Stage 1-1 완료: {dataset_name} Original ACGAN 학습 완료. (소요 시간: {elapsed_time:.2f}초)")
        
    if world_size > 1: torch.distributed.barrier()
    return elapsed_time # 💡 계산된 소요 시간 반환

# (run_baseline_retrain, run_baseline_finetune, run_stage1_2, run_stage2 함수도 동일한 방식으로 수정합니다)

def run_baseline_retrain(config, dataloader, sampler, rank, world_size, device, fixed_noise, fixed_labels):
    start_time = time.time()
    dataset_name = config["current_dataset_name"]; d_config = config["dataset_configs"][dataset_name]; img_size, img_channels = d_config["img_size"], d_config["img_channels"]
    base_dir = f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}/Baseline_Retrain"; model_save_dir = os.path.join(base_dir, "models")
    skip_flag = torch.tensor([0], device=device)
    if rank == 0:
        if os.path.exists(os.path.join(model_save_dir, "generator.pth")) and os.path.exists(os.path.join(model_save_dir, "discriminator.pth")):
            tqdm.write(f"\n✅ Baseline Retrain SKIPPED: Retrained models already exist.")
            skip_flag.fill_(1)
    if world_size > 1: torch.distributed.broadcast(skip_flag, src=0)
    if skip_flag.item() == 1:
        if world_size > 1: torch.distributed.barrier()
        return 0.0
    # ... (중간 로직 동일) ...
    writer, loss_log = None, None
    if rank == 0:
        print(f"\n--- Baseline: {dataset_name} Retraining from Scratch ---")
        img_dir, gif_dir = [os.path.join(base_dir, d) for d in ["images", "gifs"]]; os.makedirs(model_save_dir, exist_ok=True); os.makedirs(img_dir, exist_ok=True); os.makedirs(gif_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(base_dir, "runs", "Baseline_Retrain"))
        loss_log = {"epoch": [], "g_loss": [], "d_loss": [], "g_loss_gan": [], "g_loss_aux": [], "d_loss_gan": [], "d_loss_aux_real": [], "d_loss_aux_fake": []}
    generator = ACGANGenerator(config["latent_dim"], config["num_classes"], config["label_embedding_dim_g"], img_size, img_channels).to(device); discriminator = ACGANDiscriminator(config["num_classes"], img_size, img_channels).to(device)
    initialize_weights(generator); initialize_weights(discriminator)
    if world_size > 1:
        generator = DDP(generator, device_ids=[rank], find_unused_parameters=True); discriminator = DDP(discriminator, device_ids=[rank], find_unused_parameters=True)
    gan_loss_fn, aux_criterion = EfficientGANLoss().to(device), nn.CrossEntropyLoss().to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=config["lr_g"], betas=(0.5, 0.999)); d_optimizer = optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=(0.5, 0.999))
    epochs = config["original_epochs"]
    epoch_pbar = tqdm(range(epochs), desc="Baseline Retrain Epochs", position=2, leave=False, disable=(rank != 0))
    for epoch in epoch_pbar:
        if sampler is not None: sampler.set_epoch(epoch)
        epoch_g_loss, epoch_d_loss, epoch_g_loss_gan, epoch_g_loss_aux = 0.0, 0.0, 0.0, 0.0
        epoch_d_loss_gan, epoch_d_loss_aux_real, epoch_d_loss_aux_fake = 0.0, 0.0, 0.0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=3, leave=False, disable=(rank != 0))
        for real_imgs, labels in batch_pbar:
            real_imgs, labels, batch_size = real_imgs.to(device), labels.to(device), real_imgs.size(0)
            d_optimizer.zero_grad()
            real_src_pred, real_cls_pred = discriminator(real_imgs)
            z_d = torch.randn(batch_size, config["latent_dim"], device=device); gen_labels_d = torch.randint(0, config["num_classes"], (batch_size,), device=device)
            with torch.no_grad(): fake_imgs = generator(z_d, gen_labels_d)
            fake_src_pred, fake_cls_pred = discriminator(fake_imgs)
            d_loss_gan, d_loss_aux_real, d_loss_aux_fake = gan_loss_fn.discriminator_loss(real_src_pred, fake_src_pred), aux_criterion(real_cls_pred, labels), aux_criterion(fake_cls_pred, gen_labels_d)
            d_loss = d_loss_gan + config["aux_loss_weight"] * (d_loss_aux_real + d_loss_aux_fake)
            d_loss.backward(); d_optimizer.step()
            g_optimizer.zero_grad()
            z_g, gen_labels_g = torch.randn(batch_size, config["latent_dim"], device=device), torch.randint(0, config["num_classes"], (batch_size,), device=device)
            gen_imgs_g = generator(z_g, gen_labels_g)
            g_src_output, g_cls_output = discriminator(gen_imgs_g)
            g_loss_gan, g_loss_aux = gan_loss_fn.generator_loss(g_src_output), aux_criterion(g_cls_output, gen_labels_g)
            g_loss = g_loss_gan + config["aux_loss_weight"] * g_loss_aux
            g_loss.backward(); g_optimizer.step()
            epoch_d_loss += d_loss.item(); epoch_g_loss += g_loss.item(); epoch_d_loss_gan += d_loss_gan.item(); epoch_g_loss_gan += g_loss_gan.item(); epoch_d_loss_aux_real += d_loss_aux_real.item(); epoch_d_loss_aux_fake += d_loss_aux_fake.item(); epoch_g_loss_aux += g_loss_aux.item()
        if rank == 0:
            num_batches = len(dataloader)
            avg_g_loss, avg_d_loss = epoch_g_loss / num_batches, epoch_d_loss / num_batches
            avg_g_gan, avg_g_aux = epoch_g_loss_gan / num_batches, epoch_g_loss_aux / num_batches
            avg_d_gan, avg_d_aux_r, avg_d_aux_f = epoch_d_loss_gan / num_batches, epoch_d_loss_aux_real / num_batches, epoch_d_loss_aux_fake / num_batches
            loss_log["epoch"].append(epoch + 1); loss_log["g_loss"].append(avg_g_loss); loss_log["d_loss"].append(avg_d_loss); loss_log["g_loss_gan"].append(avg_g_gan); loss_log["g_loss_aux"].append(avg_g_aux); loss_log["d_loss_gan"].append(avg_d_gan); loss_log["d_loss_aux_real"].append(avg_d_aux_r); loss_log["d_loss_aux_fake"].append(avg_d_aux_f)
            writer.add_scalar('Baseline_Retrain/Loss/G_Total', avg_g_loss, epoch+1); writer.add_scalar('Baseline_Retrain/Loss/D_Total', avg_d_loss, epoch+1)
            if (epoch+1) % config["save_every"] == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    g_eval = generator.module if world_size > 1 else generator
                    g_eval.eval(); samples = g_eval(fixed_noise, fixed_labels).detach().cpu(); vutils.save_image(samples, os.path.join(img_dir, f"epoch_{epoch+1:03d}.png"), nrow=10, normalize=True)
                    writer.add_image('Baseline_Retrain/Generated_Images', vutils.make_grid(samples, nrow=10, normalize=True), epoch+1); g_eval.train()
    elapsed_time = time.time() - start_time
    if rank == 0:
        if writer is not None: writer.close()
        create_training_gif(img_dir, gif_dir, "Baseline_Retrain")
        loss_log_path, plot_path = os.path.join(base_dir, "loss_log.json"), os.path.join(base_dir, "loss_plot.png")
        with open(loss_log_path, 'w') as f: json.dump(loss_log, f, indent=4)
        save_loss_plot(loss_log_path, plot_path, "Baseline Retrain")
        g_to_save, d_to_save = (m.module if world_size > 1 else m for m in [generator, discriminator])
        torch.save(g_to_save.state_dict(), os.path.join(model_save_dir, "generator.pth")); torch.save(d_to_save.state_dict(), os.path.join(model_save_dir, "discriminator.pth"))
        tqdm.write(f"✅ Baseline Retrain 완료. (소요 시간: {elapsed_time:.2f}초)")
    if world_size > 1: torch.distributed.barrier()
    return elapsed_time

def run_baseline_finetune(config, dataloader, sampler, rank, world_size, device, fixed_noise, fixed_labels):
    start_time = time.time()
    dataset_name = config["current_dataset_name"]; d_config = config["dataset_configs"][dataset_name]; img_size, img_channels = d_config["img_size"], d_config["img_channels"]
    base_dir = f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}/Baseline_Finetune"; model_save_dir = os.path.join(base_dir, "models")
    original_model_path = f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}/Original/models/generator.pth"
    skip_flag = torch.tensor([0], device=device)
    if rank == 0:
        if os.path.exists(os.path.join(model_save_dir, "generator.pth")):
            tqdm.write(f"\n✅ Baseline Finetune SKIPPED: Finetuned model already exists.")
            skip_flag.fill_(1)
        elif not os.path.exists(original_model_path):
            tqdm.write(f"\n❌ Baseline Finetune SKIPPED: Original model not found for finetuning.")
            skip_flag.fill_(1)
    if world_size > 1: torch.distributed.broadcast(skip_flag, src=0)
    if skip_flag.item() == 1:
        if world_size > 1: torch.distributed.barrier()
        return 0.0
    # ... (중간 로직 동일) ...
    writer, loss_log = None, None
    if rank == 0:
        print(f"\n--- Baseline: {dataset_name} Simple Finetuning ---")
        img_dir, gif_dir = [os.path.join(base_dir, d) for d in ["images", "gifs"]]; os.makedirs(model_save_dir, exist_ok=True); os.makedirs(img_dir, exist_ok=True); os.makedirs(gif_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(base_dir, "runs", "Baseline_Finetune"))
        loss_log = {"epoch": [], "g_loss": [], "d_loss": [], "g_loss_gan": [], "g_loss_aux": [], "d_loss_gan": [], "d_loss_aux_real": [], "d_loss_aux_fake": []}
    g_cpu = ACGANGenerator(config["latent_dim"], config["num_classes"], config["label_embedding_dim_g"], img_size, img_channels); g_cpu.load_state_dict(torch.load(original_model_path, map_location='cpu'))
    generator = g_cpu.to(device); discriminator = ACGANDiscriminator(config["num_classes"], img_size, img_channels).to(device)
    initialize_weights(discriminator)
    if world_size > 1:
        generator = DDP(generator, device_ids=[rank], find_unused_parameters=True); discriminator = DDP(discriminator, device_ids=[rank], find_unused_parameters=True)
    gan_loss_fn, aux_criterion = EfficientGANLoss().to(device), nn.CrossEntropyLoss().to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=config["lr_g"]/10, betas=(0.5, 0.999)); d_optimizer = optim.Adam(discriminator.parameters(), lr=config["lr_d"]/10, betas=(0.5, 0.999))
    epochs = config.get("finetune_epochs", 20)
    epoch_pbar = tqdm(range(epochs), desc="Baseline Finetune Epochs", position=2, leave=False, disable=(rank != 0))
    for epoch in epoch_pbar:
        if sampler is not None: sampler.set_epoch(epoch)
        epoch_g_loss, epoch_d_loss, epoch_g_loss_gan, epoch_g_loss_aux = 0.0, 0.0, 0.0, 0.0
        epoch_d_loss_gan, epoch_d_loss_aux_real, epoch_d_loss_aux_fake = 0.0, 0.0, 0.0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=3, leave=False, disable=(rank != 0))
        for real_imgs, labels in batch_pbar:
            real_imgs, labels, batch_size = real_imgs.to(device), labels.to(device), real_imgs.size(0)
            d_optimizer.zero_grad()
            real_src_pred, real_cls_pred = discriminator(real_imgs)
            z_d = torch.randn(batch_size, config["latent_dim"], device=device); gen_labels_d = torch.randint(0, config["num_classes"], (batch_size,), device=device)
            with torch.no_grad(): fake_imgs = generator(z_d, gen_labels_d)
            fake_src_pred, fake_cls_pred = discriminator(fake_imgs)
            d_loss_gan, d_loss_aux_real, d_loss_aux_fake = gan_loss_fn.discriminator_loss(real_src_pred, fake_src_pred), aux_criterion(real_cls_pred, labels), aux_criterion(fake_cls_pred, gen_labels_d)
            d_loss = d_loss_gan + config["aux_loss_weight"] * (d_loss_aux_real + d_loss_aux_fake)
            d_loss.backward(); d_optimizer.step()
            g_optimizer.zero_grad()
            z_g, gen_labels_g = torch.randn(batch_size, config["latent_dim"], device=device), torch.randint(0, config["num_classes"], (batch_size,), device=device)
            gen_imgs_g = generator(z_g, gen_labels_g)
            g_src_output, g_cls_output = discriminator(gen_imgs_g)
            g_loss_gan, g_loss_aux = gan_loss_fn.generator_loss(g_src_output), aux_criterion(g_cls_output, gen_labels_g)
            g_loss = g_loss_gan + config["aux_loss_weight"] * g_loss_aux
            g_loss.backward(); g_optimizer.step()
            epoch_d_loss += d_loss.item(); epoch_g_loss += g_loss.item(); epoch_d_loss_gan += d_loss_gan.item(); epoch_g_loss_gan += g_loss_gan.item(); epoch_d_loss_aux_real += d_loss_aux_real.item(); epoch_d_loss_aux_fake += d_loss_aux_fake.item(); epoch_g_loss_aux += g_loss_aux.item()
        if rank == 0:
            num_batches = len(dataloader)
            avg_g_loss, avg_d_loss = epoch_g_loss / num_batches, epoch_d_loss / num_batches
            avg_g_gan, avg_g_aux = epoch_g_loss_gan / num_batches, epoch_g_loss_aux / num_batches
            avg_d_gan, avg_d_aux_r, avg_d_aux_f = epoch_d_loss_gan / num_batches, epoch_d_loss_aux_real / num_batches, epoch_d_loss_aux_fake / num_batches
            loss_log["epoch"].append(epoch + 1); loss_log["g_loss"].append(avg_g_loss); loss_log["d_loss"].append(avg_d_loss); loss_log["g_loss_gan"].append(avg_g_gan); loss_log["g_loss_aux"].append(avg_g_aux); loss_log["d_loss_gan"].append(avg_d_gan); loss_log["d_loss_aux_real"].append(avg_d_aux_r); loss_log["d_loss_aux_fake"].append(avg_d_aux_f)
            writer.add_scalar('Baseline_Finetune/Loss/G_Total', avg_g_loss, epoch+1); writer.add_scalar('Baseline_Finetune/Loss/D_Total', avg_d_loss, epoch+1)
            if (epoch+1) % config["save_every"] == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    g_eval = generator.module if world_size > 1 else generator
                    g_eval.eval(); samples = g_eval(fixed_noise, fixed_labels).detach().cpu(); vutils.save_image(samples, os.path.join(img_dir, f"epoch_{epoch+1:03d}.png"), nrow=10, normalize=True)
                    writer.add_image('Baseline_Finetune/Generated_Images', vutils.make_grid(samples, nrow=10, normalize=True), epoch+1); g_eval.train()
    elapsed_time = time.time() - start_time
    if rank == 0:
        if writer is not None: writer.close()
        create_training_gif(img_dir, gif_dir, "Baseline_Finetune")
        loss_log_path, plot_path = os.path.join(base_dir, "loss_log.json"), os.path.join(base_dir, "loss_plot.png")
        with open(loss_log_path, 'w') as f: json.dump(loss_log, f, indent=4)
        save_loss_plot(loss_log_path, plot_path, "Baseline Finetune")
        g_to_save, d_to_save = (m.module if world_size > 1 else m for m in [generator, discriminator])
        torch.save(g_to_save.state_dict(), os.path.join(model_save_dir, "generator.pth")); torch.save(d_to_save.state_dict(), os.path.join(model_save_dir, "discriminator.pth"))
        tqdm.write(f"✅ Baseline Finetune 완료. (소요 시간: {elapsed_time:.2f}초)")
    if world_size > 1: torch.distributed.barrier()
    return elapsed_time

def run_stage1_2(config, dataloader, sampler, soft_label_target, rank, world_size, device, fixed_noise, fixed_labels):
    start_time = time.time()
    dataset_name = config["current_dataset_name"]; d_config = config["dataset_configs"][dataset_name]; img_size, img_channels = d_config["img_size"], d_config["img_channels"]; forget_label, alpha = config["forget_label"], config["alpha"]
    unlearning_base_dir = f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}/Unlearning_soft_target_{soft_label_target:.1f}"
    
    writer, loss_log, skip_flag = None, None, torch.tensor([0], device=device)
    
    if rank == 0:
        model_dir = os.path.join(unlearning_base_dir, "models")
        final_gen_path = os.path.join(model_dir, "generator_step1.pth")
        final_disc_path = os.path.join(model_dir, "discriminator_step1.pth")
        if os.path.exists(final_gen_path) and os.path.exists(final_disc_path):
            tqdm.write(f"\n✅ Stage 1-2 SKIPPED: Step1 models already exist for soft_target={soft_label_target:.2f}")
            skip_flag.fill_(1)
            
    if world_size > 1: torch.distributed.broadcast(skip_flag, src=0)
    if skip_flag.item() == 1:
        if world_size > 1: torch.distributed.barrier()
        return 0.0
        
    # ... (중간 로직 동일) ...
    if rank == 0:
        print(f"\n--- Stage 1-2: D 소프트 망각 (soft_target={soft_label_target:.2f}) ---")
        model_dir, img_dir = [os.path.join(unlearning_base_dir, d) for d in ["models", "stage1_2_images"]]; os.makedirs(model_dir, exist_ok=True); os.makedirs(img_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(unlearning_base_dir, "runs", "Stage_1-2"))
        loss_log = {"epoch": [], "d_loss": [], "g_loss": [], "g_gan": [], "g_aux": [], "d_gan_real_nf": [], "d_gan_fake": [], "d_gan_forget": [], "d_aux_real": [], "d_aux_fake": []}

    generator = ACGANGenerator(config["latent_dim"], config["num_classes"], config["label_embedding_dim_g"], img_size, img_channels).to(device); discriminator = ACGANDiscriminator(config["num_classes"], img_size, img_channels).to(device)
    initialize_weights(generator); initialize_weights(discriminator)
    if world_size > 1:
        generator = DDP(generator, device_ids=[rank], find_unused_parameters=True); discriminator = DDP(discriminator, device_ids=[rank], find_unused_parameters=True)
    gan_loss_fn, aux_criterion = EfficientGANLoss().to(device), nn.CrossEntropyLoss().to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config["lr_g"], betas=(0.5, 0.999)); d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=(0.5, 0.999))
    epochs = config["step1_epochs"]
    epoch_pbar = tqdm(range(epochs), desc=f"Stage 1-2 Epochs", position=2, leave=False, disable=(rank != 0))
    for epoch in epoch_pbar:
        if sampler is not None: sampler.set_epoch(epoch)
        epoch_d_loss, epoch_g_loss, epoch_g_gan, epoch_g_aux = 0.0, 0.0, 0.0, 0.0
        epoch_d_gan_rnf, epoch_d_gan_fk, epoch_d_gan_fgt = 0.0, 0.0, 0.0
        epoch_d_aux_r, epoch_d_aux_f = 0.0, 0.0

        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=3, leave=False, disable=(rank != 0))
        for real_imgs, labels in batch_pbar:
            batch_size = real_imgs.size(0); real_imgs, labels = real_imgs.to(device), labels.to(device); d_optimizer.zero_grad()
            
            real_src_pred, real_cls_pred = discriminator(real_imgs)
            
            z_d = torch.randn(batch_size, config["latent_dim"], device=device)
            gen_labels_d = torch.randint(0, config["num_classes"], (batch_size,), device=device)
            with torch.no_grad(): fake_imgs = generator(z_d, gen_labels_d)
            
            fake_src_pred, fake_cls_pred = discriminator(fake_imgs)

            d_gan_fake = torch.mean(F.relu(1.0 + fake_src_pred)); d_loss_gan = d_gan_fake.clone()
            forget_mask = (labels == forget_label)
            d_gan_real_nf, d_gan_forget = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            if (~forget_mask).any(): d_gan_real_nf = torch.mean(F.relu(1.0 - real_src_pred[~forget_mask])); d_loss_gan += d_gan_real_nf
            if forget_mask.any():
                pred_forget = real_src_pred[forget_mask]
                d_gan_forget = alpha * torch.mean((1-soft_label_target)*F.relu(1+pred_forget) + soft_label_target*F.relu(1-pred_forget))
                d_loss_gan += d_gan_forget
            
            d_loss_aux_real = aux_criterion(real_cls_pred, labels)
            d_loss_aux_fake = aux_criterion(fake_cls_pred, gen_labels_d)
            d_loss_aux = d_loss_aux_real + d_loss_aux_fake
            
            d_loss = d_loss_gan + config["aux_loss_weight"] * d_loss_aux
            d_loss.backward(); d_optimizer.step()
            
            epoch_d_loss += d_loss.item(); epoch_d_gan_rnf += d_gan_real_nf.item(); epoch_d_gan_fk += d_gan_fake.item(); epoch_d_gan_fgt += d_gan_forget.item()
            epoch_d_aux_r += d_loss_aux_real.item(); epoch_d_aux_f += d_loss_aux_fake.item()

            g_optimizer.zero_grad()
            z_g, gen_labels_g = torch.randn(batch_size, config["latent_dim"], device=device), torch.randint(0, config["num_classes"], (batch_size,), device=device)
            gen_imgs_g = generator(z_g, gen_labels_g)
            g_src_output, g_cls_output = discriminator(gen_imgs_g)
            g_loss_gan, g_loss_aux = gan_loss_fn.generator_loss(g_src_output), config["aux_loss_weight"] * aux_criterion(g_cls_output, gen_labels_g)
            g_loss = g_loss_gan + g_loss_aux
            g_loss.backward(); g_optimizer.step()
            epoch_g_loss += g_loss.item(); epoch_g_gan += g_loss_gan.item(); epoch_g_aux += g_loss_aux.item()
            
        if rank == 0:
            num_batches = len(dataloader)
            avg_d, avg_g = epoch_d_loss/num_batches, epoch_g_loss/num_batches
            avg_g_gan, avg_g_aux = epoch_g_gan/num_batches, epoch_g_aux/num_batches
            avg_d_gan_rnf, avg_d_gan_fk, avg_d_gan_fgt = epoch_d_gan_rnf/num_batches, epoch_d_gan_fk/num_batches, epoch_d_gan_fgt/num_batches
            avg_d_aux_r, avg_d_aux_f = epoch_d_aux_r/num_batches, epoch_d_aux_f/num_batches
            
            loss_log["epoch"].append(epoch + 1); loss_log["d_loss"].append(avg_d); loss_log["g_loss"].append(avg_g)
            loss_log["g_gan"].append(avg_g_gan); loss_log["g_aux"].append(avg_g_aux); loss_log["d_gan_real_nf"].append(avg_d_gan_rnf)
            loss_log["d_gan_fake"].append(avg_d_gan_fk); loss_log["d_gan_forget"].append(avg_d_gan_fgt)
            loss_log["d_aux_real"].append(avg_d_aux_r); loss_log["d_aux_fake"].append(avg_d_aux_f)
            
            writer.add_scalar('Stage1-2/Loss/G_Total', avg_g, epoch + 1); writer.add_scalar('Stage1-2/Loss/G_GAN', avg_g_gan, epoch + 1); writer.add_scalar('Stage1-2/Loss/G_Aux', avg_g_aux, epoch + 1)
            writer.add_scalar('Stage1-2/Loss/D_Total', avg_d, epoch + 1); writer.add_scalar('Stage1-2/Loss/D_GAN_Real_NonForget', avg_d_gan_rnf, epoch + 1)
            writer.add_scalar('Stage1-2/Loss/D_GAN_Fake', avg_d_gan_fk, epoch + 1); writer.add_scalar('Stage1-2/Loss/D_GAN_Forget', avg_d_gan_fgt, epoch + 1)
            writer.add_scalar('Stage1-2/Loss/D_Aux_Real', avg_d_aux_r, epoch + 1); writer.add_scalar('Stage1-2/Loss/D_Aux_Fake', avg_d_aux_f, epoch + 1)

            if (epoch + 1) % config["save_every"] == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    g_eval = generator.module if world_size > 1 else generator
                    g_eval.eval(); samples = g_eval(fixed_noise, fixed_labels).detach().cpu(); vutils.save_image(samples, os.path.join(img_dir, f"epoch_{epoch+1:03d}.png"), nrow=10, normalize=True)
                    writer.add_image(f'Stage1-2/Generated_Images', vutils.make_grid(samples, nrow=10, normalize=True), epoch + 1); g_eval.train()
    elapsed_time = time.time() - start_time
    if rank == 0:
        if writer is not None: writer.close()
        loss_log_path, plot_path = os.path.join(unlearning_base_dir, "loss_log_stage1_2.json"), os.path.join(unlearning_base_dir, "loss_plot_stage1_2.png")
        with open(loss_log_path, 'w') as f: json.dump(loss_log, f, indent=4)
        save_loss_plot(loss_log_path, plot_path, f"Stage 1-2 (soft_target={soft_label_target:.2f})")
        model_dir = os.path.join(unlearning_base_dir, "models")
        g_to_save, d_to_save = (m.module if world_size > 1 else m for m in [generator, discriminator])
        torch.save(g_to_save.state_dict(), os.path.join(model_dir, "generator_step1.pth"))
        torch.save(d_to_save.state_dict(), os.path.join(model_dir, "discriminator_step1.pth"))
        tqdm.write(f"✅ Stage 1-2 완료. (소요 시간: {elapsed_time:.2f}초)")
    if world_size > 1: torch.distributed.barrier()
    return elapsed_time

def run_stage2(config, dataloader, sampler, soft_label_target, rank, world_size, device, fixed_noise, fixed_labels):
    start_time = time.time()
    dataset_name = config["current_dataset_name"]; d_config = config["dataset_configs"][dataset_name]; img_size, img_channels = d_config["img_size"], d_config["img_channels"]; forget_label, alpha, beta = config["forget_label"], config["alpha"], config["beta"]
    unlearning_base_dir = f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}/Unlearning_soft_target_{soft_label_target:.1f}"
    
    model_dir = os.path.join(unlearning_base_dir, "models"); final_gen_path, final_disc_path = os.path.join(model_dir, f"generator_final_soft_{soft_label_target:.1f}.pth"), os.path.join(model_dir, f"discriminator_final_soft_{soft_label_target:.1f}.pth")
    original_gen_path, stage1_2_disc_path = f"outputs/ACGAN(self_attention+residual_block+hinge_loss)/{dataset_name}/Original/models/generator.pth", os.path.join(unlearning_base_dir, "models", "discriminator_step1.pth")
    
    writer, loss_log, skip_flag = None, None, torch.tensor([0], device=device)
    
    if rank == 0:
        if os.path.exists(final_gen_path) and os.path.exists(final_disc_path):
            tqdm.write(f"\n✅ Stage 2 SKIPPED: Final models already exist for soft_target={soft_label_target:.2f}"); skip_flag.fill_(1)
        elif not os.path.exists(original_gen_path) or not os.path.exists(stage1_2_disc_path):
            tqdm.write(f"\n❌ Error: Stage 2 pre-requisite models not found for soft_target={soft_label_target:.2f}. Skipping."); skip_flag.fill_(1)
            
    if world_size > 1: torch.distributed.broadcast(skip_flag, src=0)
    if skip_flag.item() == 1:
        if world_size > 1: torch.distributed.barrier()
        return 0.0
        
    # ... (중간 로직 동일) ...
    if rank == 0:
        print(f"\n--- Stage 2: D&G 미세 조정 (soft_target={soft_label_target:.2f}) ---")
        img_dir = os.path.join(unlearning_base_dir, "stage2_images"); os.makedirs(img_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(unlearning_base_dir, "runs", "Stage_2"))
        loss_log = {"epoch": [], "d_loss": [], "g_loss": [], "g_gan_normal": [], "g_gan_forget": [], "g_aux": [], "d_gan_real_nf": [], "d_gan_fake": [], "d_gan_forget": []}

    g_cpu = ACGANGenerator(config["latent_dim"], config["num_classes"], config["label_embedding_dim_g"], img_size, img_channels)
    d_cpu = ACGANDiscriminator(config["num_classes"], img_size, img_channels)
    g_cpu.load_state_dict(torch.load(original_gen_path, map_location='cpu', weights_only=True))
    d_cpu.load_state_dict(torch.load(stage1_2_disc_path, map_location='cpu', weights_only=True))
    generator, discriminator = g_cpu.to(device), d_cpu.to(device)
    
    if world_size > 1:
        generator = DDP(generator, device_ids=[rank], find_unused_parameters=True); discriminator = DDP(discriminator, device_ids=[rank], find_unused_parameters=True)
    gan_loss_fn, aux_criterion = EfficientGANLoss().to(device), nn.CrossEntropyLoss().to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config["lr_g"], betas=(0.5, 0.999)); d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=(0.5, 0.999))
    epochs = config["step2_epochs"]
    epoch_pbar = tqdm(range(epochs), desc=f"Stage 2 Epochs", position=2, leave=False, disable=(rank != 0))
    for epoch in epoch_pbar:
        if sampler is not None: sampler.set_epoch(epoch)
        epoch_d_loss, epoch_g_loss, epoch_g_gan_n, epoch_g_gan_f, epoch_g_aux = 0.0, 0.0, 0.0, 0.0, 0.0
        epoch_d_gan_rnf, epoch_d_gan_fk, epoch_d_gan_fgt = 0.0, 0.0, 0.0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=3, leave=False, disable=(rank != 0))
        for real_imgs, labels in batch_pbar:
            batch_size = real_imgs.size(0); real_imgs, labels = real_imgs.to(device), labels.to(device); d_optimizer.zero_grad()
            real_src_pred, _ = discriminator(real_imgs)
            z_d = torch.randn(batch_size, config["latent_dim"], device=device)
            with torch.no_grad(): fake_imgs = generator(z_d, torch.randint(0, config["num_classes"], (batch_size,), device=device))
            fake_src_pred, _ = discriminator(fake_imgs)
            d_gan_fake = torch.mean(F.relu(1.0 + fake_src_pred)); d_loss_gan = d_gan_fake.clone()
            forget_mask_d = (labels == forget_label)
            d_gan_real_nf, d_gan_forget = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            if (~forget_mask_d).any(): d_gan_real_nf = torch.mean(F.relu(1.0 - real_src_pred[~forget_mask_d])); d_loss_gan += d_gan_real_nf
            if forget_mask_d.any():
                pred_forget_d = real_src_pred[forget_mask_d]
                d_gan_forget = alpha * torch.mean((1-soft_label_target)*F.relu(1+pred_forget_d) + soft_label_target*F.relu(1-pred_forget_d))
                d_loss_gan += d_gan_forget
            d_loss_gan.backward(); d_optimizer.step(); epoch_d_loss += d_loss_gan.item(); epoch_d_gan_rnf += d_gan_real_nf.item(); epoch_d_gan_fk += d_gan_fake.item(); epoch_d_gan_fgt += d_gan_forget.item()
            g_optimizer.zero_grad()
            z_g, gen_labels_g = torch.randn(batch_size, config["latent_dim"], device=device), torch.randint(0, config["num_classes"], (batch_size,), device=device)
            gen_imgs_g = generator(z_g, gen_labels_g)
            g_src_output, g_cls_output = discriminator(gen_imgs_g)
            g_loss_gan = torch.tensor(0.0, device=device); normal_mask_g, forget_mask_g = (gen_labels_g != forget_label), (gen_labels_g == forget_label)
            g_gan_normal, g_gan_forget = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            if normal_mask_g.any(): g_gan_normal = gan_loss_fn.generator_loss(g_src_output[normal_mask_g]); g_loss_gan += g_gan_normal
            if forget_mask_g.any(): g_gan_forget = beta * torch.mean(g_src_output[forget_mask_g]); g_loss_gan += g_gan_forget
            g_loss_aux = config["aux_loss_weight"] * aux_criterion(g_cls_output, gen_labels_g)
            g_loss = g_loss_gan + g_loss_aux; g_loss.backward(); g_optimizer.step()
            epoch_g_loss += g_loss.item(); epoch_g_gan_n += g_gan_normal.item(); epoch_g_gan_f += g_gan_forget.item(); epoch_g_aux += g_loss_aux.item()
        if rank == 0:
            num_batches = len(dataloader)
            avg_d, avg_g = epoch_d_loss/num_batches, epoch_g_loss/num_batches
            avg_g_gan_n, avg_g_gan_f, avg_g_aux = epoch_g_gan_n/num_batches, epoch_g_gan_f/num_batches, epoch_g_aux/num_batches
            avg_d_gan_rnf, avg_d_gan_fk, avg_d_gan_fgt = epoch_d_gan_rnf/num_batches, epoch_d_gan_fk/num_batches, epoch_d_gan_fgt/num_batches
            loss_log["epoch"].append(epoch + 1); loss_log["d_loss"].append(avg_d); loss_log["g_loss"].append(avg_g)
            loss_log["g_gan_normal"].append(avg_g_gan_n); loss_log["g_gan_forget"].append(avg_g_gan_f); loss_log["g_aux"].append(avg_g_aux)
            loss_log["d_gan_real_nf"].append(avg_d_gan_rnf); loss_log["d_gan_fake"].append(avg_d_gan_fk); loss_log["d_gan_forget"].append(avg_d_gan_fgt)
            writer.add_scalar('Stage2/Loss/G_Total', avg_g, epoch + 1); writer.add_scalar('Stage2/Loss/G_GAN_Normal', avg_g_gan_n, epoch + 1); writer.add_scalar('Stage2/Loss/G_GAN_Forget', avg_g_gan_f, epoch + 1); writer.add_scalar('Stage2/Loss/G_Aux', avg_g_aux, epoch + 1)
            writer.add_scalar('Stage2/Loss/D_Total', avg_d, epoch + 1); writer.add_scalar('Stage2/Loss/D_GAN_Real_NonForget', avg_d_gan_rnf, epoch + 1); writer.add_scalar('Stage2/Loss/D_GAN_Fake', avg_d_gan_fk, epoch + 1); writer.add_scalar('Stage2/Loss/D_GAN_Forget', avg_d_gan_fgt, epoch + 1)
            if (epoch + 1) % config["save_every"] == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    g_eval = generator.module if world_size > 1 else generator
                    g_eval.eval(); samples = g_eval(fixed_noise, fixed_labels).detach().cpu(); vutils.save_image(samples, os.path.join(img_dir, f"epoch_{epoch+1:03d}.png"), nrow=10, normalize=True)
                    writer.add_image(f'Stage2/Generated_Images', vutils.make_grid(samples, nrow=10, normalize=True), epoch + 1); g_eval.train()
    elapsed_time = time.time() - start_time
    if rank == 0:
        if writer is not None: writer.close()
        loss_log_path, plot_path = os.path.join(unlearning_base_dir, "loss_log_stage2.json"), os.path.join(unlearning_base_dir, "loss_plot_stage2.png")
        with open(loss_log_path, 'w') as f: json.dump(loss_log, f, indent=4)
        save_loss_plot(loss_log_path, plot_path, f"Stage 2 (soft_target={soft_label_target:.2f})")
        model_dir = os.path.join(unlearning_base_dir, "models")
        g_to_save, d_to_save = (m.module if world_size > 1 else m for m in [generator, discriminator])
        torch.save(g_to_save.state_dict(), final_gen_path); torch.save(d_to_save.state_dict(), final_disc_path)
        tqdm.write(f"✅ Stage 2 완료. (소요 시간: {elapsed_time:.2f}초)")
    if world_size > 1: torch.distributed.barrier()
    return elapsed_time