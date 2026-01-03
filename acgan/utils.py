import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import json

def save_loss_plot(loss_log_path, output_path, model_name):
    """
    loss_log.json 파일을 읽어 손실 곡선 그래프를 저장합니다.
    """
    try:
        with open(loss_log_path, 'r') as f:
            loss_log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not read or decode loss log file at {loss_log_path}. Skipping plot generation.")
        return

    plt.figure(figsize=(12, 8))
    
    # Generator Loss
    if "g_loss" in loss_log and "epoch" in loss_log:
        plt.subplot(2, 1, 1)
        plt.plot(loss_log["epoch"], loss_log["g_loss"], label="Total G Loss")
        if "g_loss_gan" in loss_log:
            plt.plot(loss_log["epoch"], loss_log["g_loss_gan"], label="G GAN Loss", linestyle='--')
        if "g_loss_aux" in loss_log:
            plt.plot(loss_log["epoch"], loss_log["g_loss_aux"], label="G Aux Loss", linestyle=':')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Generator Loss: {model_name}")
        plt.legend()
        plt.grid(True)

    # Discriminator Loss
    if "d_loss" in loss_log and "epoch" in loss_log:
        plt.subplot(2, 1, 2)
        plt.plot(loss_log["epoch"], loss_log["d_loss"], label="Total D Loss")
        if "d_loss_gan" in loss_log:
            plt.plot(loss_log["epoch"], loss_log["d_loss_gan"], label="D GAN Loss", linestyle='--')
        if "d_loss_aux_real" in loss_log:
            plt.plot(loss_log["epoch"], loss_log["d_loss_aux_real"], label="D Aux Real Loss", linestyle=':')
        if "d_loss_aux_fake" in loss_log:
            plt.plot(loss_log["epoch"], loss_log["d_loss_aux_fake"], label="D Aux Fake Loss", linestyle='-.')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Discriminator Loss: {model_name}")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss curve to {output_path}")

def create_training_gif(image_dir, output_path, gif_name):
    """
    지정된 디렉토리의 PNG 이미지들로 GIF를 생성합니다.
    """
    frames = []
    # glob.glob으로 파일 목록을 가져온 후, 정렬하여 순서를 보장합니다.
    file_paths = sorted(glob.glob(os.path.join(image_dir, "epoch_*.png")))
    for p in file_paths:
        try:
            frames.append(Image.open(p))
        except IOError:
            print(f"Warning: Could not open image file {p}. Skipping.")
    
    if frames:
        full_gif_path = os.path.join(output_path, f"{gif_name}_training.gif")
        frames[0].save(full_gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)
        print(f"Saved training GIF to {full_gif_path}")