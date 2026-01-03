import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SelfAttention(nn.Module):
    """Self-Attention 모듈 (SAGAN 기반)"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    """Residual Block with conditional batch normalization for Generator"""
    def __init__(self, in_channels, out_channels, num_classes=10, upsample=False):
        super(ResidualBlock, self).__init__()
        self.upsample = upsample
        self.bn1 = nn.BatchNorm2d(in_channels, affine=False)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=False)
        self.gamma1_emb = nn.Embedding(num_classes, in_channels)
        self.beta1_emb = nn.Embedding(num_classes, in_channels)
        self.gamma2_emb = nn.Embedding(num_classes, out_channels)
        self.beta2_emb = nn.Embedding(num_classes, out_channels)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.skip_conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)) if in_channels != out_channels or upsample else nn.Identity()

    def forward(self, x, labels):
        h = self.bn1(x)
        gamma1, beta1 = self.gamma1_emb(labels).unsqueeze(-1).unsqueeze(-1), self.beta1_emb(labels).unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + gamma1) + beta1
        h = F.relu(h)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = self.bn2(h)
        gamma2, beta2 = self.gamma2_emb(labels).unsqueeze(-1).unsqueeze(-1), self.beta2_emb(labels).unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + gamma2) + beta2
        h = F.relu(h)
        h = self.conv2(h)
        skip_x = x
        if self.upsample:
            skip_x = F.interpolate(skip_x, scale_factor=2, mode='nearest')
        skip_x = self.skip_conv(skip_x)
        return h + skip_x

class ACGANGenerator(nn.Module):
    """ACGAN Generator - ResNet 기반 + Self-Attention + Conditional BN"""
    def __init__(self, latent_dim=128, num_classes=10, label_embedding_dim=50, img_size=32, img_channels=1):
        super(ACGANGenerator, self).__init__()
        
        if img_size == 28:
            self.init_size, self.linear_channels = 7, 256
        elif img_size == 32:
            self.init_size, self.linear_channels = 4, 512
        elif img_size == 96:
            self.init_size, self.linear_channels = 6, 512
        else:
            raise ValueError(f"Unsupported img_size: {img_size}. Must be 28, 32, or 96.")

        self.label_emb = nn.Embedding(num_classes, label_embedding_dim)
        self.linear = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim + label_embedding_dim, self.linear_channels * self.init_size * self.init_size)),
            nn.BatchNorm1d(self.linear_channels * self.init_size * self.init_size),
            nn.ReLU(True)
        )

        if img_size == 28:
            self.blocks = nn.ModuleList([
                ResidualBlock(256, 128, num_classes, upsample=True), # 7 -> 14
                ResidualBlock(128, 64, num_classes, upsample=True), # 14 -> 28
            ])
            self.attention = SelfAttention(64)
            self.final_conv = nn.Conv2d(64, img_channels, 3, 1, 1)
        elif img_size == 32:
            self.blocks = nn.ModuleList([
                ResidualBlock(512, 256, num_classes, upsample=True), # 4 -> 8
                ResidualBlock(256, 128, num_classes, upsample=True), # 8 -> 16
                ResidualBlock(128, 64, num_classes, upsample=True), # 16 -> 32
            ])
            self.attention = SelfAttention(64)
            self.final_conv = nn.Conv2d(64, img_channels, 3, 1, 1)
        elif img_size == 96:
            self.blocks = nn.ModuleList([
                ResidualBlock(512, 512, num_classes, upsample=True), # 6 -> 12
                ResidualBlock(512, 256, num_classes, upsample=True), # 12 -> 24
                ResidualBlock(256, 128, num_classes, upsample=True), # 24 -> 48
                ResidualBlock(128, 64, num_classes, upsample=True),  # 48 -> 96
            ])
            self.attention = SelfAttention(128) # Apply attention at a slightly larger feature map
            self.final_conv = nn.Conv2d(64, img_channels, 3, 1, 1)

    def forward(self, z, labels):
        gen_input = torch.cat((z, self.label_emb(labels)), dim=-1)
        h = self.linear(gen_input)
        h = h.view(h.size(0), self.linear_channels, self.init_size, self.init_size)
        
        for i, block in enumerate(self.blocks):
            h = block(h, labels)
            # Apply attention at a predefined layer (e.g., after the 3rd block for 96x96)
            if self.attention and h.size(1) == self.attention.in_channels:
                 h = self.attention(h)
        
        h = self.final_conv(h)
        return torch.tanh(h)

class ACGANDiscriminator(nn.Module):
    """ACGAN Discriminator - Spectral Normalization + Self-Attention + Source/Class Heads"""
    def __init__(self, num_classes=10, img_size=32, img_channels=1):
        super(ACGANDiscriminator, self).__init__()
        self.num_classes = num_classes

        if img_size == 28:
            self.main = nn.Sequential(
                spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
                SelfAttention(256),
                spectral_norm(nn.Conv2d(256, 512, 3, 1, 0)), nn.LeakyReLU(0.2, inplace=True),
            )
            final_conv_channels = 512
        elif img_size == 32:
            self.main = nn.Sequential(
                spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
                SelfAttention(256),
                spectral_norm(nn.Conv2d(256, 512, 4, 1, 0)), nn.LeakyReLU(0.2, inplace=True),
            )
            final_conv_channels = 512
        elif img_size == 96:
            self.main = nn.Sequential(
                spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True), # 96->48
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),  # 48->24
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True), # 24->12
                SelfAttention(256),
                spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True), # 12->6
                spectral_norm(nn.Conv2d(512, 1024, 6, 1, 0)), nn.LeakyReLU(0.2, inplace=True),# 6->1
            )
            final_conv_channels = 1024
        else:
            raise ValueError(f"Unsupported img_size: {img_size}. Must be 28, 32, or 96.")

        self.source_output = spectral_norm(nn.Conv2d(final_conv_channels, 1, 1, 1, 0))
        self.class_output = spectral_norm(nn.Conv2d(final_conv_channels, num_classes, 1, 1, 0))

    def forward(self, x):
        h = self.main(x)
        source_pred = self.source_output(h).view(x.size(0), -1)
        class_pred = self.class_output(h).view(x.size(0), self.num_classes)
        return source_pred, class_pred

class EfficientGANLoss(nn.Module):
    """효율적인 GAN Loss - Hinge Loss 사용"""
    def __init__(self):
        super(EfficientGANLoss, self).__init__()
    
    def discriminator_loss(self, real_pred, fake_pred):
        # 이 부분이 Hinge Loss의 판별자 손실 함수입니다.
        real_loss = torch.mean(F.relu(1.0 - real_pred))
        fake_loss = torch.mean(F.relu(1.0 + fake_pred))
        return real_loss + fake_loss
    
    def generator_loss(self, fake_pred):
        # 이 부분이 Hinge Loss의 생성자 손실 함수입니다.
        return -torch.mean(fake_pred)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight is not None: nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)