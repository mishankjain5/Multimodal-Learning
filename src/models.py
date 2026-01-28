import torch
import torch.nn as nn
import torch.nn.functional as F

class RGBEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class RGBEncoderStrided(nn.Module):
    """
    Task 4: RGB encoder with stride‑2 convolutions instead of MaxPool2d
    for spatial downsampling
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 (replace first MaxPool2d)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),   # 128 -> 64

            # Block 2 (replace second MaxPool2d)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),   # 64 -> 32

            # Final conv + global pooling
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.features(x)           # B, 128, 1, 1
        x = x.view(x.size(0), -1)      # B, 128
        x = self.fc(x)                 # B, embedding_dim
        return x


class LiDAREncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class LateFusionClassifier(nn.Module):
    def __init__(self, lidar_input_dim, embedding_dim=128, num_classes=2):
        super().__init__()

        self.rgb_encoder = RGBEncoder(embedding_dim)
        self.lidar_encoder = LiDAREncoder(lidar_input_dim, embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, lidar):
        rgb_emb = self.rgb_encoder(rgb)
        lidar_emb = self.lidar_encoder(lidar)

        fused = torch.cat([rgb_emb, lidar_emb], dim=1)
        out = self.classifier(fused)
        return out

class IntermediateFusionConcat(nn.Module):
    def __init__(self, lidar_input_dim, embedding_dim=128, num_classes=2):
        super().__init__()

        self.rgb_encoder = RGBEncoder(embedding_dim)
        self.lidar_encoder = LiDAREncoder(lidar_input_dim, embedding_dim)

        # Fusion happens earlier
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, rgb, lidar):
        rgb_emb = self.rgb_encoder(rgb)
        lidar_emb = self.lidar_encoder(lidar)

        fused = torch.cat([rgb_emb, lidar_emb], dim=1)
        fused = self.fusion(fused)

        return self.classifier(fused)

class IntermediateFusionHadamard(nn.Module):
    def __init__(self, lidar_input_dim, embedding_dim=128, num_classes=2):
        super().__init__()

        self.rgb_encoder = RGBEncoder(embedding_dim)
        self.lidar_encoder = LiDAREncoder(lidar_input_dim, embedding_dim)

        # Optional refinement after Hadamard product
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU()
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, rgb, lidar):
        rgb_emb = self.rgb_encoder(rgb)
        lidar_emb = self.lidar_encoder(lidar)

        # Hadamard (element-wise) product
        fused = rgb_emb * lidar_emb

        fused = self.fusion(fused)
        return self.classifier(fused)

class IntermediateFusionAdd(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=2):
        super().__init__()

        # RGB encoder
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 128 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 64 → 32
        )

        # LiDAR encoder (MATCHES RGB SHAPE)
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(1),
        )

        # Shared layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(64, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def forward(self, rgb, lidar):
        rgb_feat = self.rgb_encoder(rgb)

        # Add channel dim to LiDAR: (B, 64, 64) → (B, 1, 64, 64)
        lidar = lidar.unsqueeze(1)
        lidar_feat = self.lidar_encoder(lidar)

        # ADDITION FUSION (SHAPES MATCH)
        fused = rgb_feat + lidar_feat

        fused = self.shared_conv(fused)
        fused = fused.view(fused.size(0), -1)

        return self.classifier(fused)
    

# -------------------------
# Task 4: MaxPool vs Strided Hadamard models
# -------------------------

class IntermediateFusionHadamardMaxPool(nn.Module):
    """
    Task 4 variant using the original MaxPool‑based RGBEncoder.
    """
    def __init__(self, lidar_input_dim, embedding_dim=128, num_classes=2):
        super().__init__()
        self.rgb_encoder = RGBEncoder(embedding_dim)
        self.lidar_encoder = LiDAREncoder(lidar_input_dim, embedding_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, rgb, lidar):
        rgb_emb = self.rgb_encoder(rgb)
        lidar_emb = self.lidar_encoder(lidar)
        fused = rgb_emb * lidar_emb
        fused = self.fusion(fused)
        return self.classifier(fused)


class IntermediateFusionHadamardStrided(nn.Module):
    """
    Task 4 variant using RGBEncoderStrided with stride2 convolutions
    instead of MaxPool2d for downsampling.
    """
    def __init__(self, lidar_input_dim, embedding_dim = 128, num_classes = 2):
        super().__init__()
        self.rgb_encoder = RGBEncoderStrided(embedding_dim)
        self.lidar_encoder = LiDAREncoder(lidar_input_dim, embedding_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, rgb, lidar):
        rgb_emb = self.rgb_encoder(rgb)
        lidar_emb = self.lidar_encoder(lidar)
        fused = rgb_emb * lidar_emb
        fused = self.fusion(fused)
        return self.classifier(fused)
