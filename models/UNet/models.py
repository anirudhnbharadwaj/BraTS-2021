import torch
import torch.nn as nn
from monai.networks.nets import UNet, AttentionUnet


class StandardUNet(nn.Module):
    """Standard 3D U-Net for brain tumor segmentation."""
    def __init__(self):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,  # 0: background, 1: ET, 2: NCR/NET, 4: ED
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )

    def forward(self, x):
        return self.model(x)

class AttentionUNet(nn.Module):
    """Attention U-Net using MONAI's AttentionUnet for enhanced focus on tumor regions."""
    def __init__(self):
        super().__init__()
        self.model = AttentionUnet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,  # 0: background, 1: ET, 2: NCR/NET, 4: ED
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.1  
        )

    def forward(self, x):
        return self.model(x)