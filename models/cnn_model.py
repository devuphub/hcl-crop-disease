"""
Crop Disease Detection CNN Architecture
Includes:
1. Custom Deep Convolutional Neural Network (CropDiseaseCNN)
2. MobileNetV2 Transfer Learning Architecture (CropDiseaseMobileNet)
3. Model Factory with support for weight initialization & feature extraction
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True, dropout_rate: float = 0.2):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CropDiseaseCNN(nn.Module):
    """
    Custom Deep Convolutional Neural Network tailored for high-accuracy crop disease identification.
    Features:
    - 4 progressive convolutional blocks (3 -> 32 -> 64 -> 128 -> 256 channels)
    - Batch Normalization for training stability
    - Adaptive Average Pooling for spatial dimension independence
    - Regularization with Dropout to minimize overfitting
    """
    def __init__(self, num_classes: int = 18, dropout_rate: float = 0.35):
        super(CropDiseaseCNN, self).__init__()
        self.num_classes = num_classes

        # Feature Extractor
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True, dropout_rate=0.15),     # Output: (B, 32, H/2, W/2)
            ConvBlock(32, 64, pool=True, dropout_rate=0.2),     # Output: (B, 64, H/4, W/4)
            ConvBlock(64, 128, pool=True, dropout_rate=0.25),   # Output: (B, 128, H/8, W/8)
            ConvBlock(128, 256, pool=True, dropout_rate=0.3)    # Output: (B, 256, H/16, W/16)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


class CropDiseaseMobileNet(nn.Module):
    """
    MobileNetV2 based architecture for ultra-lightweight and fast inference.
    Ideal for cloud and edge deployments.
    """
    def __init__(self, num_classes: int = 18, pretrained: bool = True):
        super(CropDiseaseMobileNet, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v2(weights=weights)
        self.features = base_model.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        in_features = base_model.last_channel
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(model_type: str = "cnn", num_classes: int = 18, pretrained: bool = True) -> nn.Module:
    """
    Factory function to retrieve model instance.
    """
    if model_type.lower() == "cnn":
        return CropDiseaseCNN(num_classes=num_classes)
    elif model_type.lower() in ["mobilenet", "mobilenetv2"]:
        return CropDiseaseMobileNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'cnn' or 'mobilenet'.")


if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)
    cnn = get_model("cnn", num_classes=18)
    out_cnn = cnn(dummy_input)
    print("CropDiseaseCNN output shape:", out_cnn.shape)

    mobilenet = get_model("mobilenet", num_classes=18, pretrained=False)
    out_mobilenet = mobilenet(dummy_input)
    print("CropDiseaseMobileNet output shape:", out_mobilenet.shape)
