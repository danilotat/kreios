import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ImprovedGenomicCNN(nn.Module):
    """
    Improved CNN architecture for genomic pileup image classification
    with regularization and better feature extraction
    """
    def __init__(self, input_height: int, input_width: int, compressed_feature_size: int = 128, dropout_rate: float = 0.3):
        super(ImprovedGenomicCNN, self).__init__()
        
        # Feature extraction with batch normalization and dropout
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(dropout_rate * 0.5),  # Lower dropout in early layers
            
            # Second conv block  
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Only one pooling layer
            nn.Dropout2d(dropout_rate * 0.7),
            
            # Third conv block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(dropout_rate),
            
            # Fourth conv block (added depth instead of aggressive pooling)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Second pooling layer
            nn.Dropout2d(dropout_rate),
        )
        
        # Calculate the size after convolutions (assumes only 2 pooling layers)
        # Height and width are reduced by factor of 4 (2x2 pooling twice)
        self.flattened_size = 128 * (input_height // 4) * (input_width // 4)
        
        # MLP head with explicit size calculation and regularization
        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, compressed_feature_size * 2),
            nn.BatchNorm1d(compressed_feature_size * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(compressed_feature_size * 2, compressed_feature_size),
            nn.BatchNorm1d(compressed_feature_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.5)  # Less dropout before final features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.mlp_head(x)
        return x

class GenomicClassifier(nn.Module):
    """
    Improved classifier with focal loss support and better regularization
    """
    def __init__(self, input_height: int, input_width: int, compressed_feature_size: int = 128, 
                 num_classes: int = 1, dropout_rate: float = 0.3):
        super(GenomicClassifier, self).__init__()
        self.backbone = ImprovedGenomicCNN(input_height, input_width, compressed_feature_size, dropout_rate)
        
        # More robust classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(compressed_feature_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

class FocalLoss(nn.Module):
    """
    Focal Loss implementation to handle class imbalance
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal weight: (1 - p)^gamma for positive class, p^gamma for negative class
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting (higher weight for minority class)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Training modifications
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

