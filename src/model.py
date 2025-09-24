import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SimpleGenomicCNN(nn.Module):
    """
    Simple CNN architecture for genomic pileup image classification
    Input: 6x256x256 (6 channels representing different genomic features)
    """
    def __init__(self, compressed_feature_size=128):
        super(SimpleGenomicCNN, self).__init__()
        # This is the feature extraction part of the network
        self.encoder = nn.Sequential(
            # First convolutional block
            # Input: (6, 256, 101)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (16, 128, 50)
            
            # Second convolutional block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (32, 64, 25)
            
            # Third convolutional block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (64, 32, 12)
        )
        
        # This part flattens the feature map and creates the compressed representation
        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=compressed_feature_size),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN.
        
        Args:
            x: The input tensor of shape (batch_size, 6, 256, 101).
            
        Returns:
            A compressed representation of shape (batch_size, compressed_feature_size).
        """
        x = self.encoder(x)
        x = self.mlp_head(x)
        return x
