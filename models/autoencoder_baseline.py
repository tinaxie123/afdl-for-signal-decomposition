"""
1D CNN Autoencoder Baseline for ECG Signal Reconstruction

This baseline addresses Reviewer 190C's concern about comparison with
modern deep learning methods.

Author: Haotong Xie
Institution: Shanghai University of Finance and Economics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN1DAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for ECG signal reconstruction

    This serves as a modern deep learning baseline to compare with AFDL.
    """
    def __init__(self, input_dim: int = 1, seq_len: int = 3600, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: 3600 -> 1800
            nn.Conv1d(input_dim, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Layer 2: 1800 -> 900
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Layer 3: 900 -> 450
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Layer 4: 450 -> 225
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Layer 5: 225 -> 113 (approximately)
            nn.Conv1d(256, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Calculate encoded length
        self.encoded_len = self._get_encoded_len()

        # Bottleneck (compression to latent space)
        self.fc_encode = nn.Sequential(
            nn.Linear(256 * self.encoded_len, latent_dim),
            nn.ReLU()
        )

        # Expansion from latent space
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.encoded_len),
            nn.ReLU()
        )

        # Decoder (mirror of encoder)
        self.decoder = nn.Sequential(
            # Layer 1: 113 -> 225
            nn.ConvTranspose1d(256, 256, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Layer 2: 225 -> 450
            nn.ConvTranspose1d(256, 128, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Layer 3: 450 -> 900
            nn.ConvTranspose1d(128, 64, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Layer 4: 900 -> 1800
            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Layer 5: 1800 -> 3600
            nn.ConvTranspose1d(32, input_dim, kernel_size=15, stride=2, padding=7, output_padding=1),
        )

    def _get_encoded_len(self):
        """Calculate the length after encoding"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_dim, self.seq_len)
            encoded = self.encoder(dummy_input)
            return encoded.shape[2]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        batch_size = x.shape[0]

        # Convolutional encoding
        encoded = self.encoder(x)  # [batch, 256, encoded_len]

        # Flatten
        encoded_flat = encoded.view(batch_size, -1)  # [batch, 256 * encoded_len]

        # Bottleneck
        latent = self.fc_encode(encoded_flat)  # [batch, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction"""
        batch_size = latent.shape[0]

        # Expand from latent
        decoded_flat = self.fc_decode(latent)  # [batch, 256 * encoded_len]

        # Reshape
        decoded = decoded_flat.view(batch_size, 256, self.encoded_len)  # [batch, 256, encoded_len]

        # Convolutional decoding
        reconstruction = self.decoder(decoded)  # [batch, 1, seq_len]

        # Trim or pad to exact sequence length
        if reconstruction.shape[2] != self.seq_len:
            if reconstruction.shape[2] > self.seq_len:
                reconstruction = reconstruction[:, :, :self.seq_len]
            else:
                pad_len = self.seq_len - reconstruction.shape[2]
                reconstruction = F.pad(reconstruction, (0, pad_len))

        return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [batch_size, 1, seq_len]

        Returns:
            reconstruction: [batch_size, 1, seq_len]
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    def compute_loss(self, x: torch.Tensor) -> dict:
        """
        Compute reconstruction loss

        Args:
            x: Input signal [batch_size, 1, seq_len]

        Returns:
            Dictionary of losses
        """
        reconstruction = self.forward(x)

        # MSE loss
        recon_loss = F.mse_loss(reconstruction, x)

        return {
            'total_loss': recon_loss,
            'reconstruction_loss': recon_loss
        }


class ResidualBlock(nn.Module):
    """Residual block for deeper autoencoder"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class DeepCNN1DAutoencoder(nn.Module):
    """
    Deeper 1D CNN Autoencoder with residual connections

    More powerful baseline for better comparison
    """
    def __init__(self, input_dim: int = 1, seq_len: int = 3600, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_input = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.encoder_blocks = nn.Sequential(
            ResidualBlock(64),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            ResidualBlock(128),
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            ResidualBlock(256),
            nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        # Calculate encoded length
        self.encoded_len = self._get_encoded_len()

        # Bottleneck
        self.fc_encode = nn.Linear(512 * self.encoded_len, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * self.encoded_len)

        # Decoder
        self.decoder_blocks = nn.Sequential(
            ResidualBlock(512),
            nn.ConvTranspose1d(512, 256, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            ResidualBlock(256),
            nn.ConvTranspose1d(256, 128, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            ResidualBlock(128),
            nn.ConvTranspose1d(128, 64, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.decoder_output = nn.ConvTranspose1d(64, input_dim, kernel_size=15, stride=2, padding=7, output_padding=1)

    def _get_encoded_len(self):
        """Calculate the length after encoding"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_dim, self.seq_len)
            encoded = self.encoder_input(dummy_input)
            encoded = self.encoder_blocks(encoded)
            return encoded.shape[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Encode
        encoded = self.encoder_input(x)
        encoded = self.encoder_blocks(encoded)

        # Bottleneck
        encoded_flat = encoded.view(batch_size, -1)
        latent = self.fc_encode(encoded_flat)
        decoded_flat = self.fc_decode(latent)
        decoded = decoded_flat.view(batch_size, 512, self.encoded_len)

        # Decode
        decoded = self.decoder_blocks(decoded)
        reconstruction = self.decoder_output(decoded)

        # Ensure correct length
        if reconstruction.shape[2] != self.seq_len:
            if reconstruction.shape[2] > self.seq_len:
                reconstruction = reconstruction[:, :, :self.seq_len]
            else:
                pad_len = self.seq_len - reconstruction.shape[2]
                reconstruction = F.pad(reconstruction, (0, pad_len))

        return reconstruction

    def compute_loss(self, x: torch.Tensor) -> dict:
        reconstruction = self.forward(x)
        recon_loss = F.mse_loss(reconstruction, x)
        return {
            'total_loss': recon_loss,
            'reconstruction_loss': recon_loss
        }

if __name__ == '__main__':
    print("\n[1] Standard CNN1D Autoencoder")
    model = CNN1DAutoencoder(input_dim=1, seq_len=3600, latent_dim=128)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(4, 1, 3600)
    reconstruction = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {reconstruction.shape}")

    losses = model.compute_loss(x)
    print(f"  Loss: {losses['total_loss'].item():.4f}")

    # Test deep autoencoder
    print("\n[2] Deep CNN1D Autoencoder")
    deep_model = DeepCNN1DAutoencoder(input_dim=1, seq_len=3600, latent_dim=128)
    print(f"  Total parameters: {sum(p.numel() for p in deep_model.parameters()):,}")

    reconstruction = deep_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {reconstruction.shape}")

    losses = deep_model.compute_loss(x)
    print(f"  Loss: {losses['total_loss'].item():.4f}")

    print("\n" + "="*80)
    print("Test passed!")
    print("="*80)
