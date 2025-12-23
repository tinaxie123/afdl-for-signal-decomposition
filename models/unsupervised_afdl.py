"""
Unsupervised AFDL Implementation (Simplified Version)

This directly extends AFDL but replaces supervised segmentation with
unsupervised statistical heuristics.

Author: Haotong Xie
Institution: Shanghai University of Finance and Economics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.afdl import AFDL


class UnsupervisedSegmentation(nn.Module):
    """Unsupervised signal segmentation based on local variance"""
    def __init__(self, window_size: int = 20):
        super().__init__()
        self.window_size = window_size

    def forward(self, context: torch.Tensor, signal: torch.Tensor) -> tuple:
        """
        Args:
            context: [batch_size, seq_len, context_dim]
            signal: [batch_size, 1, seq_len]

        Returns:
            logits: [batch_size, seq_len, 2] (dummy, for compatibility)
            probs: [batch_size, seq_len, 2]
        """
        batch_size, _, seq_len = signal.shape
        device = signal.device

        # Compute local variance
        signal_1d = signal.squeeze(1)  # [batch, seq_len]
        padded = F.pad(signal_1d, (self.window_size // 2, self.window_size // 2), mode='reflect')

        # Sliding windows
        windows = padded.unfold(1, self.window_size, 1)  # [batch, padded_len - window_size + 1, window_size]
        local_var = windows.var(dim=-1)  # [batch, result_len]

        # Ensure output length matches input
        if local_var.shape[1] != seq_len:
            if local_var.shape[1] > seq_len:
                local_var = local_var[:, :seq_len]
            else:
                pad_len = seq_len - local_var.shape[1]
                local_var = F.pad(local_var, (0, pad_len), value=local_var[:, -1:].mean())

        # Normalize to [0, 1]
        var_min = local_var.min(dim=1, keepdim=True)[0]
        var_max = local_var.max(dim=1, keepdim=True)[0]
        normalized_var = (local_var - var_min) / (var_max - var_min + 1e-8)

        # Spike probability = high variance
        spike_prob = normalized_var.unsqueeze(-1)  # [batch, seq_len, 1]
        smooth_prob = 1 - spike_prob

        probs = torch.cat([smooth_prob, spike_prob], dim=-1)  # [batch, seq_len, 2]

        # Create dummy logits (for compatibility with loss computation)
        logits = torch.log(probs + 1e-10)

        return logits, probs


class UnsupervisedAFDL(AFDL):
    """
    Unsupervised AFDL - Extends AFDL but replaces supervised segmentation

    Only differences from AFDL:
    1. Uses UnsupervisedSegmentation instead of SignalSegmentationNetwork
    2. No segmentation loss during training
    """
    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 3600,
        dict_size: int = 128,
        context_dim: int = 128,
        hidden_dim: int = 64,
        num_heads: int = 4,
        sparsity_weight: float = 0.05
    ):
        # Initialize parent AFDL
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            dict_size=dict_size,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            sparsity_weight=sparsity_weight
        )

        # **REPLACE** supervised segmentation with unsupervised version
        self.segmentation = UnsupervisedSegmentation(window_size=20)

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - identical to AFDL except segmentation is unsupervised

        Args:
            x: Input signal [batch_size, 1, seq_len]

        Returns:
            Same dictionary as AFDL
        """
        batch_size = x.shape[0]

        # Step 1: Extract context embedding
        context = self.encoder(x)  # [batch, seq_len, context_dim]

        # Step 2: **UNSUPERVISED** Segment signal (no gradients needed)
        with torch.no_grad():
            seg_logits, seg_probs = self.segmentation(context, x)

        # Step 3: Basis function selection
        smooth_weights = self.smooth_selector(context)
        spike_weights = self.spike_selector(context)

        # Step 4: Generate parameters
        all_head_params = self.attention_modulator(context)

        # Step 5: Reconstruct signal (same as AFDL)
        reconstructions = []

        for head_idx, head_params in enumerate(all_head_params):
            head_reconstruction = torch.zeros(batch_size, self.seq_len, device=x.device)

            # Smooth basis functions
            for i, basis_idx in enumerate(self.dictionary.smooth_indices):
                basis_fn = self.dictionary.basis_functions[basis_idx]
                params = head_params[basis_idx]
                params_mean = params.mean(dim=1)
                atom = basis_fn(params_mean)
                weights_2d = smooth_weights[:, :, i] * seg_probs[:, :, 0]
                weighted_atom = weights_2d * atom
                head_reconstruction += weighted_atom

            # Spike basis functions
            for i, basis_idx in enumerate(self.dictionary.spike_indices):
                basis_fn = self.dictionary.basis_functions[basis_idx]
                params = head_params[basis_idx]
                params_mean = params.mean(dim=1)
                atom = basis_fn(params_mean)
                weights_2d = spike_weights[:, :, i] * seg_probs[:, :, 1]
                weighted_atom = weights_2d * atom
                head_reconstruction += weighted_atom

            reconstructions.append(head_reconstruction)

        # Average across heads
        final_reconstruction = torch.stack(reconstructions, dim=0).mean(dim=0)
        final_reconstruction = final_reconstruction.unsqueeze(1)

        result = {
            'reconstruction': final_reconstruction,
            'segmentation_logits': seg_logits,
            'segmentation_probs': seg_probs,
        }

        if return_components:
            result.update({
                'context': context,
                'smooth_weights': smooth_weights,
                'spike_weights': spike_weights,
                'head_params': all_head_params
            })

        return result

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        seg_labels: Optional[torch.Tensor] = None  # IGNORED
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss WITHOUT segmentation supervision

        Args:
            x: Input signal
            outputs: Model outputs (if None, will run forward)
            seg_labels: IGNORED (not used in unsupervised version)

        Returns:
            Dictionary of losses
        """
        if outputs is None:
            outputs = self.forward(x)

        reconstruction = outputs['reconstruction']

        # 1. Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x)

        # 2. Sparsity loss (L1 on weights)
        smooth_weights = outputs.get('smooth_weights')
        spike_weights = outputs.get('spike_weights')

        if smooth_weights is not None and spike_weights is not None:
            all_weights = torch.cat([
                smooth_weights.flatten(),
                spike_weights.flatten()
            ])
            sparsity_loss = torch.mean(torch.abs(all_weights))
        else:
            # If weights not in outputs, use dummy
            sparsity_loss = torch.tensor(0.0, device=x.device)

        # Total loss (NO segmentation loss)
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'segmentation_loss': torch.tensor(0.0, device=x.device)  # Always 0
        }
if __name__ == '__main__':
    model = UnsupervisedAFDL(
        input_dim=1,
        seq_len=3600,
        dict_size=128,
        context_dim=128,
        hidden_dim=64,
        num_heads=4,
        sparsity_weight=0.05
    )

    print(f"\nModel created:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 3600)

    print(f"\nTesting forward pass...")
    print(f"  Input shape: {x.shape}")

    outputs = model(x, return_components=True)

    print(f"\nOutputs:")
    print(f"  Reconstruction shape: {outputs['reconstruction'].shape}")
    print(f"  Segmentation probs shape: {outputs['segmentation_probs'].shape}")
    print(f"  Smooth weights shape: {outputs['smooth_weights'].shape}")
    print(f"  Spike weights shape: {outputs['spike_weights'].shape}")

    # Test loss computation
    losses = model.compute_loss(x, outputs)

    print(f"\nLosses:")
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print(f"  Reconstruction loss: {losses['reconstruction_loss'].item():.4f}")
    print(f"  Sparsity loss: {losses['sparsity_loss'].item():.4f}")
    print(f"  Segmentation loss: {losses['segmentation_loss'].item():.4f} (should be 0.0)")
