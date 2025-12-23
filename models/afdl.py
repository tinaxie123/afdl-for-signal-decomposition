"""
Adaptive Functional Dictionary Learning (AFDL)
Implementation based on ICASSP 2026 paper

Author: Haotong Xie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


# ============================================================================
# 1. Hierarchical Functional Dictionary - Basis Functions
# ============================================================================

class BasisFunction(nn.Module):
    """Base class for all basis functions"""
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.t = torch.linspace(0, 1, seq_len)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            params: shape [batch_size, num_params]
        Returns:
            shape [batch_size, seq_len]
        """
        raise NotImplementedError


class GaborBasisFunction(BasisFunction):
    """Gabor wavelet: exp(-((t-mu)/sigma)^2) * cos(2*pi*freq*(t-mu))"""
    def __init__(self, seq_len: int):
        super().__init__(seq_len)
        # Parameters: [center, width, frequency]
        self.num_params = 3

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # params: [batch, 3] -> [mu, sigma, freq]
        mu = params[:, 0:1]  # [batch, 1]
        sigma = torch.clamp(params[:, 1:2], min=0.01)  # [batch, 1]
        freq = torch.clamp(params[:, 2:3], min=0.1)  # [batch, 1]

        t = self.t.to(params.device).unsqueeze(0)  # [1, seq_len]

        # Gaussian envelope
        envelope = torch.exp(-((t - mu) / sigma) ** 2)
        # Cosine modulation
        carrier = torch.cos(2 * math.pi * freq * (t - mu))

        return envelope * carrier  # [batch, seq_len]


class ExponentialDecayBasisFunction(BasisFunction):
    """Exponential decay: A * exp(-decay_rate * t) for spike-like patterns"""
    def __init__(self, seq_len: int):
        super().__init__(seq_len)
        # Parameters: [amplitude, decay_rate, onset]
        self.num_params = 3

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        amplitude = params[:, 0:1]  # [batch, 1]
        decay_rate = torch.clamp(params[:, 1:2], min=0.1)  # [batch, 1]
        onset = torch.clamp(params[:, 2:3], min=0, max=0.9)  # [batch, 1]

        t = self.t.to(params.device).unsqueeze(0)  # [1, seq_len]

        # Apply onset
        t_shifted = torch.clamp(t - onset, min=0)

        return amplitude * torch.exp(-decay_rate * t_shifted)


class GaussianBasisFunction(BasisFunction):
    """Gaussian: A * exp(-((t-mu)/sigma)^2)"""
    def __init__(self, seq_len: int):
        super().__init__(seq_len)
        # Parameters: [amplitude, center, width]
        self.num_params = 3

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        amplitude = params[:, 0:1]  # [batch, 1]
        mu = params[:, 1:2]  # [batch, 1]
        sigma = torch.clamp(params[:, 2:3], min=0.01)  # [batch, 1]

        t = self.t.to(params.device).unsqueeze(0)  # [1, seq_len]

        return amplitude * torch.exp(-((t - mu) / sigma) ** 2)


class ChirpBasisFunction(BasisFunction):
    """Chirp: cos(2*pi*(f0 + k*t)*t) with frequency modulation"""
    def __init__(self, seq_len: int):
        super().__init__(seq_len)
        # Parameters: [initial_freq, chirp_rate, amplitude]
        self.num_params = 3

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        f0 = torch.clamp(params[:, 0:1], min=0.1)  # [batch, 1]
        k = params[:, 1:2]  # chirp rate [batch, 1]
        amplitude = params[:, 2:3]  # [batch, 1]

        t = self.t.to(params.device).unsqueeze(0)  # [1, seq_len]

        phase = 2 * math.pi * (f0 * t + 0.5 * k * t ** 2)
        return amplitude * torch.cos(phase)


class LinearBasisFunction(BasisFunction):
    """Linear: slope * t + intercept"""
    def __init__(self, seq_len: int):
        super().__init__(seq_len)
        # Parameters: [slope, intercept]
        self.num_params = 2

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        slope = params[:, 0:1]  # [batch, 1]
        intercept = params[:, 1:2]  # [batch, 1]

        t = self.t.to(params.device).unsqueeze(0)  # [1, seq_len]

        return slope * t + intercept


class HierarchicalFunctionalDictionary(nn.Module):
    """
    Hierarchical dictionary with smooth and spike subdictionaries
    Distribution: 40% Gabor, 25% Exp Decay, 15% Gaussian, 15% Chirp, 5% Linear
    """
    def __init__(self, seq_len: int, dict_size: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.dict_size = dict_size

        # Calculate number of each basis function type
        num_gabor = int(dict_size * 0.40)
        num_exp_decay = int(dict_size * 0.25)
        num_gaussian = int(dict_size * 0.15)
        num_chirp = int(dict_size * 0.15)
        num_linear = dict_size - (num_gabor + num_exp_decay + num_gaussian + num_chirp)

        # Create basis functions
        self.basis_functions = nn.ModuleList()
        self.basis_types = []
        self.param_dims = []

        # Smooth subdictionary
        self.smooth_indices = []
        for _ in range(num_gabor):
            self.basis_functions.append(GaborBasisFunction(seq_len))
            self.basis_types.append('gabor')
            self.param_dims.append(3)
            self.smooth_indices.append(len(self.basis_functions) - 1)

        for _ in range(num_gaussian):
            self.basis_functions.append(GaussianBasisFunction(seq_len))
            self.basis_types.append('gaussian')
            self.param_dims.append(3)
            self.smooth_indices.append(len(self.basis_functions) - 1)

        for _ in range(num_chirp):
            self.basis_functions.append(ChirpBasisFunction(seq_len))
            self.basis_types.append('chirp')
            self.param_dims.append(3)
            self.smooth_indices.append(len(self.basis_functions) - 1)

        for _ in range(num_linear):
            self.basis_functions.append(LinearBasisFunction(seq_len))
            self.basis_types.append('linear')
            self.param_dims.append(2)
            self.smooth_indices.append(len(self.basis_functions) - 1)

        # Spike subdictionary
        self.spike_indices = []
        for _ in range(num_exp_decay):
            self.basis_functions.append(ExponentialDecayBasisFunction(seq_len))
            self.basis_types.append('exp_decay')
            self.param_dims.append(3)
            self.spike_indices.append(len(self.basis_functions) - 1)

        self.num_smooth = len(self.smooth_indices)
        self.num_spike = len(self.spike_indices)

        print(f"Dictionary initialized: {self.num_smooth} smooth + {self.num_spike} spike = {dict_size} total")

    def forward(self, params_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            params_list: List of parameter tensors for each basis function
        Returns:
            Dictionary matrix [dict_size, seq_len]
        """
        atoms = []
        for i, (basis_fn, params) in enumerate(zip(self.basis_functions, params_list)):
            atom = basis_fn(params)  # [batch, seq_len]
            atoms.append(atom)

        dictionary = torch.stack(atoms, dim=0)  # [dict_size, batch, seq_len]
        return dictionary


# ============================================================================
# 2. Signal Segmentation Component
# ============================================================================

class ContextEncoder(nn.Module):
    """
    Multi-modal feature encoder using Conv1D + BiLSTM
    Extracts local statistics, gradient features, and spectral features
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, context_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # Local feature extractor (Conv1D)
        self.local_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=11, padding=5),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Gradient feature extractor
        self.grad_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )

        # Spectral feature extractor (using learnable filters)
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=31, padding=15),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )

        # Temporal dependency modeling (BiLSTM)
        lstm_input_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # Project to context embedding
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
            nn.LeakyReLU(0.2)
        )

    def extract_gradient_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract first and second derivatives"""
        # First derivative
        first_deriv = x[:, :, 1:] - x[:, :, :-1]
        first_deriv = F.pad(first_deriv, (0, 1), mode='replicate')

        # Second derivative
        second_deriv = first_deriv[:, :, 1:] - first_deriv[:, :, :-1]
        second_deriv = F.pad(second_deriv, (0, 1), mode='replicate')

        return torch.cat([first_deriv, second_deriv], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input signal [batch_size, input_dim, seq_len]
        Returns:
            Context embedding [batch_size, seq_len, context_dim]
        """
        batch_size, _, seq_len = x.shape

        # Extract multi-modal features
        local_features = self.local_conv(x)  # [batch, hidden_dim, seq_len]

        # Gradient features
        grad_input = self.extract_gradient_features(x)
        grad_features = self.grad_conv(grad_input[:, :self.input_dim])  # [batch, hidden_dim//2, seq_len]

        # Spectral features (approximation using conv)
        spectral_features = self.spectral_conv(x)  # [batch, hidden_dim//2, seq_len]

        # Concatenate all features
        features = torch.cat([local_features, grad_features, spectral_features], dim=1)
        # [batch, lstm_input_dim, seq_len]

        # Transpose for LSTM: [batch, seq_len, features]
        features = features.transpose(1, 2)

        # BiLSTM for temporal modeling
        lstm_out, _ = self.lstm(features)  # [batch, seq_len, hidden_dim*2]

        # Project to context embedding
        context = self.projection(lstm_out)  # [batch, seq_len, context_dim]

        return context


class SignalSegmentationNetwork(nn.Module):
    """
    Segments signal into smooth and spike regions
    """
    def __init__(self, context_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.context_dim = context_dim
        self.num_classes = num_classes  # smooth and spike

        self.classifier = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.LayerNorm(context_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(context_dim // 2, num_classes)
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context: [batch_size, seq_len, context_dim]
        Returns:
            logits: [batch_size, seq_len, num_classes]
            probs: [batch_size, seq_len, num_classes] (after softmax)
        """
        logits = self.classifier(context)  # [batch, seq_len, num_classes]
        probs = F.softmax(logits, dim=-1)

        return logits, probs


# ============================================================================
# 3. Brain-Inspired Basis Function Selection Network
# ============================================================================

class BasisFunctionSelectionNetwork(nn.Module):
    """
    Region-adaptive basis function selection
    Independent selection network for each region type (smooth/spike)
    """
    def __init__(self, context_dim: int, num_basis_functions: int):
        super().__init__()
        self.context_dim = context_dim
        self.num_basis_functions = num_basis_functions

        # 3-layer fully connected architecture
        self.selection_network = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.BatchNorm1d(context_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(context_dim, context_dim // 2),
            nn.BatchNorm1d(context_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(context_dim // 2, num_basis_functions)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [batch_size, seq_len, context_dim]
        Returns:
            selection_weights: [batch_size, seq_len, num_basis_functions]
        """
        batch_size, seq_len, _ = context.shape

        # Reshape for batch norm: [batch*seq_len, context_dim]
        context_flat = context.reshape(batch_size * seq_len, -1)

        # Get selection scores
        scores = self.selection_network(context_flat)  # [batch*seq_len, num_basis]

        # Reshape back and apply softmax
        scores = scores.reshape(batch_size, seq_len, -1)
        weights = F.softmax(scores, dim=-1)

        return weights


# ============================================================================
# 4. Attention Modulator - Parameter Prediction Network
# ============================================================================

class ParameterPredictionNetwork(nn.Module):
    """
    Predicts basis function parameters from context
    One network per (region, basis_function) pair
    """
    def __init__(self, context_dim: int, num_params: int):
        super().__init__()
        self.context_dim = context_dim
        self.num_params = num_params

        # Lightweight conv-fc architecture
        self.network = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.LayerNorm(context_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(context_dim // 2, num_params)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [batch_size, seq_len, context_dim]
        Returns:
            params: [batch_size, seq_len, num_params]
        """
        return self.network(context)


class AttentionModulator(nn.Module):
    """
    Multi-head attention modulator for adaptive parameter generation
    """
    def __init__(
        self,
        context_dim: int,
        dict_size: int,
        param_dims: List[int],
        num_heads: int = 4
    ):
        super().__init__()
        self.context_dim = context_dim
        self.dict_size = dict_size
        self.param_dims = param_dims
        self.num_heads = num_heads

        # Create parameter prediction networks for each basis function and head
        self.param_networks = nn.ModuleList([
            nn.ModuleList([
                ParameterPredictionNetwork(context_dim, num_params)
                for num_params in param_dims
            ])
            for _ in range(num_heads)
        ])

    def forward(self, context: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Args:
            context: [batch_size, seq_len, context_dim]
        Returns:
            params: [num_heads][dict_size] tensors of shape [batch_size, seq_len, num_params]
        """
        all_params = []

        for head_networks in self.param_networks:
            head_params = []
            for param_net in head_networks:
                params = param_net(context)  # [batch, seq_len, num_params]
                head_params.append(params)
            all_params.append(head_params)

        return all_params


# ============================================================================
# 5. Complete AFDL Model
# ============================================================================

class AFDL(nn.Module):
    """
    Complete Adaptive Functional Dictionary Learning model
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
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.dict_size = dict_size
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.sparsity_weight = sparsity_weight

        # 1. Hierarchical Functional Dictionary
        self.dictionary = HierarchicalFunctionalDictionary(seq_len, dict_size)

        # 2. Context Encoder
        self.encoder = ContextEncoder(input_dim, hidden_dim, context_dim)

        # 3. Signal Segmentation
        self.segmentation = SignalSegmentationNetwork(context_dim, num_classes=2)

        # 4. Basis Function Selection Networks (one for smooth, one for spike)
        self.smooth_selector = BasisFunctionSelectionNetwork(
            context_dim, self.dictionary.num_smooth
        )
        self.spike_selector = BasisFunctionSelectionNetwork(
            context_dim, self.dictionary.num_spike
        )

        # 5. Attention Modulator
        self.attention_modulator = AttentionModulator(
            context_dim, dict_size, self.dictionary.param_dims, num_heads
        )

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input signal [batch_size, input_dim, seq_len]
            return_components: Whether to return intermediate components

        Returns:
            Dictionary containing:
                - reconstruction: [batch_size, input_dim, seq_len]
                - segmentation_logits: [batch_size, seq_len, 2]
                - segmentation_probs: [batch_size, seq_len, 2]
                - (optional) other components
        """
        batch_size = x.shape[0]

        # Step 1: Extract context embedding
        context = self.encoder(x)  # [batch, seq_len, context_dim]

        # Step 2: Segment signal into smooth and spike regions
        seg_logits, seg_probs = self.segmentation(context)
        # seg_probs: [batch, seq_len, 2] where [:,:,0] is smooth, [:,:,1] is spike

        # Step 3: Basis function selection for each region
        smooth_weights = self.smooth_selector(context)  # [batch, seq_len, num_smooth]
        spike_weights = self.spike_selector(context)  # [batch, seq_len, num_spike]

        # Step 4: Generate parameters for each basis function (multi-head)
        all_head_params = self.attention_modulator(context)
        # all_head_params: [num_heads][dict_size] with shape [batch, seq_len, num_params]

        # Step 5: Reconstruct signal - OPTIMIZED VERSION
        # Use mean parameters across time for efficiency (approximation)
        reconstructions = []

        for head_idx, head_params in enumerate(all_head_params):
            head_reconstruction = torch.zeros(batch_size, self.seq_len, device=x.device)

            # Process smooth basis functions
            for i, basis_idx in enumerate(self.dictionary.smooth_indices):
                basis_fn = self.dictionary.basis_functions[basis_idx]
                params = head_params[basis_idx]  # [batch, seq_len, num_params]

                # Use mean parameters for efficiency
                params_mean = params.mean(dim=1)  # [batch, num_params]

                # Evaluate basis function once
                atom = basis_fn(params_mean)  # [batch, seq_len]

                # Apply time-varying weights
                weights_2d = smooth_weights[:, :, i] * seg_probs[:, :, 0]  # [batch, seq_len]

                # Weight the atom (broadcasting)
                weighted_atom = weights_2d * atom  # [batch, seq_len]
                head_reconstruction += weighted_atom

            # Process spike basis functions
            for i, basis_idx in enumerate(self.dictionary.spike_indices):
                basis_fn = self.dictionary.basis_functions[basis_idx]
                params = head_params[basis_idx]  # [batch, seq_len, num_params]

                params_mean = params.mean(dim=1)  # [batch, num_params]
                atom = basis_fn(params_mean)  # [batch, seq_len]

                weights_2d = spike_weights[:, :, i] * seg_probs[:, :, 1]  # [batch, seq_len]
                weighted_atom = weights_2d * atom
                head_reconstruction += weighted_atom

            reconstructions.append(head_reconstruction)

        # Average across heads
        final_reconstruction = torch.stack(reconstructions, dim=0).mean(dim=0)
        # [batch, seq_len]

        # Reshape to match input
        final_reconstruction = final_reconstruction.unsqueeze(1)  # [batch, 1, seq_len]

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
        seg_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss

        Args:
            x: Input signal [batch_size, input_dim, seq_len]
            seg_labels: Optional segmentation labels [batch_size, seq_len]

        Returns:
            Dictionary of losses
        """
        outputs = self.forward(x, return_components=True)
        reconstruction = outputs['reconstruction']
        seg_logits = outputs['segmentation_logits']
        smooth_weights = outputs['smooth_weights']
        spike_weights = outputs['spike_weights']

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)

        # Sparsity loss (L1 regularization on selection weights)
        smooth_sparsity = torch.mean(torch.abs(smooth_weights))
        spike_sparsity = torch.mean(torch.abs(spike_weights))
        sparsity_loss = self.sparsity_weight * (smooth_sparsity + spike_sparsity)

        # Segmentation loss (if labels provided)
        seg_loss = torch.tensor(0.0, device=x.device)
        if seg_labels is not None:
            # Weighted cross-entropy
            seg_loss = F.cross_entropy(
                seg_logits.reshape(-1, 2),
                seg_labels.reshape(-1),
                reduction='mean'
            )

        # Total loss
        total_loss = recon_loss + sparsity_loss + 0.1 * seg_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'seg_loss': seg_loss
        }


# ============================================================================
# 6. Helper Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 7. Test and Demo
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Testing AFDL Implementation")
    print("=" * 80)

    # Model configuration
    batch_size = 4
    input_dim = 1
    seq_len = 3600  # 10 seconds at 360 Hz
    dict_size = 128

    # Create model
    model = AFDL(
        input_dim=input_dim,
        seq_len=seq_len,
        dict_size=dict_size,
        context_dim=128,
        hidden_dim=64,
        num_heads=4,
        sparsity_weight=0.05
    )

    print(f"\nModel Parameters: {count_parameters(model):,}")

    # Test with random input
    signal = torch.randn(batch_size, input_dim, seq_len)

    print(f"\nInput shape: {signal.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(signal, return_components=True)

    print("\nOutputs:")
    print(f"  Reconstruction shape: {outputs['reconstruction'].shape}")
    print(f"  Segmentation probs shape: {outputs['segmentation_probs'].shape}")
    print(f"  Smooth weights shape: {outputs['smooth_weights'].shape}")
    print(f"  Spike weights shape: {outputs['spike_weights'].shape}")

    # Compute loss
    print("\nComputing losses...")
    losses = model.compute_loss(signal)

    print("\nLosses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.6f}")

    # Check reconstruction quality
    mse = F.mse_loss(outputs['reconstruction'], signal)
    snr = 10 * torch.log10(signal.var() / mse)
    print(f"\nReconstruction SNR: {snr.item():.2f} dB")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
