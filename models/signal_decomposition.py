import torch
import torch.nn as nn


class SignalDecompositionModel(nn.Module):
    """
    Signal Decomposition Model for ICASSP 2026

    This model performs signal decomposition using deep learning.
    Replace this template with your actual model architecture.
    """

    def __init__(self,
                 input_dim=1,
                 hidden_dim=256,
                 num_layers=4,
                 num_components=3,
                 **kwargs):
        """
        Initialize the Signal Decomposition Model

        Args:
            input_dim (int): Input signal dimension
            hidden_dim (int): Hidden layer dimension
            num_layers (int): Number of layers
            num_components (int): Number of decomposed components
        """
        super(SignalDecompositionModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_components = num_components

        # Example encoder architecture
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Example decomposition layers
        self.decomposition_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers)
        ])

        # Output heads for each component
        self.component_heads = nn.ModuleList([
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
            for _ in range(num_components)
        ])

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input signal [batch_size, input_dim, seq_len]

        Returns:
            list of torch.Tensor: Decomposed signal components
        """
        # Encoder
        features = self.encoder(x)

        # Decomposition layers
        for layer in self.decomposition_layers:
            features = features + layer(features)  # Residual connection

        # Generate components
        components = []
        for head in self.component_heads:
            component = head(features)
            components.append(component)

        return components

    def get_model_params(self):
        """Get total number of model parameters"""
        return sum(p.numel() for p in self.parameters())


class DecompositionLoss(nn.Module):
    """
    Custom loss function for signal decomposition
    """

    def __init__(self, reconstruction_weight=1.0, orthogonality_weight=0.1):
        super(DecompositionLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.orthogonality_weight = orthogonality_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, components, target):
        """
        Calculate decomposition loss

        Args:
            components (list of torch.Tensor): Predicted components
            target (torch.Tensor): Original signal

        Returns:
            torch.Tensor: Total loss
        """
        # Reconstruction loss
        reconstructed = sum(components)
        reconstruction_loss = self.mse_loss(reconstructed, target)

        # Orthogonality loss (encourage components to be different)
        orthogonality_loss = 0
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                correlation = torch.mean(components[i] * components[j])
                orthogonality_loss += torch.abs(correlation)

        if len(components) > 1:
            orthogonality_loss /= (len(components) * (len(components) - 1) / 2)

        # Total loss
        total_loss = (self.reconstruction_weight * reconstruction_loss +
                     self.orthogonality_weight * orthogonality_loss)

        return total_loss, reconstruction_loss, orthogonality_loss


if __name__ == '__main__':
    # Test model
    model = SignalDecompositionModel(
        input_dim=1,
        hidden_dim=128,
        num_layers=4,
        num_components=3
    )

    # Test input
    batch_size = 8
    seq_len = 1000
    x = torch.randn(batch_size, 1, seq_len)

    # Forward pass
    components = model(x)

    print(f"Model Parameters: {model.get_model_params():,}")
    print(f"Input shape: {x.shape}")
    print(f"Number of components: {len(components)}")
    for i, comp in enumerate(components):
        print(f"Component {i+1} shape: {comp.shape}")

    # Test loss
    criterion = DecompositionLoss()
    loss, recon_loss, ortho_loss = criterion(components, x)
    print(f"\nTotal Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {recon_loss.item():.4f}")
    print(f"Orthogonality Loss: {ortho_loss.item():.4f}")
