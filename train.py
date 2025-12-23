import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models import SignalDecompositionModel
from models.signal_decomposition import DecompositionLoss
from utils import compute_snr, compute_mse, plot_training_curves


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloader(data_path, batch_size, shuffle=True):
    """
    Create DataLoader from data file

    Args:
        data_path (str): Path to data file (.npy or .pt)
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data

    Returns:
        DataLoader: PyTorch DataLoader
    """
    if data_path.endswith('.npy'):
        data = np.load(data_path)
        data = torch.from_numpy(data).float()
    elif data_path.endswith('.pt'):
        data = torch.load(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    # Assuming data shape is [num_samples, channels, seq_len]
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                           num_workers=4, pin_memory=True)

    return dataloader


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_ortho_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (data,) in enumerate(pbar):
        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        components = model(data)

        # Compute loss
        loss, recon_loss, ortho_loss = criterion(components, data)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_ortho_loss += ortho_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'ortho': f'{ortho_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_ortho_loss = total_ortho_loss / len(dataloader)

    return avg_loss, avg_recon_loss, avg_ortho_loss


def validate(model, dataloader, criterion, device):
    """Validation step"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_snr = 0

    with torch.no_grad():
        for data, in dataloader:
            data = data.to(device)

            # Forward pass
            components = model(data)

            # Compute loss
            loss, recon_loss, ortho_loss = criterion(components, data)

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()

            # Compute SNR
            reconstructed = sum(components)
            for i in range(data.size(0)):
                snr = compute_snr(data[i].cpu().numpy(),
                                (reconstructed[i] - data[i]).cpu().numpy())
                total_snr += snr

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_snr = total_snr / len(dataloader.dataset)

    return avg_loss, avg_recon_loss, avg_snr


def main(args):
    # Load configuration
    if args.config:
        config = load_config(args.config)
        # Override config with command line arguments
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config[key] = value
    else:
        config = vars(args)

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Set device
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create output directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=config['log_dir'])

    # Create model
    model = SignalDecompositionModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_components=config['num_components']
    ).to(device)

    print(f"\nModel parameters: {model.get_model_params():,}")

    # Loss and optimizer
    criterion = DecompositionLoss(
        reconstruction_weight=config['reconstruction_weight'],
        orthogonality_weight=config['orthogonality_weight']
    )

    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                          weight_decay=config['weight_decay'])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Create dataloaders
    train_loader = create_dataloader(
        config['train_data_path'],
        config['batch_size'],
        shuffle=True
    )

    val_loader = create_dataloader(
        config['val_data_path'],
        config['batch_size'],
        shuffle=False
    )

    print(f"\nTraining samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("\nStarting training...")

    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss, train_recon_loss, train_ortho_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_recon_loss, val_snr = validate(
            model, val_loader, criterion, device
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Logging
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print(f"  Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, Ortho: {train_ortho_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, SNR: {val_snr:.2f} dB)")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('SNR/val', val_snr, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['checkpoint_dir'], 'model_best.pth'))
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % config['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['checkpoint_dir'], f'model_epoch_{epoch}.pth'))

    # Plot training curves
    plot_training_curves(
        train_losses,
        val_losses,
        save_path=os.path.join(config['log_dir'], 'training_curves.png')
    )

    writer.close()
    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Signal Decomposition Model')

    # Configuration
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')

    # Data
    parser.add_argument('--train_data_path', type=str, default='data/processed/train.npy',
                       help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default='data/processed/val.npy',
                       help='Path to validation data')

    # Model
    parser.add_argument('--input_dim', type=int, default=1,
                       help='Input signal dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of layers')
    parser.add_argument('--num_components', type=int, default=3,
                       help='Number of decomposed components')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')

    # Loss
    parser.add_argument('--reconstruction_weight', type=float, default=1.0,
                       help='Reconstruction loss weight')
    parser.add_argument('--orthogonality_weight', type=float, default=0.1,
                       help='Orthogonality loss weight')

    # System
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()
    main(args)
