"""
Train 1D CNN Autoencoder Baseline

This script trains a 1D CNN Autoencoder as a modern deep learning baseline
to address Reviewer 190C's concern about broader comparisons.

Author: Haotong Xie
Institution: Shanghai University of Finance and Economics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder_baseline import CNN1DAutoencoder, DeepCNN1DAutoencoder
from data.ptb_ecg_loader import create_dataloaders
from utils.metrics import calculate_snr, calculate_prd


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (signals, labels) in enumerate(progress_bar):
        signals = signals.to(device)  # [batch, 1, seq_len]

        # Forward pass
        losses = model.compute_loss(signals)

        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        # Accumulate losses
        total_loss += losses['total_loss'].item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}"
        })

    n_batches = len(train_loader)
    return {'loss': total_loss / n_batches}


def evaluate(model, val_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_snr = 0
    total_prd = 0
    n_samples = 0

    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)  # [batch, 1, seq_len]

            # Forward pass
            reconstruction = model(signals)

            # Move to CPU for metric computation
            signals_np = signals.cpu().numpy()
            recon_np = reconstruction.cpu().numpy()

            # Compute metrics for each sample
            batch_size = signals.shape[0]
            for i in range(batch_size):
                original = signals_np[i, 0, :]
                recon = recon_np[i, 0, :]

                snr = calculate_snr(original, recon)
                prd = calculate_prd(original, recon)

                total_snr += snr
                total_prd += prd
                n_samples += 1

    avg_snr = total_snr / n_samples
    avg_prd = total_prd / n_samples

    return {'snr': avg_snr, 'prd': avg_prd}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nLoading PTB-ECG dataset from: {args.data_path}")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=4,
        seq_len=args.seq_len,
        sampling_rate=360
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"\nCreating {'Deep ' if args.use_deep else ''}1D CNN Autoencoder...")
    if args.use_deep:
        model = DeepCNN1DAutoencoder(
            input_dim=1,
            seq_len=args.seq_len,
            latent_dim=args.latent_dim
        ).to(device)
    else:
        model = CNN1DAutoencoder(
            input_dim=1,
            seq_len=args.seq_len,
            latent_dim=args.latent_dim
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )


    best_snr = 0
    best_epoch = 0

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        if epoch % args.eval_every == 0:
            val_metrics = evaluate(model, val_loader, device)

            print(f"\n[Epoch {epoch}] Results:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val SNR: {val_metrics['snr']:.2f} dB")
            print(f"  Val PRD: {val_metrics['prd']:.2f}%")
            if val_metrics['snr'] > best_snr:
                best_snr = val_metrics['snr']
                best_epoch = epoch
                checkpoint_path = os.path.join(args.save_dir, 'autoencoder_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'snr': val_metrics['snr'],
                    'prd': val_metrics['prd'],
                }, checkpoint_path)
                print(f"  [BEST MODEL] Saved to {checkpoint_path}")
            scheduler.step(val_metrics['snr'])
    checkpoint_path = os.path.join(args.save_dir, 'autoencoder_best.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {best_epoch} (SNR: {best_snr:.2f} dB)")

    test_metrics = evaluate(model, test_loader, device)

    print(f"\n[TEST RESULTS]")
    print(f"  SNR: {test_metrics['snr']:.2f} dB")
    print(f"  PRD: {test_metrics['prd']:.2f} %")

    print(f"\n[COMPARISON WITH PAPER RESULTS]")
    print(f"  AFDL (paper): 17.5 ± 1.1 dB")
    print(f"  1D CNN Autoencoder: {test_metrics['snr']:.2f} dB")

    if test_metrics['snr'] < 17.5:
        print(f"\n[NOTE] Autoencoder SNR is lower than AFDL")
        print(f"       This supports AFDL's superiority even against modern deep learning")
    else:
        print(f"\n[NOTE] Autoencoder has comparable or higher SNR")
        print(f"       However, AFDL offers superior interpretability through")
        print(f"       functional basis functions, which autoencoders lack")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 1D CNN Autoencoder Baseline')

    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/ptb_ecg_data',
                       help='Path to PTB-ECG data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--seq_len', type=int, default=3600,
                       help='Sequence length (10s at 360Hz)')

    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension size')
    parser.add_argument('--use_deep', action='store_true',
                       help='Use deep autoencoder with residual blocks')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--eval_every', type=int, default=5,
                       help='Evaluate every N epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/autoencoder',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
