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

from models.unsupervised_afdl import UnsupervisedAFDL
from data.ptb_ecg_loader import create_dataloaders
from utils.metrics import calculate_snr, calculate_prd


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_sparsity_loss = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (signals, labels) in enumerate(progress_bar):
        signals = signals.to(device)  # [batch, 1, seq_len]

        # Forward pass
        outputs = model(signals)

        # Compute loss (NO segmentation loss)
        losses = model.compute_loss(signals, outputs)

        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        # Accumulate losses
        total_loss += losses['total_loss'].item()
        total_recon_loss += losses['reconstruction_loss'].item()
        total_sparsity_loss += losses['sparsity_loss'].item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'recon': f"{losses['reconstruction_loss'].item():.4f}",
            'sparse': f"{losses['sparsity_loss'].item():.4f}"
        })

    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'sparsity_loss': total_sparsity_loss / n_batches
    }


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
            outputs = model(signals)
            reconstruction = outputs['reconstruction']

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

    # Create model
    print(f"\nCreating Unsupervised AFDL model...")
    model = UnsupervisedAFDL(
        input_dim=1,
        seq_len=args.seq_len,
        dict_size=args.dict_size,
        context_dim=128,
        hidden_dim=64,
        num_heads=args.num_heads,
        sparsity_weight=args.sparsity_weight
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"\n[NOTE] No supervised segmentation - using statistical heuristics")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    best_snr = 0
    best_epoch = 0

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate
        if epoch % args.eval_every == 0:
            val_metrics = evaluate(model, val_loader, device)

            print(f"\n[Epoch {epoch}] Results:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val SNR: {val_metrics['snr']:.2f} dB")
            print(f"  Val PRD: {val_metrics['prd']:.2f}%")

            # Check if this is the best model
            if val_metrics['snr'] > best_snr:
                best_snr = val_metrics['snr']
                best_epoch = epoch

                # Save checkpoint
                checkpoint_path = os.path.join(args.save_dir, 'unsupervised_afdl_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'snr': val_metrics['snr'],
                    'prd': val_metrics['prd'],
                }, checkpoint_path)
                print(f"  [BEST MODEL] Saved to {checkpoint_path}")

            # Update learning rate
            scheduler.step(val_metrics['snr'])
    checkpoint_path = os.path.join(args.save_dir, 'unsupervised_afdl_best.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {best_epoch} (SNR: {best_snr:.2f} dB)")

    test_metrics = evaluate(model, test_loader, device)

    print(f"\n[TEST RESULTS]")
    print(f"  SNR: {test_metrics['snr']:.2f} ± ? dB")
    print(f"  PRD: {test_metrics['prd']:.2f} ± ? %")

    print(f"\n[COMPARISON WITH BASELINES]")
    print(f"  K-SVD (from paper): 15.8 ± 1.2 dB")
    print(f"  Unsupervised AFDL:  {test_metrics['snr']:.2f} dB")

    if test_metrics['snr'] > 15.8:
        improvement = ((test_metrics['snr'] - 15.8) / 15.8) * 100
    else:
        print(f"\n[WARNING] SNR below K-SVD baseline. May need more training or tuning.")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Unsupervised AFDL')
    parser.add_argument('--data_path', type=str, default='./data/ptb_ecg_data',
                       help='Path to PTB-ECG data')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--seq_len', type=int, default=3600,
                       help='Sequence length (10s at 360Hz)')
    parser.add_argument('--dict_size', type=int, default=128,
                       help='Dictionary size')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--sparsity_weight', type=float, default=0.05,
                       help='Sparsity regularization weight')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--eval_every', type=int, default=5,
                       help='Evaluate every N epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/unsupervised_afdl',
                       help='Directory to save checkpoints')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
