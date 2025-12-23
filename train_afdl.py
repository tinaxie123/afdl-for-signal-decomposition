
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.afdl import AFDL
from data.ptb_ecg_loader import create_dataloaders
from utils.metrics import evaluate_reconstruction


class AFDLTrainer:
    """
    Trainer class for AFDL model
    """
    def __init__(
        self,
        model: AFDL,
        train_loader,
        val_loader,
        test_loader,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['scheduler_patience'],
            verbose=True
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_snr = -float('inf')
        self.train_losses = []
        self.val_losses = []

        # Logging
        log_dir = os.path.join(config['save_dir'], 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.writer = SummaryWriter(log_dir)

        # Save directory
        os.makedirs(config['save_dir'], exist_ok=True)

        print(f"Trainer initialized")
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total': [],
            'recon': [],
            'sparsity': [],
            'seg': []
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')

        for batch_idx, (signals, labels) in enumerate(pbar):
            signals = signals.to(self.device)

            # Forward pass
            losses = self.model.compute_loss(signals)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # Log losses
            epoch_losses['total'].append(losses['total_loss'].item())
            epoch_losses['recon'].append(losses['recon_loss'].item())
            epoch_losses['sparsity'].append(losses['sparsity_loss'].item())
            epoch_losses['seg'].append(losses['seg_loss'].item())

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'recon': losses['recon_loss'].item()
            })

        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

        return avg_losses

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model"""
        self.model.eval()

        epoch_losses = {
            'total': [],
            'recon': [],
            'sparsity': [],
            'seg': []
        }

        all_snr = []
        all_prd = []

        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')

        for signals, labels in pbar:
            signals = signals.to(self.device)

            # Forward pass
            losses = self.model.compute_loss(signals)
            outputs = self.model(signals)

            # Log losses
            epoch_losses['total'].append(losses['total_loss'].item())
            epoch_losses['recon'].append(losses['recon_loss'].item())
            epoch_losses['sparsity'].append(losses['sparsity_loss'].item())
            epoch_losses['seg'].append(losses['seg_loss'].item())

            # Evaluate reconstruction quality
            reconstruction = outputs['reconstruction']
            for i in range(signals.size(0)):
                metrics = evaluate_reconstruction(
                    reconstruction[i].cpu(),
                    signals[i].cpu()
                )
                all_snr.append(metrics['snr'])
                all_prd.append(metrics['prd'])

            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'snr': np.mean(all_snr) if all_snr else 0
            })

        # Average metrics
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        avg_losses['snr'] = np.mean(all_snr)
        avg_losses['prd'] = np.mean(all_prd)

        return avg_losses

    @torch.no_grad()
    def test(self) -> dict:

        self.model.eval()

        all_snr = []
        all_prd = []
        all_mse = []
        all_mae = []

        for signals, labels in tqdm(self.test_loader, desc='Testing'):
            signals = signals.to(self.device)

            # Forward pass
            outputs = self.model(signals)
            reconstruction = outputs['reconstruction']

            # Evaluate
            for i in range(signals.size(0)):
                metrics = evaluate_reconstruction(
                    reconstruction[i].cpu(),
                    signals[i].cpu()
                )
                all_snr.append(metrics['snr'])
                all_prd.append(metrics['prd'])
                all_mse.append(metrics['mse'])
                all_mae.append(metrics['mae'])

        # Compute statistics
        results = {
            'snr_mean': np.mean(all_snr),
            'snr_std': np.std(all_snr),
            'prd_mean': np.mean(all_prd),
            'prd_std': np.std(all_prd),
            'mse_mean': np.mean(all_mse),
            'mae_mean': np.mean(all_mae),
        }

        print("\nTest Results:")
    
        return results

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_snr': self.best_val_snr,
            'config': self.config
        }

        # Save latest
        latest_path = os.path.join(self.config['save_dir'], 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with SNR: {self.best_val_snr:.2f} dB")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_snr = checkpoint['best_val_snr']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best validation SNR: {self.best_val_snr:.2f} dB")

    def train(self, num_epochs: int):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting training...")
        print("="*80)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step(val_metrics['total'])

            # Log metrics
            self.writer.add_scalar('Loss/train', train_metrics['total'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['total'], epoch)
            self.writer.add_scalar('Recon Loss/train', train_metrics['recon'], epoch)
            self.writer.add_scalar('Recon Loss/val', val_metrics['recon'], epoch)
            self.writer.add_scalar('SNR/val', val_metrics['snr'], epoch)
            self.writer.add_scalar('PRD/val', val_metrics['prd'], epoch)
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['total']:.4f} | Val Loss: {val_metrics['total']:.4f}")
            print(f"Val SNR: {val_metrics['snr']:.2f} dB | Val PRD: {val_metrics['prd']:.2f}%")

            # Save checkpoint
            is_best = val_metrics['snr'] > self.best_val_snr
            if is_best:
                self.best_val_snr = val_metrics['snr']
                self.best_val_loss = val_metrics['total']

            if (epoch + 1) % self.config['save_interval'] == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)

        # Test with best model
        best_checkpoint = os.path.join(self.config['save_dir'], 'checkpoint_best.pth')
        if os.path.exists(best_checkpoint):
            self.load_checkpoint(best_checkpoint)
            test_results = self.test()

            # Save test results
            results_path = os.path.join(self.config['save_dir'], 'test_results.json')
            with open(results_path, 'w') as f:
                json.dump(test_results, f, indent=4)

        self.writer.close()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train AFDL model')
    parser.add_argument('--data_path', type=str, default='./data/ptb_ecg_data',
                        help='Path to PTB-ECG data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--dict_size', type=int, default=128,
                        help='Dictionary size')
    parser.add_argument('--seq_len', type=int, default=3600,
                        help='Sequence length')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--sparsity_weight', type=float, default=0.05,
                        help='Sparsity regularization weight')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run testing')

    args = parser.parse_args()

    # Configuration
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'scheduler_patience': 10,
        'grad_clip': 1.0,
        'save_interval': 5,
        'save_dir': args.save_dir,
        'dict_size': args.dict_size,
        'seq_len': args.seq_len,
        'num_heads': args.num_heads,
        'sparsity_weight': args.sparsity_weight,
    }

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=4 if device == 'cuda' else 0,
        seq_len=args.seq_len,
        sampling_rate=360
    )

    # Create model
    print("Creating AFDL model...")
    model = AFDL(
        input_dim=1,
        seq_len=args.seq_len,
        dict_size=args.dict_size,
        context_dim=128,
        hidden_dim=64,
        num_heads=args.num_heads,
        sparsity_weight=args.sparsity_weight
    )

    # Create trainer
    trainer = AFDLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train or test
    if args.test_only:
        trainer.test()
    else:
        trainer.train(num_epochs=args.num_epochs)


if __name__ == '__main__':
    main()
