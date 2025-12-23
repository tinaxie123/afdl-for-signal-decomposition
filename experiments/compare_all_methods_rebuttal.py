"""
Updated Method Comparison Script - Including New Baselines for Rebuttal

This script compares:
1. AFDL (supervised)
2. **NEW** Unsupervised AFDL (no supervised segmentation)
3. K-SVD, ICA, NMF, DWT (traditional baselines)
4. **NEW** 1D CNN Autoencoder (modern deep learning baseline)

Author: Haotong Xie
Institution: Shanghai University of Finance and Economics
"""

import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.afdl import AFDL
from models.unsupervised_afdl import UnsupervisedAFDL
from models.autoencoder_baseline import CNN1DAutoencoder, DeepCNN1DAutoencoder
from models.baselines import KSVDDecomposer, ICADecomposer, NMFDecomposer, DWTDecomposer
from data.ptb_ecg_loader import create_dataloaders
from utils.metrics import calculate_snr, calculate_prd


def evaluate_deep_learning_model(model, test_loader, device, model_name="Model"):
    """Evaluate AFDL, Unsupervised AFDL, or Autoencoder"""
    model.eval()
    all_snr = []
    all_prd = []

    print(f"\nEvaluating {model_name}...")
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            signals = signals.to(device)  # [batch, 1, seq_len]

            # Forward pass
            if isinstance(model, (AFDL, UnsupervisedAFDL)):
                outputs = model(signals)
                reconstruction = outputs['reconstruction']
            else:  # Autoencoder
                reconstruction = model(signals)

            # Move to CPU for metrics
            signals_np = signals.cpu().numpy()
            recon_np = reconstruction.cpu().numpy()

            # Compute metrics
            batch_size = signals.shape[0]
            for i in range(batch_size):
                original = signals_np[i, 0, :]
                recon = recon_np[i, 0, :]

                snr = calculate_snr(original, recon)
                prd = calculate_prd(original, recon)

                all_snr.append(snr)
                all_prd.append(prd)

    mean_snr = np.mean(all_snr)
    std_snr = np.std(all_snr)
    mean_prd = np.mean(all_prd)
    std_prd = np.std(all_prd)

    return {
        'snr_mean': mean_snr,
        'snr_std': std_snr,
        'prd_mean': mean_prd,
        'prd_std': std_prd
    }


def evaluate_traditional_baseline(method_class, test_signals, method_name="Method"):
    """Evaluate K-SVD, ICA, NMF, or DWT"""
    print(f"\nEvaluating {method_name}...")

    # Create method instance
    if method_name == "DWT":
        method = method_class(wavelet='db4', level=5)
    else:
        method = method_class(n_components=128)

    # Fit on test data (for K-SVD, ICA, NMF)
    if method_name != "DWT":
        print(f"  Fitting {method_name} on test data...")
        method.fit(test_signals)

    # Transform (reconstruct)
    print(f"  Reconstructing signals...")
    reconstructed, _ = method.transform(test_signals)

    # Compute metrics
    all_snr = []
    all_prd = []

    for i in tqdm(range(len(test_signals)), desc=f"Computing metrics"):
        original = test_signals[i]
        recon = reconstructed[i]

        snr = calculate_snr(original, recon)
        prd = calculate_prd(original, recon)

        all_snr.append(snr)
        all_prd.append(prd)

    mean_snr = np.mean(all_snr)
    std_snr = np.std(all_snr)
    mean_prd = np.mean(all_prd)
    std_prd = np.std(all_prd)

    return {
        'snr_mean': mean_snr,
        'snr_std': std_snr,
        'prd_mean': mean_prd,
        'prd_std': std_prd
    }


def main(args):
    print("="*80)
    print("Complete Method Comparison (Including New Rebuttal Experiments)")
    print("="*80)
    print("\nMethods to evaluate:")
    print("  [1] AFDL (Supervised) - Original method")
    print("  [2] **NEW** Unsupervised AFDL - No supervised segmentation")
    print("  [3] K-SVD, ICA, NMF, DWT - Traditional baselines")
    print("  [4] **NEW** 1D CNN Autoencoder - Modern deep learning baseline")
    print("")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print(f"Loading PTB-ECG dataset from: {args.data_path}")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=4,
        seq_len=3600,
        sampling_rate=360
    )

    # Load test signals for traditional methods
    test_signals_list = []
    for signals, _ in test_loader:
        test_signals_list.append(signals.numpy()[:, 0, :])  # Remove channel dim
    test_signals = np.concatenate(test_signals_list, axis=0)

    print(f"Test set: {test_signals.shape[0]} samples\n")

    # Results dictionary
    results = {}

    # ========================================================================
    # 1. Evaluate AFDL (Supervised)
    # ========================================================================
    if args.eval_afdl and os.path.exists(args.afdl_checkpoint):
        print("\n" + "="*80)
        print("[1] Evaluating AFDL (Supervised)")
        print("="*80)

        model = AFDL(
            input_dim=1,
            seq_len=3600,
            dict_size=128,
            context_dim=128,
            hidden_dim=64,
            num_heads=4,
            sparsity_weight=0.05
        ).to(device)

        checkpoint = torch.load(args.afdl_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        results['AFDL'] = evaluate_deep_learning_model(model, test_loader, device, "AFDL")

    # ========================================================================
    # 2. **NEW** Evaluate Unsupervised AFDL
    # ========================================================================
    if args.eval_unsupervised and os.path.exists(args.unsupervised_checkpoint):
        print("\n" + "="*80)
        print("[2] **NEW** Evaluating Unsupervised AFDL")
        print("="*80)

        model = UnsupervisedAFDL(
            input_dim=1,
            seq_len=3600,
            dict_size=128,
            context_dim=128,
            hidden_dim=64,
            num_heads=4,
            sparsity_weight=0.05
        ).to(device)

        checkpoint = torch.load(args.unsupervised_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        results['Unsupervised AFDL'] = evaluate_deep_learning_model(
            model, test_loader, device, "Unsupervised AFDL"
        )

    # ========================================================================
    # 3. Evaluate Traditional Baselines
    # ========================================================================
    if args.eval_traditional:
        print("\n" + "="*80)
        print("[3] Evaluating Traditional Baselines")
        print("="*80)

        methods = {
            'K-SVD': KSVDDecomposer,
            'ICA': ICADecomposer,
            'NMF': NMFDecomposer,
            'DWT': DWTDecomposer
        }

        for name, method_class in methods.items():
            results[name] = evaluate_traditional_baseline(
                method_class, test_signals, name
            )

    # ========================================================================
    # 4. **NEW** Evaluate 1D CNN Autoencoder
    # ========================================================================
    if args.eval_autoencoder and os.path.exists(args.autoencoder_checkpoint):
        print("\n" + "="*80)
        print("[4] **NEW** Evaluating 1D CNN Autoencoder")
        print("="*80)

        if args.use_deep_autoencoder:
            model = DeepCNN1DAutoencoder(input_dim=1, seq_len=3600, latent_dim=128).to(device)
        else:
            model = CNN1DAutoencoder(input_dim=1, seq_len=3600, latent_dim=128).to(device)

        checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        results['1D CNN Autoencoder'] = evaluate_deep_learning_model(
            model, test_loader, device, "1D CNN Autoencoder"
        )

    # ========================================================================
    # Print Results
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS - COMPLETE COMPARISON")
    print("="*80)

    print("\n{:<25} {:<20} {:<20}".format("Method", "SNR (dB)", "PRD (%)"))
    print("-"*80)

    # Sort by SNR (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['snr_mean'], reverse=True)

    for method_name, metrics in sorted_results:
        snr_str = f"{metrics['snr_mean']:.1f}±{metrics['snr_std']:.1f}"
        prd_str = f"{metrics['prd_mean']:.1f}±{metrics['prd_std']:.1f}"
        print(f"{method_name:<25} {snr_str:<20} {prd_str:<20}")

    # ========================================================================
    # Key Comparisons for Rebuttal
    # ========================================================================
    print("\n" + "="*80)
    print("KEY COMPARISONS FOR REBUTTAL")
    print("="*80)

    if 'Unsupervised AFDL' in results and 'K-SVD' in results:
        unsup_snr = results['Unsupervised AFDL']['snr_mean']
        ksvd_snr = results['K-SVD']['snr_mean']
        improvement = ((unsup_snr - ksvd_snr) / ksvd_snr) * 100

        print(f"\n[Reviewer 3F43] Unsupervised AFDL vs K-SVD:")
        print(f"  Unsupervised AFDL: {unsup_snr:.2f} dB")
        print(f"  K-SVD: {ksvd_snr:.2f} dB")
        print(f"  Improvement: {improvement:+.1f}%")

        if unsup_snr > ksvd_snr:
            print(f"  [SUCCESS] ✓ Micro-NN Dictionary outperforms K-SVD WITHOUT supervision!")
        else:
            print(f"  [WARNING] Performance below K-SVD")

    if 'AFDL' in results and 'Unsupervised AFDL' in results:
        sup_snr = results['AFDL']['snr_mean']
        unsup_snr = results['Unsupervised AFDL']['snr_mean']
        gap = sup_snr - unsup_snr

        print(f"\n[Supervision Effect] Supervised vs Unsupervised AFDL:")
        print(f"  Supervised AFDL: {sup_snr:.2f} dB")
        print(f"  Unsupervised AFDL: {unsup_snr:.2f} dB")
        print(f"  Supervision benefit: {gap:.2f} dB ({gap/sup_snr*100:.1f}%)")

    if '1D CNN Autoencoder' in results and 'AFDL' in results:
        ae_snr = results['1D CNN Autoencoder']['snr_mean']
        afdl_snr = results['AFDL']['snr_mean']

        print(f"\n[Reviewer 190C] AFDL vs Modern Deep Learning:")
        print(f"  AFDL: {afdl_snr:.2f} dB")
        print(f"  1D CNN Autoencoder: {ae_snr:.2f} dB")

        if afdl_snr > ae_snr:
            print(f"  [SUCCESS] ✓ AFDL outperforms deep learning baseline!")
        else:
            print(f"  [NOTE] Autoencoder has slightly higher SNR, but AFDL offers interpretability")

    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)

    # Save results
    results_path = os.path.join(args.save_dir, 'complete_comparison_results.txt')
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPLETE METHOD COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write("{:<25} {:<20} {:<20}\n".format("Method", "SNR (dB)", "PRD (%)"))
        f.write("-"*80 + "\n")

        for method_name, metrics in sorted_results:
            snr_str = f"{metrics['snr_mean']:.1f}±{metrics['snr_std']:.1f}"
            prd_str = f"{metrics['prd_mean']:.1f}±{metrics['prd_std']:.1f}"
            f.write(f"{method_name:<25} {snr_str:<20} {prd_str:<20}\n")

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete Method Comparison')

    # Data
    parser.add_argument('--data_path', type=str, default='./data/ptb_ecg_data',
                       help='Path to PTB-ECG data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')

    # Which methods to evaluate
    parser.add_argument('--eval_afdl', action='store_true', default=True,
                       help='Evaluate AFDL (supervised)')
    parser.add_argument('--eval_unsupervised', action='store_true', default=True,
                       help='Evaluate Unsupervised AFDL')
    parser.add_argument('--eval_traditional', action='store_true', default=True,
                       help='Evaluate traditional baselines')
    parser.add_argument('--eval_autoencoder', action='store_true', default=True,
                       help='Evaluate 1D CNN Autoencoder')

    # Checkpoints
    parser.add_argument('--afdl_checkpoint', type=str,
                       default='./checkpoints/checkpoint_best.pth',
                       help='Path to AFDL checkpoint')
    parser.add_argument('--unsupervised_checkpoint', type=str,
                       default='./checkpoints/unsupervised_afdl/unsupervised_afdl_best.pth',
                       help='Path to Unsupervised AFDL checkpoint')
    parser.add_argument('--autoencoder_checkpoint', type=str,
                       default='./checkpoints/autoencoder/autoencoder_best.pth',
                       help='Path to Autoencoder checkpoint')

    # Autoencoder options
    parser.add_argument('--use_deep_autoencoder', action='store_true',
                       help='Use deep autoencoder instead of standard')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
