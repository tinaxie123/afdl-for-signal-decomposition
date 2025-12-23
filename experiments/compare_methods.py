import torch
import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.afdl import AFDL
from models.baselines import KSVDDecomposer, ICADecomposer, NMFDecomposer, DWTDecomposer
from data.ptb_ecg_loader import create_dataloaders
from utils.metrics import evaluate_reconstruction
import matplotlib.pyplot as plt


def evaluate_baseline(method_name, method, test_signals, verbose=True):
   
    if verbose:
        print(f"\nEvaluating {method_name}...")
    if test_signals.ndim == 3:
        test_signals = test_signals.squeeze(1)
    reconstructed, _ = method.fit_transform(test_signals)
    all_snr = []
    all_prd = []
    all_mse = []

    for i in range(len(test_signals)):
        metrics = evaluate_reconstruction(reconstructed[i], test_signals[i])
        all_snr.append(metrics['snr'])
        all_prd.append(metrics['prd'])
        all_mse.append(metrics['mse'])

    results = {
        'snr_mean': np.mean(all_snr),
        'snr_std': np.std(all_snr),
        'prd_mean': np.mean(all_prd),
        'prd_std': np.std(all_prd),
        'mse_mean': np.mean(all_mse),
        'reconstructed': reconstructed
    }

    if verbose:
        print(f"  SNR: {results['snr_mean']:.2f} ± {results['snr_std']:.2f} dB")
        print(f"  PRD: {results['prd_mean']:.2f} ± {results['prd_std']:.2f} %")
        print(f"  MSE: {results['mse_mean']:.6f}")

    return results


def evaluate_afdl(model, test_loader, device, verbose=True):
    if verbose:
    model.eval()
    all_snr = []
    all_prd = []
    all_mse = []

    with torch.no_grad():
        for signals, _ in tqdm(test_loader, disable=not verbose):
            signals = signals.to(device)

            # Forward pass
            outputs = model(signals)
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

    results = {
        'snr_mean': np.mean(all_snr),
        'snr_std': np.std(all_snr),
        'prd_mean': np.mean(all_prd),
        'prd_std': np.std(all_prd),
        'mse_mean': np.mean(all_mse)
    }

    if verbose:
        print(f"  SNR: {results['snr_mean']:.2f} ± {results['snr_std']:.2f} dB")
        print(f"  PRD: {results['prd_mean']:.2f} ± {results['prd_std']:.2f} %")
        print(f"  MSE: {results['mse_mean']:.6f}")

    return results


def plot_comparison_table(results_dict, save_path):
    methods = list(results_dict.keys())
    snr_values = [results_dict[m]['snr_mean'] for m in methods]
    snr_stds = [results_dict[m]['snr_std'] for m in methods]
    prd_values = [results_dict[m]['prd_mean'] for m in methods]
    prd_stds = [results_dict[m]['prd_std'] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['green' if m == 'AFDL' else 'skyblue' for m in methods]
    bars1 = ax1.bar(range(len(methods)), snr_values, color=colors,
                   edgecolor='black', yerr=snr_stds, capsize=5)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('SNR (dB)', fontsize=12)
    ax1.set_title('Signal-to-Noise Ratio Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (bar, val, std) in enumerate(zip(bars1, snr_values, snr_stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=9)
    colors = ['green' if m == 'AFDL' else 'lightcoral' for m in methods]
    bars2 = ax2.bar(range(len(methods)), prd_values, color=colors,
                   edgecolor='black', yerr=prd_stds, capsize=5)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('PRD (%)', fontsize=12)
    ax2.set_title('Percentage Root-mean-square Difference', fontsize=14,
                 fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (bar, val, std) in enumerate(zip(bars2, prd_values, prd_stds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Unified Training and Evaluation Script')
    parser.add_argument('--data_path', type=str, default='./data/ptb_ecg_data',
                       help='Path to data')
    parser.add_argument('--batch_size', type=str, default=16,
                       help='Batch size')
    parser.add_argument('--method', type=str, default='all',
                       choices=['afdl', 'ksvd', 'ica', 'nmf', 'dwt', 'all'],
                       help='Method to evaluate')
    parser.add_argument('--train_afdl', action='store_true',
                       help='Train AFDL model (if not, load pretrained)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs for AFDL')
    parser.add_argument('--afdl_checkpoint', type=str,
                       default='./checkpoints/checkpoint_best.pth',
                       help='Path to AFDL checkpoint')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save results')

    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    _, _, test_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=0,
        seq_len=3600
    )
    test_signals_list = []
    for signals, _ in test_loader:
        test_signals_list.append(signals.numpy())
    test_signals = np.concatenate(test_signals_list, axis=0)

    print(f"Test set: {test_signals.shape[0]} samples\n")
    results_all = {}
    if args.method in ['afdl', 'all']:
        if args.train_afdl:
            print("\nTraining AFDL not implemented in this script.")
            print("Use train_afdl.py to train, then run this script.")
        else:
            # Load pretrained AFDL
            if os.path.exists(args.afdl_checkpoint):
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

                results_all['AFDL'] = evaluate_afdl(model, test_loader, device)
            else:
                print(f"\nAFDL checkpoint not found at {args.afdl_checkpoint}")
                print("Skipping AFDL evaluation.")
    if args.method in ['ksvd', 'all']:
        ksvd = KSVDDecomposer(n_components=128)
        results_all['K-SVD'] = evaluate_baseline('K-SVD', ksvd,
                                                 test_signals.squeeze(1))
    if args.method in ['ica', 'all']:
        ica = ICADecomposer(n_components=128)
        results_all['ICA'] = evaluate_baseline('ICA', ica,
                                               test_signals.squeeze(1))
    if args.method in ['nmf', 'all']:
        nmf = NMFDecomposer(n_components=128)
        results_all['NMF'] = evaluate_baseline('NMF', nmf,
                                               test_signals.squeeze(1))

    if args.method in ['dwt', 'all']:
        dwt = DWTDecomposer(wavelet='db4', level=5)
        results_all['DWT'] = evaluate_baseline('DWT', dwt,
                                               test_signals.squeeze(1))

    for method, results in results_all.items():
        snr_str = f"{results['snr_mean']:.2f} ± {results['snr_std']:.2f}"
        prd_str = f"{results['prd_mean']:.2f} ± {results['prd_std']:.2f}"
        mse_str = f"{results['mse_mean']:.6f}"
        print(f"{method:<15} {snr_str:<20} {prd_str:<20} {mse_str:<15}")
    results_json = {
        method: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in res.items() if k != 'reconstructed'}
        for method, res in results_all.items()
    }

    json_path = os.path.join(args.results_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=4)
    print(f"\nResults saved to {json_path}")
    if len(results_all) > 1:
        plot_path = os.path.join(args.results_dir, 'comparison.png')
        plot_comparison_table(results_all, plot_path)


if __name__ == '__main__':
    main()
