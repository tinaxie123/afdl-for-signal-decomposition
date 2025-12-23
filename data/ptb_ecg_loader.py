"""
PTB-ECG Dataset Loader

Preprocessed PTB Diagnostic ECG Database from PhysioNet
Dataset: 2400 10-second segments (3600 samples at 360 Hz)
Classes: Normal Sinus Rhythm (NSR), Premature Ventricular Contractions (PVC),
         Atrial Fibrillation (AF)

Reference: Bousseljot R, Kreiseler D, Schnabel A.
"Nutzung der EKG-Signaldatenbank CARDIODAT der PTB über das Internet."
Biomedizinische Technik, Band 40, Ergänzungsband 1 (1995) S 317
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, List, Optional
from scipy import signal


# ============================================================================
# Signal Processing Utilities
# ============================================================================

def normalize_signal(signal: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize signal using various methods

    Args:
        signal: Input signal
        method: 'standard' (zero mean, unit variance),
                'minmax' (scale to [0,1]),
                'maxabs' (scale to [-1,1])
    """
    if method == 'standard':
        mean = np.mean(signal)
        std = np.std(signal)
        normalized = (signal - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(signal)
        max_val = np.max(signal)
        normalized = (signal - min_val) / (max_val - min_val + 1e-8)
    elif method == 'maxabs':
        max_abs = np.max(np.abs(signal))
        normalized = signal / (max_abs + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def add_noise(signal: np.ndarray, snr_db: float = 20) -> np.ndarray:
    """
    Add Gaussian noise to signal with specified SNR

    Args:
        signal: Clean signal
        snr_db: Signal-to-Noise Ratio in dB
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    noisy_signal = signal + noise

    return noisy_signal


def add_baseline_wander(signal: np.ndarray, amplitude: float = 0.1,
                        frequency: float = 0.3) -> np.ndarray:
    """
    Add baseline wander (common in ECG)

    Args:
        signal: Input signal
        amplitude: Amplitude of baseline wander
        frequency: Frequency of baseline wander in Hz
    """
    t = np.linspace(0, len(signal) / 360, len(signal))  # Assuming 360 Hz
    baseline = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal + baseline


class PTBECGDataset(Dataset):
    """
    PTB-ECG Dataset for signal decomposition

    Loads preprocessed PTB Diagnostic ECG Database
    """
    def __init__(
        self,
        data_path: str,
        mode: str = 'train',
        seq_len: int = 3600,
        sampling_rate: int = 360,
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Args:
            data_path: Path to preprocessed PTB-ECG data
            mode: 'train', 'val', or 'test'
            seq_len: Sequence length (default 3600 for 10s at 360Hz)
            sampling_rate: Sampling rate in Hz
            normalize: Whether to normalize signals
            augment: Whether to apply data augmentation
        """
        self.data_path = data_path
        self.mode = mode
        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.augment = augment and (mode == 'train')

        # Load preprocessed data
        self.signals, self.labels = self._load_data()

        print(f"Loaded {len(self.signals)} {mode} samples")

    def _load_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load preprocessed ECG signals and labels from PTB database

        Returns:
            signals: List of ECG signal arrays
            labels: List of class labels (0: NSR, 1: PVC, 2: AF)
        """
        # Load from preprocessed numpy files
        processed_path = os.path.join(self.data_path, f'{self.mode}_data.npz')

        if not os.path.exists(processed_path):
            raise FileNotFoundError(
                f"Preprocessed data not found at {processed_path}\n"
                f"Please run the preprocessing script to prepare PTB-ECG data:\n"
                f"  python scripts/preprocess_ptb_ecg.py --raw_dir <PTB_RAW_DIR> --output_dir {self.data_path}"
            )

        print(f"Loading preprocessed data from {processed_path}")
        data = np.load(processed_path, allow_pickle=True)
        signals = data['signals'].tolist()
        labels = data['labels'].tolist()

        return signals, labels

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance"""
        return normalize_signal(signal, method='standard')

    def _augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply data augmentation for training"""
        # Random amplitude scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            signal = signal * scale

        # Random noise addition
        if np.random.rand() < 0.5:
            snr_db = np.random.uniform(15, 30)
            signal = add_noise(signal, snr_db=snr_db)

        # Random baseline wander (ECG-specific)
        if np.random.rand() < 0.3:
            amplitude = np.random.uniform(0.05, 0.15)
            frequency = np.random.uniform(0.2, 0.5)
            signal = add_baseline_wander(signal, amplitude, frequency)

        # Random baseline shift
        if np.random.rand() < 0.3:
            shift = np.random.uniform(-0.1, 0.1)
            signal = signal + shift

        return signal

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample

        Returns:
            signal: ECG signal tensor [1, seq_len]
            label: Class label (0, 1, or 2)
        """
        signal = self.signals[idx].copy()
        label = self.labels[idx]

        # Normalize
        if self.normalize:
            signal = self._normalize_signal(signal)

        # Augment
        if self.augment:
            signal = self._augment_signal(signal)

        # Convert to tensor
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0)  # [1, seq_len]

        return signal_tensor, label


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seq_len: int = 3600,
    sampling_rate: int = 360
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders

    Args:
        data_path: Path to preprocessed PTB-ECG data
        batch_size: Batch size
        num_workers: Number of worker processes
        seq_len: Sequence length
        sampling_rate: Sampling rate

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = PTBECGDataset(
        data_path=data_path,
        mode='train',
        seq_len=seq_len,
        sampling_rate=sampling_rate,
        normalize=True,
        augment=True
    )

    val_dataset = PTBECGDataset(
        data_path=data_path,
        mode='val',
        seq_len=seq_len,
        sampling_rate=sampling_rate,
        normalize=True,
        augment=False
    )

    test_dataset = PTBECGDataset(
        data_path=data_path,
        mode='test',
        seq_len=seq_len,
        sampling_rate=sampling_rate,
        normalize=True,
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
