"""
Traditional Baseline Methods for Signal Decomposition

Includes:
- K-SVD Dictionary Learning
- Independent Component Analysis (ICA)
- Non-negative Matrix Factorization (NMF)
- Discrete Wavelet Transform (DWT)

Author: Haotong Xie
Institution: Shanghai University of Finance and Economics
"""

import numpy as np
from sklearn.decomposition import FastICA, NMF as SklearnNMF, DictionaryLearning
import pywt
from typing import Tuple, Optional


class KSVDDecomposer:
    """
    K-SVD Dictionary Learning for signal decomposition
    Reference: Aharon et al., "K-SVD: An Algorithm for Designing Overcomplete
    Dictionaries for Sparse Representation" (IEEE TSP, 2006)
    """

    def __init__(self,
                 n_components: int = 128,
                 max_iter: int = 30,
                 transform_n_nonzero_coefs: int = 10,
                 random_state: int = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

        self.model = DictionaryLearning(
            n_components=n_components,
            max_iter=max_iter,
            transform_n_nonzero_coefs=transform_n_nonzero_coefs,
            random_state=random_state,
            fit_algorithm='lars',
            transform_algorithm='omp'
        )

        self.dictionary = None
        self.is_fitted = False

    def fit(self, signals: np.ndarray) -> 'KSVDDecomposer':
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        self.model.fit(signals)
        self.dictionary = self.model.components_
        self.is_fitted = True

        return self

    def transform(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform.")

        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        codes = self.model.transform(signals)
        reconstructed = codes @ self.dictionary

        return reconstructed, codes

    def fit_transform(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(signals)
        return self.transform(signals)


class ICADecomposer:
    """
    Independent Component Analysis (ICA) for signal decomposition
    Reference: Hyvärinen & Oja, "Independent component analysis:
    algorithms and applications" (Neural Networks, 2000)
    """

    def __init__(self, n_components: int = 128, max_iter: int = 200,
                 random_state: int = 42):
        self.n_components = n_components
        self.model = FastICA(
            n_components=n_components,
            max_iter=max_iter,
            random_state=random_state,
            whiten='unit-variance'
        )
        self.is_fitted = False

    def fit(self, signals: np.ndarray) -> 'ICADecomposer':
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        self.model.fit(signals.T)
        self.is_fitted = True

        return self

    def transform(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform.")

        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        sources = self.model.transform(signals.T).T
        reconstructed = self.model.inverse_transform(sources.T).T

        return reconstructed, sources

    def fit_transform(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(signals)
        return self.transform(signals)


class NMFDecomposer:
    """
    Non-negative Matrix Factorization (NMF) for signal decomposition
    Reference: Lee & Seung, "Algorithms for Non-negative Matrix Factorization"
    (NIPS, 2000)
    """

    def __init__(self, n_components: int = 128, max_iter: int = 200,
                 random_state: int = 42):
        self.n_components = n_components
        self.model = SklearnNMF(
            n_components=n_components,
            init='nndsvda',
            max_iter=max_iter,
            random_state=random_state,
            solver='mu',
            beta_loss='frobenius'
        )
        self.is_fitted = False

    def fit(self, signals: np.ndarray) -> 'NMFDecomposer':
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        # NMF requires non-negative data
        self.min_val = signals.min()
        signals_shifted = signals - self.min_val + 1e-10

        self.model.fit(signals_shifted)
        self.is_fitted = True

        return self

    def transform(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform.")

        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        signals_shifted = signals - self.min_val + 1e-10
        coefficients = self.model.transform(signals_shifted)
        reconstructed = coefficients @ self.model.components_
        reconstructed = reconstructed + self.min_val - 1e-10

        return reconstructed, coefficients

    def fit_transform(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(signals)
        return self.transform(signals)


class DWTDecomposer:
    """
    Discrete Wavelet Transform (DWT) for signal decomposition
    Reference: Mallat, "A Wavelet Tour of Signal Processing" (1999)
    """

    def __init__(self, wavelet: str = 'db4', level: int = 5,
                 mode: str = 'symmetric'):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def fit(self, signals: np.ndarray) -> 'DWTDecomposer':
        # DWT doesn't require fitting
        return self

    def transform(self, signals: np.ndarray) -> Tuple[np.ndarray, list]:
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_samples, signal_length = signals.shape
        reconstructed = np.zeros_like(signals)
        coeffs_list = []

        for i in range(n_samples):
            signal = signals[i]

            # Decomposition
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level,
                                 mode=self.mode)

            # Threshold small coefficients
            coeffs_thresholded = self._threshold_coefficients(coeffs)

            # Reconstruction
            reconstructed[i] = pywt.waverec(coeffs_thresholded, self.wavelet,
                                           mode=self.mode)[:signal_length]

            coeffs_list.append(coeffs)

        return reconstructed, coeffs_list

    def _threshold_coefficients(self, coeffs: list,
                                threshold_scale: float = 0.1) -> list:
        thresholded = []

        for i, coeff in enumerate(coeffs):
            if i == 0:
                # Keep approximation coefficients
                thresholded.append(coeff)
            else:
                # Threshold detail coefficients
                sigma = np.median(np.abs(coeff)) / 0.6745
                threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(coeff)))
                coeff_thresh = pywt.threshold(coeff, threshold, mode='soft')
                thresholded.append(coeff_thresh)

        return thresholded

    def fit_transform(self, signals: np.ndarray) -> Tuple[np.ndarray, list]:
        return self.transform(signals)


# ============================================================================
# Convenience function for testing all baselines
# ============================================================================

def test_all_baselines(signals: np.ndarray):
    """
    Test all baseline methods on given signals

    Args:
        signals: Input signals [n_samples, signal_length]

    Returns:
        Dictionary with results from all methods
    """
    results = {}

    # K-SVD
    print("Testing K-SVD...")
    ksvd = KSVDDecomposer(n_components=128)
    recon_ksvd, _ = ksvd.fit_transform(signals)
    mse_ksvd = np.mean((signals - recon_ksvd) ** 2)
    results['K-SVD'] = {'reconstruction': recon_ksvd, 'mse': mse_ksvd}
    print(f"  MSE: {mse_ksvd:.6f}")

    # ICA
    print("Testing ICA...")
    ica = ICADecomposer(n_components=128)
    recon_ica, _ = ica.fit_transform(signals)
    mse_ica = np.mean((signals - recon_ica) ** 2)
    results['ICA'] = {'reconstruction': recon_ica, 'mse': mse_ica}
    print(f"  MSE: {mse_ica:.6f}")

    # NMF
    print("Testing NMF...")
    nmf = NMFDecomposer(n_components=128)
    recon_nmf, _ = nmf.fit_transform(signals)
    mse_nmf = np.mean((signals - recon_nmf) ** 2)
    results['NMF'] = {'reconstruction': recon_nmf, 'mse': mse_nmf}
    print(f"  MSE: {mse_nmf:.6f}")

    # DWT
    print("Testing DWT...")
    dwt = DWTDecomposer(wavelet='db4', level=5)
    recon_dwt, _ = dwt.transform(signals)
    mse_dwt = np.mean((signals - recon_dwt) ** 2)
    results['DWT'] = {'reconstruction': recon_dwt, 'mse': mse_dwt}
    print(f"  MSE: {mse_dwt:.6f}")

    return results


if __name__ == '__main__':
    print("=" * 80)
    print("Testing Baseline Methods")
    print("=" * 80)

    # Generate synthetic ECG signals
    np.random.seed(42)
    n_samples = 100
    signal_length = 3600
    t = np.linspace(0, 10, signal_length)

    signals = np.zeros((n_samples, signal_length))
    for i in range(n_samples):
        # Synthetic ECG
        heart_rate = 60 + np.random.randn() * 5
        freq = heart_rate / 60

        # Components
        signal = 0.2 * np.sin(2 * np.pi * freq * 0.8 * t)  # P-wave
        signal += np.exp(-((t % (1/freq) - 0.35) ** 2) / 0.005)  # QRS
        signal += 0.3 * np.exp(-((t % (1/freq) - 0.6) ** 2) / 0.02)  # T-wave
        signal += 0.05 * np.random.randn(signal_length)  # Noise

        signals[i] = signal

    print(f"\nGenerated {n_samples} synthetic ECG signals ({signal_length} samples each)\n")

    # Test all baselines
    results = test_all_baselines(signals)

    print("\n" + "=" * 80)
    print("All baseline methods tested successfully!")
    print("=" * 80)
