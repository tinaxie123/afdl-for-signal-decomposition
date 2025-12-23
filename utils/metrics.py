import numpy as np
import torch


def compute_snr(signal, noise):
    """
    Compute Signal-to-Noise Ratio (SNR) in dB

    Args:
        signal (np.ndarray or torch.Tensor): Ground truth signal
        noise (np.ndarray or torch.Tensor): Noise or error signal

    Returns:
        float: SNR in dB
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    if isinstance(noise, torch.Tensor):
        noise = noise.detach().cpu().numpy()

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def compute_mse(pred, target):
    """
    Compute Mean Squared Error (MSE)

    Args:
        pred (np.ndarray or torch.Tensor): Predicted signal
        target (np.ndarray or torch.Tensor): Ground truth signal

    Returns:
        float: MSE value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    mse = np.mean((pred - target) ** 2)
    return mse


def compute_mae(pred, target):
    """
    Compute Mean Absolute Error (MAE)

    Args:
        pred (np.ndarray or torch.Tensor): Predicted signal
        target (np.ndarray or torch.Tensor): Ground truth signal

    Returns:
        float: MAE value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    mae = np.mean(np.abs(pred - target))
    return mae


def compute_rmse(pred, target):
    """
    Compute Root Mean Squared Error (RMSE)

    Args:
        pred (np.ndarray or torch.Tensor): Predicted signal
        target (np.ndarray or torch.Tensor): Ground truth signal

    Returns:
        float: RMSE value
    """
    mse = compute_mse(pred, target)
    return np.sqrt(mse)


def compute_correlation(signal1, signal2):
    """
    Compute correlation coefficient between two signals

    Args:
        signal1 (np.ndarray or torch.Tensor): First signal
        signal2 (np.ndarray or torch.Tensor): Second signal

    Returns:
        float: Correlation coefficient
    """
    if isinstance(signal1, torch.Tensor):
        signal1 = signal1.detach().cpu().numpy()
    if isinstance(signal2, torch.Tensor):
        signal2 = signal2.detach().cpu().numpy()

    signal1_flat = signal1.flatten()
    signal2_flat = signal2.flatten()

    corr = np.corrcoef(signal1_flat, signal2_flat)[0, 1]
    return corr


def compute_prd(pred, target):
    """
    Compute Percentage Root-mean-square Difference (PRD) in %

    PRD is commonly used in ECG signal quality assessment.
    Lower PRD indicates better reconstruction quality.

    Args:
        pred (np.ndarray or torch.Tensor): Predicted signal
        target (np.ndarray or torch.Tensor): Ground truth signal

    Returns:
        float: PRD value in percentage
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    numerator = np.sum((pred - target) ** 2)
    denominator = np.sum(target ** 2)

    if denominator == 0:
        return float('inf')

    prd = 100 * np.sqrt(numerator / denominator)
    return prd


def evaluate_decomposition(components, ground_truth_components):
    """
    Evaluate decomposition quality

    Args:
        components (list): List of predicted components
        ground_truth_components (list): List of ground truth components

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    assert len(components) == len(ground_truth_components), \
        "Number of components must match"

    metrics = {}

    for i, (pred_comp, gt_comp) in enumerate(zip(components, ground_truth_components)):
        component_name = f"component_{i+1}"

        metrics[f"{component_name}_mse"] = compute_mse(pred_comp, gt_comp)
        metrics[f"{component_name}_mae"] = compute_mae(pred_comp, gt_comp)
        metrics[f"{component_name}_snr"] = compute_snr(gt_comp, pred_comp - gt_comp)
        metrics[f"{component_name}_corr"] = compute_correlation(pred_comp, gt_comp)

    # Overall reconstruction metrics
    reconstructed = sum(components)
    original = sum(ground_truth_components)

    metrics["overall_mse"] = compute_mse(reconstructed, original)
    metrics["overall_mae"] = compute_mae(reconstructed, original)
    metrics["overall_snr"] = compute_snr(original, reconstructed - original)
    metrics["overall_prd"] = compute_prd(reconstructed, original)

    return metrics


def evaluate_reconstruction(pred, target):
    """
    Evaluate signal reconstruction quality (for methods that don't use components)

    Args:
        pred (np.ndarray or torch.Tensor): Reconstructed signal
        target (np.ndarray or torch.Tensor): Original signal

    Returns:
        dict: Dictionary containing SNR and PRD metrics
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    noise = pred - target
    snr = compute_snr(target, noise)
    prd = compute_prd(pred, target)

    return {
        'snr': snr,
        'prd': prd,
        'mse': compute_mse(pred, target),
        'mae': compute_mae(pred, target)
    }


# Aliases for compatibility
def calculate_snr(original, reconstructed):
    """
    Calculate SNR between original and reconstructed signals

    Args:
        original (np.ndarray): Original signal
        reconstructed (np.ndarray): Reconstructed signal

    Returns:
        float: SNR in dB
    """
    noise = reconstructed - original
    return compute_snr(original, noise)


def calculate_prd(original, reconstructed):
    """
    Calculate PRD between original and reconstructed signals

    Args:
        original (np.ndarray): Original signal
        reconstructed (np.ndarray): Reconstructed signal

    Returns:
        float: PRD in percentage
    """
    return compute_prd(reconstructed, original)
