# Adaptive Functional Dictionary Learning (AFDL)

Official PyTorch implementation of **"Beyond Static Bases: Adaptive Functional Dictionary Learning for Interpretable Signal Decomposition"** (ICASSP 2026).

## Paper Information

- **Title**: Beyond Static Bases: Adaptive Functional Dictionary Learning for Interpretable Signal Decomposition
- **Author**: Haotong Xie
- **Institution**: Shanghai University of Finance and Economics, Shanghai, China
- **Conference**: ICASSP 2026

## Abstract

Signal decomposition inherently struggles with a tradeoff between the adaptability of data-driven methods and the critical interpretability often lost in abstract representations. We propose Adaptive Functional Dictionary Learning (AFDL), a novel and rigorous framework that resolves this tension. AFDL reimagines atom learning as the parameter estimation of interpretable functional forms (e.g., tunable wavelets), rather than opaque abstract vectors, thereby ensuring intrinsic physical interpretability with minimal reliance on strong domain priors. On PTB-ECG, AFDL outperforms K-SVD (SNR +10.8%, PRD -27.6%) and ICA, uniting adaptability and state-of-the-art interpretability.

## Environment Setup

### Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.0 (for GPU training)

### Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd ICASSP_Signal_Decomposition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### PTB Diagnostic ECG Database

This project uses the **PTB Diagnostic ECG Database**, a publicly available dataset from PhysioNet.

**Dataset Specifications**:
- **Source**: [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/)
- **Total Samples**: 2400 10-second ECG segments
- **Sampling Rate**: 360 Hz (3600 samples per segment)
- **Clinical Classes**:
  - NSR (Normal Sinus Rhythm): 800 samples
  - PVC (Premature Ventricular Contractions): 800 samples
  - AF (Atrial Fibrillation): 800 samples

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare PTB-ECG data (requires access to PTB database)
# Place preprocessed data in ./data/ptb_ecg_data/
# Required files: train_data.npz, val_data.npz, test_data.npz

# 3. Start training
python train_afdl.py --data_path ./data/ptb_ecg_data
```

### Data Requirements

The preprocessed PTB-ECG dataset should be placed in `./data/ptb_ecg_data/` with the following structure:
- `train_data.npz` - Training set (1680 samples)
- `val_data.npz` - Validation set (360 samples)
- `test_data.npz` - Test set (360 samples)

Each .npz file contains:
- `signals`: ECG signal arrays [N, 3600]
- `labels`: Class labels [N] (0: NSR, 1: PVC, 2: AF)

### Data Availability

The PTB Diagnostic ECG Database is publicly available from:
- **PhysioNet**: https://physionet.org/content/ptbdb/1.0.0/
- **Official PTB Site**: https://www.ptb.de/en/mediathek/datenbanken/ecg-database.html

Our preprocessing scripts ensure reproducibility of the exact dataset used in the paper.

## Project Structure

```
ICASSP_Signal_Decomposition/
├── configs/                 # Configuration files
│   ├── train_config.yaml   # Training configuration
│   └── test_config.yaml    # Testing configuration
├── data/                   # Dataset directory
│   ├── raw/               # Raw data
│   └── processed/         # Preprocessed data
├── models/                 # Model definitions
│   ├── __init__.py
│   └── signal_decomposition.py
├── utils/                  # Utility functions
��   ├── __init__.py
│   ├── metrics.py         # Evaluation metrics
│   └── visualize.py       # Visualization tools
├── checkpoints/           # Saved model checkpoints
├── pretrained/            # Pretrained model weights
├── results/               # Experiment results
├── train.py              # Training script
├── test.py               # Testing/inference script
├── data_preprocessing.py # Data preprocessing pipeline
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Training

### Train AFDL from Scratch

```bash
python train_afdl.py \
    --data_path ./data/ptb_ecg_data \
    --save_dir ./checkpoints \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --dict_size 128 \
    --num_heads 4 \
    --sparsity_weight 0.05
```

### Training Arguments

- `--data_path`: Path to PTB-ECG data (default: `./data/ptb_ecg_data`)
- `--save_dir`: Directory to save checkpoints (default: `./checkpoints`)
- `--batch_size`: Batch size (default: 16)
- `--num_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--dict_size`: Dictionary size (default: 128)
- `--seq_len`: Sequence length (default: 3600 for 10s at 360Hz)
- `--num_heads`: Number of attention heads (default: 4)
- `--sparsity_weight`: Sparsity regularization weight (default: 0.05)
- `--resume`: Path to checkpoint to resume from
- `--test_only`: Only run testing

### Resume Training

```bash
python train_afdl.py --resume ./checkpoints/checkpoint_latest.pth
```

## Pretrained Models

Download pretrained models from [Link to be added]:

| Model | Dataset | Metric | Score | Download |
|-------|---------|--------|-------|----------|
| Model-A | Dataset-1 | SNR | XX.XX dB | [link] |
| Model-B | Dataset-2 | MSE | X.XXXX | [link] |

Place downloaded models in `pretrained/` directory.

## Testing / Inference

### Run inference with trained model:

```bash
python train_afdl.py --test_only --resume ./checkpoints/checkpoint_best.pth
```

### Compare AFDL with all baseline methods:

Run a comprehensive comparison of AFDL against K-SVD, ICA, NMF, and DWT:

```bash
python experiments/compare_methods.py \
    --data_path ./data/ptb_ecg_data \
    --afdl_checkpoint ./checkpoints/checkpoint_best.pth \
    --batch_size 16 \
    --num_test_samples 100 \
    --save_dir ./results
```

This will:
1. Evaluate AFDL on the test set
2. Train and evaluate all baseline methods (K-SVD, ICA, NMF, DWT)
3. Compute performance metrics (SNR, PRD) for all methods
4. Generate comparative visualizations
5. Save results to `results/` directory

**Arguments:**
- `--data_path`: Path to PTB-ECG dataset
- `--afdl_checkpoint`: Path to trained AFDL model checkpoint
- `--batch_size`: Batch size for AFDL evaluation (default: 16)
- `--num_test_samples`: Number of test samples to evaluate (default: 100)
- `--save_dir`: Directory to save results (default: ./results)

## Results

### Quantitative Results on PTB-ECG Dataset

| Method | SNR (dB) | PRD (%) |
|--------|----------|---------|
| **AFDL (Ours)** | **17.5±1.1** | **4.2±0.5** |
| K-SVD | 15.8±1.2** | 5.8±0.7** |
| ICA | 14.9±1.3** | 6.5±0.8** |
| NMF | 14.5±1.4** | 7.2±0.9** |
| DWT | 15.2±1.1** | 6.2±0.6** |

**Relative Improvement over Best Baseline (K-SVD):**
- SNR: **+10.8%**
- PRD: **-27.6%** (lower is better)

*p < 0.05, **p < 0.001 (paired t-tests vs. AFDL with Bonferroni correction)

### Ablation Study

| Model Variant | SNR (dB) | PRD (%) |
|--------------|----------|---------|
| AFDL (Full) | 17.5±1.1 | 4.2±0.5 |
| w/o Attention | 15.7±1.2** | 6.1±0.7** |
| w/o Sparsity Reg | 17.0±0.8* | 4.7±0.4* |

*p < 0.05, **p < 0.001 (paired t-tests vs. AFDL)

## Reproduced Experimental Results

All figures and tables from the ICASSP 2026 paper are based on experiments conducted with the PTB-ECG dataset. Results include:
- Signal reconstruction comparisons
- Performance metrics tables
- Ablation study results

Results are saved to `results/` directory after training and evaluation.

### Figure 2: ECG Signal Analysis

#### (a) Signal Reconstruction Comparison
<p align="center">
  <img src="results/figures/fig2a_reconstruction.png" width="95%">
</p>

*Time-domain reconstruction comparison showing AFDL's superior preservation of ECG morphology. AFDL achieves SNR=17.5dB compared to K-SVD (15.8dB), ICA (14.9dB), and NMF (14.5dB).*

#### (b) Reconstruction Error Analysis
<p align="center">
  <img src="results/figures/fig2b_error.png" width="95%">
</p>

*Reconstruction error over time demonstrates AFDL's consistent low error (MAE=0.016mV) across all cardiac cycles, significantly outperforming baselines.*

#### (c) Frequency Domain Analysis
<p align="center">
  <img src="results/figures/fig2c_frequency.png" width="95%">
</p>

*Power spectral density comparison validates frequency content preservation in the critical ECG band (0.5-40Hz). AFDL maintains spectral fidelity while achieving superior SNR.*

#### Combined Figure 2
<p align="center">
  <img src="results/figures/fig2_combined.png" width="95%">
</p>

*Complete figure showing reconstruction comparison (a), error analysis (b), and frequency domain analysis (c) in one view.*

### Performance Comparison

<p align="center">
  <img src="results/figures/performance_comparison.png" width="95%">
</p>

*Bar chart comparing AFDL with baseline methods on SNR (higher is better) and PRD (lower is better). AFDL demonstrates superior performance with statistically significant improvements (**p < 0.001).*

### Key Findings

1. **Superior Reconstruction Quality**: AFDL achieves 17.5±1.1 dB SNR, representing a **+10.8% improvement** over the best baseline (K-SVD at 15.8±1.2 dB).

2. **Reduced Reconstruction Error**: PRD of 4.2±0.5%, a **-27.6% reduction** compared to K-SVD (5.8±0.7%), indicating more accurate signal representation.

3. **Frequency Preservation**: Maintains critical ECG frequency components (0.5-40Hz) with minimal distortion, essential for clinical applications.

4. **Ablation Study Results**:
   - Removing attention mechanism reduces SNR to 15.7±1.2 dB (**p < 0.001), confirming importance of adaptive basis selection
   - Removing sparsity regularization reduces SNR to 17.0±0.8 dB (*p < 0.05), showing value of parsimony constraint

5. **Statistical Significance**: All improvements over baselines are highly significant (p < 0.001) with Cohen's d effect sizes ranging from 0.58 to 2.73.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{xie2026afdl,
  title={Beyond Static Bases: Adaptive Functional Dictionary Learning for Interpretable Signal Decomposition},
  author={Xie, Haotong},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
}
```

## Key Features

- **Hierarchical Functional Dictionary**: Interpretable basis functions (40% Gabor, 25% Exponential Decay, 15% Gaussian, 15% Chirp, 5% Linear)
- **Multi-Modal Context Encoding**: Local statistics, gradient features, and spectral features with BiLSTM
- **Signal Segmentation**: Automatic classification into smooth and spike regions
- **Brain-Inspired Basis Selection**: Region-adaptive attention for optimal basis function selection
- **Attention Modulator**: Multi-head parameter prediction for adaptive signal fitting
- **Sparse Representation**: L1 regularization for parsimonious decompositions

## Model Architecture Components

### 1. Basis Functions

| Type | Formula | Usage |
|------|---------|-------|
| Gabor | $e^{-(t-\mu)^2/\sigma^2} \cos(2\pi f(t-\mu))$ | Oscillatory patterns |
| Exponential Decay | $A e^{-\lambda t}$ | Spike-like transients |
| Gaussian | $A e^{-(t-\mu)^2/\sigma^2}$ | Smooth localized features |
| Chirp | $\cos(2\pi(f_0 + kt)t)$ | Frequency-modulated signals |
| Linear | $at + b$ | Baseline trends |

### 2. Loss Function

$$\mathcal{L} = \mathbb{E}\left[\int_0^T \|X(t) - \hat{X}(t)\|^2 dt\right] + \lambda_s \mathcal{L}_{\text{sparsity}}$$

## Acknowledgments

This work builds upon:
- K-SVD dictionary learning
- Functional data analysis principles
- Brain-inspired neural architectures
- Sparse coding theory

## License

This implementation is for research purposes only. See LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- **Haotong Xie** - Shanghai University of Finance and Economics

---

**Implementation Status**: ✅ Complete
- [x] Hierarchical Functional Dictionary with 5 basis function types
- [x] Multi-modal Context Encoder (Conv1D + BiLSTM)
- [x] Signal Segmentation Network
- [x] Brain-Inspired Basis Function Selection
- [x] Attention Modulator with multi-head parameter prediction
- [x] Complete training pipeline with evaluation metrics
- [x] Data loader for PTB-ECG dataset
- [x] Comprehensive documentation

**Files**:
- `models/afdl.py`: Complete AFDL implementation (4.7M parameters)
- `train_afdl.py`: Training script with TensorBoard logging
- `data/ptb_ecg_loader.py`: PTB-ECG dataset loader
- `utils/metrics.py`: Evaluation metrics (SNR, PRD)

