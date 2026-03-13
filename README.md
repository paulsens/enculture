# enculture

Paired sequence transformers for analyzing fMRI brain imaging data from a musical enculturation study. Developed as part of my PhD thesis at [PLACEHOLDER: University Name] ([PLACEHOLDER: link to thesis]).

## Overview

This project investigates whether self-supervised transformer pretraining on fMRI timeseries can learn representations useful for distinguishing neural responses to culturally familiar vs. unfamiliar music. Subjects listened to Western (Bach) and non-Western (Shanxi folk) music across two scanning sessions, enabling both within-session and cross-session analyses.

The project extends the [fmriBERT](https://github.com/paulsens/fmriBERT) framework with two key adaptations:

1. **Same-Session (SS) task**: A binary classification task asking whether two fMRI sequences come from the same scanning session or different sessions of the same subject — probing session-level neural consistency
2. **Next-Thought Prediction (NTP)**: Adapted from BERT's next-sentence prediction — given two consecutive windows of voxel activations, predict whether the second follows the first temporally

### Pretraining Tasks

- **Next-Thought Prediction (NTP)**: Given paired 5-TR windows of voxel activations separated by CLS/SEP tokens, predict whether the second window is the true temporal successor or a random segment
- **Same-Session Prediction (SS)**: Given paired sequences from two scanning sessions of the same subject, predict whether both come from the same session (positive) or different sessions (negative), conditioned on music style (Bach or Shanxi)

### Downstream Finetuning

Pretrained models are transferred via `pairedfinetune.py`, which loads encoder weights and evaluates on held-out folds. Cross-task transfer is supported (e.g., pretraining on genre NTP, finetuning on enculturation SS).

## Architecture

The model (`transfer_transformer.py`) is an encoder-decoder transformer adapted for fMRI voxel timeseries:

- **Input**: Flattened voxel activations from Nucleus Accumbens (NAcc) union ROI (~417 voxels + 3 token dimensions = 420), with special tokens (CLS, MSK, SEP) prepended
- **Tokenization**: Each TR (1 fMRI volume) is one token. Input is a pair of 5-TR windows: `[CLS, seq1_t1, ..., seq1_t5, SEP, seq2_t1, ..., seq2_t5]` → 12 tokens
- **Encoder**: Multi-head self-attention blocks with learned positional embeddings
- **Decoder**: Dual-head output for CLS (binary session/NTP classification) and MSK (voxel reconstruction) tasks
- **Transfer**: For finetuning, pretrained encoder weights are loaded with `strict=False`, and task-specific output heads are used

## Datasets

### Enculturation Dataset
- 5 subjects, 2 scanning sessions each (separated by weeks/months)
- 8 runs per session, 12 trials per run, 30 TRs per trial (360 TRs/run, 2880 TRs/session)
- Stimuli: Bach (Western) and Shanxi folk (non-Western) music clips
- Cross-validated with one run held out per subject

### OpenGenre Dataset (Transfer Learning)
- 5 subjects listening to 10 music genres
- 6 test runs + 12 training runs per subject
- 400 TRs per run (40 clips x 10 TRs each after dummy removal)
- Used for NTP pretraining, with 12-fold cross-validation

### ROI: Nucleus Accumbens (NAcc)
- Bilateral NAcc from Harvard-Oxford atlas, warped to each subject's space
- Union ROI across all subjects → 417 voxels
- Left/right split at MNI x-coordinate midpoint (48)

## Project Structure

```
enculture/
├── pairedpretrain.py           # Pretraining script (NTP and SS tasks)
├── pairedfinetune.py           # Finetuning with pretrained model transfer
├── transfer_transformer.py     # Transformer model (encoder-decoder architecture)
│
├── helpers.py                  # Data processing, masking, labels, metrics, Dataset classes
├── make_datasets.py            # Dataset construction (NTP and SS from raw voxel data)
├── Constants.py                # Configuration, paths, subject metadata
│
├── detrendstandardize.py       # fMRI preprocessing: ROI extraction, detrending, z-scoring
├── countROIs.py                # ROI analysis: binary masks, 3D search, lateralization, NAcc union
├── create_eventfiles.py        # BIDS event file generation from experiment logs
│
├── scripts/                    # SLURM job submission scripts
│   ├── pretrain.sh/.script     # Parameterized pretraining launcher
│   └── finetune.sh/.script     # Parameterized finetuning launcher
│
└── deprecated/                 # Old/unused files preserved for reference
    ├── scratch.py              # Incomplete ROI comparison code
    ├── FLA.py                  # First-level analysis stub
    ├── compare_smoothness.py   # One-off smoothing comparison
    ├── pleasure_check.py       # Pleasure rating extraction
    ├── rename_dirs.py          # Batch directory renaming utility
    ├── dayone.py               # Stimulus file categorization
    ├── *.png                   # Result visualizations
    ├── *.tsv, *.json           # Sample BIDS data files
    └── pairedpretrain.sh/...   # Original SLURM scripts (pre-consolidation)
```

## Usage

### 1. Data Preparation

Extract ROI voxels from preprocessed fMRI data, detrend, and standardize:

```bash
python detrendstandardize.py
```

Count and construct ROI masks (NAcc union across subjects):

```bash
python countROIs.py
```

Build NTP and/or Same-Session datasets:

```bash
python make_datasets.py
```

### 2. Pretraining

Pretrain on genre NTP (12-fold cross-validation):

```bash
bash scripts/pretrain.sh "genre_ntp_run" 0-11 0.00001 CLS_only genre_NTP
```

Pretrain on enculturation Same-Session (Bach condition):

```bash
bash scripts/pretrain.sh "enc_bach_ss" 0-0 0.00001 CLS_only enc_bachSS
```

Or call the Python script directly:

```bash
python pairedpretrain.py -m "description" -heldout_run 0 -LR 0.00001 -task CLS_only -dataset enc_shanxiSS
```

### 3. Finetuning

Transfer pretrained weights and finetune on a different dataset/task:

```bash
bash scripts/finetune.sh "transfer_exp" 0-11 0.00001 CLS_only enc_shanxiSS enc_bachSS
```

Or call the Python script directly:

```bash
python pairedfinetune.py -m "description" -heldout_run 0 -LR 0.00001 -dataset enc_shanxiSS -pretrain_task enc_bachSS
```

## Configuration

Edit `Constants.py` to set:
- `env`: `"local"` or `"discovery"` (HPC cluster)
- Paths to fMRI data, ROI masks, and output directories
- `ATTENTION_HEADS`, `EPOCHS`, and other hyperparameters
- `COOL_DIVIDEND`: Voxel space dimension (420 after token dimensions)

Available datasets for `-dataset` argument:
- `genre_NTP`: OpenGenre next-thought prediction
- `enc_NTP`: Enculturation next-thought prediction
- `enc_bachSS`: Enculturation same-session (Bach condition)
- `enc_shanxiSS`: Enculturation same-session (Shanxi condition)

## Requirements

- Python 3.9+
- PyTorch
- nibabel (for NIfTI fMRI data)
- nilearn
- numpy, scipy
- scikit-learn
- pandas

## Citation

If you use this code, please cite:

```
[PLACEHOLDER: thesis citation]
```

## License

MIT License — see [LICENSE](LICENSE) for details.
