# ReCySE: Reliability-Aware Selective Ensemble Framework for Cervical Cytology Classification

This repository contains the code, fixed data split, and saved model outputs for the manuscript

> Albzour, N., & Lam, S. S. *ReCySE: A Reliability-Aware Selective Ensemble Framework for Cervical Cytology Classification.* Manuscript under review.

ReCySE (Reliability-aware Cervical Cytology Selective Ensemble) evaluates deep ensembles for five-class cervical cell classification on SIPaKMeD. It combines three steps, each fitted on calibration data only:

1. **Reliability-aware ensemble selection.** Nine candidate models are ranked by an equal-weight composite of eight metrics (accuracy, macro-F1, AUROC, ECE, worst-class ECE, AURC, Brier score, and NLL), and the top K = 2, 3, or 4 are combined by soft voting (Hybrid-2, Hybrid-3, Hybrid-4).
2. **Probability calibration.** Temperature scaling with an exact negative log-likelihood objective, T in [0.25, 5.0].
3. **Empirical risk-targeted selective prediction.** For a target selective risk alpha, a confidence threshold is chosen so that the one-sided 90% Clopper-Pearson upper bound of the calibration selective risk does not exceed alpha. Cells below the threshold are referred for review.

The study design keeps every selection, calibration, and threshold decision away from the test data:

- **Parent-image-grouped cross-validation.** SIPaKMeD cells are cropped from 966 parent images. Cells from the same parent image never appear in more than one role within a fold. Under a cell-level split of the same images, 91.6% of test cells had a sibling cell from the same parent image in training.
- **Parent-disjoint calibration subset.** About 1/8 of the outer-training parent images (about 405 cells per fold) are held out for temperature fitting, ranking (with inner cross-fitting), and threshold selection.
- **5 outer folds x 3 training seeds (42, 123, 456) = 15 runs per model.**

The selective prediction analysis is exploratory and provides no formal risk guarantee. Grouping is by parent image, because patient and slide identifiers are not available in SIPaKMeD.

The project was previously called **Hybrid-K**. That name appears in file names, comments, and configuration labels (for example `Hybrid-2`).

## Repository structure

```
ReCySE/
├── splits/                 Fixed parent-image-grouped split
│   ├── split_manifest.csv      Role of each of the 4,049 cells in each outer fold (train / calib / test)
│   ├── split_audit.txt/.json   Fold sizes, class balance, and zero-overlap checks
│   └── make_split_manifest.py  Script that built the manifest
├── training/               Google Colab notebooks used to train the nine models (one per model)
├── analysis/               Post-training analyses (CPU only, read saved outputs)
│   ├── calib_correction_cell.py     1. Temperature scaling on the calibration subset
│   ├── crossfit_composite_cell.py   2. Grouped cross-fitting, composite score, Top-K selection
│   ├── ensemble_eval_cell.py        3. Ensemble construction and final test evaluation
│   ├── risk_targeting_cell.py       4. Empirical risk-targeted selective prediction
│   ├── stats_cell.py                5. Paired fold-level statistics (Nadeau-Bengio, Holm)
│   └── robustness_cell.py           6. Metric-weight robustness (Dirichlet sampling, leave-one-metric-out)
├── posthoc/                Post-hoc descriptive diagnostics (reliability bins, risk-coverage, class-level analysis)
│   ├── verify.py
│   └── analyze.py
├── protocols/              Pre-specified analysis protocol and training configurations
│   ├── risk_targeting_protocol.json
│   ├── ensemble_evaluation_config.json
│   └── run_configs/            Training configuration and software environment for each model
├── saved_outputs/
│   └── fold_results/       Saved outputs of all 135 training runs (9 models x 5 folds x 3 seeds)
└── reference_outputs/      Outputs reported in the manuscript, for comparison
```

Each file in `saved_outputs/fold_results/` holds, for one model, fold, and seed: test logits, raw and calibrated probabilities, labels, file names, parent-image IDs, the fitted temperature, calibration-subset logits and labels, the training history, and test and calibration metrics. All analyses in the manuscript can be reproduced from these files without retraining.

Model checkpoints and the SIPaKMeD images are not included.

## Dataset

SIPaKMeD contains 4,049 single-cell images in five classes (Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, Superficial-Intermediate), cropped from 966 Pap smear images. It is available from its original source: https://www.cs.uoi.gr/~marina/sipakmed.html

The training notebooks download the dataset with `kagglehub` from the Kaggle copy `prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed`. Please cite the original dataset paper:

> Plissiti, M. E., Dimitrakopoulos, P., Sfikas, G., Nikou, C., Krikoni, O., & Charchanti, A. (2018). SIPaKMeD: A new dataset for feature and image based classification of normal and pathological cervical cells in Pap smear images. In *2018 25th IEEE International Conference on Image Processing (ICIP)*, 3144-3148. https://doi.org/10.1109/ICIP.2018.8451588

The split manifest identifies cells by class folder and file name only. The parent-image ID is the `NNN` prefix of a crop name `NNN_MM.bmp` within its class folder. The training code verifies the manifest against the downloaded files with a SHA-256 checksum (`01163b7112fcd935bac3941b8b64489125e7e417c39765d650c2711adcdb3864`) and stops if they differ.

## Installation

The experiments were run in Google Colab with Python 3.13, PyTorch 2.11.0, torchvision 0.26.0, timm 1.0.11, and scikit-learn 1.6.1 on an NVIDIA L4 GPU (recorded in `protocols/run_configs/`).

```bash
git clone https://github.com/Nisreen91/ReCySE.git
cd ReCySE
pip install -r requirements.txt
```

The analysis and post-hoc scripts need only NumPy, pandas, SciPy, scikit-learn, and Matplotlib, and run on a CPU.

## Usage

### 1. Train the nine models (GPU)

Open each notebook in `training/` in Google Colab with a GPU runtime and run all cells. Each notebook trains one model for 5 folds x 3 seeds, using the fixed split embedded in the notebook and verified by its checksum, and writes one JSON file per run to `fold_results/` in the output folder (default `/content/drive/MyDrive/HybridK_Phase1_corrected`).

Training settings (identical for all models): ImageNet-pretrained timm backbones, 224 x 224 input, 15 epochs, batch size 32, AdamW (learning rate 1e-4, weight decay 1e-4), 2-epoch warmup then cosine decay, label smoothing 0.1, class-balanced sampling, and the final-epoch model. The test fold is evaluated once, after training.

You can skip this step and use the saved outputs in `saved_outputs/fold_results/`.

### 2. Run the analyses (CPU)

Each script reads its inputs from the folder given by the environment variable `HYBRIDK_OUT` and writes its results to a new subfolder there. To reproduce the analyses from the saved outputs:

```bash
export HYBRIDK_OUT=$(pwd)/saved_outputs
python analysis/calib_correction_cell.py     # -> calibration_corrected/
python analysis/crossfit_composite_cell.py   # -> crossfit_composite/
python analysis/ensemble_eval_cell.py        # -> hybridk_ensemble_evaluation/
python analysis/risk_targeting_cell.py       # -> risk_targeting_exploratory/
python analysis/stats_cell.py                # -> hybridk_statistical_analysis/
python analysis/robustness_cell.py           # -> hybridk_robustness_analysis/
```

Run the scripts in this order, because each step uses the previous outputs. The scripts were written as Colab cells. Outside Colab they skip the Google Drive mount. Some scripts refuse to overwrite an existing results folder; remove or rename it before running again.

### 3. Post-hoc diagnostics (CPU)

`posthoc/verify.py` checks the integrity of the saved ensemble predictions, and `posthoc/analyze.py` computes the reliability bins, risk-coverage curves, and class-level coverage, referral, and selective risk. Both read their inputs from the folder set by the variable `D` at the top of the script. Set `D` to `reference_outputs/` (or to your `HYBRIDK_OUT` folder after step 2). `analyze.py` writes its tables to the folder set by `O`, and `verify.py` writes `verification.json` to the path given in its last line; set both to a local folder before running.

### 4. Compare with the reported results

`reference_outputs/` contains the outputs reported in the manuscript, including the summary metrics of all configurations (`hybridk_ensemble_evaluation/summary_mean_sd.csv`), the selected ensemble members per fold, the risk-targeting results (`risk_targeting_exploratory/`), and the class-level analysis (`posthoc_class_level/class_level_summary.csv`).

## Main results

Means over 15 fold-by-seed runs under parent-image-grouped cross-validation (see the manuscript for full results and statistical tests):

- Compared with the calibration-selected best single model (accuracy 0.965), the ReCySE ensembles reached about 0.972 accuracy and reduced the Brier score by 20-25%, NLL by 19-28%, and AURC by 30-56%. The Brier score and NLL improved in all five folds, but no difference was significant after Holm correction.
- Accuracy-based Top-K and nine-model ensembles gave similar gains. No consistent advantage of reliability-aware over accuracy-based member selection was detected.
- In exploratory risk targeting at a 1% target, Hybrid-2 accepted 76.5% of test cells on average, compared with 53.3% for Accuracy Top-2, with substantial variation across folds.
- An overall risk target did not ensure class-level reliability: the selective risk among accepted Koilocytotic cells exceeded the overall target for every configuration at alpha = 1%, 3%, and 5%.

## Related preprint

A preliminary version of this work was posted as arXiv:2609.09189 (https://doi.org/10.48550/arXiv.2609.09189). That version used a cell-level split. This repository implements the corrected protocol described in the manuscript.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The SIPaKMeD dataset is not distributed here and remains subject to the terms of its original source.

## Contact

Nisreen Albzour, School of Systems Science and Industrial Engineering, Binghamton University (nalbzour@binghamton.edu)
