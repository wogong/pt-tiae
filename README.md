# Unsupervised Outlier Detection via Transformation Invariant Autoencoder

## Introduction
This is the official implementation of the TIAE framework presented by "Unsupervised Outlier Detection via Transformation Invariant Autoencoder
". The codes are used to reproduce experimental results of TIAE reported in the paper.

## Requirements
- Python 3.6
- PyTorch 1.3.1 (GPU)
- Keras 2.2.0 
- Tensorflow 1.8.0 (GPU)
- sklearn 0.19.1
 
## Usage

To obtain the results of TIAE with default settings, simply run the following command:

```bash
python outlier_experiments.py
```

This will automatically run TIAE reported in the manuscript.  Please see ```outlier_experiments.py``` for more details.

After training, to print UOD results for a specific algorithm in AUROC/AUPR, run:

```bash
# AUROC of TIAE on CIFAR10 with outlier ratio 0.1
python scripts/evaluate_roc_auc.py --dataset cifar10 --algo_name tiae-0.1

# AUPR of TIAE on MNIST with outlier ratio 0.25 and inliers as the postive class
python scripts/evaluate_pr_auc.py --dataset mnist --algo_name tiae-0.25 --postive inliers
```

## Credit

- https://github.com/demonzyj56/E3Outlier
- https://github.com/gilshm/anomaly-detection

## License

TIAE is released under the MIT License.
