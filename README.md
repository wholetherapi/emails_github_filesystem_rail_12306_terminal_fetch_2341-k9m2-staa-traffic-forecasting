# Sparse Temporal Attention for Long-Horizon Traffic Forecasting

This repository contains the implementation for the paper **"Sparse Temporal Attention for Long-Horizon Traffic Forecasting"**
(ACM SIGSPATIAL 2025).

The released code was written by the authors and is intended to reproduce the
main results reported in the paper.

## Repository layout

- `src/model.py` - sparse temporal attention encoder
- `src/dataset.py` - METR-LA / PEMS-BAY loaders
- `src/train.py` - training and rolling evaluation entry point

## Environment

```
python>=3.9
torch>=2.1
numpy>=1.24
```

Install the dependencies with `pip install -r requirements.txt`.

## Training

```
python -m src.train --config configs/main.yaml
```

## Citation

```
@inproceedings{sparse2025,
  title={Sparse Temporal Attention for Long-Horizon Traffic Forecasting},
  author={Scott, Jeffrey and Raman, Priya},
  booktitle={ACM SIGSPATIAL 2025},
  year={2025}
}
}
