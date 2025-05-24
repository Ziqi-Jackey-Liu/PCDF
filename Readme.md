# Predictability-Aware Compression and Decompression Framework (PCDF)

This repository contains the official implementation of our NeurIPS 2025 submission:

**Predictability-Aware Compression and Decompression Framework for Multichannel Time Series Data**

## 📦 Requirements

The code is written in Python and requires the following packages:
- `torch~=2.6.0+cu126`
- `numpy~=2.1.2`
- `pandas~=2.2.3`
- `tqdm~=4.67.1`
- `meteostat~=1.6.8`
- `matplotlib~=3.10.0`
- `scipy~=1.15.1`
- `scikit-learn~=1.6.1`
- `psutil~=5.9.0`

You can install all dependencies via:
```bash
pip install -r requirements.txt

## 📁 Project Structure:
PCDF/
│
├── Designed methods.py
├── Comparative methods.py
├── data/
├── main.py    
└── Readme.md

## 🚀 How to Run
To reproduce the experiments:
1. Select and prepare your dataset (place it under the data/ directory).
2. Modify relevant parameters in main.py according to the paper.
3. Run the experiment with: python main.py
4. The output will include the MSE and runtime of both the proposed and comparative methods.
