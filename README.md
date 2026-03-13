# MolRes-DTA

**MolRes-DTA: A Molecular-Multiview Fusion and Residue-Aware Framework for Drug–Target Affinity Prediction**

This repository contains the source code for **MolRes-DTA**, a deep learning framework for drug–target affinity (DTA) prediction. The method integrates multi-granular drug molecular features and residue-aware protein representations, aiming to improve prediction accuracy across drugs of varying complexity.

> 📌 This work is currently under preparation for submission. The paper will be linked here once available.

---

## 🔍 Highlights

- **Multi-multiview Drug Representation**: Combines atom-level topology with functional group semantics.
- **Residue-aware Protein Encoding**: Combine residue-aware with attention and multi-scale convolution.
- **State-of-the-art Performance**: Achieved SOTA level on the Davis dataset.
- The first study to systematically explore the link between molecular size and multimodal fusion.

---

## 🧬 Dataset

We use the datasets provided by the [TEFDTA project](https://github.com/lizongquan01/TEFDTA/tree/master/data). Please download the data from their repository and place it in the `./data` directory:

```bash
git clone https://github.com/lizongquan01/TEFDTA
cp -r TEFDTA/data ./data
```

---

## 🛠️ Installation

### Requirements (Software Libraries)
Tested environment (recommended):
- Python 3.10
- PyTorch (torch)
- NumPy (numpy)
- Pandas (pandas)
- SciPy (scipy)
- TensorBoard (tensorboard)
- RDKit (rdkit)  *(used for SMILES processing / MACCS keys in `process_data.py`)*
- PyTorch Geometric (torch_geometric) *(used for GCN/GAT in `model.py`)*

Install dependencies:
```bash
pip install -r requirements.txt
```

> Notes:
> - If you use GPU, please install a CUDA-enabled PyTorch build matching your CUDA version.
> - Installing `rdkit` and `torch_geometric` may require extra steps depending on OS/CUDA. Please refer to their official installation guides.

### Setup
1. Clone this repository:
```bash
git clone https://github.com/xiaohou-88/MolResDTA.git
cd MolResDTA
```

2.(Optional but recommended) Create a virtual environment:

```bash
conda create -n molres-dta python=3.10
conda activate molres-dta
```

3.Install packages:

```bash
pip install -r requirements.txt
```

## 🚀 Usage
Run training:

```bash
python main.py
```

### Test-only (evaluate a saved checkpoint)
To evaluate a trained checkpoint on the test set:

```bash
python test_simple.py --dataset davis --ckpt .\test\YOUR_BEST_MODEL.pth
```

(Optional) Save predictions to a custom file:

```bash
python test_simple.py --dataset davis --ckpt .\test\YOUR_BEST_MODEL.pth --pred_out .\predictions\davis_pred.txt
```

---

## 🆕 Train/Test with your own data

### 1) Prepare your CSV files
Create two CSV files (train and test) with the following required columns:

- `iso_smiles`: SMILES string of the compound
- `target_sequence`: amino-acid sequence of the protein target
- `affinity`: real-valued binding affinity label

Example (header only):
```csv
iso_smiles,target_sequence,affinity
CC(=O)OC1=CC=CC=C1C(=O)O,MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPT...,7.2
```

### 2) Put files into the data folder
Option A (recommended): follow the existing folder style, e.g.
```
./data/YourData/
  YourData_train.csv
  YourData_test.csv
```

### 3) Point the code to your dataset
Currently, dataset selection is controlled by the `DATASET` variable in `main.py` and `test_simple.py`.
To use a new dataset, add a new branch in:
- `main.py` (for training)
- `test_simple.py` (for test-only)

You need to specify:
- CSV paths
- `max_smiles_len` and `max_fasta_len` (sequence truncation lengths)

### 4) Run training/testing
Train:
```bash
python main.py
```

Test (after training, use the checkpoint saved in `./test/`):
```bash
python test_simple.py --dataset davis --ckpt .\test\YOUR_BEST_MODEL.pth
```

---

## 📁 Project Structure

```bash
MolRes-DTA/
│
├── data/                
├── test/
├── model.py              
├── metrics.py
├── main.py              # Main entry point
├── peocess_data.py
├── train_and_test.py
├── test_simple.py 
├── requirements.txt
└── README.md
```

## 📖 Citation
The citation will be available once the paper is published.

## 📄 License
This project is licensed under the Apache 2.0 License.
