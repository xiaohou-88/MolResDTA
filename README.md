# MolRes-DTA

**MolRes-DTA: A Molecular-Modality Fusion and Residue-Aware Framework for Drug–Target Affinity Prediction**

This repository contains the source code for **MolRes-DTA**, a deep learning framework for drug–target affinity (DTA) prediction. The method integrates multi-granular drug molecular features and residue-aware protein representations, aiming to improve prediction accuracy across drugs of varying complexity.

> 📌 This work is currently under preparation for submission. The paper will be linked here once available.

---

## 🔍 Highlights

- **Multi-modality Drug Representation**: Combines atom-level topology with functional group semantics.
- **Residue-aware Protein Encoding**: Combine residue-aware with attention and multi-scale convolution.
- **State-of-the-art Performance**: Achieved SOTA level on the Davis dataset.
- The first study to systematically explore the link between molecular size and multimodal fusion.

---

## 🧬 Dataset

We use the datasets provided by the [TEFDTA project](https://github.com/lizongquan01/TEFDTA/tree/master/data). Please download the data from their repository and place it in the `./data` directory:

```bash
git clone https://github.com/lizongquan01/TEFDTA
cp -r TEFDTA/data ./data

## 🛠️ Installation
1.Clone this repository:

```bash
git clone https://github.com/xiaohou-88/MolResDTA.git
cd MolResDTA
```

2.(Optional but recommended) Create a virtual environment:

```bash
conda create -n molres-dta python=3.10
conda activate molres-dta
```

3.Clone this repository:

```bash
pip install -r requirements.txt
```

## 🚀 Usage
Run training or evaluation by executing:

```bash
python main.py
```

## 📁 Project Structure

```bash
MolRes-DTA/
│
├── data/                # Input datasets (from TEFDTA)
├── models/              # Model architectures
├── utils/               # Utility scripts
├── main.py              # Main entry point
├── requirements.txt
└── README.md
```

## 📖 Citation
The citation will be available once the paper is published.

## 📄 License
This project is licensed under the Apache 2.0 License.
