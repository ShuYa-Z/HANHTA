# HAN-HTA: Herb-Target Association Prediction Using a Heterogeneous Attention Network

## Overview

This repository provides the implementation of **HAN-HTA (Heterogeneous Attention Network for Herb-Target Association)** for herb-target association prediction in Traditional Chinese Medicine (TCM).

HAN-HTA integrates heterogeneous graph neural networks with neural inductive matrix completion to model complex relationships among herbs, targets, diseases, symptoms, drugs, and TCM symptoms. The heterogeneous attention mechanism is used to aggregate information along different meta-paths, and the learned herb and target representations are subsequently used for herb-target association prediction.

The code and dataset provided in this repository correspond to the experiments reported in the manuscript:

> **Herb target prediction based on neural inductive matrix completion with heterogeneous graph network**

---

## Repository Structure

```text
.
├── data/ 
│     ├── herb_herb.csv
│     ├── herb_disease.csv
│     ├── herb_TCMsymptom.csv
│     ├── herb_symptom.csv
│     ├── target_drug.csv
│     ├── target_disease.csv
│     ├── target_symptom.csv
│     └── target_herb.csv
│
├── main.py
├── model.py
├── utils.py
├── predict_new_target.py
└── README.md
```

- `main.py`: Main training and evaluation script.
- `model.py`: HAN-HTA model implementation.
- `utils.py`: Dataset loading, heterogeneous graph construction, evaluation, and utility functions.
- `predict_new_target.py`: Script for the new-target prediction experiment.
- `dataset/Zdataset/`: Dataset used for herb-target association prediction.

---

## Dataset

The Zdataset integrates heterogeneous information from publicly available TCM and biomedical databases.

The following relationship matrices are used:

| File | Relationship |
|---|---|
| `herb_herb.csv` | Herb-herb similarity |
| `herb_disease.csv` | Herb-disease associations |
| `herb_TCMsymptom.csv` | Herb-TCM symptom associations |
| `herb_symptom.csv` | Herb-symptom associations |
| `target_drug.csv` | Target-drug associations |
| `target_disease.csv` | Target-disease associations |
| `target_symptom.csv` | Target-symptom associations |
| `target_herb.csv` | Herb-target associations |

The integrated data were collected from publicly available resources, including:

- SymMap
- ETCM
- HERB
- SIDER
- DrugBank

Detailed information regarding data sources, preprocessing, and dataset construction is provided in the manuscript.

---

## Heterogeneous Graph Construction

The heterogeneous graph contains multiple types of relationships on both the herb and target sides.

For herb-target association prediction, the implementation provides a control parameter, `include_herb_target`, for heterogeneous graph construction.

When `include_herb_target` is enabled, herb-target and target-herb relations are included in the heterogeneous graph through the corresponding `htg` and `gth` relations.

During each evaluation, training herb-target associations are included in the heterogeneous graph and used for node embedding. Held-out test herb-target associations are excluded from these processes and are used only as ground-truth labels for final evaluation.

---

## Model Architecture

HAN-HTA consists of two main components:

### 1. Heterogeneous Attention Network

The heterogeneous attention network aggregates information from multiple relation types and meta-paths in the heterogeneous graph.

The model performs:

- intra-meta-path information aggregation;
- inter-meta-path semantic aggregation;
- attention-based representation learning for herbs and targets.

### 2. Neural Inductive Matrix Completion

The learned herb and target representations are combined to reconstruct the herb-target association score matrix.

The resulting scores are used to evaluate the predicted herb-target associations.

---

## Training Pipeline

The overall workflow is:

Zdataset
    │
    ▼
Load heterogeneous relationship matrices
    │
    ▼
Construct herb-target prediction dataset
    │
    ▼
Cross-validation data splitting
    │
    ├── Training herb-target associations
    │
    └── Held-out test herb-target associations
    │
    ▼
Construct training heterogeneous graph
    │
    ├── Herb-side relations
    ├── Target-side relations
    └── Training herb-target relations
        controlled by `include_herb_target`
    │
    ▼
HAN-based heterogeneous representation learning
    │
    ▼
Herb and target embeddings
    │
    ▼
Neural inductive matrix completion
    │
    ▼
Herb-target association score matrix
    │
    ▼
Compare predictions with held-out test associations
    │
    ▼
Model evaluation


---

## Baseline Methods

The baseline methods were implemented following their original publications. To ensure a fair comparison, we retained the original model architectures, parameter settings, and training strategies without modification. Since the original implementations were designed for drug-target prediction, the drug-target association data were replaced with the herb-target association dataset used in this study.

---

## Requirements

The implementation uses the following major Python packages:

- Python
- PyTorch
- DGL
- NumPy
- Pandas
- SciPy
- scikit-learn

The package versions should correspond to the computational environment used to run the experiments.

---

## Usage

### 1. Prepare the dataset

Place the Zdataset files under:

```text
dataset/Zdataset/
```

The directory should contain:

```text
herb_herb.csv
herb_disease.csv
herb_TCMsymptom.csv
herb_symptom.csv
target_drug.csv
target_disease.csv
target_symptom.csv
target_herb.csv
```

### 2. Train HAN-HTA

Run:

```bash
python main.py
```

The training script loads the Zdataset, constructs the heterogeneous graph, performs the specified cross-validation procedure, trains the HAN-HTA model, and evaluates the predicted herb-target associations.

Model and training parameters can be configured through the arguments defined in `main.py`.

### 3. New-target prediction

The repository also provides:

```bash
python predict_new_target.py
```

for the new-target prediction experiment.

The corresponding dataset path and experimental parameters are defined in the script.

---

## Evaluation Metrics

The implementation evaluates herb-target association prediction using:

- AUROC
- AUPR
- Accuracy
- Precision
- Recall
- F1-score

The generated evaluation results are saved by the corresponding training scripts.

---

## Reproducibility

This repository provides the dataset and source code used for the experiments to facilitate reproducibility.

To reproduce the experiments:

1. Download or use the provided Zdataset.
2. Place the dataset files in `dataset/Zdataset/`.
3. Install the required Python dependencies.
4. Check the dataset path and model parameters in the corresponding scripts.
5. Run `main.py` for herb-target association prediction.
6. Run `predict_new_target.py` for the new-target prediction experiment.

The detailed experimental settings and methodology are described in the manuscript.

---

## Citation

If you use this implementation or dataset in your research, please cite:


---

## Contact

For questions regarding the implementation, dataset, or experiments, please contact the corresponding author.
