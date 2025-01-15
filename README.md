# README: Herb-Target Prediction Using HAN-HTA Framework

## Overview
This study proposes a novel framework, **HAN-HTA (Heterogeneous Attention Network for Herb-Target Association)**, for predicting herb-target interactions in Traditional Chinese Medicine (TCM). By leveraging a heterogeneous graph network and neural inductive matrix completion, HAN-HTA provides an advanced method for understanding the pharmacological mechanisms of herbal medicines.

## Features
- **Heterogeneous Graph Construction**: Integrates data from multiple biomedical sources, including herbs, targets, diseases, and symptoms.
- **Attention Mechanism**: Utilizes intra-metapath and inter-metapath aggregations to focus on relevant semantic information.
- **Neural Inductive Matrix Completion**: Enhances prediction accuracy by reconstructing herb-target associations.
- **Performance Metrics**: Achieves superior results in AUROC, AUPRC, accuracy, precision, recall, and F1-score compared to baseline models.

## Key Contributions
1. Introduced an end-to-end deep learning framework tailored for herb-target association prediction.
2. Developed meta-path-based aggregations to capture complex relationships in heterogeneous networks.
3. Validated the model with real-world case studies and cross-validation experiments.

## Data Sources
The dataset integrates information from:
- **SymMap**  
- **ETCM**  
- **HERB**  
- **SIDER**  
- **DrugBank**

### Statistics
- **Nodes**: 4,584 (herbs, symptoms, diseases, targets, TCM symptoms)
- **Edges**: 200,750 (herb-target, herb-disease, herb-symptom, and more)

## Results
- **Performance**: HAN-HTA consistently outperforms existing models in multiple metrics.
- **Case Studies**: Demonstrated effectiveness in predicting targets for herbs like Artemisia annua (Qinghao) and Ginkgo biloba (Yinxing).

## Usage
1. **Prerequisites**: Ensure access to the relevant datasets (e.g., SymMap, ETCM) and required software libraries for deep learning and graph analysis.
2. **Implementation**: Follow the method described in the manuscript to build the heterogeneous network and train the HAN-HTA model.
3. **Evaluation**: Use cross-validation to assess the performance of the model on your dataset.

## Citation
None.

## Contact
For questions or further information, please contact the corresponding author.
