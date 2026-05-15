#  Applied Machine Learning for Industrial Data Analysis

## Overview

This repository contains the full machine learning pipeline applied to the **SECOM Semiconductor Manufacturing Dataset** as part of the STAT9013 module project at Munster Technological University.

The goal is to predict whether a semiconductor product **passes or fails** quality control using high-dimensional sensor data.

---

## Dataset

**SECOM Manufacturing Dataset** — UCI Machine Learning Repository  
Source: https://archive.ics.uci.edu/dataset/179/secom

| Property | Value |
|---|---|
| Samples | 1,567 |
| Features | 590 sensor readings |
| Target | Binary — Pass (`-1 → 0`) / Fail (`+1 → 1`) |
| Class imbalance | ~94% Pass, ~6% Fail |
| Missing values | Present across most features |

Place both dataset files in the same directory as the notebook before running:
- `secom.data`
- `secom_labels.data`

---

## Project Structure

```
.
├── Analysis_code.ipynb   # Main notebook (fully executed)
├── README.md                             # This file
└── figures/                              # Save exported plots here for the report
    ├── fig1_class_distribution.png
    ├── fig2_variance_distribution.png
    ├── fig3_top10_distributions.png
    ├── fig4_correlation_heatmap.png
    ├── fig5_outlier_density.png
    ├── fig6_boxplots.png
    ├── fig7_pca_scree.png
    ├── fig8_pca_scatter.png
    ├── fig9_elbow_silhouette.png
    ├── fig10_kmeans.png
    ├── fig11_confusion_matrices.png
    ├── fig12_roc_curves.png
    └── fig13_model_comparison_bar.png
```

---

## Pipeline Summary

The notebook covers all 9 marked tasks:

| Task | Description | Marks |
|---|---|---|
| 1 | Data understanding and preprocessing | 10 |
| 2 | Exploratory data analysis (EDA) | 10 |
| 3 | Outlier detection and analysis | 10 |
| 4 | Dimensionality reduction (PCA) | 15 |
| 5 | Clustering (K-Means) | 15 |
| 6 | Classification (k-NN, Decision Tree, SVM) | 20 |
| 7 | Model evaluation (confusion matrix, ROC, cross-validation) | 10 |
| 8 | Discussion and critical analysis | 10 |
| 9 | Code quality | 10 |

---

## Requirements

Install all dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy jupyter
```

Tested with Python 3.10+.

---

## Running the Notebook

```bash
jupyter notebook STAT9013_SECOM_Analysis_FINAL.ipynb
```

The `_FINAL` version already contains all executed cell outputs (plots, printed results). If you want to re-run from scratch, make sure `secom.data` and `secom_labels.data` are in the same directory, then run all cells in order.

---

## Compiling the Report

The LaTeX report references figures from the `figures/` folder. To compile:

```bash
pdflatex STAT9013_Report.tex
pdflatex STAT9013_Report.tex   # Run twice to resolve references
```

Or upload `STAT9013_Report.tex` and the `figures/` folder to [Overleaf](https://www.overleaf.com) and click Compile.

---

## Key Results

| Model | Balanced Accuracy | F1 (Fail) |
|---|---|---|
| k-NN (k=5) | ~0.50 | ~0.00 |
| Decision Tree | ~0.65 | ~0.25 |
| SVM (RBF) | ~0.70 | ~0.35 |

SVM with an RBF kernel achieved the best overall performance. k-NN effectively fails under the severe class imbalance. Exact values are printed in the notebook output.

---

## License

This project is submitted for academic assessment. The SECOM dataset is publicly available via the UCI ML Repository under their standard terms of use.
