<div align="center">

# 🏠 House Price Regression & ML Classification Analysis

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)

**A dual-domain machine learning project exploring regression techniques for house price prediction and classification models for medical diagnosis — comparing performance across Linear, Polynomial, Logistic Regression, KNN, SVM, and Neural Networks.**

[📓 View Notebook](./project.ipynb) · [🏠 House Price Dataset](./House%20Price%20India.csv) · [🧬 Diagnosis Dataset](./data.csv) · [🐛 Report Bug](https://github.com/ahmedayman2825/House-Price-Regression-Analysis/issues)

</div>

---

## 📌 Overview

This project is split into two interconnected analyses built inside a single Jupyter notebook:

**Part 1 — House Price Regression (India Dataset)**  
Predicts residential property prices across India using **14,619 records and 22 features**. The analysis covers the full pipeline from correlation analysis and outlier removal, through Simple Linear, Multiple Linear, and Polynomial Regression models, with direct performance comparisons across all three.

**Part 2 — Medical Diagnosis Classification (Breast Cancer Wisconsin Dataset)**  
Predicts whether a tumor is malignant or benign using **569 records and 30 clinical features**. Five classification models are trained, tuned, and rigorously compared: Logistic Regression, K-Nearest Neighbors, SVM (with three kernels), and a Deep Neural Network — with hyperparameter optimization via GridSearchCV and ROC-AUC curve comparison.

---

## 🖼️ Sample Outputs

### 📊 Correlation Heatmap

<img width="1168" height="942" alt="image" src="https://github.com/user-attachments/assets/69f54646-5437-4ba3-9ecf-cae0ab1ceeba" />

> _Screenshot of the feature correlation heatmap (House Price dataset)_

---

### 📉 Price Distribution
<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/d3e4cd6f-d381-438a-b8c7-a0b862b6b359" />

> _Distribution of house prices with KDE overlay_

---

### 📈 Simple Linear Regression
<img width="678" height="547" alt="image" src="https://github.com/user-attachments/assets/064aa76f-a4d5-49d8-8d1b-81ad415927ac" />

> _Scatter plot of actual vs predicted prices using the single most-correlated feature_

---

### 📈 Multiple Linear Regression — Results
<img width="678" height="547" alt="image" src="https://github.com/user-attachments/assets/4822686d-ae4f-40e9-81b6-10e5ef011de7" />

> _Console output showing R², MSE, and improvement over Simple Linear Regression_

---

### 📈 Polynomial Regression (Degree 2 / 3 / 4)
<img width="678" height="547" alt="image" src="https://github.com/user-attachments/assets/8544a07b-ddf1-4ab2-95b9-acdb40f27137" />
<img width="678" height="547" alt="image" src="https://github.com/user-attachments/assets/cb95bfc2-3de2-496e-b5e2-c779e3b4d29f" />
<img width="678" height="547" alt="image" src="https://github.com/user-attachments/assets/c02f3642-ca8e-491e-a08c-f38dde8ecc29" />

> _Side-by-side curve fits for degree 2, 3, and 4 polynomial models_

---

### 🧬 Logistic Regression — Confusion Matrix
<img width="558" height="455" alt="image" src="https://github.com/user-attachments/assets/735989f0-5640-4f04-8785-39ca83ee3dfd" />

> _Confusion matrix heatmap (Benign vs Malignant predictions)_

---

### 🧠 Neural Network — Loss Curve
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/92b09b79-94cd-483d-8b37-b9b23b7ad426" />

> _Training vs validation loss over 50 epochs_

---

### 📊 Full Model Comparison Table
<img width="551" height="155" alt="image" src="https://github.com/user-attachments/assets/91c28183-993a-40d8-be7f-c4f0a8446efb" />

> _Side-by-side Accuracy, Precision, Recall, F1 for all classifiers_

---

### 📉 ROC-AUC Curves
<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/ea1aa4ff-7bf6-4a85-b5ed-58db287a114d" />

> _Overlaid ROC curves for SVM (all kernels) and Neural Network_

---

## ✨ Features

### 🏠 Part 1 — House Price Regression

| Step | Description |
|---|---|
| **EDA & Correlation Analysis** | Full correlation heatmap; top-5 features selected by correlation with `Price` |
| **Outlier Removal** | IQR-based outlier filtering applied per feature |
| **Feature Scaling** | `StandardScaler` applied to train/test splits |
| **Simple Linear Regression** | Single most-correlated feature; scatter plot with fit line |
| **Multiple Linear Regression** | Top-5 correlated features; R² and MSE improvement over simple model |
| **Polynomial Regression** | Degrees 2, 3, and 4 tested; visual curve fits compared |
| **Model Comparison** | Side-by-side R² and MSE across all regression approaches |

### 🧬 Part 2 — Classification

| Step | Description |
|---|---|
| **Data Preprocessing** | Label encoding, 60/20/20 stratified train/val/test split |
| **Logistic Regression** | Baseline classifier with confusion matrix visualization |
| **K-Nearest Neighbors** | Tuned `k`; 5-Fold cross-validation for robust accuracy estimate |
| **Support Vector Machine** | Three kernels compared: `linear`, `poly`, `rbf` |
| **Neural Network** | 3-layer Keras MLP (16→8→1); binary cross-entropy loss; 50 epochs |
| **Hyperparameter Tuning** | GridSearchCV for SVM, Logistic Regression, and KNN |
| **Full Model Comparison** | Accuracy, Precision, Recall, F1-score table across all classifiers |
| **ROC-AUC Curves** | Overlaid ROC curves for SVM kernels and Neural Network |

---

## 🗂️ Datasets

### House Price India (`House Price India.csv`) — 14,619 rows × 23 columns

| Feature | Type | Description |
|---|---|---|
| `number of bedrooms` | int | Total bedrooms |
| `number of bathrooms` | float | Total bathrooms |
| `living area` | int | Living area in sq ft |
| `lot area` | int | Total lot area in sq ft |
| `number of floors` | float | Number of floors |
| `waterfront present` | int | Waterfront flag (0/1) |
| `grade of the house` | int | Overall grade assigned to the property |
| `condition of the house` | int | Overall condition rating |
| `Built Year` / `Renovation Year` | int | Year built and last renovated |
| `Number of schools nearby` | int | Nearby schools count |
| `Distance from the airport` | int | Distance to nearest airport |
| `Price` | int | **Target variable** — sale price |

### Breast Cancer Wisconsin (`data.csv`) — 569 rows × 33 columns

| Feature | Type | Description |
|---|---|---|
| `diagnosis` | str (M/B) | **Target variable** — Malignant or Benign |
| `radius_mean`, `texture_mean`, … | float | 30 clinical measurements of cell nuclei |

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | latest | Data loading, cleaning, and manipulation |
| `numpy` | latest | Numerical operations |
| `matplotlib` / `seaborn` | latest | Visualizations, heatmaps, ROC curves |
| `scikit-learn` | latest | All ML models, scaling, cross-validation, GridSearchCV |
| `tensorflow` / `keras` | latest | Neural network architecture and training |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ahmedayman2825/House-Price-Regression-Analysis.git
cd House-Price-Regression-Analysis

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

### Run the Notebook

```bash
jupyter notebook project.ipynb
```

Run all cells top-to-bottom (`Kernel → Restart & Run All`) to reproduce the full analysis.

> **Note:** The Neural Network section requires TensorFlow. If you only want to run the regression and classical ML sections, TensorFlow can be skipped.

---

## 📁 Repository Structure

```
House-Price-Regression-Analysis/
│
├── project.ipynb              # Main notebook — all models and analysis
├── House Price India.csv      # Regression dataset (14,619 records, 23 features)
├── data.csv                   # Classification dataset (569 records, 33 features)
└── README.md                  # Project documentation
```

---

## 📈 Analysis Walkthrough

### Part 1 — Regression Pipeline

```
Load 'House Price India.csv'
    → Correlation Heatmap (22 features vs Price)
    → Select top-5 correlated features
    → IQR Outlier Removal per feature
    → Train/Test Split (80/20) + StandardScaler
    → Simple Linear Regression   (1 feature)
    → Multiple Linear Regression (5 features)
    → Polynomial Regression      (degree 2, 3, 4)
    → Compare R² and MSE across all models
```

### Part 2 — Classification Pipeline

```
Load 'data.csv' (Breast Cancer Wisconsin)
    → Label Encode diagnosis (M=1, B=0)
    → Stratified Split: 60% train / 20% val / 20% test
    → StandardScaler
    → Logistic Regression       → Confusion Matrix
    → KNN (k=7)                 → 5-Fold Cross-Validation
    → SVM (linear, poly, rbf)   → Per-kernel metrics
    → Neural Network (MLP)      → Loss/accuracy curves
    → GridSearchCV tuning (SVM, LR, KNN)
    → Model Comparison Table    (Accuracy, Precision, Recall, F1)
    → ROC-AUC Curves            (all models overlaid)
```

---

## 🔭 Future Improvements

- [ ] Add **Ridge and Lasso regression** with regularization strength tuning for the house price model
- [ ] Introduce **feature engineering** — price per sq ft, age of property, renovation flag
- [ ] Extend the Neural Network with **dropout layers** and **early stopping** to reduce overfitting
- [ ] Add a **Gradient Boosting model** (XGBoost / LightGBM) to the classification comparison
- [ ] Build a **Streamlit web app** allowing users to input house features and get a real-time price prediction
- [ ] Apply **SHAP values** for model explainability — understanding which features drive each prediction

---

## 👤 Author

**Ahmed Ayman**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/eng-ahmed-ayman/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ahmedayman2825)

**Ahmed Tawfik**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ah-mo-tawfik/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/tAwFiK2005)

**Ahmed Abdelwahab**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmed-abdalwahab/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ahmedabdalwahab)

**Ahmed Emad**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmed-zamel09/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/AhmedZamel09) 

---

<div align="center">
  <sub>If you found this project useful, consider giving it a ⭐</sub>
</div>
