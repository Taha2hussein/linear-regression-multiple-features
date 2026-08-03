# 🚗 Car Selling Price Prediction

A Machine Learning project for predicting used car prices using multiple regression algorithms.

---

## 📌 Project Overview

This project predicts the selling price of used cars based on their specifications.

The project includes:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Model Comparison
- Model Evaluation

---

## 📂 Dataset

**Source**

Car Details Dataset (Kaggle)

Target Variable

- Price

---

## 📊 Exploratory Data Analysis

Performed:

- Missing Values Analysis
- Duplicate Detection
- Outlier Detection
- Correlation Analysis
- Histograms
- Boxplots
- Scatter Plots

---

## 🤖 Machine Learning Models

The following regression algorithms were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- SGD Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regressor

---

## 🏆 Best Model

Gradient Boosting Regressor

Performance

| Metric | Value |
|--------|--------:|
| R² | **0.8775** |
| MAE | 305,512 |
| RMSE | 925,043 |

Gradient Boosting achieved the best performance because it effectively captured nonlinear relationships while maintaining strong generalization.

---

## 📁 Project Structure

```text
linear-regression-multiple-features
│
├── data
│   └── raw
│
├── outputs
│
├── src
│   ├── config
│   ├── data
│   ├── models
│   ├── utils
│   └── visualization
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Taha2hussein/linear-regression-multiple-features.git

cd linear-regression-multiple-features

pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python src/main.py
```

---

## 📈 Results

The project compares all implemented regression algorithms and automatically selects the best-performing model based on the evaluation metrics.

---

## 🛠 Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 🚀 Future Improvements

- Hyperparameter Optimization
- Feature Selection
- Cross Validation
- XGBoost
- LightGBM
- CatBoost

---

## 👨‍💻 Author

**Taha Hussein**

Senior iOS Engineer | Machine Learning Engineer
