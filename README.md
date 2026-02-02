# 💎 InsurRisk AI: Enterprise Claim Forecasting System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Enterprise-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/sklearn-Ensemble-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**InsurRisk AI** is a next-generation actuarial forecasting engine designed to predict insurance claim costs and classify risk severity in real-time. 

Built with a **Glassmorphism UI**, it leverages an **Ensemble Voting Regressor** (Random Forest + Gradient Boosting + Linear Regression) to achieve high-fidelity predictions, offering explainability via **SHAP** for regulatory compliance.

---

## 🚀 Key Features

### 🧠 Advanced AI Core
* **Ensemble Modeling:** Combines three algorithms (`RandomForest`, `GradientBoosting`, `LinearRegression`) via a `VotingRegressor` to minimize variance and improve R² accuracy.
* **Dynamic Risk Tiering:** Automatically classifies applicants into **Low**, **Medium**, or **High** risk tiers based on live quantile analysis of the dataset.

### 🎨 Next-Gen UI/UX
* **Glassmorphism Design:** Custom CSS implementation featuring translucent cards, background blurs (`backdrop-filter`), and a mesh gradient aesthetic.
* **Interactive Analytics:** deeply integrated **Plotly** charts with transparent backgrounds for market benchmarking and risk visualization.

### 🔍 Explainable AI (XAI)
* **SHAP Integration:** embedds Shapley Additive Explanations to visualize exactly *why* a specific price was predicted (e.g., "Smoker status added +$12k, BMI added +$4k").

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Custom CSS/HTML injection)
* **Machine Learning:** Scikit-Learn (Pipelines, ColumnTransformer, VotingRegressor)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly Express, Plotly Graph Objects, SHAP (JS wrappers)

---

## 📦 Installation & Setup

### Prerequisites
* Python 3.8+
* pip

### 1. Clone the Repository
```bash
git clone [https://github.com/sumitshresht/InsurRisk-AI.git](https://github.com/sumitshresht/InsurRisk-AI.git)
cd insurrisk-ai

