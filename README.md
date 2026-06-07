# AIML-Internship-Task5
# Task 5: Classification with Decision Trees and Random Forests

This repository contains the official implementation of **Task 5: Tree-Based Models & Ensemble Learning** as part of the AI & ML Internship. The project evaluates the structural and predictive differences between standard unconstrained Decision Trees and ensemble architectures like Random Forests.

---

## 🚀 Live Interactive Notebook
GitHub can occasionally experience rendering timeouts with large model plots or complex data frames. To view the complete verified dashboard, graphs, and performance scores seamlessly, use the link below:

👉 **[Click Here to View the Project on NBViewer](https://nbviewer.org/)**

---

## 📊 Project Architecture & Implementation

1. **Dataset Pipeline:** Leveraged structured diagnostic features from the built-in wine attribute dataset, mapped onto a binary classification target to align with evaluation criteria.
2. **Overfitting Analysis & Pruning:** Built a baseline unconstrained Decision Tree showing standard variance leaks ($100\%$ training alignment), and successfully resolved it via explicit cost-complexity control parameters (`max_depth=3`).
3. **Ensemble Modeling:** Trained a multi-estimator `RandomForestClassifier` with $100$ parallel estimators to stabilize out-of-sample predictions.
4. **Validation Strategy:** Implemented a robust $5$-Fold Stratified Cross-Validation loop to ensure structural generalization across different data splits.

---

## 📈 Official Task 5 Performance Report

The models generated highly stable and consistent predictive results upon execution:
* **Unconstrained Tree Train Accuracy:** `1.0000` (Perfect training baseline / Overfitting Indicator)
* **Pruned Decision Tree Test Accuracy:** `0.9444` 
* **Random Forest Model Test Accuracy:** **`0.9722` (Ensemble Performance Boost)**
* **5-Fold Random Forest CV Mean Score:** **`0.9887` (Extremely Stable Across Folds)**

---

## 🛠️ Tools & Technologies Used
* **Language:** Python 3
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 🎯 Validation Dashboards
Below is the structural breakdown and attribute contribution charts generated during the evaluation step:


