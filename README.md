# 🩺 Diabetes Prediction System — Machine Learning Project  

![Diabetes Prediction Project](https://img.shields.io/badge/Diabetes-Prediction-blue?style=for-the-badge&logo=python&logoColor=white)

---

<p align="center">
  <img src="Diabetes_png.png" alt="Project Banner" width="100%" >
</p>

---

A **Machine Learning project** that predicts whether a person is likely to have **Diabetes** based on key health indicators such as glucose level, BMI, insulin, and age.  
This project demonstrates a complete **end-to-end ML workflow** — from data preprocessing to model training and evaluation using **Python** and **Scikit-learn**.

---

## 📁 Project Overview

The **PIMA Indians Diabetes Dataset** from the UCI repository is used in this project.  
It aims to build a **classification model** that can accurately predict diabetic conditions based on patient data.

### Dataset Features:
- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  
- Outcome (0 = Non-Diabetic, 1 = Diabetic)

---

## 🎯 Project Objectives

1. Load and clean the dataset.  
2. Handle missing or zero values.  
3. Scale numerical features for model stability.  
4. Train the **Logistic Regression model**.  
5. Perform **Hyperparameter Tuning** using GridSearchCV.  
6. Evaluate model performance using accuracy, confusion matrix, and classification report.  
7. Save the trained model using **Pickle**.  
8. Integrate model with a simple **Streamlit Web App** for real-time prediction.

---

## ⚙️ Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python |
| **Libraries** | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |
| **Model** | Logistic Regression |
| **Deployment Tool** | Streamlit |
| **IDE** | Jupyter Notebook / VS Code |

---

## 🧠 Machine Learning Concepts Used

- Data Cleaning & Feature Scaling  
- Logistic Regression  
- Train-Test Split  
- Hyperparameter Tuning (`GridSearchCV`)  
- Confusion Matrix & ROC Curve  
- Model Saving with Pickle  
- Streamlit Integration  

---

## 📊 Model Evaluation Metrics

| Metric | Description |
|---------|--------------|
| **Accuracy** | Measures correct predictions |
| **Precision & Recall** | Evaluate false positives/negatives |
| **F1 Score** | Balance between precision and recall |
| **Confusion Matrix** | Visual representation of prediction classes |

---

## 🧩 Key Code Snippets

### ▶️ Model Training
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
```
## 🧾 Conclusion

This **Diabetes Prediction System** successfully applies **Machine Learning** to assist in early detection of diabetes using health indicators such as glucose, BMI, and age.  
By leveraging **Logistic Regression** and robust **data preprocessing**, the model achieves reliable predictive performance and provides valuable insights into key risk factors.  

The project demonstrates how **data-driven healthcare** can support timely diagnosis and prevention strategies.  
It also serves as a strong foundation for future advancements such as integrating:
- **Advanced ML models** like Random Forest or XGBoost  
- **Real-time web deployment** using Streamlit or Flask  
- **Database integration** for patient management and reporting  

Overall, this project emphasizes the potential of **AI-powered predictive systems** in improving healthcare outcomes and aiding clinical decision-making.

