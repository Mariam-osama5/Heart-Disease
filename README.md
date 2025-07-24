# 🧠 Heart Disease Prediction - Full Machine Learning Pipeline

This project is a comprehensive end-to-end machine learning pipeline designed to analyze and predict the risk of heart disease using the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease).

---

## Project Goals

- ✅ Clean and preprocess raw medical data  
- ✅ Perform exploratory data analysis (EDA)  
- ✅ Apply dimensionality reduction using PCA  
- ✅ Select the most important features  
- ✅ Train and evaluate classification models  
- ✅ Use clustering to discover patterns in data  
- ✅ Tune model hyperparameters  
- ✅ Export the best model for deployment 

---

## ML Models Used

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  

### Metrics:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

---

## Dimensionality Reduction

- Applied **PCA** to reduce noise and retain 95% variance.

---

## Feature Selection

- Feature Importance using Random Forest  
- Recursive Feature Elimination (RFE)  
- Chi-Square Test  

---

## Clustering

- K-Means (with Elbow Method)  
- Hierarchical Clustering (Dendrogram)

---

## Hyperparameter Tuning

- Used `GridSearchCV` & `RandomizedSearchCV` to optimize model performance.

---

## Model Deployment

- Best model saved as `.pkl` using `joblib`  
- Streamlit web interface created for prediction
