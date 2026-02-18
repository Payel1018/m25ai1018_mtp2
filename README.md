# m25ai1018_mtp2
M Tech Project

**Disease Prediction Using Lifestyle Data (Multi-Class Classification)**
---------------------------------------------------------------------------

A Machine Learning project that predicts multiple disease types based on lifestyle-related factors such as BMI, smoking habits, physical activity, sleep, and diet. The project compares multiple classification models and identifies the best-performing one.

**Project Objective**

The goal of this project is to:

Predict disease categories using lifestyle data
Implement multiple machine learning classification models
Compare model performance using appropriate evaluation metrics
Identify the most effective model for multi-class disease prediction

**Problem Type**

Multi-Class Classification

**Example disease classes:**

Cardiovascular - blood_pressure, heart_rate, cholesterol
Metabolic - glucose, insulin
Lifestyle - smoking, alcohol, diet, exercise, sleep
Mental health - stress_level, mental_health_score
Socioeconomic - income, education_level, healthcare_access
Genetic - family_history
Behavioral - screen_time, device_usage

**Dataset**

The dataset contains lifestyle attributes such as:

Age, Gender, BMI, Smoking status, Alcohol consumption, Physical activity level, Sleep duration, Stress level

**Technologies Used**

Python
Pandas
NumPy
Scikit-learn
Google Colab Notebook

**Project Workflow**
1. Data Preprocessing

Handling missing values
Encoding categorical variables
Label encoding target variable

2. Train-Test Split

80% training
20% testing

3. Models Implemented

Logistic Regression
Naive Bayes
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Decision Tree
Random Forest

**Evaluation Metrics**

For proper multi-class evaluation, the following metrics were used:
Accuracy
Precision (Macro Average)
Recall (Macro Average)
F1-Score (Macro Average)
Confusion Matrix 
