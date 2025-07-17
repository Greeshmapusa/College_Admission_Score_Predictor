# College_Admission_Score_Predictor
A machine learning mini-project that predicts the probability of a student getting admitted into a graduate program based on academic performance and research background using Linear Regression.


# Problem Statement

The goal is to predict the Chance of Admission (on a scale of 0 to 1) based on the following features:

- GRE Score
- TOEFL Score
- University Rating
- Statement of Purpose (SOP)
- Letter of Recommendation (LOR)
- CGPA
- Research Experience (0 or 1)


# Dataset

- Source: [Graduate Admissions Dataset – Kaggle](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)
- Size: 500 records
- Format: CSV

# Algorithm Used

- **Linear Regression** from scikit-learn


#  Tech Stack

- Python
- Pandas & NumPy (Data Processing)
- Matplotlib & Seaborn (Visualization)
- Scikit-learn (Model Building & Evaluation)


# Model Evaluation

- Mean Squared Error (MSE)
- R² Score

These metrics were used to evaluate how well the model predicts the admission probability.


# Workflow

1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Train-Test Split
4. Linear Regression Model Training
5. Evaluation using MSE and R²
6. Prediction on new student data


# Example Prediction

python
# Sample input: [GRE, TOEFL, Univ Rating, SOP, LOR, CGPA, Research]
sample_input = [[322, 112, 4, 4.5, 4, 9.1, 1]]
model.predict(sample_input)
# Output
Predicted Admission Chance: 0.82

