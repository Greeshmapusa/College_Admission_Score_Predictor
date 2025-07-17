#Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#Load the Dataset
data = pd.read_csv("College_Admission_Predict.csv")
# Display the first few rows
print(data.head())

#Data Preparation
data = data.drop(columns=["Serial No."])
data.rename(columns={"Chance of Admit ": "Chance"}, inplace=True)
# Check for missing values
print("\nMissing values in each column:\n", data.isnull().sum())

#Perform the Exploratory data analysis
plt.figure(figsize=(10, 7))
sns.heatmap(data.corr(), annot=True, cmap='Blues')
plt.title("Feature Correlation Matrix")
plt.show()

# Visualize relationships between GPA, GRE, TOEFL vs Admission Chance
sns.pairplot(data[['GRE Score', 'TOEFL Score', 'CGPA', 'Chance']])
plt.show()

#Feature Selection and splitting
features = data.drop("Chance", axis=1)
target = data["Chance"]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=1)

#Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("Intercept (Bias):", lr_model.intercept_)
print("Feature Coefficients:", list(zip(features.columns, lr_model.coef_)))

#Model Evaluation
predictions = lr_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("Mean Squared Error:", round(mse, 4))
print("R² Score:", round(r2, 4))

#Visualization
plt.figure(figsize=(6,6))
plt.scatter(y_test, predictions, color='green')
plt.plot([0,1], [0,1], linestyle='--', color='red')
plt.xlabel("Actual Chance")
plt.ylabel("Predicted Chance")
plt.title("Actual vs Predicted Admission Chances")
plt.grid(True)
plt.show()
