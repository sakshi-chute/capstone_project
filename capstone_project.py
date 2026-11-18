
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders
os.makedirs("dataset", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --------------------------------
# 1. Generate Student Dataset
# --------------------------------

np.random.seed(42)
n = 200

study_hours = np.random.randint(1, 9, n)
attendance = np.random.randint(50, 101, n)
previous_score = np.random.randint(35, 96, n)
assignment_score = np.random.randint(30, 101, n)
sleep_hours = np.random.randint(4, 10, n)

exam_score = (
    study_hours * 4
    + attendance * 0.20
    + previous_score * 0.20
    + assignment_score * 0.15
    + sleep_hours * 1.5
    + np.random.normal(0, 5, n)
)

exam_score = np.clip(exam_score, 0, 100)

df = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Previous_Score": previous_score,
    "Assignment_Score": assignment_score,
    "Sleep_Hours": sleep_hours,
    "Exam_Score": exam_score.round(2)
})

# Save dataset
df.to_csv("dataset/student_performance.csv", index=False)

print("Dataset created successfully!")
print(df.head())

# --------------------------------
# 2. Data Cleaning
# --------------------------------

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.drop_duplicates()

df = df.fillna(df.mean(numeric_only=True))

print("\nDataset after cleaning:")
print(df.info())

# --------------------------------
# 3. Exploratory Data Analysis
# --------------------------------

print("\nStatistical Summary:")
print(df.describe())

print("\nAverage Exam Score:")
print(df["Exam_Score"].mean())

print("\nHighest Exam Score:")
print(df["Exam_Score"].max())

print("\nLowest Exam Score:")
print(df["Exam_Score"].min())

# --------------------------------
# 4. Data Visualization
# --------------------------------

# Exam score distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Exam_Score"], bins=15, kde=True)
plt.title("Student Exam Score Distribution")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("outputs/performance_distribution.png")
plt.close()

# Study hours vs exam score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Study_Hours", y="Exam_Score")
plt.title("Study Hours vs Exam Score")
plt.tight_layout()
plt.savefig("outputs/study_hours_analysis.png")
plt.close()

# Attendance vs exam score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Attendance", y="Exam_Score")
plt.title("Attendance vs Exam Score")
plt.tight_layout()
plt.savefig("outputs/attendance_analysis.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Student Performance Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

print("\nAll charts saved in the outputs folder.")

# --------------------------------
# 5. Machine Learning Model
# --------------------------------

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Score",
        "Assignment_Score",
        "Sleep_Hours"
    ]
]

y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# --------------------------------
# 6. Model Evaluation
# --------------------------------

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nMachine Learning Model Results")
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# --------------------------------
# 7. Actual vs Predicted Chart
# --------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Student Performance")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# --------------------------------
# 8. Feature Importance
# --------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 5))
importance.plot(kind="bar")
plt.title("Factors Affecting Student Performance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# --------------------------------
# 9. Predict New Student Score
# --------------------------------

new_student = pd.DataFrame([{
    "Study_Hours": 6,
    "Attendance": 85,
    "Previous_Score": 75,
    "Assignment_Score": 80,
    "Sleep_Hours": 7
}])

predicted_score = model.predict(new_student)

print("\nNew Student Predicted Exam Score:")
print(round(predicted_score[0], 2))

print("\nCapstone Project Completed Successfully!")
