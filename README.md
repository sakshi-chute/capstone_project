# Student Performance Analysis and Prediction Using Python and Machine Learning

## Project Overview

The Student Performance Analysis and Prediction System is a Data Science Capstone Project developed using Python and Machine Learning.

The project analyzes student academic performance based on different factors such as study hours, attendance, previous examination scores, assignment scores, and sleep hours.

It uses Python libraries to perform data cleaning, data analysis, data visualization, and predictive modelling. A Random Forest Regression algorithm is implemented to predict student examination scores.

The main objective is to demonstrate the practical application of Data Science and Machine Learning concepts acquired during the internship.

## Objectives

- Analyze student academic performance.
- Generate and prepare a student performance dataset.
- Perform data cleaning and preprocessing.
- Use Pandas and NumPy for data analysis.
- Create visualizations using Matplotlib and Seaborn.
- Identify relationships between student performance factors.
- Develop a Machine Learning prediction model.
- Evaluate model performance using suitable metrics.
- Generate meaningful insights from the analysis.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical calculations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| VS Code | Development environment |

## Project Structure

```text
Final_Data_Science_Capstone/
│
├── dataset/
│   └── student_performance.csv
│
├── outputs/
│   ├── performance_distribution.png
│   ├── study_hours_analysis.png
│   ├── attendance_analysis.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
│
├── capstone_project.py
├── README.md
└── requirements.txt
```

## Dataset Description

The project uses a synthetically generated dataset containing 200 student records.

The dataset includes the following attributes:

- Study_Hours
- Attendance
- Previous_Score
- Assignment_Score
- Sleep_Hours
- Exam_Score

The dataset is generated using NumPy and stored in CSV format using Pandas.

**Note:** The dataset is created for educational and demonstration purposes. It does not contain real student records.

## Project Workflow

1. Generate the student performance dataset.
2. Load the dataset using Pandas.
3. Check and clean the data.
4. Perform exploratory data analysis.
5. Calculate statistical measures.
6. Generate charts and visualizations.
7. Split the dataset into training and testing sets.
8. Train a Random Forest Regression model.
9. Predict examination scores.
10. Evaluate the model using performance metrics.

## Data Visualization

The project generates the following visualizations:

1. **Performance Distribution:** Displays the distribution of student examination scores.
2. **Study Hours Analysis:** Shows the relationship between study hours and examination scores.
3. **Attendance Analysis:** Examines attendance in relation to examination performance.
4. **Correlation Heatmap:** Displays correlations among numerical variables.
5. **Actual vs Predicted:** Compares actual examination scores with predicted scores.
6. **Feature Importance:** Displays the relative importance of input features in the trained model.

All generated charts are saved in the `outputs` folder.

## Machine Learning Algorithm

**Random Forest Regression** is used to predict student examination scores.

The input features are:

- Study Hours
- Attendance
- Previous Score
- Assignment Score
- Sleep Hours

The target variable is `Exam_Score`.

The dataset is divided into:
- 80% training data
- 20% testing data

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-Squared Score

## Installation and Setup

### Step 1: Install Python

Download and install Python from:

https://www.python.org/downloads/

### Step 2: Install Required Libraries

Open the VS Code terminal and execute:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 3: Run the Project

Open the project folder in Visual Studio Code.

Execute the following command:

```bash
python capstone_project.py
```

### Step 4: View Results

After successful execution:

- The student dataset is generated in the `dataset` folder.
- Analysis results are displayed in the terminal.
- Machine Learning predictions are generated.
- Visualization images are saved in the `outputs` folder.

## Expected Outcomes

The project demonstrates:

- Practical understanding of Python programming.
- Data manipulation using Pandas.
- Numerical operations using NumPy.
- Data visualization using Matplotlib and Seaborn.
- Exploratory Data Analysis techniques.
- Machine Learning model development.
- Model evaluation and prediction.
- Interpretation of analytical results.

## Learning Outcomes

Through this capstone project, I gained practical experience in:

1. Understanding the complete Data Science workflow.
2. Working with structured datasets.
3. Cleaning and preprocessing data.
4. Performing statistical analysis.
5. Creating meaningful visualizations.
6. Training and testing Machine Learning models.
7. Evaluating predictive models.
8. Converting data into meaningful insights.

## Future Enhancements

- Use real-world student academic datasets.
- Add more academic and behavioural features.
- Compare different Machine Learning algorithms.
- Develop an interactive dashboard.
- Create a web application for student score prediction.
- Improve model performance through hyperparameter tuning.

## Conclusion

This project demonstrates an end-to-end Data Science workflow using Python and Machine Learning. It combines data generation, data cleaning, analysis, visualization, model training, and prediction in a single project.

It helped develop practical knowledge of important Data Science tools and libraries and provided hands-on experience in building a Machine Learning solution.

## Author

**Name:**Sakshi Girirdhar Chute

**Course:** MCA

**Project:** Final Data Science Capstone Project

**Domain:** Data Science and Machine Learning
