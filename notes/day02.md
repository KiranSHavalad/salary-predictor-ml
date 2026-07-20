# Day 02 - Exploratory Data Analysis (EDA)

## Name
Kiran Havalad

## Project
Smart Employee Salary Predictor

## Topic
Exploratory Data Analysis (EDA)

---

## 1. What is Exploratory Data Analysis (EDA)?

Exploratory Data Analysis (EDA) is the process of understanding and examining a dataset before building a Machine Learning model. It helps us understand the structure of the data, identify missing values, detect duplicate records, find outliers, and understand the relationships between different features. EDA is an important first step because the quality of the model depends on the quality of the data.

---

## 2. Why do we use `df.head()`?

The `df.head()` function displays the first five rows of the dataset. It helps us verify that the dataset has been loaded correctly and allows us to quickly inspect the column names and sample data.

---

## 3. What does `df.info()` tell us?

The `df.info()` function provides a summary of the dataset. It shows:
- Total number of rows and columns
- Column names
- Data types of each column
- Number of non-null values
- Memory usage

This helps us identify incorrect data types and missing values.

---

## 4. Why do we check for missing values?

Machine Learning models cannot work properly with missing data. Missing values can reduce model accuracy or cause errors during training. Therefore, identifying and handling missing values is an essential preprocessing step.

---

## 5. What is the difference between a Feature and a Target Variable?

A **Feature (X)** is an input variable used by the Machine Learning model to make predictions.

Examples from our project:
- work_year
- experience_level
- employment_type
- job_title
- remote_ratio
- company_location
- company_size

A **Target Variable (Y)** is the output that the model tries to predict.

For this project:
- `salary_in_usd`

---

## Key Learnings

Today I learned:

- The importance of understanding the dataset before training a model.
- How to inspect the dataset using Pandas.
- How to check the size of the dataset.
- How to identify missing values and duplicate records.
- How to distinguish between features and the target variable.
- Why EDA is the foundation of every Machine Learning project.

---

## Commands Practiced

```python
import pandas as pd

df = pd.read_csv("../data/raw/salaries.csv")

df.head()

df.shape

df.columns

df.info()

df.describe()

df.isnull().sum()

df.duplicated().sum()

df.nunique()
```

---

## Reflection

Today I understood that Machine Learning is not only about training models. The first step is to understand the data thoroughly. Performing EDA helps identify potential issues early, ensuring better model performance and more reliable predictions.