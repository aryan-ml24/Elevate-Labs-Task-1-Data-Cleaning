# Elevate Labs AI & ML Internship - Task 1

## Task: Data Cleaning and Preprocessing

### Objective
The objective of this task is to clean and preprocess the Titanic dataset for machine learning.

## Dataset
Titanic Dataset

## Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Tasks Performed

### 1. Dataset Loading
Loaded the Titanic dataset using Pandas and Seaborn.

### 2. Data Inspection
Checked:
- Dataset shape
- Column names
- Data types
- Missing values

### 3. Missing Value Handling
- Filled missing `age` values using the median.
- Filled missing `embarked` and `embark_town` values using the mode.
- Removed the `deck` column because it contained a large number of missing values.

### 4. Categorical Encoding
Applied One-Hot Encoding to categorical features such as `sex` and `embarked`.

### 5. Feature Scaling
Used StandardScaler to standardize numerical features.

### 6. Outlier Detection
Used boxplots and the IQR method to identify potential outliers in `age` and `fare`.

### 7. Outlier Handling
Applied IQR-based capping to the `age` and `fare` features.

### 8. Data Visualization
Created boxplots and histograms to understand feature distributions and identify potential outliers.

## Conclusion
The Titanic dataset was cleaned and preprocessed by handling missing values, encoding categorical features, standardizing numerical features, detecting potential outliers, and visualizing the data.

## Author
Elevate Labs AI & ML Internship
