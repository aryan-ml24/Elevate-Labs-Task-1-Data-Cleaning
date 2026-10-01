# ============================================================
# ELEVATE LABS AI & ML INTERNSHIP - TASK 1
# Data Cleaning and Preprocessing - Titanic Dataset
# ============================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Libraries imported successfully!")


# ============================================================
# 2. Load Titanic Dataset
# ============================================================

# Titanic dataset from GitHub
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

print("\nDataset loaded successfully!")
print("Dataset Shape:", df.shape)


# ============================================================
# 3. Display First 5 Rows
# ============================================================

print("\nFirst 5 rows:")
display(df.head())


# ============================================================
# 4. Basic Information
# ============================================================

print("\nDataset Information:")
df.info()


# ============================================================
# 5. Statistical Summary
# ============================================================

print("\nStatistical Summary:")
display(df.describe())


# ============================================================
# 6. Check Missing Values
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 7. Check Duplicate Rows
# ============================================================

print("\nNumber of Duplicate Rows:", df.duplicated().sum())


# ============================================================
# 8. Remove Duplicate Rows
# ============================================================

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)


# ============================================================
# 9. Handle Missing Values
# ============================================================

# Fill Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin because it contains many missing values
df = df.drop(columns=["Cabin"])

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# 10. Check Data Types
# ============================================================

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 11. Convert Categorical Data into Numerical Data
# ============================================================

# Convert Sex into numerical values
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Convert Embarked using One-Hot Encoding
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

print("\nDataset after encoding:")
display(df.head())


# ============================================================
# 12. Convert Boolean Columns to Integer
# ============================================================

for column in df.select_dtypes(include="bool").columns:
    df[column] = df[column].astype(int)


# ============================================================
# 13. Select Features and Target
# ============================================================

X = df.drop(columns=["Survived"])
y = df["Survived"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Survived")


# ============================================================
# 14. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ============================================================
# 15. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")


# ============================================================
# 16. Convert Scaled Data Back to DataFrame
# ============================================================

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)

print("\nScaled Training Data:")
display(X_train_scaled.head())


# ============================================================
# 17. Visualization - Missing Values
# ============================================================

plt.figure(figsize=(10, 5))

sns.heatmap(
    df.isnull(),
    cbar=False
)

plt.title("Missing Values After Data Cleaning")
plt.show()


# ============================================================
# 18. Visualization - Survival Count
# ============================================================

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Survived"
)

plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()


# ============================================================
# 19. Visualization - Age Distribution
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Age"],
    bins=30,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# ============================================================
# 20. Correlation Heatmap
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()


# ============================================================
# 21. Final Dataset Information
# ============================================================

print("\nFinal Dataset Shape:", df.shape)

print("\nFinal Dataset:")
display(df.head())

print("\nFinal Missing Values:")
print(df.isnull().sum())

print("\nFinal Data Types:")
print(df.dtypes)


# ============================================================
# 22. Save Cleaned Dataset
# ============================================================

df.to_csv("titanic_cleaned.csv", index=False)

print("\nCleaned dataset saved as: titanic_cleaned.csv")

print("\n================================================")
print("DATA CLEANING AND PREPROCESSING COMPLETED!")
print("================================================")
