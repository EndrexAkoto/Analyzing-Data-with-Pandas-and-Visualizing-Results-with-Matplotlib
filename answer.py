import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset

# Load dataset (Iris dataset for this example)
try:
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    print("Dataset successfully loaded!")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check the data types and missing values
print("\nDataset Information:")
print(df.info())

print("\nCheck for missing values:")
print(df.isnull().sum())

# Clean the dataset (No missing values in this dataset)
# If there were missing values, we could use df.fillna() or df.dropna()

# Task 2: Basic Data Analysis

# Compute basic statistics of numerical columns
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Grouping by species and computing the mean of numerical columns
grouped_data = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped_data)

# Observations:
# Sepal and petal dimensions vary significantly across species.
# For example, setosa has smaller petal lengths compared to virginica and versicolor.

# Task 3: Data Visualization

# Line chart (Example: Trend over hypothetical time for petal length in each species)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x=range(len(df)), y="petal_length", hue="species")
plt.title("Petal Length Trend Across Species")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# Bar chart (Average petal length per species)
plt.figure(figsize=(8, 5))
grouped_data["petal_length"].plot(kind="bar", color=['skyblue', 'orange', 'green'])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram (Distribution of petal lengths)
plt.figure(figsize=(8, 5))
sns.histplot(df["petal_length"], bins=15, kde=True, color="purple")
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot (Sepal length vs. petal length)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sepal_length", y="petal_length", hue="species", palette="deep")
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
