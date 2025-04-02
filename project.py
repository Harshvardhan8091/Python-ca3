import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, norm

# Load the dataset (using only first 10,000 rows)
df = pd.read_csv("vgsales.csv", nrows=10000)  # Replace with actual file name

# Objective 1: Data Understanding and Cleaning
print(df.head())  # Display first few rows
print(df.dtypes)  # Display data types
print(df.isnull().sum())  # Check missing values
df.fillna(0, inplace=True)  # Handle missing values

def categorize_sales(global_sales, threshold=10):
    return "High Sales" if global_sales >= threshold else "Low Sales"
df["Sales_Category"] = df["Global_Sales"].apply(lambda x: categorize_sales(x, 10))

# Objective 2: Data Exploration and Manipulation
genre_counts = df["Genre"].value_counts()
publisher_sales = df.groupby("Publisher")["Global_Sales"].sum()
platform_sales = df.groupby("Platform")["Global_Sales"].sum()
average_sales_per_year = df.groupby("Year")["Global_Sales"].mean()

total_sales = df["Global_Sales"].sum()
print("Total Global Sales:", total_sales)

# Objective 3: Data Visualization
plt.figure(figsize=(10, 5))
df["Genre"].value_counts().plot(kind="bar")
plt.xlabel("Genre")
plt.ylabel("Number of Games")
plt.title("Number of Games per Genre")
plt.show()

sales_trend = df.groupby("Year")["Global_Sales"].sum()
sales_trend.plot(kind="line", figsize=(10, 5))
plt.xlabel("Year")
plt.ylabel("Total Sales (millions)")
plt.title("Sales Trend Over Time")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Sales Correlation Heatmap")
plt.show()

# Objective 4: Statistical Analysis
print(f"Mean Sales: {df['Global_Sales'].mean():.2f}")
print(f"Median Sales: {df['Global_Sales'].median():.2f}")
print(f"Standard Deviation: {df['Global_Sales'].std():.2f}")

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Global_Sales"])
plt.title("Outlier Detection in Global Sales")
plt.show()

genre1_sales = df[df["Genre"] == "Action"]["Global_Sales"]
genre2_sales = df[df["Genre"] == "Sports"]["Global_Sales"]
t_stat, p_value = ttest_ind(genre1_sales, genre2_sales, nan_policy='omit')
print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.5f}")

# Objective 5: Probability Distribution and Hypothesis Testing
sales_data = df["Global_Sales"].dropna()
mu, sigma = norm.fit(sales_data)
plt.figure(figsize=(8, 5))
sns.histplot(sales_data, kde=True, bins=30, stat="density")
plt.title("Normal Distribution Fit for Global Sales")
plt.xlabel("Global Sales (millions)")
plt.ylabel("Density")
plt.show()
