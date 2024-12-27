import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("Mall_Customers.csv")

# Function to categorize income into ranges
def categorize_income(data):
    bins = [0, 30, 60, 90, 120, 150]  # Define income ranges
    labels = ["0-30k", "30-60k", "60-90k", "90-120k", "120-150k"]
    data["Income Range"] = pd.cut(data["Annual Income (k$)"], bins=bins, labels=labels)
    return data

# Main Streamlit app
def main():
    st.title("Mall Customers Analysis")
    data = load_data()

    # Categorize income
    data = categorize_income(data)

    # Display raw data
    if st.checkbox("Show Raw Data"):
        st.write(data)

    # Correlation Matrix
    st.subheader("Correlation Matrix (Excluding Gender)")
    numeric_data = data.select_dtypes(include=["number"])  # Select only numeric columns
    correlation_matrix = numeric_data.corr()
    st.write(correlation_matrix)

    # Visualization: Distribution of Annual Income
    st.subheader("Distribution of Annual Income (k$)")
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(data["Annual Income (k$)"], kde=True, color="blue")
    plt.title("Distribution of Annual Income (k$)", fontsize=20)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Count")
    st.pyplot(plt)

    # Visualization: Distribution of Age
    st.subheader("Distribution of Age")
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(data["Age"], kde=True, color="green")
    plt.title("Distribution of Age", fontsize=20)
    plt.xlabel("Age")
    plt.ylabel("Count")
    st.pyplot(plt)

    # Scatter Plot: Spending Score vs Annual Income
    st.subheader("Spending Score vs Annual Income")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", palette="coolwarm")
    plt.title("Spending Score vs Annual Income", fontsize=20)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    st.pyplot(plt)

    # Bar Plot: Count of Customers by Gender
    st.subheader("Count of Customers by Gender")
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    sns.countplot(data=data, x="Gender", palette="viridis")
    plt.title("Count of Customers by Gender", fontsize=20)
    plt.xlabel("Gender")
    plt.ylabel("Count")
    st.pyplot(plt)

    # Bar Plot: Count of Customers by Income Range
    st.subheader("Count of Customers by Income Range")
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    sns.countplot(data=data, x="Income Range", palette="coolwarm")
    plt.title("Count of Customers by Income Range", fontsize=20)
    plt.xlabel("Income Range (k$)")
    plt.ylabel("Count")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
