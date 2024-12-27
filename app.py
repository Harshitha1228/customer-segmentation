import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("Mall_Customers.csv")

# Main Streamlit app
def main():
    st.title("Mall Customers Analysis")
    data = load_data()

    # Display raw data
    if st.checkbox("Show Raw Data"):
        st.write(data)

    # Correlation Matrix
    st.subheader("Correlation Matrix (Excluding Gender)")
    correlation_matrix = data.drop(columns=["Gender"]).corr()
    st.write(correlation_matrix)

    # Visualization: Distribution of Annual Income
    st.subheader("Distribution of Annual Income (k$)")
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(data["Annual Income (k$)"], kde=True)
    plt.title("Distribution of Annual Income (k$)", fontsize=20)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Count")
    st.pyplot(plt)

    # Visualization: Distribution of Age
    st.subheader("Distribution of Age")
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(data["Age"], kde=True)
    plt.title("Distribution of Age", fontsize=20)
    plt.xlabel("Age")
    plt.ylabel("Count")
    st.pyplot(plt)

    # Additional Visualization: Spending Score vs Annual Income
    st.subheader("Spending Score vs Annual Income")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender")
    plt.title("Spending Score vs Annual Income", fontsize=20)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
