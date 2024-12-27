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

    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.write(data)

    # Correlation Matrix
    st.subheader("Correlation Matrix (Excluding Gender)")
    correlation_matrix = data.drop(columns=["Gender"]).corr()
    st.write(correlation_matrix)

    # Distribution of Annual Income
    st.subheader("Distribution of Annual Income (k$)")
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(data["Annual Income (k$)"], kde=True)
    plt.title("Distribution of Annual Income (k$)", fontsize=20)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Count")
    st.pyplot(plt)

    # Distribution of Age
    st.subheader("Distribution of Age")
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(data["Age"], kde=True)
    plt.title("Distribution of Age", fontsize=20)
    plt.xlabel("Age")
    plt.ylabel("Count")
    st.pyplot(plt)

    # Distribution of Spending Score
    st.subheader("Distribution of Spending Score (1-100)")
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(data["Spending Score (1-100)"], kde=True, color="purple")
    plt.title("Distribution of Spending Score (1-100)", fontsize=20)
    plt.xlabel("Spending Score (1-100)")
    plt.ylabel("Count")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
