import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("Mall_Customers.csv")

# Function to calculate the Elbow Method for K Value
def calculate_elbow_method(data, max_clusters=10):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
    return wcss

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

    # Count of Customers by Gender
    st.subheader("Count of Customers by Gender")
    plt.figure(figsize=(8, 5))
    sns.set(style="whitegrid")
    gender_counts = data["Gender"].value_counts()
    sns.barplot(x=gender_counts.index, y=gender_counts.values, palette="pastel")
    plt.title("Count of Customers by Gender", fontsize=18)
    plt.xlabel("Gender", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    st.pyplot(plt)

    # Spending Score vs Annual Income
    st.subheader("Spending Score (1-100) vs Annual Income (k$)")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", palette="coolwarm")
    plt.title("Spending Score vs Annual Income", fontsize=20)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    st.pyplot(plt)

    # K Value (Elbow Method)
    st.subheader("Optimal K Value (Elbow Method)")
    clustering_data = data[["Annual Income (k$)", "Spending Score (1-100)"]]
    wcss = calculate_elbow_method(clustering_data)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
    plt.title("Elbow Method for Optimal K", fontsize=20)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.xticks(range(1, 11))
    st.pyplot(plt)

if __name__ == "__main__":
    main()
