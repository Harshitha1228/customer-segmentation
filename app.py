import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("Mall_Customers.csv")

# Main Streamlit app
def main():
    st.title("Mall Customers Analysis")
    data = load_data()

    # Show raw data
    if st.checkbox("Show Raw Data", key="raw_data_checkbox"):
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

    # Elbow Curve
    st.subheader("Elbow Curve for Optimal Number of Clusters")
    X = data[["Annual Income (k$)", "Spending Score (1-100)"]]
    wcss = []  # Within-cluster sum of squares

    # Calculate WCSS for different numbers of clusters
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plotting the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', color='red', linewidth=2)
    plt.title("Elbow Method for Optimal k", fontsize=20)
    plt.xlabel("Number of Clusters (k)", fontsize=14)
    plt.ylabel("WCSS", fontsize=14)
    st.pyplot(plt)

    # Additional WSS Calculation for k=1 to k=10
    st.subheader("WSS (Within-Cluster Sum of Squared Errors) for Different k Values")
    X2 = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]  # Using additional features
    wcss_2 = []  # List to store WCSS for k=1 to k=10

    # Calculate WCSS for different numbers of clusters (using X2 with 3 features)
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init="k-means++")
        kmeans.fit(X2)
        wcss_2.append(kmeans.inertia_)

    # Plotting WSS for k=1 to k=10
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 11), wcss_2, linewidth=2, color="red", marker="8")
    plt.xlabel("K Value", fontsize=14)
    plt.xticks(np.arange(1, 11, 1))
    plt.ylabel("WSS (Within-Cluster Sum of Squared Errors)", fontsize=14)
    st.pyplot(plt)

    # KMeans Clustering with the Optimal Number of Clusters
    optimal_k = st.slider("Select the optimal number of clusters", min_value=1, max_value=10, value=5, step=1)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X2)

    # Scatterplot of the Clusters
    st.subheader(f"Clusters Scatterplot with k={optimal_k}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", palette="viridis", s=100, alpha=0.6)
    plt.title(f"Clusters of Spending Score vs Annual Income (k={optimal_k})", fontsize=20)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    st.pyplot(plt)

    # 3D Scatter Plot for Clusters with Age, Annual Income, and Spending Score
    st.subheader("3D Scatter Plot of Clusters")
    fig = px.scatter_3d(data, x="Age", y="Annual Income (k$)", z="Spending Score (1-100)", color="Cluster",
                        title="3D Clusters of Customers Based on Age, Annual Income, and Spending Score",
                        labels={"Age": "Age", "Annual Income (k$)": "Annual Income (k$)", "Spending Score (1-100)": "Spending Score (1-100)"})
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
