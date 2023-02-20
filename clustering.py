"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
"""

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


mpl.use('TkAgg')


# Function to plot features in pairs
def plot_pairs(data: pd.DataFrame) -> None:
    data = pd.DataFrame(data)
    Y = data.iloc[:, 1]
    pd.plotting.scatter_matrix(data, c=Y, figsize=(50, 50), marker='o')
    plt.savefig('images/cx.png')


# Function to plot the clusters with labels
def plot_cluster_with_label(label: any, data: pd.DataFrame, k: int, col_1: str, col_2: str) -> None:
    # Getting unique labels
    u_labels = np.unique(label)
    # Plotting the results:
    for i in u_labels:
        plt.scatter(data.iloc[label == i, 0], data.iloc[label == i, 1], label=i)
    plt.legend()
    name = f"images/clustering with k = {k} for {col_1} and {col_2}.png"
    plt.savefig(name)


# Function to calculate minimum description length
def calculate_minimum_description_length(k: int, n: int, representation_error: float) -> float:
    minimum_description_length = representation_error + (k / 2) * math.log2(n)
    return minimum_description_length


# Function to calculate representation error
def calculate_representation_error(kmeans_model: KMeans, data: pd.DataFrame, k: int) -> float:
    # Get the squared distances
    distances = kmeans_model.transform(data) ** 2
    # Find the distance to the closest cluster mean
    distances = distances.min(axis=1)
    # Get the sum-squared error of those distances
    representation_error = 0
    for i in range(0, len(distances)):
        representation_error += distances[i]
    # Verify if the representation error is calculated correctly
    if round(kmeans_model.inertia_, 3) != round(representation_error, 3):
        print(f"Representation error is not calculated correctly using k={k}")
    return representation_error


# Function to generate clusters with k means algorithm on a dataset with input k
def k_means(data: pd.DataFrame, k: int, if_plot: bool = True, col_1: str = '', col_2: str = '') -> float:
    kmeans = KMeans(n_clusters=k, n_init="auto")
    label = kmeans.fit_predict(data)
    if if_plot:
        plot_cluster_with_label(label, data, k, col_1, col_2)
    representation_error = calculate_representation_error(kmeans, data, k)
    return representation_error


# Cluster features in pairs
def cluster_pairs(data: pd.DataFrame, col_1: str, col_2: str) -> None:
    new_df = data.loc[:, [col_1, col_2]]
    print("new_df: ")
    print(new_df)
    plt.scatter(new_df.iloc[:, 0], new_df.iloc[:, 1])
    plt.show()
    # k = 6
    # representation_error = k_means(new_df, k, True, col_1, col_2)
    # minimum_description_length = calculate_minimum_description_length(k, 2, representation_error)
    # print(f"representation error for {col_1} and {col_2} is {representation_error}")
    # print(f"minimum description length for {col_1} and {col_2} is {minimum_description_length}")


def cluster_features(data: pd.DataFrame) -> None:
    # Cluster all features
    n = len(data.iloc[:, 0])
    minimum_description_lengths = []
    representation_errors = []
    k_values = []
    for k in range(6, 20):
        representation_error = k_means(data, k, False)
        minimum_description_length = calculate_minimum_description_length(k, n, representation_error)
        minimum_description_lengths.append(minimum_description_length)
        representation_errors.append(representation_error)
        k_values.append(k)
    plt.plot(k_values, minimum_description_lengths)
    plt.savefig('images/MDL.png')
    plt.plot(k_values, representation_errors)
    plt.savefig('images/representation_error.png')

    # Cluster features in pairs
    # From what we found from the plotting, we decide only to cluster the ones worth clustering
    cluster_pairs(data, 'Age', 'RestingBP')
    cluster_pairs(data, 'Age', 'Cholesterol')
    cluster_pairs(data, 'Age', 'MaxHR')
    cluster_pairs(data, 'Age', 'Oldpeak')
    cluster_pairs(data, 'RestingBP', 'Cholesterol')
    cluster_pairs(data, 'RestingBP', 'MaxHR')
    cluster_pairs(data, 'RestingBP', 'Oldpeak')
    cluster_pairs(data, 'MaxHR', 'Cholesterol')
    cluster_pairs(data, 'Oldpeak', 'Cholesterol')
    cluster_pairs(data, 'MaxHR', 'Oldpeak')