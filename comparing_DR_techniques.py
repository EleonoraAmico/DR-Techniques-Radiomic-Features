# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:54:10 2025

@author: Ele_p
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pacmap
import os
#%%
def evaluate_local_structure(original_data, reduced_data, k=5):
    """
    Evaluate local structure preservation using k-nearest neighbors
    
    Parameters:
    -----------
    original_data : array-like
        Original high-dimensional data
    reduced_data : array-like
        Low-dimensional projection
    k : int
        Number of nearest neighbors to consider
        
    Returns:
    --------
    float
        Fraction of k-nearest neighbors preserved
    """
    # Compute pairwise distances in original and reduced space
    original_dist = euclidean_distances(original_data)
    reduced_dist = euclidean_distances(reduced_data)
    
    # Get k-nearest neighbors for each point in both spaces
    orig_neighbors = np.argsort(original_dist, axis=1)[:, 1:k+1]  # exclude self
    red_neighbors = np.argsort(reduced_dist, axis=1)[:, 1:k+1]
    
    # Calculate preservation ratio
    preservation_scores = []
    for i in range(len(original_data)):
        orig_set = set(orig_neighbors[i])
        red_set = set(red_neighbors[i])
        preservation = len(orig_set.intersection(red_set)) / k
        preservation_scores.append(preservation)
    
    return np.mean(preservation_scores)

def evaluate_global_structure(original_data, reduced_data, n_triplets=1000):
    """
    Evaluate global structure preservation using random triplets
    
    Parameters:
    -----------
    original_data : array-like
        Original high-dimensional data
    reduced_data : array-like
        Low-dimensional projection
    n_triplets : int
        Number of random triplets to evaluate
        
    Returns:
    --------
    dict
        Dictionary containing triplet accuracy and distance correlation
    """
    n_samples = len(original_data)
    
    # Calculate pairwise distances
    original_dist = euclidean_distances(original_data)
    reduced_dist = euclidean_distances(reduced_data)
    
    # Random triplet accuracy
    triplets = np.random.choice(n_samples, size=(n_triplets, 3), replace=True)
    correct_count = 0
    
    for i, j, k in triplets:
        # Check if relative distances are preserved
        orig_closer = original_dist[i, j] < original_dist[i, k]
        red_closer = reduced_dist[i, j] < reduced_dist[i, k]
        if orig_closer == red_closer:
            correct_count += 1
            
    triplet_accuracy = correct_count / n_triplets
    
    # Distance correlation
    # Get upper triangular part of distance matrices
    orig_dist_flat = original_dist[np.triu_indices(n_samples, k=1)]
    red_dist_flat = reduced_dist[np.triu_indices(n_samples, k=1)]
    distance_correlation = stats.spearmanr(orig_dist_flat, red_dist_flat)[0]
    
    return {
        'triplet_accuracy': triplet_accuracy,
        'distance_correlation': distance_correlation
    }

def evaluate_dr_method(original_data, reduced_data, method_name):
    """
    Evaluate a dimensionality reduction method using all metrics
    """
    local_score = evaluate_local_structure(original_data, reduced_data)
    global_scores = evaluate_global_structure(original_data, reduced_data)
    
    return {
        'method': method_name,
        'local_preservation': local_score,
        'triplet_accuracy': global_scores['triplet_accuracy'],
        'distance_correlation': global_scores['distance_correlation']
    }

def plot_evaluation_results(results_df):
    """
    Create bar plots for each evaluation metric
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot local preservation
    sns.barplot(x='method', y='local_preservation', data=results_df, ax=axes[0])
    axes[0].set_title('Local Structure Preservation')
    axes[0].set_ylim(0, 1)
    
    # Plot triplet accuracy
    sns.barplot(x='method', y='triplet_accuracy', data=results_df, ax=axes[1])
    axes[1].set_title('Triplet Accuracy')
    axes[1].set_ylim(0, 1)
    
    # Plot distance correlation
    sns.barplot(x='method', y='distance_correlation', data=results_df, ax=axes[2])
    axes[2].set_title('Distance Correlation')
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def main():
    # Load your data
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to your CSV file
    file_path = os.path.join(script_dir, 'binarized_data.csv')

    # Now use file_path in your code
    data = pd.read_csv(file_path)
    
    # Initialize DR methods
    dr_methods = {
        'PCA': PCA(n_components=2),
        'TSNE': TSNE(n_components=2),
        'UMAP': UMAP(n_components=2),
        'PaCMAP': pacmap.PaCMAP(n_components=2)
    }
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Evaluate each method
    results = []
    for method_name, method in dr_methods.items():
        print(f"Evaluating {method_name}...")
        reduced_data = method.fit_transform(data_scaled)
        evaluation = evaluate_dr_method(data_scaled, reduced_data, method_name)
        results.append(evaluation)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(results_df)
    
    # Plot results
    fig = plot_evaluation_results(results_df)
    plt.show()

if __name__ == "__main__":
    main()