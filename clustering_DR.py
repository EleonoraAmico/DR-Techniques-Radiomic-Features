# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:38:39 2025

@author: Ele_p
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pacmap
import os
from typing import Tuple, Dict, Any, Optional
import hdbscan
from sklearn.metrics.cluster import adjusted_rand_score, silhouette_score
from scipy import stats

def get_target_labels(target_name: str, target_values: np.ndarray) -> list:
    """
    Generate appropriate labels based on the target variable name and its values.
    
    Parameters:
    -----------
    target_name : str
        Name of the target variable (e.g., 'OS_EVENT', 'R_ISS', etc.)
    target_values : np.ndarray
        Array of target values
        
    Returns:
    --------
    list : List of labels for each unique target value
    """
    # Dictionary mapping target names to their label formats
    label_formats = {
        'OS_EVENT': {0: 'No Event', 1: 'Event'},
        'PFS_I_EVENT': {0: 'No Progression', 1: 'Progression'},
        'SEX': {0: 'Female', 1: 'Male'},
        'R_ISS': {1: 'R-ISS I', 2: 'R-ISS II', 3: 'R-ISS III'}
    }
    
    unique_values = np.sort(np.unique(target_values))
    
    # If we have predefined labels for this target
    if target_name in label_formats:
        return [label_formats[target_name][val] for val in unique_values]
    
    # Default format if target not in dictionary
    return [f'{target_name} = {val}' for val in unique_values]
def interpret_ari(ari: float) -> str:
    """Interpret Adjusted Rand Index value"""
    if ari < 0:
        return "Poor agreement (worse than random)"
    elif ari < 0.2:
        return "Slight agreement"
    elif ari < 0.4:
        return "Fair agreement"
    elif ari < 0.6:
        return "Moderate agreement"
    elif ari < 0.8:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

def interpret_silhouette(silhouette: float) -> str:
    """Interpret Silhouette score"""
    if silhouette < 0:
        return "Poor clustering (potential overlapping)"
    elif silhouette < 0.2:
        return "Weak structure"
    elif silhouette < 0.5:
        return "Reasonable structure"
    elif silhouette < 0.7:
        return "Strong structure"
    else:
        return "Very strong structure"
def validate_clustering(
    cluster_labels: np.ndarray,
    outcome_values: np.ndarray,
    feature_matrix: np.ndarray,
    outcome_name: str,
) -> Dict[str, Any]:
    """
    Validate clustering results using multiple metrics.
    
    Parameters:
    -----------
    cluster_labels : np.ndarray
        Cluster assignments from HDBSCAN
    outcome_values : np.ndarray
        Known outcome values to compare against
    feature_matrix : np.ndarray
        Original feature matrix used for clustering
    outcome_name : str
        Name of the outcome variable
        
    Returns:
    --------
    Dict containing validation metrics and statistical tests
    """
    # Remove noise points (-1) from HDBSCAN
    valid_mask = cluster_labels != -1
    clean_clusters = cluster_labels[valid_mask]
    clean_outcomes = outcome_values[valid_mask]
    clean_features = feature_matrix[valid_mask]
    
    # 1. Adjusted Rand Index
    ari = adjusted_rand_score(clean_outcomes, clean_clusters)
    
    # 2. Chi-square test
    contingency = pd.crosstab(clean_outcomes, clean_clusters)
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    
    # 3. Silhouette score
    silhouette = silhouette_score(clean_features, clean_clusters)
    
    # Create results dictionary
    results = {
        'metrics': {
            'adjusted_rand_index': ari,
            'silhouette_score': silhouette,
            'chi_square': chi2,
            'p_value': p_value
        },
        'cluster_sizes': {
            f'cluster_{i}': sum(clean_clusters == i) 
            for i in np.unique(clean_clusters)
        },
        'noise_points': sum(~valid_mask),
        'valid_samples': sum(valid_mask),
        'contingency_table': contingency
    }
    
    # Add interpretation
    results['interpretation'] = {
        'ari_interpretation': interpret_ari(ari),
        'silhouette_interpretation': interpret_silhouette(silhouette),
        'statistical_significance': p_value < 0.05
    }
    
    return results
def compare_clustering_methods(
    data: pd.DataFrame,
    target: np.ndarray = None,
    target_name: str = None,
    random_state: int = 42,
    perplexity: int = 30,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    pacmap_neighbors: int = 10,
    pacmap_MN_ratio: float = 0.5,
    pacmap_FP_ratio: float = 2.0,
    min_cluster_size: int = 5,
    min_samples: int = None,
    metric: str = 'euclidean',
    alpha: float = 0.6
) -> Dict[str, Dict]:
    """
    Compare different dimensionality reduction methods with both target variable coloring
    and HDBSCAN clustering results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data matrix
    target : np.ndarray, optional
        Target variable for coloring the plots
    target_name : str, optional
        Name of the target variable
    random_state : int
        Random state for reproducibility
    perplexity : int
        Perplexity parameter for t-SNE
    umap_neighbors : int
        Number of neighbors for UMAP
    umap_min_dist : float
        Minimum distance parameter for UMAP
    pacmap_neighbors : int
        Number of neighbors for PaCMAP
    pacmap_MN_ratio : float
        MN ratio for PaCMAP
    pacmap_FP_ratio : float
        FP ratio for PaCMAP
    min_cluster_size : int
        Minimum cluster size for HDBSCAN
    min_samples : int
        Minimum samples for HDBSCAN
    metric : str
        Distance metric for HDBSCAN
    alpha : float
        Transparency for plot points
    
    Returns:
    --------
    dict : Dictionary containing results and metrics for each method
    """
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Initialize results dictionary
    results = {}
    
    # Set up methods
    methods = {
        'PCA': PCA(n_components=2, random_state=random_state),
        't-SNE': TSNE(n_components=2, random_state=random_state, perplexity=perplexity),
        'UMAP': umap.UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state
        ),
        'PaCMAP': pacmap.PaCMAP(
            n_components=2,
            n_neighbors=pacmap_neighbors,
            MN_ratio=pacmap_MN_ratio,
            FP_ratio=pacmap_FP_ratio,
            random_state=random_state
        )
    }
    
    # Create figures: one for target variable, one for clustering
    fig_target, axes_target = plt.subplots(2, 2, figsize=(20, 20))
    fig_cluster, axes_cluster = plt.subplots(2, 2, figsize=(20, 20))
    
    axes_target = axes_target.ravel()
    axes_cluster = axes_cluster.ravel()
    
    # Process each method
    for idx, (method_name, reducer) in enumerate(methods.items()):
        # Perform dimensionality reduction
        reduced_data = reducer.fit_transform(scaled_data)
        
        # Store results
        results[method_name.lower()] = {
            'result': reduced_data,
            'reducer': reducer
        }
        
        if method_name == 'PCA':
            results[method_name.lower()]['explained_variance_ratio'] = reducer.explained_variance_ratio_
        
        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            gen_min_span_tree=True
        )
        cluster_labels = clusterer.fit_predict(reduced_data)
        results[method_name.lower()]['clustering'] = {
            'labels': cluster_labels,
            'clusterer': clusterer
        }
        
        # Plot 1: Target variable coloring
        if target is not None:
            unique_values = np.sort(np.unique(target))
            class_labels = get_target_labels(target_name, target) if target_name else [f'Class {val}' for val in unique_values]
            
            for i, target_class in enumerate(unique_values):
                mask = target == target_class
                axes_target[idx].scatter(
                    reduced_data[mask, 0],
                    reduced_data[mask, 1],
                    label=class_labels[i],
                    alpha=alpha
                )
            axes_target[idx].legend()
        else:
            axes_target[idx].scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=alpha)
            
        # Plot 2: HDBSCAN clustering
        unique_clusters = np.unique(cluster_labels)
        for cluster in unique_clusters:
            mask = cluster_labels == cluster
            label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
            color = 'gray' if cluster == -1 else None
            axes_cluster[idx].scatter(
                reduced_data[mask, 0],
                reduced_data[mask, 1],
                label=label,
                color=color,
                alpha=alpha
            )
        axes_cluster[idx].legend()
        
        # Add titles
        title = method_name
        if method_name == 'PCA':
            title += (f'\nExplained variance: {reducer.explained_variance_ratio_[0]:.3f}, '
                     f'{reducer.explained_variance_ratio_[1]:.3f}\n'
                     f'Total: {sum(reducer.explained_variance_ratio_):.3f}')
        
        axes_target[idx].set_title(f'{title}\nTarget: {target_name}')
        axes_cluster[idx].set_title(f'{title}\nHDBSCAN Clusters')
        
        # Add labels
        for ax in [axes_target[idx], axes_cluster[idx]]:
            ax.set_xlabel('First component')
            ax.set_ylabel('Second component')
    
    # Adjust layout
    fig_target.suptitle(f'Dimensionality Reduction Results Colored by {target_name}', y=1.02, fontsize=16)
    fig_cluster.suptitle(f'Dimensionality Reduction Results with HDBSCAN Clustering ', y=1.02, fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and add quality metrics
    if target is not None:
        for method_name in methods.keys():
            method_key = method_name.lower()
            reduced_data = results[method_key]['result']
            cluster_labels = results[method_key]['clustering']['labels']
            
            # Calculate metrics
            metrics = validate_clustering(
                cluster_labels=cluster_labels,
                outcome_values=target,
                feature_matrix=reduced_data,
                outcome_name=target_name if target_name else "target"
            )
            
            results[method_key]['metrics'] = metrics
    
    return results

# Example usage:
def run_clustering_comparison(data_all, variables_to_analyze, feature_sets):
    """
    Run clustering comparison for multiple feature sets and target variables.
    
    Parameters:
    -----------
    data_all : pd.DataFrame
        Complete dataset
    variables_to_analyze : list
        List of target variables to analyze
    feature_sets : dict
        Dictionary of feature sets to analyze
    """
    for feature_name, feature_params in feature_sets.items():
        print(f"\nAnalyzing {feature_name}")
        data = feature_params['data']
        
        for target_var in variables_to_analyze:
            print(f"\nTarget variable: {target_var}")
            
            # Remove target variable from features if present
            data_without_target = data.drop(columns=[target_var]) if target_var in data.columns else data
            
            results = compare_clustering_methods(
                data=data_without_target,
                target=data_all[target_var].values,
                target_name=target_var,
                perplexity=30,
                umap_neighbors=15,
                umap_min_dist=0.1,
                pacmap_neighbors=10,
                min_cluster_size=5
            )
            
            # Print quality metrics
            print("\nQuality Metrics:")
            for method, method_results in results.items():
                if 'metrics' in method_results:
                    metrics = method_results['metrics']
                    print(f"\n{method.upper()}:")
                    print(f"Silhouette Score: {metrics['metrics']['silhouette_score']:.2e}")
                    print(f"ARI: {metrics['metrics']['adjusted_rand_index']:.2e}")
                    print(f"Chi-square p-value: {metrics['metrics']['p_value']:.2e}")
                    print(f"Interpretation: {metrics['interpretation']['ari_interpretation']}")
                    

#%%
def preprocess_dataset_riss(dataset: pd.DataFrame, riss_column: str = 'R_ISS') -> pd.DataFrame:
    """
    Preprocess R-ISS values in a dataset to standardize them to 1, 2, or 3.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Input DataFrame containing R-ISS values
    riss_column : str, optional
        Name of the column containing R-ISS values (default: 'R_ISS')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with standardized R-ISS values
    """
    # Create a copy of the dataset to avoid modifying the original
    processed_dataset = dataset.copy()
    
    # Convert to numeric if not already
    riss_numeric = pd.to_numeric(processed_dataset[riss_column], errors='coerce')
    
    # Function to round to nearest stage
    def round_to_stage(x):
        if pd.isna(x):
            return np.nan
        elif x < 1.5:  # Values closer to 1
            return 1
        elif x < 2.5:  # Values closer to 2
            return 2
        else:  # Values closer to 3
            return 3
    
    # Apply rounding function
    processed_dataset[riss_column] = riss_numeric.apply(round_to_stage)
    
    # Print summary of changes
    print("\nR-ISS Standardization Summary:")
    print("-" * 30)
    #print("Original unique values:", dataset[riss_column].unique())
    print("Standardized unique values:", 
          processed_dataset[riss_column].dropna().unique())
    
    # Count samples in each category
    value_counts = processed_dataset[riss_column].value_counts().sort_index()
    for value, count in value_counts.items():
        print(f"R-ISS {int(value)}: {count} samples")
    return processed_dataset

def extract_feature_subset(df, feature_type='clinical', image_type=None, category=None):
    """
    Extract specific feature subsets from the dataset
    
    Parameters:
    df : DataFrame
        Input dataset
    feature_type : str
        'clinical', 'radiomic', or 'all'
    image_type : str, optional
        'original', 'wavelet', etc.
    category : str, optional
        'firstorder', 'GLCM', etc.
    """
    # Basic feature groups
    survival_cols = df.columns[:4]
    clinical_cols = df.columns[4:12]
    radiomic_cols = df.columns[13:]
    
    if feature_type == 'clinical':
        return df[clinical_cols]
    
    elif feature_type == 'radiomic':
        if image_type and category:
            # Filter radiomic features by both image type and category
            selected_cols = [col for col in radiomic_cols 
                           if image_type in col and category in col]
        elif image_type:
            # Filter by image type only
            selected_cols = [col for col in radiomic_cols 
                           if image_type in col]
        elif category:
            # Filter by category only
            selected_cols = [col for col in radiomic_cols 
                           if category in col]
        else:
            selected_cols = radiomic_cols
            
        return df[selected_cols]
    
    else:  # 'all'
        return df[list(clinical_cols) + list(radiomic_cols)]
# Get the directory containing your Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to your CSV file
file_path = os.path.join(script_dir, 'binarized_data.csv')

# Now use file_path in your code
data_all = pd.read_csv(file_path)
data_all = preprocess_dataset_riss(data_all)
# Then extract different feature subsets
clinical_features = extract_feature_subset(data_all, feature_type='clinical')
all_radiomic_features = extract_feature_subset(data_all, feature_type='radiomic')
original_features = extract_feature_subset(data_all, feature_type='radiomic', image_type='original')
firstorder_features = extract_feature_subset(data_all, feature_type='radiomic', category='firstorder')
original_glcm_features = extract_feature_subset(data_all, feature_type='radiomic', 
                                              image_type='original', category='glcm')
#%%
glcm_features = extract_feature_subset(data_all, feature_type='radiomic', category='glcm')
log_sigma_features = extract_feature_subset(data_all, feature_type='radiomic', image_type='log-sigma')
wavelet_features = extract_feature_subset(data_all, feature_type='radiomic', image_type='wavelet')
glrlm_features = extract_feature_subset(data_all, feature_type='radiomic', category='glrlm')
glszm_features = extract_feature_subset(data_all, feature_type='radiomic', category='glszm')
shape_features = extract_feature_subset(data_all, feature_type='radiomic', category='shape')
glcm_wavelet_features=extract_feature_subset(data_all, feature_type='radiomic', image_type='wavelet', category='glcm')

#clinical_features = data_all.drop(columns=['SEX'])
variables_to_analyze = ['OS_EVENT', 'SEX', 'PFS_I_EVENT', 'R_ISS']
variable_to_analyze = ['OS_EVENT', 'PFS_I_EVENT', 'R_ISS', 'SEX']
# if __name__ == "__main__":
#     # Example usage for all feature types
feature_sets = {
    'all_dataset':{'data': data_all, 'type': 'all'},
    # 'clinical_features': {'data': clinical_features, 'type': 'clinical'},
    # 'radiomic_features': {'data': all_radiomic_features, 'type': 'radiomic'},
    # 'original_features': {'data': original_features, 'type': 'radiomic', 'image_type': 'original'},
    # 'firstorder_features': {'data': firstorder_features, 'type': 'radiomic', 'category': 'firstorder'},
    # 'glcm_features': {'data': glcm_features, 'type': 'radiomic', 'category': 'glcm'},
    # 'log_sigma_features': {'data': log_sigma_features, 'type': 'radiomic', 'image_type': 'log-sigma'},
    # 'wavelet_features': {'data': wavelet_features, 'type': 'radiomic', 'image_type': 'wavelet'},
    # 'glrlm_features': {'data': glrlm_features, 'type': 'radiomic', 'category': 'glrlm'},
    # 'glszm_features': {'data': glszm_features, 'type': 'radiomic', 'category': 'glszm'},
    # 'shape_features': {'data': shape_features, 'type': 'radiomic', 'category': 'shape'},
    # 'glcm_wavelet_features': {'data': glcm_wavelet_features, 'type': 'radiomic', 'image_type': 'wavelet', 'category': 'glcm'}
}



run_clustering_comparison(
    data_all=data_all,
    variables_to_analyze=variable_to_analyze,
    feature_sets=feature_sets
)