# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:31:27 2025

@author: Ele_p
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import hdbscan
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import confusion_matrix
from scipy import stats
import os
#%%
def create_pca_visualizations(data_dict, output_file=None):
    """
    Create interactive PCA visualizations using Plotly.
    
    Parameters:
    data_dict (dict): Dictionary containing different PCA results
        Each entry should have the format:
        {
            'name': str,
            'data': array-like,
            'n_components': int
        }
    output_file (str, optional): Path to save the HTML output
    """
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Individual Explained Variance', 'Cumulative Explained Variance'),
        vertical_spacing=0.2
    )
    
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 
              'rgb(44, 160, 44)', 'rgb(214, 39, 40)']
    
    for idx, (name, dataset) in enumerate(data_dict.items()):
        # Perform PCA
        pca = PCA(n_components=dataset['n_components'])
        pca.fit(dataset['data'])
        
        # Calculate explained variance ratio
        explained_variance = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(explained_variance)
        
        # Add traces for individual variance
        fig.add_trace(
            go.Bar(
                x=[f'PC{i+1}' for i in range(len(explained_variance))],
                y=explained_variance,
                name=f'{name} - Individual',
                marker_color=colors[idx],
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add traces for cumulative variance
        fig.add_trace(
            go.Scatter(
                x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
                y=cumulative_variance,
                name=f'{name} - Cumulative',
                line=dict(color=colors[idx], width=2),
                mode='lines+markers',
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='PCA Analysis Results',
        template='plotly_white',
        showlegend=True,
        height=1000,  # Set height in layout instead of make_subplots
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Explained Variance (%)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Explained Variance (%)", row=2, col=1)
    
    if output_file:
        fig.write_html(output_file)
    
    return fig

def process_pca_results(name="pca_results.html"):
    # Create a dictionary with your different PCA results
    pca_results = {
        'Raw Data': {
            'data': data,  # Your original data
            'n_components': 10
        },
        'Scaled Data': {
            'data': scaled_data,  # Your scaled data
            'n_components': 10
        },
        'Scaled Selected Data': {
            'data': scaled_selected_data,  # Your feature-selected data
            'n_components': 10
        },
        'Selected Data': {
            'data':selected_data,  # Your scaled and selected data
            'n_components': 10
        }
    }
    
    # Create and save the visualizations
    fig = create_pca_visualizations(pca_results, name)
    fig.show()

#%%


# Get the directory containing your Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to your CSV file
file_path = os.path.join(script_dir, 'binarized_data.csv')

# Now use file_path in your code
data = pd.read_csv(file_path)
#%%
# Sample data (replace with your actual dataset)
#data = pd.read_csv('binarized_data.csv')
# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Feature Selection
selector = VarianceThreshold(threshold=0.2)  # Adjust threshold as needed
scaled_selected_data = selector.fit_transform(scaled_data)

# Scale the data
selector = VarianceThreshold(threshold=0.2) 
selected_data = selector.fit_transform(data)

# Run the analysis
#process_pca_results()
# Run the analysis
if __name__ == "__main__":
    process_pca_results(name="pca_results_all_data.html")
#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Try different scalers
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

# Apply different scaling methods
scaled_standard = standard_scaler.fit_transform(data)
scaled_minmax = minmax_scaler.fit_transform(data)
scaled_robust = robust_scaler.fit_transform(data)

# Perform PCA on each scaled version
pca_standard = PCA().fit(scaled_standard)
pca_minmax = PCA().fit(scaled_minmax)
pca_robust = PCA().fit(scaled_robust)

# Compare cumulative explained variance
cumulative_variance_standard = np.cumsum(pca_standard.explained_variance_ratio_ * 100)
cumulative_variance_minmax = np.cumsum(pca_minmax.explained_variance_ratio_ * 100)
cumulative_variance_robust = np.cumsum(pca_robust.explained_variance_ratio_ * 100)

print(cumulative_variance_standard)
print(cumulative_variance_minmax)
print(cumulative_variance_robust)

#%%

import pandas as pd
import numpy as np

# Assuming your data is in a DataFrame called 'df'
def analyze_dataset_structure(df):
    # Print basic information about the dataset
    print("Dataset Shape:", df.shape)
    print("\nFirst few column names:")
    print(df.columns.tolist()[:15])  # Print first 15 column names
    
    # Create feature groups
    survival_cols = df.columns[:4]
    clinical_cols = df.columns[4:12]  # 9 clinical variables after survival
    radiomic_cols = df.columns[13:]   # Rest are radiomic features
    
    print("\nFeature Groups:")
    print(f"Survival columns ({len(survival_cols)}):", survival_cols.tolist())
    print(f"\nClinical columns ({len(clinical_cols)}):", clinical_cols.tolist())
    print(f"\nNumber of radiomic features: {len(radiomic_cols)}")
    
    # Analyze radiomic feature names to identify categories
    if len(radiomic_cols) > 0:
        # Assuming radiomic features follow a naming pattern
        # Example: original_firstorder_Mean, wavelet_GLCM_Correlation
        categories = set()
        image_types = set()
        for col in radiomic_cols:
            parts = col.split('_')
            if len(parts) >= 2:
                image_types.add(parts[0])  # original, wavelet, etc.
                categories.add(parts[1])    # firstorder, GLCM, etc.
        
        print("\nRadiomic feature categories found:", sorted(categories))
        print("Image types found:", sorted(image_types))
    
    return survival_cols, clinical_cols, radiomic_cols

# Function to extract specific feature sets
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

# Example usage:
# First analyze the dataset structure
survival_cols, clinical_cols, radiomic_cols = analyze_dataset_structure(data)

# Then extract different feature subsets
clinical_features = extract_feature_subset(data, feature_type='clinical')
all_radiomic_features = extract_feature_subset(data, feature_type='radiomic')
original_features = extract_feature_subset(data, feature_type='radiomic', image_type='original')
firstorder_features = extract_feature_subset(data, feature_type='radiomic', category='firstorder')
original_glcm_features = extract_feature_subset(data, feature_type='radiomic', 
                                              image_type='original', category='glcm')
#%%
glcm_features = extract_feature_subset(data, feature_type='radiomic', category='glcm')
log_sigma_features = extract_feature_subset(data, feature_type='radiomic', image_type='log-sigma')
wavelet_features = extract_feature_subset(data, feature_type='radiomic', image_type='wavelet')
glrlm_features = extract_feature_subset(data, feature_type='radiomic', category='glrlm')
glszm_features = extract_feature_subset(data, feature_type='radiomic', category='glszm')
shape_features = extract_feature_subset(data, feature_type='radiomic', category='shape')
glcm_wavelet_features=extract_feature_subset(data, feature_type='radiomic', image_type='wavelet', category='glcm')
#%%
data = data.copy
data = clinical_features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clinical_features)

# Feature Selection
selector = VarianceThreshold(threshold=0.2)  # Adjust threshold as needed
selected_data_for_pca = selector.fit_transform(scaled_data)

# Scale the data
selector = VarianceThreshold(threshold=0.2) 
selected_data = selector.fit_transform(clinical_features)

# Run the analysis
#process_pca_results()
# Run the analysis
if __name__ == "__main__":
    process_pca_results(name="pca_clinical_features.html")
    
#%%
data = data.copy
data = all_radiomic_features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(all_radiomic_features)

# Feature Selection
selector = VarianceThreshold(threshold=0.2)  # Adjust threshold as needed
selected_data_for_pca = selector.fit_transform(scaled_data)

# Scale the data
selector = VarianceThreshold(threshold=0.2) 
selected_data = selector.fit_transform(all_radiomic_features)

# Run the analysis
#process_pca_results()
# Run the analysis
if __name__ == "__main__":
    process_pca_results(name="pca_radiomics_features.html")

#%%
data = data.copy
data = original_features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(original_features)

# Feature Selection
selector = VarianceThreshold(threshold=0.2)  # Adjust threshold as needed
selected_data_for_pca = selector.fit_transform(scaled_data)

# Scale the data
selector = VarianceThreshold(threshold=0.01) 
selected_data = selector.fit_transform(original_features)

# Run the analysis
#process_pca_results()
# Run the analysis
if __name__ == "__main__":
    process_pca_results(name="pca_radiomics_original_features.html")

#%%
data = data.copy
data = firstorder_features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(firstorder_features)

# Feature Selection
selector = VarianceThreshold(threshold=0.2)  # Adjust threshold as needed
selected_data_for_pca = selector.fit_transform(scaled_data)

# Scale the data
selector = VarianceThreshold(threshold=0.01) 
selected_data = selector.fit_transform(firstorder_features)

# Run the analysis
#process_pca_results()
# Run the analysis
if __name__ == "__main__":
    process_pca_results(name="pca_radiomics_first_order_features.html")
#%%
data = data.copy
data = original_glcm_features
print(original_glcm_features.head)
print(len(original_glcm_features))
scaler = StandardScaler()
scaled_data = scaler.fit_transform(original_glcm_features)

# Feature Selection
selector = VarianceThreshold(threshold=0.2)  # Adjust threshold as needed
selected_data_for_pca = selector.fit_transform(scaled_data)

# Scale the data
selector = VarianceThreshold(threshold=0.01) 
selected_data = selector.fit_transform(original_glcm_features)

# Run the analysis
#process_pca_results()
# Run the analysis
if __name__ == "__main__":
    process_pca_results(name="pca_radiomics_original_glcm_features.html")

#%%
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(original_features)

# Calculate variances for each feature
variances = np.var(scaled_features, axis=0)

# Sort variances in ascending order
sorted_variances = np.sort(variances)

print("Minimum variance:", sorted_variances[0])
print("Maximum variance:", sorted_variances[-1])
print("Mean variance:", np.mean(variances))

# Plot variance distribution
plt.figure(figsize=(10, 6))
plt.plot(sorted_variances, 'b-')
plt.axhline(y=1.0, color='r', linestyle='--', label='Standard Variance')
plt.xlabel('Features (sorted by variance)')
plt.ylabel('Variance')
plt.title('Feature Variance Distribution After Scaling')
plt.legend()
plt.grid(True)
plt.show()
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# First, standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(original_features)

# Create correlation matrix
correlation_matrix = pd.DataFrame(scaled_features).corr()

# Visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Function to identify highly correlated feature pairs
def find_correlated_features(correlation_matrix, threshold=0.8):
    high_correlations = np.where(np.abs(correlation_matrix) > threshold)
    correlated_pairs = []
    
    for i, j in zip(*high_correlations):
        if i < j:  # Avoid duplicate pairs
            correlated_pairs.append((
                correlation_matrix.index[i], 
                correlation_matrix.index[j], 
                correlation_matrix.iloc[i, j]
            ))
    
    return pd.DataFrame(
        correlated_pairs, 
        columns=['Feature 1', 'Feature 2', 'Correlation']
    ).sort_values('Correlation', ascending=False)

# Find highly correlated features
correlated_features = find_correlated_features(correlation_matrix, threshold=0.8)
print("\nHighly correlated feature pairs:")
print(correlated_features)

#%%
import pandas as pd
# Get the directory containing your Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to your CSV file
file_path = os.path.join(script_dir, 'binarized_data.csv')

# Now use file_path in your code
data = pd.read_csv(file_path)
# To check for completely empty columns
empty_columns = data.columns[data.isna().all()].tolist()

# To check for partially empty columns and get the percentage of missing values
missing_percentages = (data.isna().sum() / len(data)) * 100

# To get columns with any missing values
columns_with_missing = data.columns[data.isna().any()].tolist()

def analyze_missing_data(df):
    total_rows = len(df)
    missing_summary = pd.DataFrame({
        'Missing Values': df.isna().sum(),
        'Percentage Missing': (df.isna().sum() / total_rows) * 100,
        'Data Type': df.dtypes
    })
    return missing_summary.sort_values('Percentage Missing', ascending=False)

analyze_missing_data(data)
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from typing import Optional, Tuple, Dict, Any

def process_feature_subset(
    data: pd.DataFrame,
    feature_type: str,
    variance_threshold: float = 0.2,
    n_components: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, PCA, Dict[str, Any]]:
    """
    Process a feature subset through scaling, variance thresholding, and PCA.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data containing features
    feature_type : str
        Type of features being processed (for logging)
    variance_threshold : float
        Threshold for variance-based feature selection
    n_components : Optional[int]
        Number of PCA components to keep
    **kwargs : 
        Additional filtering criteria (image_type, category)
    
    Returns:
    --------
    Tuple containing:
    - Transformed PCA data
    - Fitted PCA object
    - Dictionary with analysis metrics
    """
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Variance thresholding
    selector = VarianceThreshold(threshold=variance_threshold)
    selected_data = selector.fit_transform(scaled_data)
    
    # Determine number of components if not specified
    if n_components is None:
        n_components = min(selected_data.shape[1], 10)  # Default to 10 or max possible
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(selected_data)
    
    # Calculate metrics
    metrics = {
        'n_features_original': data.shape[1],
        'n_features_selected': selected_data.shape[1],
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
        'feature_type': feature_type,
        **kwargs
    }
    
    return pca_result, pca, metrics

#%%

def create_pca_visualization(
    pca_result: np.ndarray,
    metrics: Dict[str, Any],
    output_file: str
) -> None:
    """
    Create seaborn visualizations for PCA results.
    
    Parameters:
    -----------
    pca_result : np.ndarray
        Transformed PCA data
    metrics : Dict[str, Any]
        Dictionary containing analysis metrics
    output_file : str
        Path to save the output figure
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 15))
    
    # Create grid for subplots
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])  # Scree plot takes full width
    ax2 = fig.add_subplot(gs[1, 0])  # 2D plot
    ax3 = fig.add_subplot(gs[1, 1], projection='3d')  # 3D plot
    
    # Plot 1: Scree plot with cumulative explained variance
    components = list(range(1, len(metrics['explained_variance_ratio']) + 1))
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Component': components,
        'Explained Variance Ratio': metrics['explained_variance_ratio']
    })
    
    # Bar plot for individual explained variance
    sns.barplot(data=plot_data,
                x='Component',
                y='Explained Variance Ratio',
                color='skyblue',
                alpha=0.5,
                ax=ax1)
    
    # Line plot for cumulative explained variance
    ax1_twin = ax1.twinx()
    plot_data['Cumulative Variance Ratio'] = metrics['cumulative_variance_ratio']
    sns.lineplot(data=plot_data,
                 x='Component',
                 y='Cumulative Variance Ratio',
                 color='red',
                 marker='o',
                 ax=ax1_twin)
    
    # Customize first plot
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1_twin.set_ylabel('Cumulative Explained Variance Ratio')
    ax1.set_title(f'Scree Plot - {metrics["feature_type"]} Features')
    
    # Plot 2: First two components scatter plot if available
    if pca_result.shape[1] >= 2:
        scatter_data = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1]
        })
        
        sns.scatterplot(data=scatter_data,
                       x='PC1',
                       y='PC2',
                       ax=ax2)
        
        # Add a 2D kernel density estimate
        sns.kdeplot(data=scatter_data,
                   x='PC1',
                   y='PC2',
                   levels=5,
                   color='red',
                   alpha=0.3,
                   ax=ax2)
        
        ax2.set_title(f'PC1 vs PC2 - {metrics["feature_type"]} Features')
        
        # Add 3D scatter plot if we have at least 3 components
        if pca_result.shape[1] >= 3:
            scatter_data['PC3'] = pca_result[:, 2]
            
            # Create 3D scatter plot
            ax3.scatter(scatter_data['PC1'], 
                       scatter_data['PC2'], 
                       scatter_data['PC3'],
                       c=scatter_data['PC1'],  # Color by PC1 value
                       cmap='viridis',
                       alpha=0.6)
            
            ax3.set_xlabel('PC1')
            ax3.set_ylabel('PC2')
            ax3.set_zlabel('PC3')
            ax3.set_title(f'3D PCA - {metrics["feature_type"]} Features')
            
            # Add the cumulative explained variance for 3 components
            cum_var_3d = metrics['cumulative_variance_ratio'][2]
            ax3.text2D(0.05, 0.95, f'Cumulative Variance (3D): {cum_var_3d:.2%}', 
                     transform=ax3.transAxes)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def perform_hdbscan_clustering(
    pca_result: np.ndarray,
    n_components: int,
    min_cluster_size: int = 5,
    min_samples: int = None,
    metric: str = 'euclidean'
) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
    """
    Perform HDBSCAN clustering on PCA results.
    
    Parameters:
    -----------
    pca_result : np.ndarray
        PCA transformed data
    n_components : int
        Number of components to use for clustering
    min_cluster_size : int
        The minimum size of clusters
    min_samples : int
        The number of samples in a neighborhood for a point to be considered a core point
    metric : str
        The metric to use for distance computations
        
    Returns:
    --------
    Tuple containing:
    - Cluster labels
    - HDBSCAN object
    """
    # Select components
    data_for_clustering = pca_result[:, :n_components]
    
    # Initialize HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        gen_min_span_tree=True  # For visualization if needed
    )
    
    # Fit and predict
    clusterer.fit(data_for_clustering)
    
    return clusterer.labels_, clusterer

def visualize_clusters_projections(
    pca_result: np.ndarray,
    clusters: np.ndarray,
    n_components: int,
    metrics: Dict[str, Any],
    output_file: str
) -> None:
    """
    Create visualizations of clustering results in different PC projections.
    
    Parameters:
    -----------
    pca_result : np.ndarray
        PCA transformed data
    clusters : np.ndarray
        Cluster labels from HDBSCAN
    n_components : int
        Number of components used
    metrics : Dict[str, Any]
        Dictionary containing analysis metrics
    output_file : str
        Base path for saving output files
    """
    # Calculate number of plots needed
    n_plots = min(4, n_components)
    fig = plt.figure(figsize=(15, 15))
    
    # Create grid for plots
    if n_plots > 2:
        n_rows = 2
        n_cols = 2
    else:
        n_rows = 1
        n_cols = 2
    
    # Plot different PC combinations
    for i in range(n_plots):
        ax = plt.subplot(n_rows, n_cols, i+1)
        pc1 = i
        pc2 = (i + 1) % n_components
        
        scatter = ax.scatter(
            pca_result[:, pc1],
            pca_result[:, pc2],
            c=clusters,
            cmap='viridis',
            alpha=0.6
        )
        
        ax.set_xlabel(f'PC{pc1+1}')
        ax.set_ylabel(f'PC{pc2+1}')
        ax.set_title(f'Clusters in PC{pc1+1} vs PC{pc2+1}')
        
        # Add variance explained text
        var_explained = metrics['explained_variance_ratio'][pc1:pc2+1].sum()
        ax.text(0.05, 0.95, f'Var. Explained: {var_explained:.2%}',
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.colorbar(scatter)
    plt.tight_layout()
    
    # Save clustering visualization
    cluster_output = output_file.replace('.png', '_clusters.png')
    plt.savefig(cluster_output, dpi=300, bbox_inches='tight')
    plt.close()
    
def perform_3d_hdbscan(pca_result, min_cluster_size=5):
    """
    Perform HDBSCAN clustering on first 3 PCA components.
    """
    # Use only first 3 components
    data_3d = pca_result[:, :3]
    
    # Perform clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        gen_min_span_tree=True
    )
    
    cluster_labels = clusterer.fit_predict(data_3d)
    return cluster_labels, clusterer

def visualize_3d_clusters(pca_result, clusters, metrics, output_file):
    """
    Create interactive 3D visualization of clusters.
    """
    fig = plt.figure(figsize=(15, 15))
    
    # Create 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with clusters
    scatter = ax.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        pca_result[:, 2],
        c=clusters,
        cmap='viridis',
        alpha=0.6
    )
    
    # Add variance explained for each component
    var_text = f"Variance Explained:\nPC1: {metrics['explained_variance_ratio'][0]:.2%}\n"
    var_text += f"PC2: {metrics['explained_variance_ratio'][1]:.2%}\n"
    var_text += f"PC3: {metrics['explained_variance_ratio'][2]:.2%}\n"
    var_text += f"Total: {sum(metrics['explained_variance_ratio'][:3]):.2%}"
    
    ax.text2D(0.05, 0.95, var_text, transform=ax.transAxes,
              bbox=dict(facecolor='white', alpha=0.8))
    
    # Labels
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D Clustering Results')
    
    # Add colorbar
    plt.colorbar(scatter)
    
    # Add multiple viewing angles
    views = [(45, 45), (0, 0), (90, 0), (0, 90)]
    for i, (elev, azim) in enumerate(views):
        ax.view_init(elev, azim)
        view_file = output_file.replace('.png', f'_view{i+1}.png')
        plt.savefig(view_file, dpi=300, bbox_inches='tight')
    
    plt.close()

def analyze_3d_clusters(pca_result, clusters):
    """
    Analyze the quality of 3D clustering.
    """
    n_clusters = len(np.unique(clusters[clusters >= 0]))
    noise_points = np.sum(clusters == -1)
    cluster_sizes = np.bincount(clusters[clusters >= 0])
    
    # Calculate cluster separation in 3D space
    data_3d = pca_result[:, :3]
    
    analysis = {
        'n_clusters': n_clusters,
        'noise_points': noise_points,
        'cluster_sizes': cluster_sizes,
        'noise_percentage': noise_points / len(clusters) * 100,
        'data_separation': data_3d
    }
    
    return analysis

def process_3d_clustering(data, name, feature_type, **kwargs):
    """
    Main function for 3D PCA and clustering analysis.
    """
    # Get PCA results
    pca_result, pca, metrics = process_feature_subset(data, feature_type)
    
    # Check if 3D explains enough variance (e.g., >60%)
    total_var_3d = sum(metrics['explained_variance_ratio'][:3])
    print(f"\nTotal variance explained in 3D: {total_var_3d:.2%}")
    
    # Perform 3D clustering
    cluster_labels, clusterer = perform_3d_hdbscan(pca_result)
    
    # Create visualizations
    folder_name = f"pca_{name}"
    os.makedirs(folder_name, exist_ok=True)
    output_file = os.path.join(folder_name, f"pca_3d_{name}_{feature_type}.png")
    
    visualize_3d_clusters(pca_result, cluster_labels, metrics, output_file)
    
    # Analyze clusters
    cluster_analysis = analyze_3d_clusters(pca_result, cluster_labels)
    
    # Print results
    print("\nClustering Results:")
    print(f"Number of clusters: {cluster_analysis['n_clusters']}")
    print(f"Noise points: {cluster_analysis['noise_points']} ({cluster_analysis['noise_percentage']:.1f}%)")
    print("Cluster sizes:", cluster_analysis['cluster_sizes'])
    print(f"Output saved to: {output_file}")
    
    return cluster_labels, metrics
def process_pca_results(
    data: pd.DataFrame,
    name: str,
    feature_type: str,
    min_cluster_size: int = 5,
    **kwargs
) -> None:
    """
    Main function to process PCA results and create visualizations with clustering.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input feature data
    name : str
        Output filename
    feature_type : str
        Type of features being processed
    min_cluster_size : int
        Minimum cluster size for HDBSCAN
    **kwargs :
        Additional filtering criteria
    """
    # Process the data
    pca_result, pca, metrics = process_feature_subset(
        data,
        feature_type=feature_type,
        **kwargs
    )
    
    # Determine number of components for 70% variance
    n_components = np.argmax(metrics['cumulative_variance_ratio'] >= 0.7) + 1
    
    # Perform clustering
    cluster_labels, clusterer = process_3d_clustering(
        data,
        name,
        feature_type,
        min_cluster_size=min_cluster_size
    )
    
    #results, cluster_labels = perform_clinical_analysis(pca_result, data_all, metrics, output_prefix= name)
    # Create folder if it doesn't exist
    folder_name = f"pca_{name}"
    os.makedirs(folder_name, exist_ok=True)
    output_file = os.path.join(folder_name, f"pca_{name}_{feature_type}.png")
    
    # Create standard PCA visualization
    #create_pca_visualization(pca_result, metrics, output_file)
    
    # Create clustering visualization
    #visualize_clusters_projections(pca_result, cluster_labels, n_components, metrics, output_file)
    
    # Print summary statistics
    print(f"\nAnalysis Summary for {feature_type} features:")
    print(f"Original features: {metrics['n_features_original']}")
    print(f"Selected features: {metrics['n_features_selected']}")
    print(f"Components needed for 70% variance: {n_components}")
    print(f"Number of clusters found: {len(np.unique(cluster_labels))}")
    print(f"Number of noise points: {np.sum(cluster_labels == -1)}")
    print(f"Cluster sizes: {np.bincount(cluster_labels[cluster_labels >= 0])}")
    print(f"Output saved to: {output_file}")
    return pca_result, metrics

def analyze_cluster_os_association(
    cluster_labels: np.ndarray,
    os_values: np.ndarray,
    output_prefix: str
) -> Dict[str, Any]:
    """
    Analyze the association between cluster assignments and OS groups.
    
    Parameters:
    -----------
    cluster_labels : np.ndarray
        Cluster labels from HDBSCAN
    os_values : np.ndarray
        Binary OS values (0/1)
    output_prefix : str
        Prefix for output files
        
    Returns:
    --------
    Dict containing statistical results and metrics
    """
    # Remove noise points (-1) for analysis
    valid_indices = cluster_labels != -1
    clean_clusters = cluster_labels[valid_indices]
    clean_os = os_values[valid_indices]
    
    # Calculate association metrics
    ari = adjusted_rand_score(clean_os, clean_clusters)
    ami = adjusted_mutual_info_score(clean_os, clean_clusters)
    
    # Perform chi-square test
    contingency_table = pd.crosstab(clean_os, clean_clusters)
    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    
    # Create visualizations
    os.makedirs(f"pca_{output_prefix}", exist_ok=True)
    
    # 1. Enhanced Heatmap with proportions
    plt.figure(figsize=(12, 8))
    
    # Calculate proportions within each cluster
    prop_table = contingency_table.div(contingency_table.sum(axis=0), axis=1) * 100
    
    # Plot heatmap with both counts and percentages
    sns.heatmap(contingency_table, 
                annot=np.array([[f'{val}\n({prop:.1f}%)' 
                                for val, prop in zip(row, prop_row)] 
                               for row, prop_row in zip(contingency_table.values, 
                                                      prop_table.values)]),
                fmt='',
                cmap='YlOrRd')
    
    plt.title('OS Groups Distribution Across Clusters\n(Count and Percentage)')
    plt.xlabel('Cluster')
    plt.ylabel('OS Group')
    
    plt.savefig(f"pca_{output_prefix}/os_cluster_heatmap.png", 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # 2. Stacked Bar Plot
    plt.figure(figsize=(12, 6))
    
    contingency_table_percentage = contingency_table.div(contingency_table.sum(axis=0), axis=1) * 100
    contingency_table_percentage.plot(kind='bar', 
                                    stacked=True,
                                    color=['#ff9999', '#66b3ff'])
    
    plt.title('OS Distribution in Each Cluster')
    plt.xlabel('OS Group')
    plt.ylabel('Percentage')
    plt.legend(title='Cluster')
    plt.xticks(rotation=0)
    
    # Add percentage labels on bars
    for c in plt.gca().containers:
        plt.gca().bar_label(c, fmt='%.1f%%', label_type='center')
    
    plt.savefig(f"pca_{output_prefix}/os_cluster_bars.png", 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # 3. Pie Charts Grid for each cluster
    n_clusters = len(contingency_table.columns)
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)
    
    for i, cluster in enumerate(contingency_table.columns):
        row = i // n_cols
        col = i % n_cols
        
        cluster_data = contingency_table[cluster]
        total = cluster_data.sum()
        
        axes[row, col].pie(cluster_data, 
                          labels=[f'OS=0\n{cluster_data[0]/total*100:.1f}%',
                                 f'OS=1\n{cluster_data[1]/total*100:.1f}%'],
                          colors=['#ff9999', '#66b3ff'],
                          autopct='%1.1f%%')
        axes[row, col].set_title(f'Cluster {cluster}\n(n={total})')
    
    # Remove empty subplots if any
    for i in range(n_clusters, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])
    
    plt.suptitle('OS Distribution Within Each Cluster', y=1.02, fontsize=16)
    plt.tight_layout()
    
    plt.savefig(f"pca_{output_prefix}/os_cluster_pies.png", 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # Calculate proportions within each cluster
    cluster_proportions = {}
    for cluster in np.unique(clean_clusters):
        cluster_mask = clean_clusters == cluster
        cluster_os = clean_os[cluster_mask]
        prop_os_1 = np.mean(cluster_os)
        cluster_proportions[cluster] = {
            'OS=1': prop_os_1,
            'OS=0': 1 - prop_os_1,
            'size': sum(cluster_mask)
        }
    
    # Perform Fisher's exact test for each cluster vs others
    fisher_results = {}
    for cluster in np.unique(clean_clusters):
        # Create 2x2 contingency table for this cluster vs others
        cluster_mask = clean_clusters == cluster
        table = np.array([
            [sum((clean_clusters == cluster) & (clean_os == 1)),
             sum((clean_clusters != cluster) & (clean_os == 1))],
            [sum((clean_clusters == cluster) & (clean_os == 0)),
             sum((clean_clusters != cluster) & (clean_os == 0))]
        ])
        
        # Perform Fisher's exact test
        _, p_value = stats.fisher_exact(table)
        fisher_results[cluster] = p_value
    
    results = {
        'adjusted_rand_index': ari,
        'adjusted_mutual_info': ami,
        'chi2_statistic': chi2,
        'chi2_p_value': p_value,
        'cluster_proportions': cluster_proportions,
        'fisher_exact_tests': fisher_results,
        'contingency_table': contingency_table,
        'n_valid_samples': len(clean_clusters),
        'n_noise_points': sum(~valid_indices)
    }
    
    return results


def print_cluster_os_analysis(results: Dict[str, Any]) -> None:
    """
    Print formatted analysis results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Dictionary containing analysis results
    """
    print("\nCluster-OS Association Analysis")
    print("-" * 40)
    
    print("\nSample Size:")
    print(f"Valid samples: {results['n_valid_samples']}")
    print(f"Noise points: {results['n_noise_points']}")
    
    print("\nGlobal Association Metrics:")
    print(f"Adjusted Rand Index: {results['adjusted_rand_index']:.3f}")
    print(f"Adjusted Mutual Information: {results['adjusted_mutual_info']:.3f}")
    print(f"Chi-square test p-value: {results['chi2_p_value']:.3e}")
    
    print("\nCluster-specific Analysis:")
    for cluster, props in results['cluster_proportions'].items():
        print(f"\nCluster {cluster}:")
        print(f"Size: {props['size']} samples")
        print(f"OS=1 proportion: {props['OS=1']:.2%}")
        print(f"OS=0 proportion: {props['OS=0']:.2%}")
        print(f"Fisher's exact test p-value: {results['fisher_exact_tests'][cluster]:.3e}")
    
    print("\nContingency Table:")
    print(results['contingency_table'])

def perform_clinical_analysis(
    pca_result: np.ndarray,
    data_all: pd.DataFrame,
    metrics: Dict[str, Any],
    output_prefix: str,
    min_cluster_size: int = 5
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Perform clustering and analyze associations with clinical outcomes.
    
    Parameters:
    -----------
    pca_result : np.ndarray
        PCA transformed data
    data_all : pd.DataFrame
        Complete dataset including OS
    metrics : Dict[str, Any]
        Dictionary containing PCA metrics
    output_prefix : str
        Prefix for output files
    min_cluster_size : int
        Minimum cluster size for HDBSCAN
        
    Returns:
    --------
    Tuple containing:
    - Dictionary with analysis results
    - Cluster labels
    """
    n_components = np.argmax(metrics['cumulative_variance_ratio'] >= 0.8) + 1
    # Perform clustering
    cluster_labels, _ = perform_hdbscan_clustering(pca_result, n_components, min_cluster_size)
    
    # Analyze association with OS
    os_values = data_all['SEX'].values
    results = analyze_cluster_os_association(cluster_labels, os_values, output_prefix)
    
    # Print results
    print_cluster_os_analysis(results)
    
    return results, cluster_labels

# Get the directory containing your Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to your CSV file
file_path = os.path.join(script_dir, 'binarized_data.csv')

# Now use file_path in your code
data = pd.read_csv(file_path)
data_all = data
print(data_all.head)
if __name__ == "__main__":
    # Example usage for all feature types
    feature_sets = {
        'data':{'data': data, 'type': 'all'},
        'clinical_features': {'data': clinical_features, 'type': 'clinical'},
        'radiomic_features': {'data': all_radiomic_features, 'type': 'radiomic'},
        'original_features': {'data': original_features, 'type': 'radiomic', 'image_type': 'original'},
        'firstorder_features': {'data': firstorder_features, 'type': 'radiomic', 'category': 'firstorder'},
        'glcm_features': {'data': glcm_features, 'type': 'radiomic', 'category': 'glcm'},
        'log_sigma_features': {'data': log_sigma_features, 'type': 'radiomic', 'image_type': 'log-sigma'},
        'wavelet_features': {'data': wavelet_features, 'type': 'radiomic', 'image_type': 'wavelet'},
        'glrlm_features': {'data': glrlm_features, 'type': 'radiomic', 'category': 'glrlm'},
        'glszm_features': {'data': glszm_features, 'type': 'radiomic', 'category': 'glszm'},
        'shape_features': {'data': shape_features, 'type': 'radiomic', 'category': 'shape'},
        'glcm_wavelet_features': {'data': glcm_wavelet_features, 'type': 'radiomic', 'image_type': 'wavelet', 'category': 'glcm'}
    }
    
    # Process each feature set
    for name, params in feature_sets.items():
        # Create folder if it doesn't exist
        output_file = f"pca_{name}"
        data = params.pop('data')
        type_data = params.pop('type')
        pca_results, metrics = process_pca_results(data, output_file, type_data, **params)
        results, cluster_labels = perform_clinical_analysis(pca_results, data_all, metrics, output_prefix=name + "_SEX_")
        
#%%
import numpy as np
import pandas as pd
from typing import Tuple

import numpy as np
import pandas as pd
from typing import Tuple

def preprocess_riss(riss_values: np.ndarray) -> np.ndarray:
    """
    Preprocess R-ISS values to standardize them to 1, 2, or 3.
    
    Parameters:
    -----------
    riss_values : np.ndarray
        Array of R-ISS values that may include decimal values
        
    Returns:
    --------
    np.ndarray
        Standardized R-ISS values (1, 2, or 3)
    """
    # Convert to numeric if not already
    riss_numeric = pd.to_numeric(riss_values, errors='coerce')
    
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
    standardized_riss = np.array([round_to_stage(x) for x in riss_numeric])
    
    # Print summary of changes
    print("\nR-ISS Standardization Summary:")
    print("-" * 30)
    #○print("Original unique values:", np.unique(riss_values))
    print("Standardized unique values:", np.unique(standardized_riss[~np.isnan(standardized_riss)]))
    
    # Count samples in each category
    unique, counts = np.unique(standardized_riss[~np.isnan(standardized_riss)], return_counts=True)
    for value, count in zip(unique, counts):
        print(f"R-ISS {int(value)}: {count} samples")
    
    return standardized_riss

def analyze_cluster_riss_association(
    cluster_labels: np.ndarray,
    riss_values: np.ndarray,
    output_prefix: str
) -> dict:
    """
    Analyze the association between cluster assignments and R-ISS stages.
    
    Parameters:
    -----------
    cluster_labels : np.ndarray
        Cluster labels from HDBSCAN
    riss_values : np.ndarray
        R-ISS values (1, 2, or 3)
    output_prefix : str
        Prefix for output files
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Remove noise points (-1) and any NaN R-ISS values
    valid_indices = (cluster_labels != -1) & (~np.isnan(riss_values))
    clean_clusters = cluster_labels[valid_indices]
    clean_riss = riss_values[valid_indices]
    
    # Create contingency table
    contingency_table = pd.crosstab(clean_riss, clean_clusters)
    
    # Create visualizations
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Ensure output directory exists
    os.makedirs(f"pca_{output_prefix}", exist_ok=True)
    
    # 1. Enhanced Heatmap
    plt.figure(figsize=(12, 8))
    prop_table = contingency_table.div(contingency_table.sum(axis=0), axis=1) * 100
    
    sns.heatmap(contingency_table, 
                annot=np.array([[f'{val}\n({prop:.1f}%)' 
                                for val, prop in zip(row, prop_row)] 
                               for row, prop_row in zip(contingency_table.values, 
                                                      prop_table.values)]),
                fmt='',
                cmap='YlOrRd')
    
    plt.title('R-ISS Stage Distribution Across Clusters\n(Count and Percentage)')
    plt.xlabel('Cluster')
    plt.ylabel('R-ISS Stage')
    
    plt.savefig(f"pca_{output_prefix}/riss_cluster_heatmap.png", 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # 2. Stacked Bar Plot
    plt.figure(figsize=(12, 6))
    contingency_table_percentage = contingency_table.div(contingency_table.sum(axis=0), axis=1) * 100
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']  # One color for each R-ISS stage
    contingency_table_percentage.plot(kind='bar', 
                                    stacked=True,
                                    color=colors[:len(contingency_table_percentage.index)])
    
    plt.title('R-ISS Distribution in Each Cluster')
    plt.xlabel('R-ISS Stage')
    plt.ylabel('Percentage')
    plt.legend(title='Cluster')
    plt.xticks(rotation=0)
    
    # Add percentage labels on bars
    for c in plt.gca().containers:
        plt.gca().bar_label(c, fmt='%.1f%%', label_type='center')
    
    plt.savefig(f"pca_{output_prefix}/riss_cluster_bars.png", 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # Perform statistical tests
    from scipy import stats
    
    # Chi-square test
    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    
    # Calculate cluster proportions
    cluster_proportions = {}
    for cluster in np.unique(clean_clusters):
        cluster_mask = clean_clusters == cluster
        cluster_riss = clean_riss[cluster_mask]
        proportions = {
            f'R-ISS={stage}': np.mean(cluster_riss == stage)
            for stage in [1, 2, 3]
        }
        proportions['size'] = sum(cluster_mask)
        cluster_proportions[cluster] = proportions
    
    # Store results
    results = {
        'contingency_table': contingency_table,
        'chi2_statistic': chi2,
        'chi2_p_value': p_value,
        'cluster_proportions': cluster_proportions,
        'n_valid_samples': len(clean_clusters),
        'n_noise_points': sum(cluster_labels == -1),
        'n_missing_riss': sum(np.isnan(riss_values))
    }
    
    # Print summary
    print("\nR-ISS - Cluster Association Analysis")
    print("-" * 40)
    print("\nSample Size:")
    print(f"Valid samples: {results['n_valid_samples']}")
    print(f"Noise points: {results['n_noise_points']}")
    print(f"Missing R-ISS: {results['n_missing_riss']}")
    print(f"\nChi-square test p-value: {results['chi2_p_value']:.3e}")
    
    print("\nCluster-specific Analysis:")
    for cluster, props in results['cluster_proportions'].items():
        print(f"\nCluster {cluster}:")
        print(f"Size: {props['size']} samples")
        for stage in [1, 2, 3]:
            print(f"R-ISS={stage} proportion: {props[f'R-ISS={stage}']:.2%}")
    
    return results
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os
from datetime import datetime
def create_analysis_report(
    results: dict,
    metrics: dict,
    n_components: int,
    feature_type: str,
    output_prefix: str
) -> None:
    """
    Create a PDF report summarizing the clustering analysis results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results
    metrics : dict
        Dictionary containing PCA metrics
    n_components : int
        Number of PCA components used
    feature_type : str
        Type of features analyzed
    output_prefix : str
        Prefix for output file name
    """
    # Create output directory if it doesn't exist
    os.makedirs(f"pca_{output_prefix}", exist_ok=True)
    
    # Initialize PDF document
    pdf_file = f"pca_{output_prefix}/analysis_report_{feature_type}.pdf"
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create custom style for small text
    small_style = ParagraphStyle(
        'SmallText',
        parent=styles['Normal'],
        fontSize=8,
        leading=10
    )
    
    # Initialize story (content) for the PDF
    story = []
    
    # Add title
    story.append(Paragraph(f"Clustering Analysis Report - {feature_type}", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", small_style))
    story.append(Spacer(1, 20))
    
    # Add PCA Information
    story.append(Paragraph("PCA Information", heading_style))
    story.append(Paragraph(
        f"Number of PCA components used: {n_components} " +
        f"(explaining {metrics['cumulative_variance_ratio'][n_components-1]:.1%} of variance)",
        normal_style
    ))
    story.append(Spacer(1, 20))
    
    # Add Sample Size Information
    story.append(Paragraph("Sample Information", heading_style))
    sample_data = [
        ["Metric", "Count"],
        ["Valid samples", str(results['n_valid_samples'])],
        ["Noise points", str(results['n_noise_points'])],
        ["Missing R-ISS", str(results['n_missing_riss'])]
    ]
    sample_table = Table(sample_data, colWidths=[2*inch, 1*inch])
    sample_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(sample_table)
    story.append(Spacer(1, 20))
    
    # Add Statistical Analysis
    story.append(Paragraph("Statistical Analysis", heading_style))
    story.append(Paragraph(
        f"Chi-square test p-value: {results['chi2_p_value']:.3e}",
        normal_style
    ))
    story.append(Spacer(1, 20))
    
    # Add Cluster-specific Analysis
    story.append(Paragraph("Cluster-specific Analysis", heading_style))
    
    # Create table data for cluster analysis
    cluster_data = [["Cluster", "Size"] + [f"R-ISS {stage}" for stage in [1, 2, 3]]]
    for cluster, props in results['cluster_proportions'].items():
        row = [
            f"Cluster {cluster}",
            str(props['size'])
        ]
        for stage in [1, 2, 3]:
            row.append(f"{props[f'R-ISS={stage}']:.1%}")
        cluster_data.append(row)
    
    # Create and style the cluster analysis table
    cluster_table = Table(cluster_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    cluster_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(cluster_table)
    story.append(Spacer(1, 20))
    
    # Add figures reference
    story.append(Paragraph("Visualizations", heading_style))
    story.append(Paragraph(
        "The following visualizations have been generated and saved separately:",
        normal_style
    ))
    story.append(Paragraph(
        "1. R-ISS Stage Distribution Heatmap (riss_cluster_heatmap.png)",
        normal_style
    ))
    story.append(Paragraph(
        "2. R-ISS Distribution Bar Plot (riss_cluster_bars.png)",
        normal_style
    ))
    
    # Add interpretation guide
    story.append(Spacer(1, 20))
    story.append(Paragraph("Interpretation Guide", heading_style))
    story.append(Paragraph(
        "1. Check if the number of clusters matches the expected R-ISS groups (3).",
        normal_style
    ))
    story.append(Paragraph(
        "2. Look for clusters that have high proportions of specific R-ISS stages.",
        normal_style
    ))
    story.append(Paragraph(
        "3. Consider the chi-square p-value for statistical significance of the association.",
        normal_style
    ))
    story.append(Paragraph(
        "4. Pay attention to the number of noise points, as high noise might indicate poor clustering.",
        normal_style
    ))
    
    # Build the PDF
    doc.build(story)
    print(f"\nAnalysis report saved as: {pdf_file}")

def update_perform_riss_analysis(
    pca_result: np.ndarray,
    data_all: pd.DataFrame,
    metrics: dict,
    output_prefix: str,
    feature_type: str,
    min_cluster_size: int = 5
) -> None:
    """
    Updated function to perform analysis and generate report.
    """
    # Your existing analysis code here...
    results, cluster_labels = perform_riss_analysis(
        pca_result,
        data_all,
        metrics,
        output_prefix,
        min_cluster_size
    )
    
    # Determine number of components for 70% variance
    n_components = np.argmax(metrics['cumulative_variance_ratio'] >= 0.7) + 1
    
    # Generate the PDF report
    create_analysis_report(
        results,
        metrics,
        n_components,
        feature_type,
        output_prefix
    )
    
    return results, cluster_labels
def perform_riss_analysis(
    pca_result: np.ndarray,
    data_all: pd.DataFrame,
    metrics: dict,
    output_prefix: str,
    min_cluster_size: int = 5
) -> Tuple[dict, np.ndarray]:
    """
    Perform clustering and analyze associations with R-ISS stages.
    
    Parameters:
    -----------
    pca_result : np.ndarray
        PCA transformed data
    data_all : pd.DataFrame
        Complete dataset including R-ISS
    metrics : dict
        Dictionary containing PCA metrics
    output_prefix : str
        Prefix for output files
    min_cluster_size : int
        Minimum cluster size for HDBSCAN
        
    Returns:
    --------
    Tuple containing:
    - Dictionary with analysis results
    - Cluster labels
    """
    # Determine number of components for 70% variance
    n_components = np.argmax(metrics['cumulative_variance_ratio'] >= 0.9) + 1
    print(f"\nUsing {n_components} PCA components (explaining {metrics['cumulative_variance_ratio'][n_components-1]:.1%} of variance)")
    
    # Perform clustering using optimal number of components
    from hdbscan import HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(pca_result[:, :n_components])
    
    # Preprocess R-ISS values
    riss_values = preprocess_riss(data_all['R_ISS'].values)
    
    # Analyze association with R-ISS
    results = analyze_cluster_riss_association(cluster_labels, riss_values, output_prefix)
    
    return results, cluster_labels


# Get the directory containing your Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to your CSV file
file_path = os.path.join(script_dir, 'binarized_data.csv')

# Now use file_path in your code
data = pd.read_csv(file_path)

data_all = data

if __name__ == "__main__":
    # Example usage for all feature types
    feature_sets = {
        'data':{'data': data_all, 'type': 'all'},
        'clinical_features': {'data': clinical_features, 'type': 'clinical'},
        'radiomic_features': {'data': all_radiomic_features, 'type': 'radiomic'},
        'original_features': {'data': original_features, 'type': 'radiomic', 'image_type': 'original'},
        'firstorder_features': {'data': firstorder_features, 'type': 'radiomic', 'category': 'firstorder'},
        'glcm_features': {'data': glcm_features, 'type': 'radiomic', 'category': 'glcm'},
        'log_sigma_features': {'data': log_sigma_features, 'type': 'radiomic', 'image_type': 'log-sigma'},
        'wavelet_features': {'data': wavelet_features, 'type': 'radiomic', 'image_type': 'wavelet'},
        'glrlm_features': {'data': glrlm_features, 'type': 'radiomic', 'category': 'glrlm'},
        'glszm_features': {'data': glszm_features, 'type': 'radiomic', 'category': 'glszm'},
        'shape_features': {'data': shape_features, 'type': 'radiomic', 'category': 'shape'},
        'glcm_wavelet_features': {'data': glcm_wavelet_features, 'type': 'radiomic', 'image_type': 'wavelet', 'category': 'glcm'}
    }
    
    # Process each feature set
    for name, params in feature_sets.items():
        # Create folder if it doesn't exist
        output_file = f"pca_{name}"
        data = params.pop('data')
        type_data = params.pop('type')
        pca_results, pca, metrics = process_feature_subset(data, type_data, variance_threshold=0.8, **params)
        results, cluster_labels = update_perform_riss_analysis(pca_results, data_all, metrics, output_prefix=name + "_R_ISS_", feature_type=type_data)
        
#%%
def analyze_cluster_os_association(
    cluster_labels: np.ndarray,
    os_values: np.ndarray,
    output_prefix: str
) -> dict:
    """
    Analyze the association between cluster assignments and OS groups.
    
    Parameters:
    -----------
    cluster_labels : np.ndarray
        Cluster labels from HDBSCAN
    os_values : np.ndarray
        OS values (0 or 1)
    output_prefix : str
        Prefix for output files
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Remove noise points (-1) and any NaN OS values
    valid_indices = (cluster_labels != -1) & (~np.isnan(os_values))
    clean_clusters = cluster_labels[valid_indices]
    clean_os = os_values[valid_indices]
    
    # Create contingency table
    contingency_table = pd.crosstab(clean_os, clean_clusters)
    
    # Create visualizations
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Ensure output directory exists
    os.makedirs(f"pca_{output_prefix}", exist_ok=True)
    
    # 1. Enhanced Heatmap
    plt.figure(figsize=(12, 8))
    prop_table = contingency_table.div(contingency_table.sum(axis=0), axis=1) * 100
    
    sns.heatmap(contingency_table, 
                annot=np.array([[f'{val}\n({prop:.1f}%)' 
                                for val, prop in zip(row, prop_row)] 
                               for row, prop_row in zip(contingency_table.values, 
                                                      prop_table.values)]),
                fmt='',
                cmap='YlOrRd')
    
    plt.title('OS Distribution Across Clusters\n(Count and Percentage)')
    plt.xlabel('Cluster')
    plt.ylabel('OS Group')
    
    plt.savefig(f"pca_{output_prefix}/os_cluster_heatmap.png", 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # 2. Stacked Bar Plot
    plt.figure(figsize=(12, 6))
    contingency_table_percentage = contingency_table.div(contingency_table.sum(axis=0), axis=1) * 100
    
    colors = ['#ff9999', '#66b3ff']  # One color for each OS group
    contingency_table_percentage.plot(kind='bar', 
                                    stacked=True,
                                    color=colors)
    
    plt.title('OS Distribution in Each Cluster')
    plt.xlabel('OS Group')
    plt.ylabel('Percentage')
    plt.legend(title='Cluster')
    plt.xticks(rotation=0)
    
    # Add percentage labels on bars
    for c in plt.gca().containers:
        plt.gca().bar_label(c, fmt='%.1f%%', label_type='center')
    
    plt.savefig(f"pca_{output_prefix}/os_cluster_bars.png", 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # 3. Pie Charts for each cluster
    n_clusters = len(np.unique(clean_clusters))
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    for i, cluster in enumerate(sorted(np.unique(clean_clusters))):
        plt.subplot(n_rows, n_cols, i+1)
        cluster_data = contingency_table[cluster]
        plt.pie(cluster_data, 
               labels=[f'OS=0\n{val/sum(cluster_data)*100:.1f}%' for val in cluster_data],
               colors=colors,
               autopct='%1.1f%%')
        plt.title(f'Cluster {cluster}\n(n={sum(cluster_data)})')
    
    plt.tight_layout()
    plt.savefig(f"pca_{output_prefix}/os_cluster_pies.png", 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # Perform statistical tests
    from scipy import stats
    
    # Chi-square test
    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    
    # Fisher's exact test for each cluster vs others
    fisher_results = {}
    for cluster in np.unique(clean_clusters):
        # Create 2x2 table for this cluster vs others
        cluster_mask = clean_clusters == cluster
        table = np.array([
            [sum((clean_clusters == cluster) & (clean_os == 1)),
             sum((clean_clusters != cluster) & (clean_os == 1))],
            [sum((clean_clusters == cluster) & (clean_os == 0)),
             sum((clean_clusters != cluster) & (clean_os == 0))]
        ])
        _, p_value_fisher = stats.fisher_exact(table)
        fisher_results[cluster] = p_value_fisher
    
    # Calculate cluster proportions
    cluster_proportions = {}
    for cluster in np.unique(clean_clusters):
        cluster_mask = clean_clusters == cluster
        cluster_os = clean_os[cluster_mask]
        proportions = {
            'OS=0': np.mean(cluster_os == 0),
            'OS=1': np.mean(cluster_os == 1)
        }
        proportions['size'] = sum(cluster_mask)
        cluster_proportions[cluster] = proportions
    
    # Store results
    results = {
        'contingency_table': contingency_table,
        'chi2_statistic': chi2,
        'chi2_p_value': p_value,
        'fisher_exact_tests': fisher_results,
        'cluster_proportions': cluster_proportions,
        'n_valid_samples': len(clean_clusters),
        'n_noise_points': sum(cluster_labels == -1),
        'n_missing_os': sum(np.isnan(os_values))
    }
    
    # Print summary
    print("\nOS - Cluster Association Analysis")
    print("-" * 40)
    print("\nSample Size:")
    print(f"Valid samples: {results['n_valid_samples']}")
    print(f"Noise points: {results['n_noise_points']}")
    print(f"Missing OS: {results['n_missing_os']}")
    print(f"\nChi-square test p-value: {results['chi2_p_value']:.3e}")
    
    print("\nCluster-specific Analysis:")
    for cluster, props in results['cluster_proportions'].items():
        print(f"\nCluster {cluster}:")
        print(f"Size: {props['size']} samples")
        print(f"OS=0 proportion: {props['OS=0']:.2%}")
        print(f"OS=1 proportion: {props['OS=1']:.2%}")
        print(f"Fisher's exact test p-value: {results['fisher_exact_tests'][cluster]:.3e}")
    
    return results