import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_tf_cell_counts(h5ad_path, csv_path, obs_column='gene', output_image='tf_cell_counts.png'):
    """
    Plots the number of cells corresponding to specific transcription factors.
    
    Parameters:
    - h5ad_path: Path to the .h5ad AnnData file.
    - csv_path: Path to the .csv file containing the list of TFs.
    - obs_column: The column in adata.obs representing the editing compound/gene.
    - output_image: Filename for the saved plot.
    """
    
    print(f"Loading dataset from {h5ad_path}...")
    try:
        adata = sc.read_h5ad(h5ad_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {h5ad_path}")
        return

    print(f"Loading TF list from {csv_path}...")
    try:
        # Assuming the CSV has a header. If the TF names are in the first column:
        tf_df = pd.read_csv(csv_path)
        # We take the first column as the list of TFs
        target_tfs = tf_df.iloc[:, 0].unique().tolist()
        print(f"Found {len(target_tfs)} unique TFs in CSV.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Check if the column exists in the dataset
    if obs_column not in adata.obs.columns:
        print(f"Error: Column '{obs_column}' not found in adata.obs.")
        print(f"Available columns: {list(adata.obs.columns)}")
        return

    # Filter the metadata (obs) for cells where the gene is in our target list
    print("Filtering cells...")
    filtered_obs = adata.obs[adata.obs[obs_column].isin(target_tfs)]
    
    if filtered_obs.empty:
        print("Warning: No cells found matching the provided Transcription Factors.")
        return

    # Count cells per TF
    counts = filtered_obs[obs_column].value_counts().reset_index()
    counts.columns = ['Transcription Factor', 'Cell Count']

    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Create bar plot
    ax = sns.barplot(
        data=counts,
        x='Transcription Factor',
        y='Cell Count',
        palette='viridis'
    )
    
    # Add labels on top of bars
    for i in ax.containers:
        ax.bar_label(i, padding=3)

    plt.title('Number of Cells per Transcription Factor Edit', fontsize=16)
    plt.xlabel('Transcription Factor', fontsize=12)
    plt.ylabel('Number of Cells', fontsize=12)
    plt.xticks(rotation=45, ha='right') # Rotate labels for better visibility
    plt.tight_layout()

    # Save
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved successfully to {output_image}")

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = 'datasets/k562.h5ad'
    TF_LIST_PATH = 'datasets/DatabaseExtract_v_1.01.csv'
    
    # Based on your dataset info, 'gene' seems to be the column holding the 
    # editing compound name (e.g., the target gene).
    OBS_COLUMN_NAME = 'gene' 

    plot_tf_cell_counts(DATASET_PATH, TF_LIST_PATH, OBS_COLUMN_NAME)