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
        # Load CSV and extract Ensembl IDs
        tf_df = pd.read_csv(csv_path)
        if 'Ensembl ID' not in tf_df.columns:
            print("Error: 'Ensembl ID' column not found in CSV.")
            return
            
        target_tfs = tf_df['Ensembl ID'].unique().tolist()
        print(f"Found {len(target_tfs)} unique Ensembl IDs in CSV.")

        # Create mapping from Ensembl ID to HGNC symbol
        if 'HGNC symbol' in tf_df.columns:
            id_to_gene = tf_df.set_index('Ensembl ID')['HGNC symbol'].to_dict()
        else:
            print("Warning: 'HGNC symbol' column not found in CSV. Gene names will not be available.")
            id_to_gene = {}

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
    # Convert to string to avoid counting unused categories if column is categorical
    counts = filtered_obs[obs_column].astype(str).value_counts().reset_index()
    counts.columns = ['Ensembl ID', 'Cell Count']

    # Add Gene Name column
    counts['Gene Name'] = counts['Ensembl ID'].map(id_to_gene)

    # Save full list to CSV
    output_csv = 'tf_cell_counts_data.csv'
    counts.to_csv(output_csv, index=False)
    print(f"Full TF cell counts saved to {output_csv}")

    # Filter top 10 for plotting
    top_10_counts = counts.sort_values('Cell Count', ascending=False).head(10)

    # Plotting
    print("Generating plot for top 10 TFs...")
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Create bar plot
    # Use Gene Name for X-axis if available, otherwise Ensembl ID
    x_col = 'Gene Name' if not top_10_counts['Gene Name'].isna().all() else 'Ensembl ID'
    
    ax = sns.barplot(
        data=top_10_counts,
        x=x_col,
        y='Cell Count',
        palette='viridis'
    )
    
    # Add labels on top of bars
    for i in ax.containers:
        ax.bar_label(i, padding=3)

    plt.title('Number of Cells per Transcription Factor Edit (Top 10)', fontsize=16)
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
    
    # Based on your dataset info, we expect a column holding the 
    # Ensembl ID (e.g., 'ensembl_id').
    OBS_COLUMN_NAME = 'gene_id' 

    plot_tf_cell_counts(DATASET_PATH, TF_LIST_PATH, OBS_COLUMN_NAME)