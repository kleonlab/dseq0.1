import anndata
import os
import numpy as np

# Make sure you have anndata installed: pip install anndata

file_path = "datasets/k562.h5ad"
output_path = "datasets/k562_500k.h5ad"
target_n_cells = 50000

# Load the entire AnnData object
# NOTE: This requires significant RAM (~128 GB is often recommended for 61 GB H5AD files)
if os.path.exists(file_path):
    print(f"Loading {file_path}...")
    adata = anndata.read_h5ad(file_path)
    print("AnnData object loaded successfully!")
    
    print("\n--- Dataset Info ---")
    print(f"Shape: {adata.shape}")
    print(f"Obs (Cell) variables: {list(adata.obs.columns)}")
    print(f"Var (Gene) variables: {list(adata.var.columns)}")
    print(f"\nFull AnnData object:\n{adata}")

    # Subsample
    if adata.n_obs > target_n_cells:
        print(f"\nSubsampling to {target_n_cells} cells...")
        # Use numpy to generate random indices without replacement
        indices = np.random.choice(adata.n_obs, target_n_cells, replace=False)
        adata_sampled = adata[indices].copy()
        
        print(f"Sampled Shape: {adata_sampled.shape}")
        
        print(f"Saving sampled dataset to {output_path}...")
        adata_sampled.write_h5ad(output_path)
        print("Saved successfully.")
    else:
        print(f"\nDataset has fewer than {target_n_cells} cells. Saving copy as is.")
        adata.write_h5ad(output_path)
        print("Saved successfully.")
        
else:
    print(f"Error: File not found at {file_path}")
