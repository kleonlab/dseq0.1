import scanpy as sc
import pandas as pd
import sys
import os

def view_h5ad_structure(h5ad_path):
    """
    View all columns and metadata structure of an h5ad file.
    
    Parameters:
    - h5ad_path: Path to the .h5ad AnnData file.
    """
    
    print("="*80)
    print(f"Loading h5ad file from: {h5ad_path}")
    print("="*80)
    
    try:
        adata = sc.read_h5ad(h5ad_path)
    except FileNotFoundError:
        print(f"Error: File not found at {h5ad_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Basic structure
    print(f"\n{'BASIC STRUCTURE':^80}")
    print("-"*80)
    print(f"Shape: {adata.shape[0]} observations (cells) × {adata.shape[1]} variables (genes)")
    
    # DATA MATRIX X
    print(f"\n{'DATA MATRIX (adata.X)':^80}")
    print("-"*80)
    if hasattr(adata.X, 'toarray'):
        # It's a sparse matrix
        print("Type: Sparse Matrix")
        print(f"Non-zero elements: {adata.X.nnz}")
        # Get a small dense chunk to show values
        x_sample = adata.X[:10, :10].toarray()
        print("\nFirst 10x10 values (converted to dense):")
        print(x_sample)
    else:
        # It's a dense array
        print("Type: Dense Array")
        x_sample = adata.X[:10, :10]
        print("\nFirst 10x10 values:")
        print(x_sample)
        
    print(f"\nRange of values in sample: Min={x_sample.min()}, Max={x_sample.max()}")

    # Check for other data representations (layers, obsm)
    print(f"\n{'OTHER DATA REPRESENTATIONS':^80}")
    print("-"*80)
    
    # Check layers
    if len(adata.layers.keys()) > 0:
        print(f"\nLayers (adata.layers):")
        for key in adata.layers.keys():
            shape = adata.layers[key].shape if hasattr(adata.layers[key], 'shape') else 'N/A'
            dtype = adata.layers[key].dtype if hasattr(adata.layers[key], 'dtype') else 'N/A'
            print(f"  - '{key}': shape={shape}, dtype={dtype}")
            
            # Check if it might be control data
            if 'control' in key.lower():
                print(f"    (POTENTIAL CONTROL DATA FOUND: {key})")
    else:
        print("No additional layers found.")

    # Check obsm (multi-dimensional observations)
    if len(adata.obsm.keys()) > 0:
        print(f"\nMulti-dimensional observations (adata.obsm):")
        for key in adata.obsm.keys():
            shape = adata.obsm[key].shape if hasattr(adata.obsm[key], 'shape') else 'N/A'
            print(f"  - '{key}': shape={shape}")
            
            # Check content of small keys that might be interesting (not just embeddings)
            if shape != 'N/A' and len(shape) == 2 and shape[1] < 10:
                print(f"    First few rows of {key}:")
                try:
                    print(adata.obsm[key][:3])
                except:
                    pass
    else:
        print("No multi-dimensional observations found.")

    # Check uns (unstructured data) for keywords like 'control'
    print(f"\nChecking adata.uns for 'control' or related keywords:")
    found_control = False
    for key in adata.uns.keys():
        if 'control' in key.lower() or 'ref' in key.lower():
            print(f"  - Found potential control info in uns['{key}']: {type(adata.uns[key])}")
            found_control = True
    
    if not found_control:
        print("  - No explicit 'control' or 'ref' keys found in adata.uns")

    # Observations (cell) metadata
    print(f"\n{'OBSERVATIONS METADATA (adata.obs)':^80}")
    print("-"*80)
    if adata.obs.shape[1] > 0:
        print(f"Number of columns: {adata.obs.shape[1]}")
        print(f"Number of cells (rows): {adata.obs.shape[0]}")
        print(f"\nColumn names and unique values:")
        
        for col in adata.obs.columns:
            n_unique = adata.obs[col].nunique()
            print(f"\n--- Column: '{col}' (Length: {len(adata.obs[col])}) ---")
            
            # Check if this looks like the "cell states" column
            if n_unique < 50:
                print(f"Unique values ({n_unique}):")
                val_counts = adata.obs[col].value_counts()
                for val, count in val_counts.items():
                    print(f"  - '{val}': {count} cells")
            else:
                print(f"Too many unique values ({n_unique}). First 5 values:")
                print(adata.obs[col].head(5).tolist())
    else:
        print("No observation metadata found.")
    
    # Variables (gene) metadata
    print(f"\n{'VARIABLES METADATA (adata.var)':^80}")
    print("-"*80)
    if adata.var.shape[1] > 0:
        print(f"Number of columns: {adata.var.shape[1]}")
        print(f"Number of genes: {adata.var.shape[0]}")
        print(f"\nColumn names:")
        for col in adata.var.columns:
             print(f"  - {col}")
             
        # Also check the index, which often holds gene names
        print(f"\nIndex (first 10 values): {adata.var.index[:10].tolist()}")
    else:
        print("No variable metadata found. Gene names might be in the index.")
        print(f"Index (first 10 values): {adata.var.index[:10].tolist()}")
    

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    # Default paths
    default_paths = [
        'datasets/k562_5k.h5ad',
    ]
    
    # Check if path was provided as argument
    if len(sys.argv) > 1:
        h5ad_path = sys.argv[1]
        view_h5ad_structure(h5ad_path)
    else:
        # Try default paths
        print("No path provided. Trying default paths...")
        found = False
        for path in default_paths:
            if os.path.exists(path):
                found = True
                view_h5ad_structure(path)
                print("\n\n")
        
        if not found:
            print("Usage: python view_andata.py <path_to_h5ad_file>")
            print(f"\nOr place one of these files in the datasets directory:")
            for path in default_paths:
                print(f"  - {path}")

