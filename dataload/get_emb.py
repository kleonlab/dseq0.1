"""
Generate cell embeddings using the SE-600M model.
This script loads an h5ad file and generates embeddings using the Arc State SE-600M model.
"""
import os
import sys
import scanpy as sc
import subprocess
import numpy as np

# Configuration
MODEL_FOLDER = "/home/b5cc/sanjukta.b5cc/dseq0.1/models/se600m"
CHECKPOINT_PATH = "/home/b5cc/sanjukta.b5cc/dseq0.1/models/se600m/se600m_epoch16.ckpt"
INPUT_DIR = "/home/b5cc/sanjukta.b5cc/dseq0.1/datasets"
OUTPUT_DIR = "/home/b5cc/sanjukta.b5cc/dseq0.1/datasets/embeddings"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_embeddings(input_file, output_file=None):
    """
    Generate embeddings for a given h5ad file using the SE-600M model.
    
    Parameters:
    -----------
    input_file : str
        Path to input h5ad file
    output_file : str, optional
        Path to output h5ad file (with embeddings). If None, will use input filename.
    """
    # Validate input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Validate model folder exists
    if not os.path.exists(MODEL_FOLDER):
        raise FileNotFoundError(f"Model folder not found: {MODEL_FOLDER}")
    
    # Validate checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    # Set output file path
    if output_file is None:
        basename = os.path.basename(input_file)
        output_file = os.path.join(OUTPUT_DIR, basename.replace('.h5ad', '_embeddings.h5ad'))
    
    # Check if output already exists
    if os.path.exists(output_file):
        print(f"✅ Output file already exists: {output_file}")
        print(f"   Loading to verify...")
        adata = sc.read_h5ad(output_file)
        print(f"   Shape: {adata.shape}")
        if 'X_state' in adata.obsm:
            print(f"   Embeddings found in .obsm['X_state']: shape {adata.obsm['X_state'].shape}")
        return output_file
    
    # Load input file to check structure
    print(f"\n📂 Loading input file: {input_file}")
    adata = sc.read_h5ad(input_file)
    print(f"   Shape: {adata.shape}")
    print(f"   Observations (cells): {adata.n_obs}")
    print(f"   Variables (genes): {adata.n_vars}")
    
    # Check for NaNs in X and fix if needed
    if hasattr(adata.X, 'toarray'):
        is_nan = np.isnan(adata.X.data).any()
    else:
        is_nan = np.isnan(adata.X).any()
        
    actual_input_file = input_file
    temp_file = None
    
    if is_nan:
        print(f"\n⚠️  Warning: NaN values found in adata.X. Replacing with 0...")
        if hasattr(adata.X, 'toarray'):
            # Sparse matrix
            adata.X.data = np.nan_to_num(adata.X.data, nan=0.0)
        else:
            # Dense array
            adata.X = np.nan_to_num(adata.X, nan=0.0)
            
        # Save to temporary file
        temp_file = input_file.replace('.h5ad', '_cleaned_temp.h5ad')
        print(f"   Saving cleaned data to temporary file: {temp_file}")
        adata.write_h5ad(temp_file)
        actual_input_file = temp_file

    # Generate embeddings using the state CLI
    print(f"\n🚀 Generating embeddings using SE-600M model...")
    print(f"   Model: {MODEL_FOLDER}")
    print(f"   Checkpoint: {os.path.basename(CHECKPOINT_PATH)}")
    
    try:
        # Set environment variables to prevent OpenBLAS threading issues
        env = os.environ.copy()
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['MKL_NUM_THREADS'] = '1'
        env['OMP_NUM_THREADS'] = '1'
        
        cmd = [
            "state", "emb", "transform",
            "--model-folder", MODEL_FOLDER,
            "--checkpoint", CHECKPOINT_PATH,
            "--input", actual_input_file,
            "--output", output_file
        ]
        
        print(f"\n   Running command: {' '.join(cmd)}")
        print(f"   (with OPENBLAS_NUM_THREADS=1 to prevent crashes)")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print(result.stdout)
        
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            print(f"   Removing temporary file: {temp_file}")
            os.remove(temp_file)
        
        print(f"\n✅ Successfully generated embeddings!")
        print(f"   Output saved to: {output_file}")
        
        # Load and display results
        adata_out = sc.read_h5ad(output_file)
        print(f"\n📊 Output file info:")
        print(f"   Shape: {adata_out.shape}")
        if 'X_state' in adata_out.obsm:
            print(f"   Embeddings in .obsm['X_state']: shape {adata_out.obsm['X_state'].shape}")
        
        return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error generating embeddings:")
        print(f"   {e}")
        if e.stderr:
            print(f"   Error output: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"\n❌ Error: 'state' command not found.")
        print(f"   Please install arc-state: uv tool install arc-state")
        raise


def main():
    """
    Main function to generate embeddings for available h5ad files.
    """
    # Default input files
    input_files = [
        os.path.join(INPUT_DIR, "k562_5k.h5ad"),
        # Add more files here if needed
        # os.path.join(INPUT_DIR, "k562_500k.h5ad"),
    ]
    
    # Process each file
    for input_file in input_files:
        if os.path.exists(input_file):
            print(f"\n{'='*70}")
            print(f"Processing: {os.path.basename(input_file)}")
            print(f"{'='*70}")
            try:
                output_file = generate_embeddings(input_file)
                print(f"\n✅ Successfully processed {os.path.basename(input_file)}")
            except Exception as e:
                print(f"\n❌ Failed to process {os.path.basename(input_file)}: {e}")
        else:
            print(f"⚠️  Skipping {input_file} (file not found)")


if __name__ == "__main__":
    main()