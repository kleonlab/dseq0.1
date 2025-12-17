import scanpy as sc
import scipy.sparse

h5ad_file = "/home/b5cc/sanjukta.b5cc/dseq0.1/datasets/k562_5k.h5ad"
clean_file = "/home/b5cc/sanjukta.b5cc/dseq0.1/datasets/k562_5k_clean.h5ad"

print(f"Loading {h5ad_file}...")
adata = sc.read_h5ad(h5ad_file)
sc.pp.filter_cells(adata, min_counts=1)  # REQUIRED: Removes cells with 0 reads (NaN source)
sc.pp.filter_genes(adata, min_cells=1)

adata.write(clean_file)
print(f"Saved cleaned data to: {clean_file}")


print(f"Cleaned shape: {adata.shape}")

X = adata.X
print(f"Matrix shape: {X.shape}")
print(f"Matrix type: {type(X)}")
print(f"Number of cells (obs): {adata.n_obs}")
print(f"Number of genes (vars): {adata.n_vars}")
print(X[0])
print(len(X[0]))


if adata.raw is not None:
    print("its here")
else: 
    print("run the original breakdown again")



from cnmf import cNMF
import numpy as np

cnmf_obj = cNMF(output_dir="demo_output", name="k562_demo")

# Step A: Prepare
# We filter for high-variance genes (HVGs) to reduce noise, just like the paper
cnmf_obj.prepare(
    counts_fn=clean_file,
    components= 10, # In paper they tested K=30, 60, etc.
    n_iter=10,                   # Paper used 100+ iterations
    seed=14,
    num_highvar_genes=2000
)

# Step B: Factorize (Matrix Decomposition)
# This runs NMF multiple times. In a real run, you parallelize this (worker_i).
print("Running Factorization...")
cnmf_obj.factorize(worker_i=0, total_workers=1)

# Step C: Combine & Consensus
# Merges the iterations to find stable "Programs"
cnmf_obj.combine()
cnmf_obj.consensus(k=10, density_threshold=0.05, show_clustering=True)

print("cNMF Complete. Programs Discovered.")