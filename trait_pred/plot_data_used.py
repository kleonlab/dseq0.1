import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_cell_counts_per_gene(
    h5ad_path: str,
    obs_column: str = "gene_id",
    top_n: int | None = 30,
    output_image: str = "cell_counts_per_gene.png",
    output_csv: str = "cell_counts_per_gene.csv",
):
    """
    Bar-plot the number of cells per perturbed gene (counts of obs_column values).

    Parameters
    ----------
    h5ad_path : str
        Path to the .h5ad AnnData file.
    obs_column : str
        Column in adata.obs that encodes which gene/perturbation each cell belongs to.
    top_n : int | None
        If set, plot only the top_n most frequent genes. If None, plot all (can get huge).
    output_image : str
        Filename for the saved plot.
    output_csv : str
        Filename for saving the full counts table.
    """
    print(f"Loading dataset from {h5ad_path}...")
    try:
        adata = sc.read_h5ad(h5ad_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {h5ad_path}")
        return

    if obs_column not in adata.obs.columns:
        print(f"Error: Column '{obs_column}' not found in adata.obs.")
        print(f"Available columns: {list(adata.obs.columns)}")
        return

    # Fast + safe counts:
    # - if categorical, include only observed categories (not unused)
    # - always treat missing as missing (not the string "nan")
    s = adata.obs[obs_column]

    # Drop missing values (optional; keeps plot clean)
    s = s.dropna()

    if isinstance(s.dtype, pd.CategoricalDtype):
        # Remove unused categories to mimic observed=True and avoid errors on older pandas
        s = s.cat.remove_unused_categories()
        counts = s.value_counts(dropna=False)
    else:
        counts = s.astype(str).value_counts(dropna=False)

    counts_df = counts.rename_axis(obs_column).reset_index(name="Cell Count")
    counts_df.to_csv(output_csv, index=False)
    print(f"Full cell-counts table saved to {output_csv}")

    plot_df = counts_df.sort_values("Cell Count", ascending=False)
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    if plot_df.empty:
        print("Warning: No valid entries found to plot.")
        return

    print(f"Generating plot{' (top ' + str(top_n) + ')' if top_n is not None else ''}...")
    plt.figure(figsize=(max(10, 0.35 * len(plot_df)), 6))
    plt.bar(plot_df[obs_column].astype(str), plot_df["Cell Count"])
    plt.title(f"Number of Cells per Perturbed Gene" + (f" (Top {top_n})" if top_n else ""))
    plt.xlabel(obs_column)
    plt.ylabel("Number of Cells")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved successfully to {output_image}")


if __name__ == "__main__":
    DATASET_PATH = "datasets/k562_5k.h5ad"
    OBS_COLUMN_NAME = "gene"  # change if your obs uses a different key
    plot_cell_counts_per_gene(
        DATASET_PATH,
        obs_column=OBS_COLUMN_NAME,
        top_n=None,  # set None to plot all genes (careful: can be unreadable)
    )