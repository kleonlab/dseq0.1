
csv_path = "/home/b5cc/sanjukta.b5cc/dseq0.1/cell_counts_per_gene.csv"
img_path = "/home/b5cc/sanjukta.b5cc/dseq0.1/cell_counts_per_gene.png"

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
df = pd.read_csv(csv_path)

# Remove the first row (first value pair)
df = df.iloc[1:]

# Assuming the first column is gene names and second is counts
# We can access them by position if names are unknown, or by expected names
gene_col = df.columns[0]
count_col = df.columns[1]

plt.figure(figsize=(12, 6))

# Create bar plot
plt.bar(df[gene_col], df[count_col])

# Remove x-axis ticks and labels
plt.xticks([])
plt.xlabel("Genes")  # Optional: keep the label but remove individual gene names if desired, or remove entirely. 
# User said "no ticks or names on the xaxis", usually means no individual labels. 
# I will keep the axis label "Gene" generic or remove it if strictly "no names" means no label either.
# I'll leave a generic label but remove ticks/ticklabels.

plt.ylabel("Count, with 180 control cells")
plt.title("Cell Counts per Gene, for 3k perturbed genes")

# Save the plot
plt.tight_layout()
plt.savefig(img_path, dpi=300)
print(f"Plot saved to {img_path}")
