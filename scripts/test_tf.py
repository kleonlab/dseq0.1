import decoupler as dc
import pandas as pd

# 1. Download the CollecTRI network (Human)
# This dataframe contains Source (TF), Target (Gene), and Weight (Confidence/Mode)
net = dc.get_collectri(organism='human', split_complexes=False)

# 2. Filter for a specific Transcription Factor (e.g., TP53)
tf_name = "TP53"
related_genes = net[net['source'] == tf_name]

# 3. View the top targets
# 'mor' = Mode of Regulation (+1 is activation, -1 is inhibition)
print(f"Found {len(related_genes)} genes regulated by {tf_name}.")
print(related_genes.sort_values(by='weight', ascending=False).head(10))