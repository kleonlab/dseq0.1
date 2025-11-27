from get_uniprot_seq import map_ensembl_to_uniprot, fetch_fasta_for_accessions, fetch_uniprot_metadata
import time
from io import StringIO
import requests
import pandas as pd
from pathlib import Path

rootpath = Path(__file__).resolve().parents[1]
tf_raw_path = rootpath / "datasets" / "DatabaseExtract_v_1.01.csv"
root_output_path = rootpath / "datasets" / "fasta_data"
failed_ids_path = rootpath / "datasets" / "failed_ensembl_ids.txt"

# Ensure output directory exists
root_output_path.mkdir(parents=True, exist_ok=True)

tf_pd = pd.read_csv(tf_raw_path)
def save_metadata_csv(meta_data, output_folder, filename="uniprot_metadata.csv"):
    """
    Save combined UniProt metadata to a CSV file.

    Args:
        meta_data: list of DataFrames OR a single DataFrame
        output_folder: Path or str to directory where CSV will be saved
        filename: name of the output CSV file
    """

    # Ensure folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Normalize input: meta_data may be a single DF or a list of DFs
    if isinstance(meta_data, pd.DataFrame):
        combined_df = meta_data

    elif isinstance(meta_data, list):
        # Filter out None or empty items
        valid_frames = [
            df for df in meta_data
            if isinstance(df, pd.DataFrame) and not df.empty
        ]

        if not valid_frames:
            print("Warning: No valid metadata to save.")
            return

        combined_df = pd.concat(valid_frames, ignore_index=True)

    else:
        raise ValueError("meta_data must be a DataFrame or list of DataFrames.")

    # Save to CSV
    output_path = output_folder / filename
    combined_df.to_csv(output_path, index=False)

    print(f"Metadata saved to {output_path}")
    return output_path

def save_fasta(fasta_dict, output_path):
    """
    Save a dict {id: sequence} to a FASTA file.

    Args:
        fasta_dict (dict): mapping acc_id -> AA sequence
        output_path (str): where to save .fasta
    """
    with open(output_path, "w") as f:
        for acc, seq in fasta_dict.items():
            f.write(f">{acc}\n")
            # Wrap text at 60 chars per line (FASTA convention)
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

print(tf_pd.columns)

failed_ids = []
batch_size =  50

ensembl_ids = [row['Ensembl ID'] for idx, row in tf_pd.iterrows()]
meta_data = []

uni_accessions = []
for each_batch in range(0, len(tf_pd), batch_size):
    batch = ensembl_ids[each_batch:each_batch+batch_size]
    mapping_df = map_ensembl_to_uniprot(batch)

    accession_col = None
    for col in ["Entry"]:
        if col in mapping_df.columns:
            accession_col = col
            break
    if accession_col is None:
        print(f"Warning: No accession column found for {batch}. Columns: {mapping_df.columns.tolist()}")
        failed_ids.extend(batch)
        continue

    uni_accessions.extend(mapping_df[accession_col].dropna().unique().tolist())
    if not uni_accessions:
        print(f"No UniProt accessions found for {batch}")
        failed_ids.extend(batch)
        continue

    fasta_dict = fetch_fasta_for_accessions(uni_accessions)

    if not fasta_dict:
        print(f"No sequences retrieved for {batch} (accessions: {uni_accessions})")
        failed_ids.append(batch)
        continue
    output_path = root_output_path / f"batch_{each_batch}.fasta"
    save_fasta(fasta_dict, output_path)
    print(f"Saved {output_path}")
    metadata_df = fetch_uniprot_metadata(uni_accessions)
    if metadata_df is None or metadata_df.empty:
        print(f"No metadata retrieved for batch {batch} (accessions: {uni_accessions})")
        failed_ids.append(batch)
        continue

    meta_data.append(metadata_df)

save_metadata_csv(
    meta_data,
    output_folder=root_output_path,
    filename="all_uniprot_metadata.csv",
)

combined_meta = pd.concat(meta_data, ignore_index=True)
print("Combined metadata shape:", combined_meta.shape)


