import time
from io import StringIO

import requests
import pandas as pd

import time
import requests
import pandas as pd
from io import StringIO



def map_ensembl_to_uniprot(ensembl_ids, from_db="Ensembl", to_db="UniProtKB"):
    """
    Given a list of Ensembl gene IDs, map them to UniProtKB accessions using UniProt API.
    Returns a pandas DataFrame with columns: ensembl_id, uni_prot_accession, status, etc.
    """
    API_URL = "https://rest.uniprot.org"
    
    # 1. Submit job
    url_submit = f"{API_URL}/idmapping/run"
    params = {
        "from": from_db,
        "to": to_db,
        "ids": ",".join(ensembl_ids)
    }
    resp = requests.post(url_submit, data=params)
    resp.raise_for_status()
    job_id = resp.json()["jobId"]
    
    # 2. Poll job status
    url_status = f"{API_URL}/idmapping/status/{job_id}"
    while True:
        resp = requests.get(url_status)
        resp.raise_for_status()
        payload = resp.json()
        
        # Check if results are ready
        if "results" in payload or "failedIds" in payload:
            # Job is complete
            break
        
        # Check job status
        if "jobStatus" in payload:
            status = payload["jobStatus"]
            if status == "RUNNING":
                time.sleep(1)
                continue
            elif status == "FINISHED":
                break
            else:
                # Job failed
                raise RuntimeError(f"Mapping job failed with status: {status}")
        
        # If neither results nor jobStatus, wait and retry
        time.sleep(1)
    
    # 3. Retrieve results as TSV
    url_results = f"{API_URL}/idmapping/uniprotkb/results/{job_id}"
    resp = requests.get(url_results, params={"format": "tsv"})
    resp.raise_for_status()
    
    # Parse TSV response
    df = pd.read_csv(StringIO(resp.text), sep="\t")
    return df

ensembl_ids = ["ENSG00000137203"]  # example human gene IDs
mapping_df = map_ensembl_to_uniprot(ensembl_ids)
#print(mapping_df.head())
print(mapping_df.columns)
print(mapping_df.size)
print(mapping_df.shape[0])
print(mapping_df.shape[1])
print(mapping_df)

import requests

def fetch_fasta_for_accessions(accessions):
    """
    Given a list of UniProt accessions, fetch the FASTA sequences.
    Returns dict: accession -> sequence (string)
    """
    fasta_dict = {}
    batch_size = 100
    
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:i+batch_size]
        
        # Build query with accession_id field for exact matches
        query_parts = [f"accession:{acc}" for acc in batch]
        query = " OR ".join(query_parts)
        
        url_fasta = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": query,
            "format": "fasta"
        }
        
        resp = requests.get(url_fasta, params=params)
        resp.raise_for_status()
        fasta_text = resp.text
        
        # Parse FASTA
        for record in fasta_text.strip().split(">")[1:]:
            lines = record.split("\n")
            header = lines[0]
            # Extract accession from header: sp|P12345|PROT_HUMAN or tr|A0A6E1|...
            # The accession is the second field when split by |
            parts = header.split("|")
            if len(parts) >= 2:
                accession = parts[1].split("-")[0]  # Remove isoform suffix if present
                sequence = "".join(lines[1:])
                fasta_dict[accession] = sequence
    
    return fasta_dict


import requests
import pandas as pd
from io import StringIO

def fetch_uniprot_metadata(accessions, fields=None, batch_size=80):
    """
    Fetch UniProt metadata for a list of accessions in batches.

    Args:
        accessions: List of UniProt accession IDs
        fields: List of field names to retrieve (if None, gets common fields)
        batch_size: How many accessions per REST query

    Returns:
        pandas DataFrame with requested fields (possibly empty if nothing retrieved)
    """
    if fields is None:
        fields = [
            "accession",
            "id",
            "gene_names",
            "organism_name",
            "protein_name",
            "cc_function",
            "go",
            "ft_domain",
            "ft_region",
        ]

    url = "https://rest.uniprot.org/uniprotkb/search"
    all_frames = []

    # Process in chunks
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i : i + batch_size]
        if not batch:
            continue

        query_parts = [f"accession:{acc}" for acc in batch]
        query = " OR ".join(query_parts)

        params = {
            "query": query,
            "format": "tsv",
            "fields": ",".join(fields),
            # "size": batch_size,  # optional; UniProt defaults are fine for accession queries
        }

        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"[WARN] HTTP error for metadata batch {i // batch_size}: {e}")
            # If you want to debug further:
            # print("Response text:", resp.text)
            continue

        # Empty response?
        if not resp.text.strip():
            print(f"[INFO] Empty metadata response for batch {i // batch_size}")
            continue

        df_batch = pd.read_csv(StringIO(resp.text), sep="\t")
        if df_batch.empty:
            print(f"[INFO] No rows parsed for metadata batch {i // batch_size}")
            continue

        all_frames.append(df_batch)

    if not all_frames:
        print("[INFO] No metadata retrieved for any batch.")
        return pd.DataFrame()

    metadata_df = pd.concat(all_frames, ignore_index=True)
    return metadata_df



uni_accessions = mapping_df["Entry"].unique().tolist()
print(uni_accessions)
fasta_dict = fetch_fasta_for_accessions(uni_accessions)
print({acc: len(seq) for acc, seq in fasta_dict.items()})
print({acc: seq for acc, seq in fasta_dict.items()})

metadata_df = fetch_uniprot_metadata(uni_accessions)
print(metadata_df)
print(metadata_df.size)

print(metadata_df[metadata_df['Entry'] == 'P05549'])

