import requests
import time
import json
import sys
import os
import pandas as pd

# --- Configuration ---
# CSV file path containing Ensembl IDs
CSV_FILE_PATH = "datasets/DatabaseExtract_v_1.01.csv"
# Column name containing Ensembl IDs (based on the CSV file)
ENSEMBL_ID_COLUMN = "Ensembl ID" 

API_URL = "https://rest.uniprot.org"
POLL_INTERVAL = 3  # seconds
TIMEOUT = 300      # max time to wait for a job (seconds)
OUTPUT_DIR = "datasets/"  # Configurable output directory


def get_ensembl_ids_from_csv(csv_path: str, column_name: str) -> list:
    """
    Reads Ensembl IDs from a specified column in a CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
        if column_name not in df.columns:
             # Fallback: try to find a column that looks like it contains Ensembl IDs if exact match fails
             # Or just raise error. The user specified "first column" but also the file has headers.
             # Let's trust the header name first, if provided.
             # If user said "first column", we can also look at index 1 (since 0 might be index)
             pass

        if column_name in df.columns:
            ids = df[column_name].dropna().unique().tolist()
        else:
            # If specific column not found, try to use the 2nd column (index 1) 
            # as index 0 often contains row numbers or generic IDs in these datasets
            print(f"Column '{column_name}' not found. Using the second column as default.")
            ids = df.iloc[:, 1].dropna().unique().tolist()
        
        # Filter for valid-looking Ensembl IDs to avoid garbage
        ids = [i for i in ids if isinstance(i, str) and i.startswith("ENS")]
        
        print(f"Loaded {len(ids)} unique Ensembl IDs from {csv_path}")
        return ids
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {e}")


def submit_id_mapping(from_db: str, to_db: str, ids):
    """
    Submit an ID mapping job and return the UniProt jobId.
    """
    if isinstance(ids, (list, tuple, set)):
        ids_str = ",".join(ids)
    else:
        ids_str = str(ids)

    payload = {
        "from": from_db,
        "to": to_db,
        "ids": ids_str,
    }

    resp = requests.post(f"{API_URL}/idmapping/run", data=payload)
    resp.raise_for_status()
    job_id = resp.json().get("jobId")
    if not job_id:
        raise RuntimeError(f"Failed to get jobId from response: {resp.text}")
    return job_id


def wait_for_job(job_id: str, poll_interval=POLL_INTERVAL, timeout=TIMEOUT):
    """
    Poll the job status endpoint until the job is finished or fails.
    Handles both 'status' and 'jobStatus' fields for compatibility.
    """
    status_url = f"{API_URL}/idmapping/status/{job_id}"
    start = time.time()

    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"Job {job_id} did not finish within {timeout} seconds.")

        res = requests.get(status_url)
        res.raise_for_status()
        j = res.json()

        # Newer API uses "status", older snippets used "jobStatus"
        status = j.get("status") or j.get("jobStatus")

        if status in ("NEW", "RUNNING", "PENDING"):
            print(f"Job {job_id} is {status}. Waiting {poll_interval}s...")
            time.sleep(poll_interval)
            continue
        elif status in ("FINISHED", "COMPLETE"):
            print(f"Job {job_id} finished.")
            return
        elif status in ("FAILED", "ERROR"):
            raise RuntimeError(f"Job {job_id} failed: {j}")
        else:
            # Some implementations return results directly without a status field
            if "results" in j or "failedIds" in j:
                print(f"Job {job_id} finished (no explicit status field).")
                return
            print(f"Unknown status for job {job_id}: {j}")
            time.sleep(poll_interval)


def get_tsv_results(job_id: str):
    """
    Fetch mapping results as TSV and return list of (from_id, to_id) pairs.
    """
    results_url = f"{API_URL}/idmapping/results/{job_id}"
    params = {"format": "tsv"}  # simpler than JSON for mapping
    res = requests.get(results_url, params=params)
    res.raise_for_status()

    lines = res.text.strip().splitlines()
    if not lines:
        return []

    header = lines[0].split("\t")
    try:
        from_idx = header.index("From")
        to_idx = header.index("To")
    except ValueError:
        raise RuntimeError(f"Unexpected TSV header: {header}")

    pairs = []
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) <= max(from_idx, to_idx):
            continue
        pairs.append((cols[from_idx], cols[to_idx]))
    return pairs


def run_mapping(from_db: str, to_db: str, ids):
    """
    Helper: submit mapping job, wait, then return list of (from, to) pairs.
    """
    print(f"Submitting mapping {from_db} → {to_db} for {len(ids)} IDs...")
    
    # The API might have limits on how many IDs can be submitted at once.
    # If the list is huge, it might be better to chunk it. 
    # For now, we assume the API handles reasonable batch sizes (e.g. < 100k).
    # If it fails, we can implement chunking.
    
    try:
        job_id = submit_id_mapping(from_db, to_db, ids)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400 and "ids" in str(e.response.content):
             print("Error submitting job. The list of IDs might be too long or contain invalid format.")
             raise
        raise

    print(f"Job ID: {job_id}")
    wait_for_job(job_id)
    print("Retrieving results...")
    return get_tsv_results(job_id)


def map_ensembl_to_pdb(ensembl_ids):
    """
    Two-step mapping:
      1) Ensembl (gene) → UniProtKB accession
      2) UniProtKB accession → PDB ID
    Returns: dict[ensembl_id] -> list[pdb_ids]
    """
    if not ensembl_ids:
        print("No Ensembl IDs provided.")
        return {}

    # Step 1: Ensembl → UniProtKB
    pairs_e2u = run_mapping("Ensembl", "UniProtKB", ensembl_ids)
    ensembl_to_uniprot = {}
    for src, dest in pairs_e2u:
        ensembl_to_uniprot.setdefault(src, []).append(dest)

    print("\nEnsembl → UniProtKB mapping done.")
    print(f"Got UniProt IDs for {len(ensembl_to_uniprot)} Ensembl IDs.")

    # Collect all unique UniProt accessions
    all_uniprot_ids = sorted({u for lst in ensembl_to_uniprot.values() for u in lst})
    if not all_uniprot_ids:
        print("No UniProt IDs found; skipping PDB mapping.")
        return {eid: [] for eid in ensembl_ids}

    # Step 2: UniProtKB accession → PDB
    # Note: mapping from UniProtKB to PDB uses 'UniProtKB_AC-ID' as the 'from' DB
    pairs_u2p = run_mapping("UniProtKB_AC-ID", "PDB", all_uniprot_ids)
    uniprot_to_pdb = {}
    for src, dest in pairs_u2p:
        uniprot_to_pdb.setdefault(src, []).append(dest)

    print("\nUniProtKB → PDB mapping done.")

    # Combine into Ensembl → PDB
    ensembl_to_pdb = {}
    for eid in ensembl_ids:
        pdb_ids = set()
        for u in ensembl_to_uniprot.get(eid, []):
            for pdb in uniprot_to_pdb.get(u, []):
                pdb_ids.add(pdb)
        ensembl_to_pdb[eid] = sorted(pdb_ids)

    return ensembl_to_pdb


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Load IDs from CSV
        print(f"Reading Ensembl IDs from '{CSV_FILE_PATH}'...")
        ensembl_ids = get_ensembl_ids_from_csv(CSV_FILE_PATH, ENSEMBL_ID_COLUMN)
        
        # Run mapping
        mapping = map_ensembl_to_pdb(ensembl_ids)

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Construct full output path
        output_file = os.path.join(OUTPUT_DIR, "ensembl_to_pdb.json")

        # Save to JSON
        with open(output_file, "w") as f:
            json.dump(mapping, f, indent=2)

        print(f"\nSuccessfully mapped {len(mapping)} Ensembl IDs.")
        print(f"Results saved to '{output_file}'")

        # Preview first few
        print("\nSample mapping (first 5):")
        for k, v in list(mapping.items())[:5]:
            print(f"{k} -> {v}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
