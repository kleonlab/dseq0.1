import requests
import sys
import time
import json
import pandas as pd
import os

def get_gene_sequences_bulk(ensembl_ids, seq_type='genomic', batch_size=50):
    """
    Fetches sequences for a list of Ensembl IDs using the Ensembl POST API.
    Handles batching to respect API limits.
    
    Parameters:
    - ensembl_ids: List of Ensembl ID strings.
    - seq_type: 'genomic', 'cds', or 'protein'.
    - batch_size: Number of IDs to send in one request (default 50).
    
    Returns:
    - dict: Dictionary mapping {ensembl_id: sequence_string}
    """
    server = "https://rest.ensembl.org"
    ext = "/sequence/id"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    results = {}
    
    # Remove duplicates just in case
    unique_ids = list(set(ensembl_ids))
    total_ids = len(unique_ids)
    print(f"Preparing to fetch sequences for {total_ids} unique IDs...")
    
    # Process in batches
    for i in range(0, total_ids, batch_size):
        batch = unique_ids[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_ids + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} IDs)...")
        
        payload = json.dumps({
            "ids": batch,
            "type": seq_type
        })
        
        # Retry logic for the batch
        max_retries = 3
        success = False
        attempt = 0
        
        while not success and attempt < max_retries:
            try:
                response = requests.post(server + ext, headers=headers, data=payload)
                
                if response.ok:
                    batch_results = response.json()
                    # Response is a list of objects
                    for item in batch_results:
                        if 'seq' in item:
                            results[item['id']] = item['seq']
                    success = True
                
                elif response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', 2))
                    print(f"  Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    attempt += 1
                
                else:
                    print(f"  Error {response.status_code}: {response.text}")
                    attempt += 1
                    time.sleep(1)
            
            except requests.exceptions.RequestException as e:
                print(f"  Network error: {e}")
                attempt += 1
                time.sleep(1)
        
        if not success:
            print(f"  Failed to retrieve batch {batch_num} after retries.")
            
        # Be polite to the server
        time.sleep(0.1)

    return results

def save_to_fasta(sequences_dict, filename):
    """Saves a dictionary of sequences to a FASTA file."""
    count = 0
    with open(filename, "w") as f:
        for gene_id, seq in sequences_dict.items():
            if seq:
                f.write(f">{gene_id}\n{seq}\n")
                count += 1
    print(f"Saved {count} sequences to {filename}")

if __name__ == "__main__":
    # Configuration
    CSV_PATH = 'datasets/DatabaseExtract_v_1.01.csv'
    OUTPUT_FILE = 'gene_sequences.fasta'
    ID_COLUMN = 'gene_id' # We determined this is the correct column earlier
    
    if os.path.exists(CSV_PATH):
        print(f"Reading IDs from {CSV_PATH}...")
        try:
            df = pd.read_csv(CSV_PATH)
            
            if ID_COLUMN in df.columns:
                # Get list of IDs
                id_list = df[ID_COLUMN].dropna().astype(str).tolist()
                
                # Fetch sequences
                print("Starting bulk retrieval...")
                sequences = get_gene_sequences_bulk(id_list, seq_type='genomic')
                
                # Ensure output directory exists
                output_dir = "assets/gene_sequences"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Save results
                output_path = os.path.join(output_dir, OUTPUT_FILE)
                save_to_fasta(sequences, output_path)
                
                print(f"Complete. Retrieved {len(sequences)}/{len(id_list)} requested sequences.")
            else:
                print(f"Error: Column '{ID_COLUMN}' not found in CSV.")
                print(f"Columns found: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error processing CSV: {e}")
            
    else:
        # Fallback test if CSV is missing
        print(f"CSV file not found at {CSV_PATH}. Running test with example IDs...")
        test_ids = ["ENSG00000136158", "ENSG00000204531", "ENSG00000012048"]
        sequences = get_gene_sequences_bulk(test_ids)
        print(sequences)
        print(sequences.keys())

        # Ensure output directory exists
        output_dir = "assets/gene_sequences"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save results
        output_path = os.path.join(output_dir, OUTPUT_FILE)
        save_to_fasta(sequences, output_path)