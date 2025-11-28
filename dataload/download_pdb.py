import os
import json
import requests
import concurrent.futures
from pathlib import Path

# --- Configuration ---
# Input: The JSON file you created in the previous step
MAPPING_FILE = "./datasets/ensembl_to_pdb.json" 
OUTPUT_DIR = "./datasets/structures_cif"
MAX_WORKERS = 10  # Number of parallel downloads (don't go too high or RCSB might block you)

def download_cif(pdb_id, output_folder):
    """
    Downloads the .cif file for a single PDB ID.
    """
    pdb_id = pdb_id.lower()
    # RCSB PDB File Server URL
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    
    save_path = output_folder / f"{pdb_id}.cif"
    
    # Skip if already exists
    if save_path.exists():
        return f"Skipped {pdb_id} (exists)"

    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return f"Downloaded {pdb_id}"
        elif response.status_code == 404:
            return f"Error: {pdb_id} not found on PDB server"
        else:
            return f"Error: {pdb_id} returned status {response.status_code}"
            
    except Exception as e:
        return f"Failed {pdb_id}: {str(e)}"

def main():
    # 1. Load the Mapping
    print(f"Loading IDs from {MAPPING_FILE}...")
    with open(MAPPING_FILE, "r") as f:
        data = json.load(f)

    # Extract unique PDB IDs (flatten the list of lists)
    # The JSON is { "ENSG...": ["1a2b", "3x4y"], ... }
    all_pdb_ids = set()
    for pdb_list in data.values():
        if isinstance(pdb_list, list):
            for pid in pdb_list:
                all_pdb_ids.add(pid)
        else:
            all_pdb_ids.add(pdb_list) # Handle cases where it might be a single string

    print(f"Found {len(all_pdb_ids)} unique PDB structures to download.")

    # 2. Setup Output Directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # 3. Download in Parallel
    print(f"Starting batch download with {MAX_WORKERS} threads...")
    
    success_count = 0
    error_count = 0

    # Using ThreadPoolExecutor to run downloads concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a dictionary to map futures to PDB IDs
        future_to_pdb = {
            executor.submit(download_cif, pdb_id, output_path): pdb_id 
            for pdb_id in all_pdb_ids
        }

        # Process as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_pdb)):
            result = future.result()
            
            # Simple progress log
            if "Error" in result or "Failed" in result:
                error_count += 1
                print(f"[{i+1}/{len(all_pdb_ids)}] {result}")
            else:
                success_count += 1
                # Optional: Don't print every success to keep terminal clean
                if i % 50 == 0: 
                    print(f"[{i+1}/{len(all_pdb_ids)}] Progress... ({success_count} success)")

    print(f"\nCompleted.")
    print(f"Successfully downloaded: {success_count}")
    print(f"Errors/Missing: {error_count}")
    print(f"Files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()