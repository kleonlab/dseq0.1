import os
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
REPO_ID = "arcinstitute/SE-600M"
LOCAL_DIR = "./models/se600m" 
# ---------------------

def download_weights():
    print(f"Starting download for {REPO_ID}...")
    print(f"Files will be saved to: {os.path.abspath(LOCAL_DIR)}")
    
    # We use snapshot_download to get the folder structure correct.
    # We filter to get only the essential files to save space/time.
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False, # standard file download (easier to move around)
        allow_patterns=[
            "config.yaml",            # Crucial: Model architecture settings
            "*.ckpt",                 # Crucial: PyTorch Lightning checkpoints (for finetuning)
            "*.pt",                   # Essential: Protein embeddings helper files
            "*.safetensors",          # Optional: Optimized weights for pure inference
            "*.md"                    # Documentation/License
        ]
    )
    print("\n✅ Download complete!")
    print("Files currently in your folder:")
    for root, dirs, files in os.walk(LOCAL_DIR):
        for file in files:
            print(f" - {file}")

if __name__ == "__main__":
    # Ensure you have the library installed: pip install huggingface_hub
    download_weights()