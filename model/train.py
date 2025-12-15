import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import scanpy as sc
import numpy as np

# Ensure the project root is in sys.path to allow absolute imports like 'model.xxx'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our models and utils
from model.joint_backbone import DNALanguageModel, VirtualCell
from model.utils import gumbel_softmax, steering_loss
from model.data_loader import create_dataloaders

def train_one_epoch(dataloader, dna_model, vc_model, optimizer, device='cuda'):
    """
    Fine-tune the DNA-LM head to be generally good at steering.
    """
    dna_model.train()
    vc_model.eval() # VC model is always frozen/eval mode in this phase
    
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move data to device
        control = batch['control_state'].to(device)
        target = batch['target_state'].to(device)
        
        # Calculate delta if not provided
        if 'delta' in batch:
            delta = batch['delta'].to(device)
        else:
            delta = target - control
            
        # Step 1: Run DNA-LM -> Output Logits
        logits = dna_model(delta)
        
        # Step 2: Apply Gumbel-Softmax -> Soft Sequence
        soft_seq = gumbel_softmax(logits, temperature=1.0, hard=True)
        
        # Step 3: Run VC Model -> Predicted Cell State
        pred_state = vc_model(control, soft_seq)
        
        # Step 4: Calculate Loss (Steering Loss)
        loss = steering_loss(pred_state, target)
        
        # Step 5: Zero Gradients
        optimizer.zero_grad()
        
        # Step 6: Backward Pass
        loss.backward()
        
        # Step 7: Update Weights
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def save_checkpoint(dna_model, optimizer, epoch, loss, save_dir):
    """
    Save model checkpoint.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': dna_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def main():
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 5 # Reduced for testing
    LR = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_PATH = "datasets/k562_5k.h5ad"
    SAVE_DIR = "models/model1"
    
    # Paths to pretrained weights
    VC_WEIGHTS_PATH = "models/se600m/se600m_epoch16.ckpt"
    DNA_WEIGHTS_PATH = "models/evo/savanna_evo2_1b_base.pt"
    
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
        
    print(f"Loading data from {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    
    # Preprocessing checks for the dataloader
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()

    # --- In main(), before create_dataloaders ---

    if np.isnan(adata.X).any():
        print("Warning: Raw input contains NaNs. Replacing with zeros.")
        adata.X = np.nan_to_num(adata.X)

    # --- STEP 2: Check for Negatives (Prevents log1p crash) ---
    min_val = adata.X.min()
    max_val = adata.X.max()
    print(f"Initial Data Range: {min_val:.2f} to {max_val:.2f}")

    if min_val < 0:
        print(">>> DETECTED NEGATIVE VALUES: Data is likely already scaled.")
        print(">>> SKIPPING normalize_total and log1p to avoid NaNs.")
        # Optional: We still re-scale to ensure variance is controlled for the NN
        # But we don't log-transform negative numbers.
    else:
        print(">>> Data is non-negative (likely raw counts). Applying normalization.")
        
        # 1. Filter empty cells (prevents divide-by-zero errors)
        sc.pp.filter_cells(adata, min_counts=1)
        
        # 2. Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
    # --- STEP 3: Final Scaling (Crucial for Neural Net Stability) ---
    # We apply this regardless, to ensure inputs are roughly -10 to 10
    print("Applying final scaling...")
    sc.pp.scale(adata, max_value=10)

    # Final Check
    if np.isnan(adata.X).any():
        print("!!! CRITICAL: NaNs still exist after processing. Replacing with 0.")
        adata.X = np.nan_to_num(adata.X)
        
    print(f"Final Data Range: {adata.X.min():.2f} to {adata.X.max():.2f}")
    
    # Ensure 'Gene Name' column exists (using 'condition' or creating dummy if needed for test)
    if 'Gene Name' not in adata.obs.columns:
        if 'condition' in adata.obs.columns:
            adata.obs['Gene Name'] = adata.obs['condition']
        else:
            print("Warning: 'Gene Name' column missing. Creating dummy labels for testing.")
            n_samples = adata.n_obs
            adata.obs['Gene Name'] = ['GeneA'] * (n_samples // 2) + ['GeneB'] * (n_samples - n_samples // 2)
            # Ensure some controls exist
            adata.obs.iloc[:10, adata.obs.columns.get_loc('Gene Name')] = 'non-targeting'

    # Create DataLoaders
    print("Creating DataLoaders...")
    try:
        train_loader, test_loader = create_dataloaders(adata, batch_size=BATCH_SIZE)
        print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return

    # 2. Initialize Models
    print("Initializing models...")
    cell_dim = adata.n_vars
    
    # Instantiate with correct dimensions
    # Load Pretrained weights immediately upon creation if possible, or after
    dna_model = DNALanguageModel(vocab_size=4, hidden_dim=2048, num_layers=24, seq_len=100, input_dim=cell_dim).to(DEVICE) 
    
    vc_model = VirtualCell(cell_dim=cell_dim, seq_len=100, pretrained_model_path=VC_WEIGHTS_PATH).to(DEVICE) 
    
    # Load DNA Model weights
    if os.path.exists(DNA_WEIGHTS_PATH):
        dna_model.load_weights(DNA_WEIGHTS_PATH, device=DEVICE)
    else:
        print(f"Warning: Pretrained DNA model not found at {DNA_WEIGHTS_PATH}. Training from random init.")
    
    # 3. Setup Freezing
    print("Freezing DNA-LM body and VC model...")
    for param in vc_model.parameters():
        param.requires_grad = False
    
    # If we loaded pretrained weights, we freeze. If not, freezing random weights is bad (as user noted).
    # Logic: Always freeze body if we intend to fine-tune head. 
    # If random init, freezing body = bad.
    # So, check if weights existed.
    if os.path.exists(DNA_WEIGHTS_PATH):
        dna_model.freeze_body()
    else:
        print("Pretrained DNA weights missing: Skipping freeze to allow full training from scratch.")
    
    # 4. Setup Optimizer
    trainable_params = [p for p in dna_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LR)
    print(f"Optimizer setup with {len(trainable_params)} trainable tensors.")
    
    # 5. Training Loop
    print(f"Starting training loop. Checkpoints will be saved to {SAVE_DIR}")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        try:
            avg_loss = train_one_epoch(dataloader=train_loader, 
                                     dna_model=dna_model, 
                                     vc_model=vc_model, 
                                     optimizer=optimizer, 
                                     device=DEVICE)
            
            # Save checkpoint
            save_checkpoint(dna_model, optimizer, epoch, avg_loss, SAVE_DIR)
            
        except RuntimeError as e:
            print(f"Runtime Error during training: {e}")
            print("Tip: Check dimension mismatches between dataset and model.")
            break
            
    print("Training run complete.")

if __name__ == "__main__":
    main()
