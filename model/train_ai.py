import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import scanpy as sc
import numpy as np

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORTS ---
from model.joint_backbone import DNALanguageModel, VirtualCell
from model.utils import gumbel_softmax, steering_loss
try:
    from model.data_loader import create_dataloaders
except ImportError:
    from dataset import create_dataloaders

def train_one_epoch(dataloader, dna_model, vc_model, optimizer, device='cuda'):
    """
    Fine-tune the DNA-LM head to be generally good at steering.
    """
    dna_model.train()
    vc_model.eval() # VC model is always frozen
    
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # 1. Move data to device
        control = batch['control_state'].to(device)
        target = batch['target_state'].to(device)
        
        if 'delta' in batch:
            delta = batch['delta'].to(device)
        else:
            delta = target - control
            
        # 2. Forward Pass
        # DNA-LM -> Logits
        logits = dna_model(delta)
        
        # Gumbel -> Discrete-ish Sequence (hard=True is critical for VC)
        soft_seq = gumbel_softmax(logits, temperature=1.0, hard=True)
        
        # VC -> Prediction
        pred_state = vc_model(control, soft_seq)
        
        # 3. Loss Calculation
        loss = steering_loss(pred_state, target)
        
        # 4. Backward Pass & Optimization
        optimizer.zero_grad()
        loss.backward()
        
        # --- CRITICAL ADDITION: Gradient Clipping ---
        # Prevents exploding gradients which cause NaN loss
        torch.nn.utils.clip_grad_norm_(dna_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(dataloader, dna_model, vc_model, device='cuda'):
    """
    Validation loop to check generalization.
    """
    dna_model.eval()
    vc_model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            control = batch['control_state'].to(device)
            target = batch['target_state'].to(device)
            delta = batch.get('delta', target - control).to(device)
            
            logits = dna_model(delta)
            # Use hard=True even in validation to match inference conditions
            soft_seq = gumbel_softmax(logits, temperature=1.0, hard=True)
            pred_state = vc_model(control, soft_seq)
            
            loss = steering_loss(pred_state, target)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def save_checkpoint(dna_model, optimizer, epoch, train_loss, val_loss, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': dna_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def main():
    # --- CONFIGURATION ---
    BATCH_SIZE = 32
    EPOCHS = 2
    LR = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    DATA_PATH = "datasets/k562_5k.h5ad"
    SAVE_DIR = "models/model1"
    VC_WEIGHTS_PATH = "models/se600m/se600m_epoch16.ckpt"
    DNA_WEIGHTS_PATH = "models/evo/savanna_evo2_1b_base.pt"
    
    print(f"Using device: {DEVICE}")
    
    # --- 1. DATA LOADING ---
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
        
    print(f"Loading data from {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    if hasattr(adata.X, "toarray"): adata.X = adata.X.toarray()

    # --- CRITICAL FIX: CLEAN DATA ---
    # This prevents the NaNs that were poisoning your checkpoints!
    if np.isnan(adata.X).any():
        print("Warning: Input data contains NaNs. Replacing with zeros.")
        adata.X = np.nan_to_num(adata.X)
    
    # Handle Metadata Columns
    if 'Gene Name' not in adata.obs.columns:
        if 'condition' in adata.obs.columns:
            adata.obs['Gene Name'] = adata.obs['condition']
        else:
            print("Warning: Missing 'Gene Name'. Creating dummy labels.")
            adata.obs['Gene Name'] = 'Gene_X'
            # Create fake non-targeting controls for logic to work
            adata.obs.iloc[:10, -1] = 'non-targeting'

    try:
        train_loader, test_loader = create_dataloaders(adata, batch_size=BATCH_SIZE)
        print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return

    # --- 2. MODEL INITIALIZATION ---
    print("Initializing models...")
    cell_dim = adata.n_vars
    
    # Initialize Models
    dna_model = DNALanguageModel(vocab_size=4, hidden_dim=512, num_layers=4, seq_len=100, input_dim=cell_dim).to(DEVICE) 
    vc_model = VirtualCell(cell_dim=cell_dim, seq_len=100, pretrained_model_path=VC_WEIGHTS_PATH).to(DEVICE) 
    
    # Load DNA Weights (if available)
    if os.path.exists(DNA_WEIGHTS_PATH):
        dna_model.load_weights(DNA_WEIGHTS_PATH, device=DEVICE)
        # Freeze Body ONLY if we loaded pre-trained weights
        print("Pretrained weights found. Freezing DNA-LM body.")
        dna_model.freeze_body()
    else:
        print(f"Warning: {DNA_WEIGHTS_PATH} not found. Training FROM SCRATCH (No Freezing).")
    
    # Freeze VC (Always)
    for param in vc_model.parameters():
        param.requires_grad = False
    
    # --- 3. OPTIMIZER ---
    trainable_params = [p for p in dna_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LR)
    print(f"Optimizer targeting {len(trainable_params)} tensor groups.")
    
    # --- 4. TRAINING LOOP ---
    print(f"Starting training. Saving to {SAVE_DIR}")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train Step
        train_loss = train_one_epoch(train_loader, dna_model, vc_model, optimizer, device=DEVICE)
        
        # Validation Step (New)
        val_loss = validate(test_loader, dna_model, vc_model, device=DEVICE)
        
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        save_checkpoint(dna_model, optimizer, epoch, train_loss, val_loss, SAVE_DIR)
            
    print("Training run complete.")

if __name__ == "__main__":
    main()
