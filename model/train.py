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
        soft_seq = gumbel_softmax(logits, temperature=1.0, hard=False)
        
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
    dna_model = DNALanguageModel(vocab_size=4, hidden_dim=512, num_layers=4, seq_len=100, input_dim=cell_dim).to(DEVICE) 
    
    vc_model = VirtualCell(cell_dim=cell_dim, seq_len=100).to(DEVICE) 
    
    # 3. Setup Freezing
    print("Freezing DNA-LM body and VC model...")
    for param in vc_model.parameters():
        param.requires_grad = False
    
    dna_model.freeze_body()
    
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
