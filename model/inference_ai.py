import torch
import torch.optim as optim
import os
import sys
import scanpy as sc
import numpy as np

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.joint_backbone import DNALanguageModel, VirtualCell
from model.utils import gumbel_softmax
from model.tokenize_seq import decode_seq
from model.metric import GENEMetrics

# --- SAFETY FIX 1: Define Robust Loss Locally ---
def robust_steering_loss(pred, target):
    """
    Calculates 1 - CosineSimilarity with NaN protection.
    """
    # eps=1e-8 prevents division by zero if pred is a zero-vector
    cosine_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1e-8)
    
    # Check for NaNs immediately
    if torch.isnan(cosine_sim).any():
        return None
        
    # Return scalar loss
    return 1 - cosine_sim.mean()

def optimize_sequence(dna_model, vc_model, control, target, steps=50, lr=0.05, device='cuda'):
    """
    Generates a novel sequence by optimizing logits directly.
    """
    dna_model.eval()
    vc_model.eval()
    
    delta = target - control
    
    # Step 1: Initial Guess from Model
    with torch.no_grad():
        logits_init = dna_model(delta)
        
    # --- SAFETY FIX 5: Check for Corrupted Model Weights ---
    # If the previous training run crashed with NaNs, the saved checkpoint 
    # likely generates NaNs immediately. We must detect and fix this.
    if torch.isnan(logits_init).any() or torch.isinf(logits_init).any():
        print("!!! WARNING: DNA Model produced NaN logits. The checkpoint weights are likely corrupted.")
        print("!!! Falling back to RANDOM INITIALIZATION to allow optimization to proceed.")
        # Initialize with small random noise instead of broken model output
        logits_init = torch.randn_like(logits_init) * 0.1

    # Clone and Detach for Optimization
    logits_curr = logits_init.detach().clone()
    logits_curr.requires_grad = True
    
    # --- SAFETY FIX 2: Use Adam ---
    logit_optimizer = optim.Adam([logits_curr], lr=lr)
    
    print(f"Starting optimization for {steps} steps...")
    
    best_loss = float('inf')
    best_logits = logits_curr.detach().clone()
    
    for i in range(steps):
        logit_optimizer.zero_grad()
        
        # Forward Pass
        # hard=True sends discrete-like DNA to VC, but keeps gradients soft
        soft_seq = gumbel_softmax(logits_curr, temperature=1.0, hard=True)
        
        # Check if Gumbel produced NaNs (rare but possible if logits exploded)
        if torch.isnan(soft_seq).any():
            print(f"  Step {i+1}: Gumbel output NaN. Reinitializing logits.")
            logits_curr.data = torch.randn_like(logits_curr) * 0.1
            continue

        pred = vc_model(control, soft_seq)
        
        # Check VC output
        if torch.isnan(pred).any():
            print(f"  Step {i+1}: Virtual Cell predicted NaN. Skipping step.")
            # If VC is unstable, we can't step. Just continue or break.
            break

        # Calculate Safe Loss
        loss = robust_steering_loss(pred, target)
        
        # --- SAFETY FIX 3: NaN Brake ---
        if loss is None or torch.isnan(loss):
            print(f"  Step {i+1}: Loss is NaN! Reverting to previous best and stopping.")
            logits_curr.data = best_logits.data
            break
            
        # Keep track of best result
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_logits = logits_curr.detach().clone()
            
        loss.backward()
        
        # --- SAFETY FIX 4: Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_([logits_curr], max_norm=1.0)
        
        logit_optimizer.step()
        
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{steps}, Loss: {loss.item():.4f}")
            
    # Final Decode
    final_tokens = best_logits.argmax(dim=-1).squeeze(0)
    final_seq_str = decode_seq(final_tokens)
    
    return final_seq_str

def main():
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_PATH = "datasets/k562_5k.h5ad"
    CHECKPOINT_DIR = "models/model1"
    
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Error: Checkpoint directory {CHECKPOINT_DIR} not found.")
        return
        
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_epoch_")])
    if not checkpoints:
        print("No checkpoints found.")
        return
        
    latest_ckpt = checkpoints[-1]
    ckpt_path = os.path.join(CHECKPOINT_DIR, latest_ckpt)
    print(f"Loading checkpoint: {ckpt_path}")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    
    print(f"Loading data metadata from {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    
    # --- DATA CLEANING FIX ---
    # Ensure no NaNs in the input data itself
    if np.isnan(adata.X).any():
        print("Warning: Input data contains NaNs. Replacing with zeros to prevent crash.")
        adata.X = np.nan_to_num(adata.X)
        
    cell_dim = adata.n_vars
    
    # 2. Initialize Models
    print("Initializing models...")
    dna_model = DNALanguageModel(vocab_size=4, hidden_dim=512, num_layers=4, seq_len=100, input_dim=cell_dim).to(DEVICE)
    vc_model = VirtualCell(cell_dim=cell_dim, seq_len=100).to(DEVICE)
    
    # 3. Load Weights
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    dna_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # 4. Select Test Case
    idx = np.random.randint(0, len(adata))
    target_row = adata.X[idx]
    
    # Use global mean as control
    control_row = adata.X.mean(axis=0)
    
    control = torch.tensor(control_row, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    target = torch.tensor(target_row, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    print(f"Optimization Target: Cell Index {idx}")
    
    # Check inputs for NaNs before starting
    if torch.isnan(control).any() or torch.isnan(target).any():
        print("Error: Control or Target tensors contain NaNs after loading. Aborting.")
        return

    # 5. Run Inference
    optimized_sequence = optimize_sequence(dna_model, vc_model, control, target, steps=50, lr=0.05)
    
    print("\n" + "="*40)
    print("RESULTING DNA SEQUENCE")
    print("="*40)
    print(optimized_sequence)
    print("="*40)
    
    # Metrics
    scorer = GENEMetrics(device=DEVICE)
    codon_stats = scorer.calculate_codon_usage(optimized_sequence)
    print(f"2. Codon Usage: GC={codon_stats['GC_Content']*100:.1f}%, Valid ORF={codon_stats['Valid_ORF']}")
    
    print("3. Structural Viability (Querying ESMFold)...")
    fold_stats = scorer.calculate_foldability_esm(optimized_sequence)
    print(f"   pLDDT Score: {fold_stats['pLDDT']} ({fold_stats['Note']})")
    
    func_stats = scorer.calculate_functional_score(vc_model, control, target, optimized_sequence)
    print(f"4. In-Silico Growth/Function:")
    print(f"   MSE (Lower is better): {func_stats['MSE_Loss']:.4f}")
    print(f"   Directionality (1.0 is perfect): {func_stats['Directionality']:.4f}")

    # Save
    assets_dir = "datasets/assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    output_file = os.path.join(assets_dir, "optimized_sequences.txt")
    with open(output_file, "a") as f:
        f.write(f"--- Run Index: {idx} ---\nSequence: {optimized_sequence}\nGC: {codon_stats['GC_Content']}\n\n")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()