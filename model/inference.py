import torch
import torch.optim as optim
import os
import sys
import scanpy as sc
import numpy as np

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.joint_backbone import DNALanguageModel, VirtualCell
from model.utils import gumbel_softmax, steering_loss
from model.tokenize_seq import decode_seq

from model.metrics import GENEMetrics

def optimize_sequence(dna_model, vc_model, control, target, steps=50, lr=0.1, device='cuda'):
    """
    Generates a novel sequence for a specific problem without changing model weights.
    Optimizes the logits directly.
    
    Parameters:
    -----------
    dna_model : DNALanguageModel
        The trained generator (used for initial guess).
    vc_model : VirtualCell
        The frozen simulator.
    control : torch.Tensor
        Control cell state (1, Cell_Dim).
    target : torch.Tensor
        Target cell state (1, Cell_Dim).
    steps : int
        Number of optimization steps.
        
    Returns:
    --------
    str
        The optimized DNA sequence string.
    """
    # Ensure models are in eval mode (though we don't update their weights)
    dna_model.eval()
    vc_model.eval()
    
    # Compute delta (The prompt)
    delta = target - control
    
    # Step 1 (Initialize): Get initial guess from DNA-LM
    with torch.no_grad():
        logits_init = dna_model(delta)
        
    # Detach and clone to create the tensor we will optimize
    # We optimize the logits directly, starting from the model's best guess
    logits_curr = logits_init.detach().clone()
    logits_curr.requires_grad = True
    
    # Define optimizer for the logits
    # SGD is often used for this kind of "latent space" or "input" optimization
    logit_optimizer = optim.SGD([logits_curr], lr=lr)
    
    print(f"Starting optimization for {steps} steps...")
    
    # Step 2 (The Loop)
    for i in range(steps):
        logit_optimizer.zero_grad()
        
        # Forward pass: Logits -> Soft Sequence -> Virtual Cell -> Predicted State
        soft_seq = gumbel_softmax(logits_curr, temperature=1.0, hard=False)
        pred = vc_model(control, soft_seq)
        
        # Calculate loss
        loss = steering_loss(pred, target)
        
        # Backward pass (Gradients flow to logits_curr)
        loss.backward()
        
        # Update logits
        logit_optimizer.step()
        
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{steps}, Loss: {loss.item():.4f}")
            
    # Step 3 (Finalize): Decode the optimized logits
    # Take argmax to get discrete tokens
    final_tokens = logits_curr.argmax(dim=-1).squeeze(0) # Shape: (SeqLen,)
    final_seq_str = decode_seq(final_tokens)
    
    return final_seq_str

def main():
    # Configuration matching train.py
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_PATH = "datasets/k562_5k.h5ad"
    CHECKPOINT_DIR = "models/model1"
    
    # Load latest checkpoint
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
    
    # 1. Load Data Context (needed for dimensions and test samples)
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
        
    print(f"Loading data metadata from {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
        
    cell_dim = adata.n_vars
    
    # 2. Initialize Models (Must match train.py architecture)
    print("Initializing models...")
    dna_model = DNALanguageModel(vocab_size=4, hidden_dim=512, num_layers=4, seq_len=100, input_dim=cell_dim).to(DEVICE)
    vc_model = VirtualCell(cell_dim=cell_dim, seq_len=100).to(DEVICE)
    
    # 3. Load Weights
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    dna_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # 4. Select a Test Case
    # Let's pick a random pair from the dataset to optimize
    idx = np.random.randint(0, len(adata))
    target_row = adata.X[idx]
    
    # Use global mean as control (similar to dataloader default)
    control_row = adata.X.mean(axis=0)
    
    # Convert to tensors
    control = torch.tensor(control_row, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    target = torch.tensor(target_row, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    print(f"Optimization Target: Cell Index {idx}")
    
    # 5. Run Inference / Optimization
    optimized_sequence = optimize_sequence(dna_model, vc_model, control, target, steps=50)
    
    print("\n" + "="*40)
    print("RESULTING DNA SEQUENCE")
    print("="*40)
    print(optimized_sequence)
    print("="*40)
    print("\nCalculated Metrics (Nature Paper Benchmarks):")
    
    scorer = GENEMetrics(device=DEVICE)
    
    # 1. Nucleic Acid: Sequence Recovery
    # Since we generated a NEW sequence, we compare it to the 'Initial Guess' 
    # (Let's assume the initial guess was what the DNA model predicted at step 0)
    # For now, we compare to the "wildtype" or mean if you have it. 
    # If not, we skip or compare to a dummy string.
    # recovery = scorer.calculate_sequence_recovery(optimized_sequence, known_gene_seq)
    # print(f"1. Sequence Recovery: {recovery}% (vs known gene)")
    
    # 2. Nucleic Acid: Codon Usage
    codon_stats = scorer.calculate_codon_usage(optimized_sequence)
    print(f"2. Codon Usage: GC={codon_stats['GC_Content']*100:.1f}%, Valid ORF={codon_stats['Valid_ORF']}")
    
    # 3. Protein/Function: TM-Score (Proxied by Foldability/pLDDT)
    # Note: This makes a web request to ESMFold
    print("3. Structural Viability (Querying ESMFold)...")
    fold_stats = scorer.calculate_foldability_esm(optimized_sequence)
    print(f"   pLDDT Score: {fold_stats['pLDDT']} ({fold_stats['Note']})")
    print(f"   (Answer to: Does the DNA code for a 3D shape?)")

    # 4. Protein/Function: Growth Assay (Proxied by Virtual Cell Score)
    func_stats = scorer.calculate_functional_score(vc_model, control, target, optimized_sequence)
    print(f"4. In-Silico Growth/Function:")
    print(f"   MSE (Lower is better): {func_stats['MSE_Loss']:.4f}")
    print(f"   Directionality (1.0 is perfect): {func_stats['Directionality']:.4f}")
    print(f"   (Answer to: Does the gene actually move the cell state?)")

if __name__ == "__main__":
    main()
