import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our models and utils
from model.joint_backbone import DNALanguageModel, VirtualCell
from model.utils import gumbel_softmax, steering_loss
# Assuming we have a Dataset class, let's import or mock it for now
# from model.data_loader import MyDataset 

def train_one_epoch(dataloader, dna_model, vc_model, optimizer, device='cuda'):
    """
    Fine-tune the DNA-LM head to be generally good at steering.
    
    Parameters:
    -----------
    dataloader : torch.utils.data.DataLoader
        Batch iterator.
    dna_model : DNALanguageModel
        The generator to be trained.
    vc_model : VirtualCell
        The frozen simulator.
    optimizer : torch.optim.Optimizer
        Optimizer for dna_model parameters.
    device : str
        'cuda' or 'cpu'
    """
    dna_model.train()
    vc_model.eval() # VC model is always frozen/eval mode in this phase
    
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move data to device
        # Assuming batch is a dict with keys: 'control', 'target', 'delta'
        # Note: 'delta' might need to be computed if not in batch: target - control
        control = batch['control'].to(device)
        target = batch['target'].to(device)
        
        # Calculate delta if not provided (assuming simplistic vector difference for now)
        if 'delta' in batch:
            delta = batch['delta'].to(device)
        else:
            delta = target - control
            
        # Step 1: Run DNA-LM -> Output Logits
        logits = dna_model(delta)
        
        # Step 2: Apply Gumbel-Softmax -> Soft Sequence
        # Using hard=False to ensure gradients flow through nicely, 
        # or hard=True with straight-through estimator (implemented in utils).
        # For steering optimization, hard=True is often better for realism, 
        # but hard=False (soft) is safer for gradient flow initially. 
        # Let's use the default from utils (hard=True usually).
        soft_seq = gumbel_softmax(logits, temperature=1.0, hard=False)
        
        # Step 3: Run VC Model -> Predicted Cell State
        # VC model takes control state and the "action" (DNA sequence)
        pred_state = vc_model(control, soft_seq)
        
        # Step 4: Calculate Loss (Steering Loss)
        loss = steering_loss(pred_state, target)
        
        # Step 5: Zero Gradients
        optimizer.zero_grad()
        
        # Step 6: Backward Pass
        # Error flows: Loss -> pred_state -> VC -> soft_seq -> Gumbel -> Logits -> DNA-LM Weights
        loss.backward()
        
        # Step 7: Update Weights
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch Average Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # 1. Initialize Models
    print("Initializing models...")
    dna_model = DNALanguageModel(vocab_size=4, hidden_dim=512, num_layers=12).to(DEVICE)
    vc_model = VirtualCell(cell_dim=768).to(DEVICE) # Load pretrained weights if available
    
    # 2. Setup Freezing (Crucial Step)
    print("Freezing DNA-LM body and VC model...")
    # Freeze VC Model entirely (it's the environment/simulator)
    for param in vc_model.parameters():
        param.requires_grad = False
        
    # Freeze DNA-LM Body (Layers 1-10), unfreeze Head (Layer 11/Projection)
    dna_model.freeze_body()
    
    # 3. Setup Optimizer
    # Only pass trainable parameters to optimizer
    trainable_params = [p for p in dna_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LR)
    print(f"Optimizer setup with {len(trainable_params)} trainable tensors.")
    
    # 4. Data Loading (Mocking for now, replace with actual DataLoader)
    # dummy_data = [{'control': torch.randn(768), 'target': torch.randn(768)} for _ in range(100)]
    # dataloader = DataLoader(dummy_data, batch_size=BATCH_SIZE, shuffle=True)
    # Real usage:
    # from model.data_loader import get_dataloader
    # dataloader = get_dataloader(batch_size=BATCH_SIZE)
    
    # For demonstration, we'll skip the loop if no dataloader
    print("Starting training loop...")
    # for epoch in range(EPOCHS):
    #     print(f"\nEpoch {epoch+1}/{EPOCHS}")
    #     train_one_epoch(dataloader, dna_model, vc_model, optimizer, device=DEVICE)
    
    print("Training setup complete. (Uncomment data loading to run)")

if __name__ == "__main__":
    main()
