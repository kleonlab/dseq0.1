import torch
import torch.nn as nn

class BioSystem(nn.Module):
    def __init__(self, cell_dim=768, seq_len=1000):
        super().__init__()
        
        # --- COMPONENT 1: THE DESIGNER ---
        # Input: Concatenated Cell States (A + B) -> Output: Soft DNA
        self.designer = nn.Sequential(
            nn.Linear(cell_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len * 4) # Output raw logits for every base pair
        )
        self.seq_len = seq_len
        
        # --- COMPONENT 2: THE PHYSICS ENGINE ---
        # Input: Soft DNA -> Output: Matrix M (Low Rank U, V)
        # Note: We process the [Seq_Len, 4] input
        self.dna_encoder = nn.Linear(seq_len * 4, 256) 
        
        self.hyper_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, (cell_dim * 64) * 2) # Low Rank (rank=64)
        )
        self.rank = 64
        self.cell_dim = cell_dim

    def forward(self, cell_A, cell_B):
        batch_size = cell_A.shape[0]
        
        # 1. DESIGNER: Generate DNA Probabilities
        designer_input = torch.cat([cell_A, cell_B], dim=1)
        dna_logits = self.designer(designer_input)
        
        # Reshape to [Batch, Seq_Len, 4]
        dna_logits = dna_logits.view(batch_size, self.seq_len, 4)
        
        # Softmax creates the "Soft DNA" (Differentiable probabilities)
        soft_dna = torch.softmax(dna_logits, dim=-1) 
        
        # 2. PHYSICS: Embed the Soft DNA
        # Flatten [Batch, Seq, 4] -> [Batch, Seq*4] for the linear layer
        flat_dna = soft_dna.view(batch_size, -1)
        dna_features = self.dna_encoder(flat_dna)
        
        # 3. HYPERNET: Generate Matrix M
        weights = self.hyper_net(dna_features)
        u_flat, v_flat = torch.chunk(weights, 2, dim=-1)
        U = u_flat.view(batch_size, self.cell_dim, self.rank)
        V = v_flat.view(batch_size, self.cell_dim, self.rank)
        M = torch.bmm(U, V.transpose(1, 2))
        
        # 4. SIMULATION: Apply M to A
        pred_B = torch.bmm(M, cell_A.unsqueeze(-1)).squeeze(-1)
        
        return pred_B, M, soft_dna