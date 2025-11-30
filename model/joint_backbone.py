import torch
import torch.nn as nn
import torch.nn.functional as F

# Retaining the original BioSystem class to avoid breaking existing code
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


# --- NEW ARCHITECTURE COMPONENTS ---

class DNALanguageModel(nn.Module):
    """
    A Transformer-based Language Model that generates DNA sequences 
    conditioned on a 'delta' vector representing cell state change.
    """
    def __init__(self, vocab_size=5, hidden_dim=512, num_layers=12, seq_len=1000):
        """
        vocab_size: Number of DNA bases (usually 4: A, C, G, T) + padding/special if needed.
                    Let's assume 4 bases + 1 special or just 4 outputs.
                    Output shape per position will be 4.
        hidden_dim: Dimension of the transformer embedding.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.vocab_size = 4 # A, C, G, T
        
        # Conditional Input Mapping
        # Maps the delta vector (which might be variable size, let's assume it's projected to hidden_dim)
        # If delta vector size is unknown, we might need an input_dim param.
        # For now, assuming delta is projected or matches hidden_dim before entering or we have a projection.
        # Let's add a projection for flexibility.
        self.delta_projection = nn.Linear(768, hidden_dim) # Assuming cell embedding dim is 768
        
        # Positional Encoding (Learnable or Sinusoidal)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        
        # Transformer Body
        # Using TransformerEncoder for a stack of blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final Projection Head
        # Projects hidden state to vocabulary logits (4 bases)
        self.head = nn.Linear(hidden_dim, self.vocab_size)
        
        self.num_layers = num_layers

    def freeze_body(self):
        """
        Freezes layers 1 to N (the transformer body), leaving only the 
        final projection head (Layer N+1) trainable.
        """
        # Freeze the transformer body
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # Freeze positional embeddings and delta projection
        self.pos_embedding.requires_grad = False
        for param in self.delta_projection.parameters():
            param.requires_grad = False
            
        # Ensure head is trainable
        for param in self.head.parameters():
            param.requires_grad = True
            
        print("DNALanguageModel: Body frozen. Only projection head is trainable.")

    def forward(self, conditional_delta):
        """
        Input: 
            conditional_delta: Tensor of shape (Batch, Input_Dim) representing the desired state change.
        
        Output:
            logits: Tensor of shape (Batch, SeqLen, 4).
        """
        batch_size = conditional_delta.shape[0]
        
        # Project delta to hidden dimension
        # Shape: (Batch, Hidden_Dim)
        delta_emb = self.delta_projection(conditional_delta)
        
        # Expand delta to cover sequence length? 
        # Or prepend as a token?
        # Strategy: Use delta as context for every position (simple conditioning)
        # Shape: (Batch, SeqLen, Hidden_Dim)
        x = delta_emb.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Add positional embeddings
        # Shape: (Batch, SeqLen, Hidden_Dim)
        x = x + self.pos_embedding
        
        # Pass through Transformer
        # Shape: (Batch, SeqLen, Hidden_Dim)
        features = self.transformer(x)
        
        # Project to logits
        # Shape: (Batch, SeqLen, 4)
        logits = self.head(features)
        
        return logits


class VirtualCell(nn.Module):
    """
    A differentiable simulator that predicts the next cell state given 
    a current state and a DNA sequence (as a soft probability matrix).
    """
    def __init__(self, pretrained_model_path=None, cell_dim=768, seq_len=1000):
        super().__init__()
        self.cell_dim = cell_dim
        self.seq_len = seq_len
        
        # Load your simulator or define its architecture.
        # For this example, we'll implement a "Physics Engine" style simulator 
        # similar to the BioSystem physics component but standalone.
        
        # Input: Soft DNA (Batch, Seq_Len*4) -> Hidden
        self.dna_encoder = nn.Linear(seq_len * 4, 256)
        
        # Hypernetwork to generate transformation matrix M
        self.hyper_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, (cell_dim * 64) * 2) # Low Rank (rank=64)
        )
        self.rank = 64
        
        if pretrained_model_path:
            self.load_weights(pretrained_model_path)

    def load_weights(self, path):
        # Placeholder for loading weights
        pass

    def forward(self, control_state, sequence_prob_matrix):
        """
        Input:
            control_state: Tensor (Batch, Cell_Dim).
            sequence_prob_matrix: Tensor (Batch, SeqLen, 4) - Soft probabilities (Gumbel output).
        
        Output:
            predicted_state: Tensor (Batch, Cell_Dim).
        """
        batch_size = control_state.shape[0]
        
        # Flatten the probability matrix for the linear encoder
        # Shape: (Batch, SeqLen * 4)
        flat_dna = sequence_prob_matrix.view(batch_size, -1)
        
        # Encode DNA
        dna_features = self.dna_encoder(flat_dna)
        
        # Generate transformation parameters
        weights = self.hyper_net(dna_features)
        
        # Construct Matrix M (Low Rank)
        u_flat, v_flat = torch.chunk(weights, 2, dim=-1)
        U = u_flat.view(batch_size, self.cell_dim, self.rank)
        V = v_flat.view(batch_size, self.cell_dim, self.rank)
        
        # M = U * V^T
        M = torch.bmm(U, V.transpose(1, 2))
        
        # Apply transformation: State_New = M * State_Old
        # (Or State_New = State_Old + M * State_Old if residual)
        # Using direct multiplication as per previous BioSystem
        pred_state = torch.bmm(M, control_state.unsqueeze(-1)).squeeze(-1)
        
        return pred_state
