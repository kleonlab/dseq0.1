import torch
import numpy as np

# Configuration
MAX_LEN = 128
PAD_TOKEN = 5
EOS_TOKEN = 4
VOCAB_SIZE_SIMPLE = 6

# ---------------------------------------------------------
# 1. Simple Tokenizer (Your Baseline)
# ---------------------------------------------------------
SIMPLE_MAP = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 5} # N usually treated as PAD or separate
SIMPLE_INV_MAP = {v: k for k, v in SIMPLE_MAP.items()}
SIMPLE_INV_MAP[4] = '<EOS>'
SIMPLE_INV_MAP[5] = '<PAD>' # Explicitly map 5 to PAD if N is also 5, context matters. Usually N is distinct.
# Adjusting maps to be consistent with user request: A-0, T-1, C-2, G-3
# PAD_TOKEN is 5, EOS is 4.

def tokenize_simple(sequence_string):
    """
    Standard character-level tokenization with EOS and PAD.
    Returns: LongTensor of shape [MAX_LEN]
    """
    # 1. Convert to integers
    indices = [SIMPLE_MAP.get(c, 5) for c in sequence_string.upper()] # Default to 5 if unknown
    
    # 2. Truncate if too long (leaving room for EOS)
    if len(indices) > MAX_LEN - 1:
        indices = indices[:MAX_LEN - 1]
        
    # 3. Add EOS
    indices.append(EOS_TOKEN)
    
    # 4. Pad
    pad_amount = MAX_LEN - len(indices)
    indices.extend([PAD_TOKEN] * pad_amount)
    
    return torch.tensor(indices, dtype=torch.long)

def decode_seq(tokens):
    """
    Decodes a sequence of simple tokens back to a DNA string.
    
    Parameters:
    -----------
    tokens : torch.Tensor or list
        Sequence of integer tokens (0=A, 1=T, 2=C, 3=G, 4=EOS, 5=PAD/N).
        
    Returns:
    --------
    str
        The decoded DNA sequence.
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
        
    seq = []
    for token in tokens:
        if token == EOS_TOKEN:
            break
        if token == PAD_TOKEN:
            continue # Skip padding
            
        char = SIMPLE_INV_MAP.get(token, 'N')
        if char not in ['<EOS>', '<PAD>']:
            seq.append(char)
            
    return "".join(seq)

# ---------------------------------------------------------
# 2. p-adic Codon Tokenizer (Dragovich)
# ---------------------------------------------------------
PADIC_NUC_MAP = {'C': 1, 'A': 2, 'T': 3, 'G': 4}
# Reverse map for decoding: Value -> Nucleotide
PADIC_INV_NUC_MAP = {v: k for k, v in PADIC_NUC_MAP.items()}

def tokenize_padic_codons(sequence_string):
    """
    Converts DNA string into p-adic integers representing Codons.
    Input: 'ATGCG...'
    Output: Tensor of integers (Values 31-124)
    """
    # 1. Clean sequence
    seq = sequence_string.upper().replace('U', 'T')
    
    # 2. Ensure divisible by 3 (Truncate incomplete codons at end)
    remainder = len(seq) % 3
    if remainder != 0:
        seq = seq[:-remainder]
        
    padic_tokens = []
    
    # 3. Iterate over triplets
    for i in range(0, len(seq), 3):
        codon = seq[i:i+3]
        
        # Get digits (Default to 0 if N)
        n1 = PADIC_NUC_MAP.get(codon[0], 0)
        n2 = PADIC_NUC_MAP.get(codon[1], 0)
        n3 = PADIC_NUC_MAP.get(codon[2], 0)
        
        if n1 == 0 or n2 == 0 or n3 == 0:
            # Handle codon with 'N' -> Map to specific unknown token
            val = 0 
        else:
            # The Dragovich Formula
            # N1 is lowest power, N3 is highest power (wobble)
            val = n1 + (n2 * 5) + (n3 * 25)
            
        padic_tokens.append(val)
        
    # 4. Pad (Note: Max len is now MAX_LEN // 3 roughly)
    target_len = MAX_LEN // 3
    if len(padic_tokens) > target_len:
        padic_tokens = padic_tokens[:target_len]
    else:
        padic_tokens.extend([0] * (target_len - len(padic_tokens)))
        
    return torch.tensor(padic_tokens, dtype=torch.long)

def decode_padic_seq(padic_tokens):
    """
    Decodes a sequence of p-adic integer tokens back to a DNA string.
    Reverses the formula: Val = N1 + (N2 * 5) + (N3 * 25)
    
    Parameters:
    -----------
    padic_tokens : torch.Tensor or list
        Sequence of integer tokens (p-adic values).
        
    Returns:
    --------
    str
        The decoded DNA sequence.
    """
    if isinstance(padic_tokens, torch.Tensor):
        padic_tokens = padic_tokens.tolist()
        
    seq = []
    
    for val in padic_tokens:
        # 0 represents Padding or Unknown/Mask in this scheme
        if val == 0:
            continue
            
        # Reverse the base-5 expansion
        # Val = n1 + 5*n2 + 25*n3
        
        # n1 is the remainder when dividing by 5
        n1 = val % 5
        
        # Remove n1 contribution, then divide by 5 to get to n2 level
        remaining = (val - n1) // 5
        
        # n2 is now the remainder
        n2 = remaining % 5
        
        # Remove n2 contribution, then divide by 5 to get to n3 level
        n3 = (remaining - n2) // 5
        
        # Map numbers back to nucleotides
        c1 = PADIC_INV_NUC_MAP.get(n1, 'N')
        c2 = PADIC_INV_NUC_MAP.get(n2, 'N')
        c3 = PADIC_INV_NUC_MAP.get(n3, 'N')
        
        seq.append(c1 + c2 + c3)
        
    return "".join(seq)

# ---------------------------------------------------------
# Usage Example
# ---------------------------------------------------------
if __name__ == "__main__":
    test_seq = "ATGTTTCCCTGA" # Met, Phe, Pro, Stop
    
    print("Original:", test_seq)
    
    # 1. Simple Roundtrip
    simple_toks = tokenize_simple(test_seq)
    decoded_simple = decode_seq(simple_toks)
    print(f"\nSimple Tokens: {simple_toks.tolist()[:10]}")
    print(f"Decoded Simple: {decoded_simple}")
    
    # 2. p-adic Roundtrip
    padic_toks = tokenize_padic_codons(test_seq)
    decoded_padic = decode_padic_seq(padic_toks)
    print(f"\np-adic Tokens: {padic_toks.tolist()[:4]}")
    print(f"Decoded p-adic: {decoded_padic}")
    
    assert decoded_simple.startswith(test_seq), "Simple decode mismatch"
    assert decoded_padic == test_seq, "p-adic decode mismatch"
    print("\nTests Passed!")
