# Vocabulary mapping
# 0:A, 1:T, 2:C, 3:G, 4:EOS, 5:PAD

MAX_LEN = 128 # Define a reasonable upper limit

def encode_dna(sequence_string):
    # 1. Convert to indices
    indices = [char_to_int[c] for c in sequence_string]
    
    # 2. Add EOS
    indices.append(4) 
    
    # 3. Pad to MAX_LEN
    pad_amount = MAX_LEN - len(indices)
    indices.extend([5] * pad_amount)
    
    return indices

# Resulting tensor shape: [Batch_Size, 128, 6] (One-hot encoded)