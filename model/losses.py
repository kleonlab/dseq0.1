

def entropy_loss(soft_dna):
    # Minimize entropy to encourage the model to be "confident" (sharp choices)
    # soft_dna shape: [Batch, Seq, 4]
    return -torch.sum(soft_dna * torch.log(soft_dna + 1e-9))