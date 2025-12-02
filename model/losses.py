

def entropy_loss(soft_dna):
    # Minimize entropy to encourage the model to be "confident" (sharp choices)
    # soft_dna shape: [Batch, Seq, 4]
    return -torch.sum(soft_dna * torch.log(soft_dna + 1e-9))

def steering_loss(pred, target):
    # eps=1e-8 prevents division by zero
    return 1 - torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1e-8).mean()