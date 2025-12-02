import requests
import sys
import torch
import torch.nn.functional as F

def gumbel_softmax(logits, temperature=1.0, hard=True):
    """
    Applies the Gumbel-Softmax trick to logits.
    
    Parameters:
    -----------
    logits : torch.Tensor
        Unnormalized log-probabilities. Shape: (Batch, SeqLen, Vocab)
    temperature : float
        Temperature parameter. Smaller values -> closer to argmax.
    hard : bool
        If True, the returned samples will be one-hot vectors,
        but gradients will still flow through the soft probabilities.
        
    Returns:
    --------
    y : torch.Tensor
        Soft or hard Gumbel-Softmax samples. Shape same as logits.
    """
    # 1. Sample Gumbel noise
    # Gumbel(0, 1) can be sampled by -log(-log(Uniform(0, 1)))
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    
    # 2. Add noise to logits and apply softmax with temperature
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    
    if not hard:
        return y_soft
        
    # 3. Hard Gumbel-Softmax (Straight-Through Estimator)
    # Convert soft probabilities to one-hot
    index = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
    
    # This line is the magic: (y_hard - y_soft).detach() + y_soft
    # Forward pass: returns y_hard
    # Backward pass: gradients flow through y_soft
    ret = y_hard - y_soft.detach() + y_soft
    
    return ret

def steering_loss(predicted_state, target_state):
    """
    Calculates the steering loss as 1 - CosineSimilarity.
    
    Parameters:
    -----------
    predicted_state : torch.Tensor
        Predicted cell state vector. Shape: (Batch, Dim)
    target_state : torch.Tensor
        Target cell state vector. Shape: (Batch, Dim)
        
    Returns:
    --------
    loss : torch.Tensor
        Scalar loss value.
    """
    # Cosine similarity returns values between -1 and 1
    # We want to maximize similarity, so we minimize (1 - similarity)
    # F.cosine_similarity computes similarity along dim 1 by default for 2D tensors
    
    cosine_sim = F.cosine_similarity(predicted_state, target_state, dim=1)
    
    # Average over the batch
    loss = 1.0 - cosine_sim.mean()
    
    return loss

def steering_loss(pred, target):
    # eps=1e-8 prevents division by zero
    return 1 - torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1e-8).mean()

def get_enrichr_genes(library_name, cell_type):
    """
    Fetch genes for a cell type from an Enrichr library.
    """
    base_url = "https://maayanlab.cloud/Enrichr/geneSetLibrary"
    query_string = "?mode=text&libraryName={}"
    url = base_url + query_string.format(library_name)
    
    try:
        # Timeout increased for larger libraries
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"Failed to fetch {library_name}: {response.status_code}")
            return []
        
        # Iterate over lines
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            term = parts[0]
            # Simple case-insensitive substring match
            if cell_type.lower() in term.lower():
                # Gene list starts after term and description (usually index 2)
                # Some libraries might have Term \t Gene1 ... but standard is Term \t Description \t Gene1 ...
                # We take from index 2 onwards to be safe for standard GMT, 
                # but we check if index 1 is a gene (usually uppercase short string). 
                # For safety in Enrichr GMTs, index 2 is safer start for genes.
                genes = parts[2:]
                # Remove any empty strings and convert to upper case for consistency
                return [g.upper() for g in genes if g]
                
        return []
        
    except Exception as e:
        print(f"Error querying Enrichr: {e}")
        return []

def get_delta(control, target, n=100):
    """
    Returns a set of gene names that are differentially expressed / markers 
    for the control and target cell states, using external databases (Enrichr).
    
    Parameters:
    -----------
    control : str
        Name of the control cell state/type.
    target : str
        Name of the target cell state/type.
    n : int
        Number of top genes to consider from each state.
        
    Returns:
    --------
    set
        Set of gene names.
    """
    # List of libraries to try, prioritized by relevance to cell types
    libraries = [
        "PanglaoDB_Augmented_2021", 
        "Azimuth_Cell_Types_2021",
        "Cell_Marker_Augmented_2021"
    ]
    
    control_markers = []
    target_markers = []
    
    print(f"Fetching markers for '{control}' and '{target}' from external databases...")
    
    for lib in libraries:
        # Only search if we haven't found markers yet
        if not control_markers:
            markers = get_enrichr_genes(lib, control)
            if markers:
                control_markers = markers
                print(f"  Found {len(control_markers)} markers for '{control}' in {lib}")
        
        if not target_markers:
            markers = get_enrichr_genes(lib, target)
            if markers:
                target_markers = markers
                print(f"  Found {len(target_markers)} markers for '{target}' in {lib}")
                
        if control_markers and target_markers:
            break
            
    # If still not found, warn
    if not control_markers:
        print(f"  Warning: Could not find markers for control state '{control}'")
    if not target_markers:
        print(f"  Warning: Could not find markers for target state '{target}'")
        
    # Take top n from each
    c_top = control_markers[:n]
    t_top = target_markers[:n]
    
    # Combine
    gene_names = set(c_top).union(set(t_top))
    
    print(f"Returning {len(gene_names)} unique gene names.")
    return gene_names
