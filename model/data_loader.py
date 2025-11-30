import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_split(df, min_cells=0, max_cells_train=2000):
    # 1. Remove the "Tail" (Too noisy)
    df_clean = df[df['Cell Count'] >= min_cells].copy()
    print(f"Dropped {len(df) - len(df_clean)} genes due to low cell counts.")
    
    # 2. Stratified Split
    # We use the Gene Name to ensure every gene is in both train and test
    train_df, test_df = train_test_split(
        df_clean, 
        test_size=0.15, 
        stratify=df_clean['Gene Name'],
        random_state=42
    )
    
    # 3. Cap the "Head" in Training only (Optional but recommended)
    # This prevents PINK1 from dominating the training gradients
    train_df_capped = train_df.groupby('Gene Name').apply(
        lambda x: x.sample(n=min(len(x), max_cells_train), random_state=42)
    ).reset_index(drop=True)
    
    return train_df_capped, test_df

# Usage
#train_set, test_set = prepare_split(df)



def prepare_split(df, min_cells=0, max_cells_train=2000, split_col='Gene Name'):
    """
    Splits the metadata dataframe into train and test sets using stratified sampling.
    
    Args:
        df (pd.DataFrame): Metadata dataframe containing at least 'Cell Count' and split_col.
        min_cells (int): Minimum cell count required to include a gene category.
        max_cells_train (int): Maximum number of cells to sample per gene for training (capping).
        split_col (str): Column to stratify by (default 'Gene Name').
        
    Returns:
        tuple: (train_df, test_df) - The split dataframes.
    """
    # 1. Remove the "Tail" (Too noisy / low representation)
    if 'Cell Count' in df.columns:
        df_clean = df[df['Cell Count'] >= min_cells].copy()
        print(f"Dropped {len(df) - len(df_clean)} cells due to low cell counts.")
    else:
        df_clean = df.copy()

    # 2. Stratified Split
    # We use the Gene Name to ensure every gene is in both train and test
    try:
        train_df, test_df = train_test_split(
            df_clean, 
            test_size=0.15, 
            stratify=df_clean[split_col],
            random_state=42
        )
    except ValueError as e:
        print(f"Warning: Stratification failed (likely singleton classes). Falling back to random split. Error: {e}")
        train_df, test_df = train_test_split(df_clean, test_size=0.15, random_state=42)
    
    # 3. Cap the "Head" in Training only (Prevent dominant classes)
    if max_cells_train > 0:
        train_df_capped = train_df.groupby(split_col).apply(
            lambda x: x.sample(n=min(len(x), max_cells_train), random_state=42)
        ).reset_index(drop=True)
        print(f"Training set capped. Size reduced from {len(train_df)} to {len(train_df_capped)}")
        train_df = train_df_capped
    
    return train_df, test_df

class PerturbationDataset(Dataset):
    """
    A PyTorch Dataset for loading single-cell perturbation pairs.
    It returns a triplet: (Control State, Target State, Delta).
    """
    def __init__(self, 
                 expression_matrix, 
                 metadata_df, 
                 control_indices=None,
                 transform=None):
        """
        Args:
            expression_matrix (np.ndarray or torch.Tensor): Matrix of shape (N_cells, N_genes).
                                                          Should typically be restricted to HVGs.
            metadata_df (pd.DataFrame): Dataframe matching the rows of expression_matrix.
            control_indices (list or np.array, optional): Indices in expression_matrix corresponding 
                                                        to 'Control' (Non-targeting) cells.
                                                        If None, we assume the dataset *is* the control.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = torch.tensor(expression_matrix, dtype=torch.float32)
        self.metadata = metadata_df.reset_index(drop=True)
        self.transform = transform
        
        # Store control pool for sampling
        if control_indices is not None:
            self.control_pool = self.data[control_indices]
        else:
            # If no specific controls provided, we might calculate a global mean 
            # or expect the user to handle this differently. 
            # Here we default to using the global mean as a static control if pool is missing.
            self.control_pool = self.data.mean(dim=0).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the training triplet.
        """
        # 1. Get the Target (Outcome) Cell State
        target_state = self.data[idx]
        
        # 2. Get the Control Cell State
        # Strategy: Randomly sample one control cell from the pool to pair with this target.
        # This acts as a form of data augmentation (pairing different controls with targets).
        if len(self.control_pool) > 1:
            control_idx = torch.randint(0, len(self.control_pool), (1,)).item()
            control_state = self.control_pool[control_idx]
        else:
            control_state = self.control_pool[0] # Use mean/single control
            
        # 3. Compute Delta (The Biological Prompt)
        # Delta = Target - Control
        delta = target_state - control_state
        
        # 4. Optional: Get Gene Name for tracking
        gene_name = self.metadata.iloc[idx].get('Gene Name', 'Unknown')
        
        sample = {
            'delta': delta,
            'control_state': control_state,
            'target_state': target_state,
            'gene_name': gene_name
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# --- Usage Example Helper ---
def create_dataloaders(adata, batch_size=32):
    """
    Helper to create train/test loaders from an AnnData object or similar.
    Assumes adata.X is the expression matrix and adata.obs is metadata.
    """
    # 1. Identify Controls (e.g., 'non-targeting' or 'NTC')
    # Adjust 'condition' and 'NTC' to match your specific column/label
    control_mask = adata.obs['Gene Name'] == 'non-targeting'
    control_indices = np.where(control_mask)[0]
    
    # 2. Split Metadata
    train_obs, test_obs = prepare_split(adata.obs, min_cells=50)
    
    # 3. Map back to integer indices for the matrix
    # (This assumes the indices in split DFs match original matrix order if not shuffled yet)
    # A safer way is to rely on index matching if adata.obs has unique indices
    train_indices = adata.obs.index.get_indexer(train_obs.index)
    test_indices = adata.obs.index.get_indexer(test_obs.index)
    
    # 4. Create Datasets
    # We pass the FULL matrix, but slice the metadata. 
    # The Dataset class normally takes aligned inputs, so we slice both here:
    
    train_dataset = PerturbationDataset(
        expression_matrix=adata.X[train_indices],
        metadata_df=train_obs,
        control_indices=np.where(control_mask[train_indices])[0] if len(control_indices) > 0 else None
    )
    
    test_dataset = PerturbationDataset(
        expression_matrix=adata.X[test_indices],
        metadata_df=test_obs,
        control_indices=np.where(control_mask[test_indices])[0] if len(control_indices) > 0 else None
    )
    
    # 5. Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

