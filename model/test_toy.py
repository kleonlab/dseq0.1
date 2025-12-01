import unittest
import os
import torch
import scanpy as sc
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import sys

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from model.data_loader import PerturbationDataset, create_dataloaders, prepare_split

class TestDataLoader(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Path to the specific test file mentioned
        cls.h5ad_path = "datasets/k562_5k.h5ad"
        
        # Verify file exists
        if not os.path.exists(cls.h5ad_path):
            raise FileNotFoundError(f"Test file not found: {cls.h5ad_path}")
            
        print(f"\nLoading test data from: {cls.h5ad_path}")
        cls.adata = sc.read_h5ad(cls.h5ad_path)
        print(f"Loaded AnnData with shape: {cls.adata.shape}")
        
        # Ensure we have a dense matrix for testing if it's sparse
        if hasattr(cls.adata.X, "toarray"):
            cls.adata.X = cls.adata.X.toarray()
            
        # Mock 'Cell Count' if missing (required for prepare_split logic)
        if 'Cell Count' not in cls.adata.obs.columns:
             # Just assign a dummy count
            cls.adata.obs['Cell Count'] = 100 
            
        # Ensure 'Gene Name' exists or use a proxy
        if 'Gene Name' not in cls.adata.obs.columns:
            # Try to find a suitable column or create dummy
            if 'condition' in cls.adata.obs.columns:
                cls.adata.obs['Gene Name'] = cls.adata.obs['condition']
            else:
                # Create dummy gene names for testing stratification
                # Ensure at least 2 samples per gene for stratification
                n_samples = cls.adata.n_obs
                cls.adata.obs['Gene Name'] = ['GeneA'] * (n_samples // 2) + ['GeneB'] * (n_samples - n_samples // 2)
                
        # Ensure we have a 'non-targeting' control for the create_dataloaders logic
        if 'non-targeting' not in cls.adata.obs['Gene Name'].values:
            print("Injecting 'non-targeting' labels for test...")
            cls.adata.obs.iloc[:10, cls.adata.obs.columns.get_loc('Gene Name')] = 'non-targeting'

    def test_perturbation_dataset_structure(self):
        """Test if the Dataset class returns the correct dictionary structure."""
        print("\nTesting PerturbationDataset direct instantiation...")
        
        # Create a small subset - Use explicit iloc for subsetting and copying to avoid view issues
        data_matrix = self.adata.X[:20]
        metadata = self.adata.obs.iloc[:20].copy() # Make a copy to be safe
        
        dataset = PerturbationDataset(
            expression_matrix=data_matrix, 
            metadata_df=metadata
        )
        
        self.assertEqual(len(dataset), 20)
        
        sample = dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn('delta', sample)
        self.assertIn('control_state', sample)
        self.assertIn('target_state', sample)
        self.assertIn('gene_name', sample)
        print(sample)
        print(dataset[15])
        print(dataset[9])
        
        # Check shapes
        n_genes = self.adata.n_vars
        self.assertEqual(sample['delta'].shape[0], n_genes)
        self.assertEqual(sample['control_state'].shape[0], n_genes)
        self.assertEqual(sample['target_state'].shape[0], n_genes)
        print(n_genes)
        
        print("PerturbationDataset structure verified.")

    def test_create_dataloaders_integration(self):
        """Test the helper function that creates train/test loaders."""
        print("\nTesting create_dataloaders integration...")
        
        batch_size = 16
        
        # Ensure we have enough data for the split. 
        # The error `ValueError: With n_samples=0...` suggests prepare_split is filtering everything out.
        # Check the 'Cell Count' logic in prepare_split.
        # If 'Cell Count' is used for filtering (min_cells=50 default in create_dataloaders), 
        # we need to ensure our dummy 'Cell Count' satisfies this.
        # In setUpClass, we set 'Cell Count' to 100, which should pass min_cells=50.
        
        # However, if stratified split fails because a class has too few samples, it might be an issue.
        # Let's check class counts.
        print("Class counts in adata.obs['Gene Name']:")
        print(self.adata.obs['Gene Name'].value_counts())
        
        try:
            train_loader, test_loader = create_dataloaders(self.adata, batch_size=batch_size)
            
            # Check if we got DataLoader objects
            self.assertIsInstance(train_loader, DataLoader)
            self.assertIsInstance(test_loader, DataLoader)
            
            # Fetch one batch
            if len(train_loader) > 0:
                batch = next(iter(train_loader))
                
                self.assertIn('delta', batch)
                self.assertIn('control_state', batch)
                
                current_batch_size = batch['delta'].shape[0]
                self.assertTrue(current_batch_size <= batch_size)
                self.assertEqual(batch['delta'].shape[1], self.adata.n_vars)
            else:
                print("Warning: Train loader is empty.")

            print(f"Train Loader Batches: {len(train_loader)}")
            print(f"Test Loader Batches: {len(test_loader)}")
            print("create_dataloaders verified.")
            
        except ValueError as e:
            print(f"Caught expected ValueError during split if dataset is too small/imbalanced: {e}")
            # In a real test environment with the full 5k dataset, this shouldn't fail if setup is correct.
            # If it fails here, it's likely due to the specific subset or filtering logic.
            raise e

    def test_delta_calculation(self):
        """Test that Delta = Target - Control."""
        print("\nTesting Delta calculation logic...")
        
        data_matrix = self.adata.X[:10]
        metadata = self.adata.obs.iloc[:10].copy()
        
        # Define explicit control indices (e.g., index 0 is control)
        dataset = PerturbationDataset(
            expression_matrix=data_matrix, 
            metadata_df=metadata,
            control_indices=[0] 
        )
        
        # Get sample at index 1
        idx = 1
        sample = dataset[idx]
        
        target = sample['target_state']
        control = sample['control_state']
        delta = sample['delta']
        
        # Allow small floating point error
        diff = (target - control) - delta
        self.assertTrue(torch.allclose(diff, torch.zeros_like(diff), atol=1e-6))
        
        print("Delta calculation verified.")

if __name__ == '__main__':
    unittest.main()
