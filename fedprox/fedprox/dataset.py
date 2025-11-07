"""MNIST dataset utilities for federated learning."""

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
import os
from fedprox.dataset_preparation import _partition_data, create_domain_shifted_loaders, DataSplitManager,make_pathmnist_clients_with_domains


def get_split_path(num_clients ,split_type: str,seed=42,domain_shift=False):
        """Get path for split file."""
        
        if domain_shift:
          splits_dir = os.path.join(os.getcwd(), 'data_shift_splits')
        
        else:
         splits_dir = os.path.join(os.getcwd(), 'data_splits')
        os.makedirs(splits_dir, exist_ok=True)
        return os.path.join(
            splits_dir, 
            f'splits_clients_{num_clients}_seed_{seed}_{split_type}.pt'
        )
    

def _get_indices( dataset):
        """Extract indices from dataset."""
        if hasattr(dataset, 'indices'):
            return dataset.indices
        return list(range(len(dataset)))



def _get_labels( dataset):
        """Extract labels from dataset."""
        if hasattr(dataset, 'targets'):
            return dataset.targets
        if hasattr(dataset, 'labels'):
            return dataset.labels
        return None
    
def save_splits(num_clients ,trainloaders, valloaders, testloader,domain_shift=False):
        """Save data splits to files."""
        
        # Extract indices and labels from dataloaders
        train_splits = [
            {
                'indices': _get_indices(loader.dataset),
                'labels': _get_labels(loader.dataset)
            }
            for loader in trainloaders
        ]
        
        val_splits = [
            {
                'indices': _get_indices(loader.dataset),
                'labels': _get_labels(loader.dataset)
            }
            for loader in valloaders
        ]
        
        test_split = {
            'indices':_get_indices(testloader.dataset),
            'labels': _get_labels(testloader.dataset)
        }
        
        # Save splits to files
        torch.save(train_splits,get_split_path(num_clients,'train',domain_shift=domain_shift))
        torch.save(val_splits, get_split_path(num_clients,'val',domain_shift=domain_shift))
        torch.save(test_split, get_split_path(num_clients,'test',domain_shift=domain_shift))
        print(f"✓ Saved splits")





def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    batch_size: int = 13,
    seed: Optional[int] = 42,
    domain_shift=False, 
    
) -> Tuple[DataLoader, DataLoader, DataLoader]:
  
    print(f"Dataset partitioning config: {config}")
    
    #print(f' train loader example {datasets[0]}')
    # Create domain-shifted dataloaders
    if domain_shift==True:
      d=3
      trainset, valsets,domain_assignment = make_pathmnist_clients_with_domains(
        num_clients,
        d,
        batch_size
        ,
    
      )
      trainloaders = []
      valloaders = []

       
        
    return trainset, valsets,domain_assignment
