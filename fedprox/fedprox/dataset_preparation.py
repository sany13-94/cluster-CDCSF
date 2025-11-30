
"""Functions for dataset download and processing."""

from typing import List, Optional, Tuple,Dict
import numpy as np
import torch
from hashlib import md5

import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.datasets import MNIST
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader, ConcatDataset
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import math
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, random_split
from typing import List, Tuple
import math
from collections import defaultdict
import medmnist
def  normalize_tensor(x: torch.Tensor):
    
        return x / 255.0 if x.max() > 1.0 else x
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset
import warnings # NOUVEAU: Pour la gestion des avertissements

# --- Filtrage des avertissements ---
# Cette ligne ignore l'avertissement bruyant de PyTorch/NumPy concernant les tableaux non inscriptibles.
# La correction est faite ci-dessous avec .copy(), mais le filtre sert de filet de sécurité.
warnings.filterwarnings(
    "ignore", 
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.", 
    category=UserWarning
)

def build_augmentation_transform():  
    
    t = []
    t.append(transforms.RandomHorizontalFlip(p=0.5))
    t.append(transforms.RandomRotation(degrees=45))
    t.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5))
    return transforms.Compose(t)


# --- Définitions des classes (Mise à jour) ---
class SameModalityDomainShift:
    def __init__(self, domain_id: int, modality: str = "CT", seed: int = 42):
        """
        Initialize domain shift based on domain_id (NOT client_id).
        
        Args:
            domain_id: Domain identifier (0=high-end, 1=mid-range, 2=older-model)
            modality: Type of medical imaging (not used in current implementation)
            seed: Random seed (for potential future stochastic operations)
        """
        self.domain_id = domain_id
        self.seed = seed
        self.characteristics = self._generate_domain_characteristics()
        
    def _generate_domain_characteristics(self) -> Dict:
        """
        Generate characteristics based on domain_id.
        Each domain represents different equipment quality.
        """
        equipment_profiles = {
    0: {  # High-End Equipment
        'name': 'high_end',
        'noise_level': 0.00,
        'contrast_scale': 1.0,
        'brightness_shift': 0.0,
    },
    1: {  # Mid-Range Equipment
        'name': 'mid_range',
        'noise_level': 0.08,        # Reduced from 0.15
        'contrast_scale': 0.75,     # Reduced from 0.7
        'brightness_shift': 0.15,   # Reduced from 0.3
    },
    2: {  # Older Model Equipment
        'name': 'older_model',
        'noise_level': 0.15,        # Keep or slightly reduce
        'contrast_scale': 0.60,     # Increased from 0.5
        'brightness_shift': 0.20,   # CRITICAL: Reduced from 0.5
    }
}
        
        # Get profile for this domain (with fallback to high-end)
        profile = equipment_profiles.get(self.domain_id, equipment_profiles[0])
        
        characteristics = {
            'name': profile['name'],
            'noise_level': profile['noise_level'],
            'contrast_scale': profile['contrast_scale'],
            'brightness_shift': profile['brightness_shift'],
        }
        
        return characteristics
    
    def apply_transform(self, img: torch.Tensor) -> torch.Tensor:
        
        # Domain 0 (high-end) remains unchanged
        if self.domain_id == 0:
            return img
        
        # Apply transformations for other domains
        # 1. Contrast adjustment
        img = img * self.characteristics['contrast_scale']
        
        # 2. Brightness shift
        img = img + self.characteristics['brightness_shift']
        
        # 3. Add Gaussian noise
        if self.characteristics['noise_level'] > 0:
            noise = torch.randn_like(img) * self.characteristics['noise_level']
            img = img + noise
        
        return img


class DomainShiftedPathMNIST(Dataset):
    def __init__(self, base_ds: Dataset, domain_id: int, seed: int = 42):
        """
        Apply domain shift to a base dataset.
        
        Args:
            base_ds: Base dataset returning images in [0, 1] range
            domain_id: Domain identifier for shift characteristics
            seed: Random seed
        """
        self.base = base_ds
        self.domain_id = domain_id
        self.shift = SameModalityDomainShift(domain_id=domain_id, seed=seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = self.shift.apply_transform(img)
        return img, label


class LazyPathMNIST(Dataset):
    def __init__(self, split: str, transform=None):
        import medmnist
        
        self.to_tensor_converter = transforms.ToTensor()

        if split not in ['train', 'test', 'val']:
            raise ValueError("Split must be one of 'train', 'test', or 'val'.")
            
        self.split = split
        
        dataset_class = getattr(medmnist.dataset, 'PathMNIST')
        self.medmnist_ds = dataset_class(split=split, transform=None, download=True)
        
        self.imgs = self.medmnist_ds.imgs          # (N, H, W, C)
        self.labels = self.medmnist_ds.labels.flatten()  # (N,)
        self.targets = self.labels                 # <--- ADD THIS LINE

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = self.imgs[idx].copy()
        img_tensor = self.to_tensor_converter(img)
        
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor


class AugmentationWrapper(Dataset):
    """
    Apply augmentation transformations after domain shift.
    Used only for training data.
    """
    def __init__(self, base_ds: Dataset,client_id,domain_id, augmentation_transform: transforms.Compose):
        self.base = base_ds
        self.augmentation = augmentation_transform
        self.client_id=client_id
        self.domain_id=domain_id

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = self.augmentation(img)
        return img, label

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset


def _get_partition_targets(partition):
    """Extract labels from Subset or Dataset."""
    if isinstance(partition, Subset):
        ds = partition.dataset
        indices = partition.indices
    else:
        ds = partition
        indices = np.arange(len(ds))

    if hasattr(ds, "targets"):
        targets = ds.targets
    elif hasattr(ds, "labels"):
        targets = ds.labels
    else:
        raise AttributeError("Dataset missing 'targets' or 'labels'.")

    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    else:
        targets = np.array(targets)

    return targets[indices]


def visualize_non_iid_bar(
    client_partitions,
    num_classes,
    class_names=None,
    normalize=True,
    figsize=(16, 8),
    cols=4,
    domain_assignment=None,
    output_path=None,
    show=True,
    test_partition=None,
    test_domain_id=None,
):
    """
    Visualizes class distribution across clients using bar plots,
    and optionally includes the test client.

    Args:
        client_partitions: list of datasets/subsets for training clients.
        test_partition: optional dataset/subset for the test client.
        test_domain_id: domain id for the test client (optional).
    """
    # Build final list of partitions
    partitions = list(client_partitions)
    domains = list(domain_assignment) if domain_assignment is not None else None

    # Append test client if provided
    if test_partition is not None:
        partitions.append(test_partition)
        if domains is not None:
            domains.append(test_domain_id)

    num_clients = len(partitions)
    rows = int(np.ceil(num_clients / cols))

    if class_names is None:
        class_names = [f"C{i}" for i in range(num_classes)]

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for client_id, partition in enumerate(partitions):
        targets = _get_partition_targets(partition)
        counts = np.bincount(targets, minlength=num_classes)

        if normalize:
            total = counts.sum()
            values = counts / total * 100 if total > 0 else np.zeros_like(counts)
            ylabel = "Percentage (%)"
        else:
            values = counts
            ylabel = "Count"

        ax = axes[client_id]
        ax.bar(range(num_classes), values, edgecolor="black")

        # Title with domain assignment
        if domains is not None and client_id < len(domains):
            dom_id = domains[client_id]
            ax.set_title(f"Client {client_id} (Domain {dom_id})")
        else:
            ax.set_title(f"Client {client_id}")

        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.2 if values.max() > 0 else 1)

    # Hide empty axes
    for i in range(num_clients, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def make_pathmnist_clients_with_domains(
    k: int = 15,
    d: int = 3,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    seed: int = 42,
    save_samples: bool = True,
    alpha: float = 0.5,
    min_size_ratio: float = 0.1
):
    augmentation_transform = build_augmentation_transform()
    
    ds_train = LazyPathMNIST(split='train')
    ds_test = LazyPathMNIST(split='test')
    
    num_train_clients = k - 1
    
    # Assign clients to domains (unchanged)
    clients_per_domain = num_train_clients // d
    domain_assignment = []
    
    for domain_id in range(d):
        for _ in range(clients_per_domain):
            domain_assignment.append(domain_id)
    
    for i in range(num_train_clients - len(domain_assignment)):
        domain_assignment.append(i % d)
    
    print(f"\n[Domain Assignment]")
    print(f"  Total clients: {k} ({num_train_clients} training + 1 test)")
    print(f"  Number of domains: {d}")
    for domain_id in range(d):
        clients_in_domain = [i for i, d_id in enumerate(domain_assignment) if d_id == domain_id]
        print(f"  Domain {domain_id}: Clients {clients_in_domain}")
    print(f"  Test Domain (unshifted): Client {k-1}")
    
    # ========= NEW: Dirichlet non-IID partitioning over ds_train =========
    raw_train_partitions = _dirichlet_split(
        trainset=ds_train,
        num_partitions=num_train_clients,
        alpha=alpha,
        min_size_ratio=min_size_ratio,
        seed=seed
    )
    # =====================================================================
    
    train_loaders, val_loaders = [], []
    
    # Create loaders for training clients
    for client_id in range(num_train_clients):
        partition_ds = raw_train_partitions[client_id]
        
        # Split each client's partition into train/val
        n_val = int(len(partition_ds) * val_ratio)
        n_trn = len(partition_ds) - n_val
        g_split = torch.Generator().manual_seed(seed + client_id)
        trn_base, val_base = random_split(partition_ds, [n_trn, n_val], generator=g_split)
        
        domain_id = domain_assignment[client_id]
        
        # Save sample images once per domain
        if save_samples and client_id == domain_assignment.index(domain_id):
            save_domain_samples(partition_ds, domain_id, client_id)
        
        # Apply domain shift (per client / domain)
        shifted_trn_ds = DomainShiftedPathMNIST(
            base_ds=trn_base,
            domain_id=domain_id,
            seed=seed
        )
        shifted_val_ds = DomainShiftedPathMNIST(
            base_ds=val_base,
            domain_id=domain_id,
            seed=seed
        )
        
        # Augmentation only for train
        augmented_trn_ds = AugmentationWrapper(
            shifted_trn_ds,
            client_id=client_id,
            domain_id=domain_id,
            augmentation_transform=augmentation_transform
        )
        
        train_loaders.append(
            DataLoader(
                augmented_trn_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
        )
        val_loaders.append(
            DataLoader(
                shifted_val_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
        )
    
    # Test client (no domain shift / or domain_id=0 if you want high-end)
    n_val_test = int(len(ds_test) * val_ratio)
    n_trn_test = len(ds_test) - n_val_test
    g_test_split = torch.Generator().manual_seed(seed + k - 1)
    trn_test_base, val_test_base = random_split(
        ds_test,
        [n_trn_test, n_val_test],
        generator=g_test_split
    )
    
    # Optional: show per-client signature on *Dirichlet* partitions
    client_signatures = []
    for client_id, partition_ds in enumerate(raw_train_partitions):
        signature = get_client_signature(partition_ds)
        print(f"Client {client_id} | Signature: {signature}")
        client_signatures.append(signature)
    
    # For test client, you can keep it "clean" (no domain shift),
    # or give it a separate domain_id (e.g., 3) as you had.
    augmented_trn_test_ds = AugmentationWrapper(
        base_ds=trn_test_base,
        client_id=num_train_clients,
        domain_id=3,  # or 0 if you want high-end test
        augmentation_transform=augmentation_transform
    )
    
    train_loaders.append(
        DataLoader(
            augmented_trn_test_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
    )
    val_loaders.append(
        DataLoader(
            val_test_base,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
    )

    NUM_CLASSES = 9  # PathMNIST
    CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]  # or real names

    visualize_non_iid_bar(
    client_partitions=raw_train_partitions,
    num_classes=9,         # PathMNIST has 9 classes
    class_names=[f"class_{i}" for i in range(9)],
    normalize=True,
    figsize=(22, 12),
    cols=5,
    domain_assignment=domain_assignment,
    test_partition=trn_test_base,      # add the test dataset
    test_domain_id=3,                  # or 0 if test domain should be unshifted
    output_path="non_iid_with_test.png",
    show=True
)

    # domain_assignment + [3] : last id is test client domain
    return train_loaders, val_loaders, domain_assignment + [3]

def _dirichlet_split(
    trainset: Dataset,
    num_partitions: int,
    alpha: float = 0.5,
    min_size_ratio: float = 0.1,
    seed: int = 42
) -> List[Dataset]:
    import numpy as np
    
    np.random.seed(seed)
    
    # Get targets from dataset
    if hasattr(trainset, "targets"):
        targets = trainset.targets
    elif hasattr(trainset, "labels"):
        targets = trainset.labels
    else:
        raise AttributeError("Dataset must have 'targets' or 'labels' attribute.")
    
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    else:
        targets = np.array(targets)
    
    num_classes = len(np.unique(targets))
    num_samples = len(targets)
    min_samples_per_partition = int(num_samples * min_size_ratio / num_partitions)
    
    print(f"Total samples: {num_samples}")
    print(f"Minimum samples per partition: {min_samples_per_partition}")
    
    max_attempts = 10
    for attempt in range(max_attempts):
        client_proportions = np.random.dirichlet(alpha=[alpha] * num_partitions, size=num_classes)
        
        partition_sizes = np.zeros(num_partitions)
        for class_id in range(num_classes):
            class_indices = np.where(targets == class_id)[0]
            proportions = client_proportions[class_id]
            num_samples_per_client = (proportions * len(class_indices)).astype(int)
            partition_sizes += num_samples_per_client
        
        if np.all(partition_sizes >= min_samples_per_partition):
            break
        print(f"Attempt {attempt + 1}: Regenerating distribution due to size constraint...")
    
    print(f"\nFinal Dirichlet distribution shape: {client_proportions.shape}")
    print(f"Client proportions per class:")
    for i in range(num_classes):
        print(f"Class {i}: {client_proportions[i]}")
    
    partition_indices = [[] for _ in range(num_partitions)]
    
    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        np.random.shuffle(class_indices)
        
        proportions = client_proportions[class_id]
        num_samples_per_client = (proportions * len(class_indices)).astype(int)
        
        remaining = len(class_indices) - num_samples_per_client.sum()
        if remaining > 0:
            idx = np.argpartition(num_samples_per_client, remaining)[:remaining]
            num_samples_per_client[idx] += 1
        
        start_idx = 0
        for client_id, num_samples in enumerate(num_samples_per_client):
            end_idx = start_idx + num_samples
            partition_indices[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx
    
    partitions = []
    for client_id, indices in enumerate(partition_indices):
        partition = Subset(trainset, indices)
        partitions.append(partition)
        
        # Debug: show label distribution per partition
        part_targets = targets[indices]
        dist = np.bincount(part_targets, minlength=num_classes)
        print(f"\nClient {client_id}:")
        print(f"Samples per class: {dist}")
        print(f"Total samples: {sum(dist)}")
        percentages = dist / sum(dist) * 100
        print(f"Class distribution: {percentages.round(2)}%")
    
    return partitions

import hashlib

def get_client_signature(dataset_subset):
    """Return a deterministic signature for a Subset based on its indices."""
    indices_bytes = str(dataset_subset.indices).encode()
    return md5(indices_bytes).hexdigest()

# Example for y


def save_domain_samples(partition_ds, domain_id, client_id):
    """Save sample images for visualization (one per domain)."""
    loader = DataLoader(partition_ds, batch_size=8, shuffle=True)
    images, labels = next(iter(loader))
    
    save_path = f"domain_{domain_id}_client_{client_id}_original.png"
    
    img_grid = make_grid(images, nrow=4, normalize=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.title(f"Original Images - Domain {domain_id} - Client {client_id}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"  Saved sample images to {save_path}")

def create_domain_shifted_loaders(
   root_path,
    num_clients: int,
    batch_size: int
,
    transform
    ,domain_shift,
    balance=False,
    iid=True,seed=42
) -> Tuple[List[DataLoader], List[DataLoader]]:
   """Create domain-shifted dataloaders for each client."""
   print(f"Dataset partitioning config: {iid}")
 
   root_path=os.getcwd()
   der = DataSplitManager(
        num_clients=num_clients,
        batch_size=batch_size,
        seed=42,
        domain_shift=True
    )

   datasets = []
   client_validsets = []

  
   trainset_base=BreastMnistDataset(root_path,prefix='train',transform=transform)
   validset_base=BreastMnistDataset(root_path,prefix='valid',transform=transform)
   trainloaders=[]
   valloaders = []
   batch_size=13
   New_split=False
   try:
    
    New_split=False
    train_splits, val_splits= der.load_splits()
    print(f"Loading existing splits for domain shift data... {len(train_splits)}")
    #for client_id in range(num_clients):
    for client_id , (train_split, val_split) in enumerate(zip(train_splits, val_splits)):
            print(f'== client id for sanaa {client_id}')
            # Create subsets using saved splits
            # Create subsets without overwriting base datasets
            train_subset = Subset(trainset_base, train_split['indices'])
            val_subset = Subset(validset_base, val_split['indices'])  # Use original validset_base
            train_subset = DomainShiftedDataset(train_subset, client_id)
            val_subset = DomainShiftedDataset(val_subset, client_id)
          
            # Validate
            if len(val_subset) != len(val_split['indices']):
                print(f"Error: Client {client_id} val_subset length {len(val_subset)} != {len(val_split['indices'])}")
            if max(val_split['indices']) >= len(validset_base):
                print(f"Error: Client {client_id} max index {max(val_split['indices'])} >= {len(validset_base)}")

            datasets.append(train_subset)
            client_validsets.append(val_subset)
   except Exception as e:
       
        print(f"No existing splits found. Creating new splits with domain shift... {e}")
        # Create new splits with non iid or iid distribution

        if balance:
              trainset_base = _balance_classes(trainset_base, seed)

        partition_size = int(len(trainset_base) / num_clients)
        print(f' par {partition_size} and len of train is {len(trainset_base)}')
        lengths = [partition_size] * num_clients
        partition_size_valid = int(len(validset_base) / num_clients)
        lengths_valid = [partition_size_valid] * num_clients
    
        if iid:
              client_validsets = random_split(validset_base, lengths_valid, torch.Generator().manual_seed(seed))

              datasets = random_split(trainset_base, lengths, torch.Generator().manual_seed(seed))
        else:

          #drishlet distribution
          # Non-IID splitting using Dirichlet distribution
          alpha=0.5
          min_size_ratio = 0.1  # Ensures each partition has at least 10% of average size
          datasets = _dirichlet_split(
                    trainset_base,
                    num_clients,
                    alpha=alpha,
                    min_size_ratio = 0.1,  # Ensures each partition has at least 10% of average size,
                    seed=seed
                )
          #print(f'dataset drichlet {datasets[0][0]}')    
         
          partition_size_valid = len(validset_base) // num_clients  # Integer division
          remainder_valid = len(validset_base) % num_clients    # Remainder
          lengths_valid = [partition_size_valid] * num_clients
          for i in range(remainder_valid):
            lengths_valid[i] += 1
          client_validsets = random_split(validset_base, lengths_valid, torch.Generator().manual_seed(seed))
          datasets = [DomainShiftedDataset(dataset, client_id) for client_id, dataset in enumerate(datasets)]
          client_validsets = [DomainShiftedDataset(client_validset, client_id) for client_id, client_validset in enumerate(client_validsets)]
   testset=BreastMnistDataset(root_path,prefix='test',transform=transform)    
         

    
   return datasets, client_validsets , testset,New_split


def makeBreastnistdata(root_path, prefix):
  print(f' root path {root_path}')
  data_path=os.path.join(root_path,'dataset')
  medmnist_data=os.path.join(data_path,'breastmnist.npz')
  print(f'dataset path: {medmnist_data}')
  data=np.load(medmnist_data)
  if prefix=='train':
    train_data=data['train_images']
    train_label=data['train_labels']
    print(f'train_data shape:{train_data.shape}')
    return train_data , train_label
  elif prefix=='test':
    val_data=data['test_images']
    val_label=data['test_labels']
    print( f'test data shape {val_data.shape}')
    return val_data , val_label
  elif prefix=='valid':
    val_data=data['val_images']
    val_label=data['val_labels']
    print( f'valid data shape {val_data.shape}')
    return val_data , val_label
#we define the data partitions of heterogeneity and domain shift
#then the purpose of this code is split a dataset among a number of clients and choose the way of spliting if it is iid or no iid etc
class BreastMnistDataset(data.Dataset):
      
    def __init__(self,root,prefix='valid', transform=None,client_id=0, num_clients=0, domain_shifti=False ):
      data,labels= makeBreastnistdata(root, prefix=prefix)
      self.data=data
      self.labels  = labels  
      self.domain_shifti=domain_shifti
      print(f' domain shift : client id {client_id} and {domain_shifti}')
      if self.domain_shifti==True and client_id is not None:
         #print(f' domain shift enabled')
         modality="MRI"
         
         # Domain shift transform (fixed per client)
         self.domain_shift = SameModalityDomainShift(
            client_id=client_id,
            modality=modality,
            seed=42
          )
        
      if transform:
        # Data augmentation (random per image)

        self.transform=transform
         
  
    def __len__(self):
        self.filelength = len(self.labels)
        return self.filelength

    def __getitem__(self, idx):
        #print(f'data : {self.data[idx]}')
        image =self.data[idx]
        if self.domain_shifti :
       
          # Normalize if needed
          image = normalize_tensor(image)
          image = self.domain_shift.apply_transform(image)

        label = self.labels[idx]
        if self.transform:
          if self.domain_shifti:
           #we already have torch type
           # If it's a tensor, ensure proper format
           image = image.float()
           if len(image.shape) == 2:
              image = image.unsqueeze(0)
           
          image = self.transform(image)
        
        return image, label
    @property
    def targets(self):
        self.labels = np.squeeze(self.labels)
        return self.labels

def _download_data() -> Tuple[Dataset, Dataset]:
    """Download (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)
    return trainset, testset



class DataSplitManager:
    def __init__(self, num_clients: int, batch_size: int, seed: int = 42 ,domain_shift=False):
       
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.seed = seed
        print(f' domain shift is {domain_shift}')
        if domain_shift:
          self.splits_dir = os.path.join(os.getcwd(), 'data_shift_splits')
        
        else:
         self.splits_dir = os.path.join(os.getcwd(), 'data_splits')
        os.makedirs(self.splits_dir, exist_ok=True)
        
    def splits_exist(self):
        """Check if splits already exist."""
        return (
            os.path.exists(self.get_split_path('train')) and
            os.path.exists(self.get_split_path('val')) 
        )
    
    def get_split_path(self, split_type: str):
        """Get path for split file."""
        return os.path.join(
            self.splits_dir, 
            f'splits_clients_{self.num_clients}_seed_{self.seed}_{split_type}.pt'
        )
    def _get_indices(self, dataset):
        """Extract indices from dataset."""
        if hasattr(dataset, 'indices'):
            return dataset.indices
        return list(range(len(dataset)))
    
    def _get_labels(self, dataset):
        """Extract labels from dataset."""
        if hasattr(dataset, 'targets'):
            return dataset.targets
        if hasattr(dataset, 'labels'):
            return dataset.labels
        return None
    def save_splits(self, trainloaders, valloaders, testloader=None):
        """Save data splits to files."""
        # Extract indices and labels from dataloaders
        
        train_splits = [
            {
                'indices': self._get_indices(loader.dataset),
                'labels': self._get_labels(loader.dataset)
            }
            for loader in trainloaders
        ]
        
        val_splits = [
            {
                'indices': self._get_indices(loader.dataset),
                'labels': self._get_labels(loader.dataset)
            }
            for loader in valloaders
        ]
        '''
        if self.testloader:
          test_split = {
            'indices': self._get_indices(testloader.dataset),
            'labels': self._get_labels(testloader.dataset)
          }
          torch.save(test_split, self.get_split_path('test'))

        '''
        # Save splits to files
        torch.save(train_splits, self.get_split_path('train'))
        torch.save(val_splits, self.get_split_path('val'))
        print(f"✓ Saved splits of domain shift to {self.splits_dir}")
    def load_splits(self):
        """Load splits and create dataloaders."""
        
        if not self.splits_exist():
            print("No existing splits found. Creating new splits...")
            return False
        
        print("Loading existing splits...")
        train_splits = torch.load(self.get_split_path('train'))
        val_splits = torch.load(self.get_split_path('val'))
        
       
        print(f"Loaded splits format:")
        print(f"Train splits type: {type(train_splits)}")
        print(f"Sample indices type: {type(train_splits[0]['indices'])}")
        return train_splits , val_splits 
  

# pylint: disable=too-many-locals
def _partition_data(
    num_clients,
    dataset_name,
    transform,
    iid: Optional[bool] = True,
    power_law: Optional[bool] = True,
    balance: Optional[bool] = False,
    seed: Optional[int] = 42,
    domain_shift=False,
   
    
) -> Tuple[List[Dataset], Dataset]:
    root_path=os.getcwd()
    
    if dataset_name=='breastmnist':
      root_path=os.getcwd()
      trainset=BreastMnistDataset(root_path,prefix='train',transform=transform)
      testset=BreastMnistDataset(root_path,prefix='test',transform=transform)
      validset=BreastMnistDataset(root_path,prefix='valid',transform=transform)
      trainloaders=[]
      valloaders = []
      batch_size=13
      New_split=False
      der=DataSplitManager(
   
        num_clients=num_clients,
        batch_size=13,
        seed=42
        )
      try:
        datasets=[]
        client_validsets=[]
        train_splits, val_splits= der.load_splits()
        # Create client-specific dataloaders
        i=0
        for train_split, val_split in zip(train_splits, val_splits):
            # Create subset datasets
            train_subset = Subset(trainset, train_split['indices'])

            valid_subset = Subset(validset, val_split['indices'])
            #testset=Subset(testset, test_splits['indices'])
            train_indices = train_split['indices']
            val_indices = val_split['indices']
            # Append to lists
            # Print first few indices to verify consistency
            print(f"\nClient {i} data points:")
            print(f"Last 5 training indices: {train_indices[5:]}")
            print(f"Number of training samples: {len(train_indices)}")
            datasets.append(train_subset)
            client_validsets.append(valid_subset)
            i+=1
        print(f' took the already splitting data')
      except  Exception as e:
        print(e)
        print(f'new data splitting')
        # Save the splits
        New_split=True
        if balance:
          trainset = _balance_classes(trainset, seed)

        partition_size = int(len(trainset) / num_clients)
        print(f' par {partition_size} and len of train is {len(trainset)}')
        lengths = [partition_size] * num_clients
        partition_size_valid = int(len(validset) / num_clients)
        lengths_valid = [partition_size_valid] * num_clients
    
        if iid:
          client_validsets = random_split(validset, lengths_valid, torch.Generator().manual_seed(seed))

          datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
        else:

          #drishlet distribution
          # Non-IID splitting using Dirichlet distribution
          alpha=0.5
          min_size_ratio = 0.1  # Ensures each partition has at least 10% of average size
          datasets = _dirichlet_split(
                    trainset,
                    num_clients,
                    alpha=alpha,
                    min_size_ratio = 0.1,  # Ensures each partition has at least 10% of average size,
                    seed=seed
                )
          print(f'dataset drichlet {datasets[0]}')      
          partition_size_valid = int(len(validset) / num_clients)
          lengths_valid = [partition_size_valid] * num_clients
          client_validsets = random_split(validset, lengths_valid, 
                                              torch.Generator().manual_seed(seed))

    return datasets, testset , client_validsets , New_split


def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    Dataset
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in np.cumsum(class_counts):
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(trainset.targets[idxs[int(count) : int(count + smallest)]])
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled


def _sort_by_class(trainset: Dataset) -> Dataset:
    """Sort dataset by class/label.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be sorted.

    Returns
    -------
    Dataset
        The sorted training dataset.
    """
    # Convert targets to numpy array if they're tensors
    if isinstance(trainset.targets, torch.Tensor):
        targets = trainset.targets.numpy()
    else:
        targets = np.array(trainset.targets)
    
    # Calculate class counts
    class_counts = np.bincount(targets)
    
    # Sort indices
    idxs = np.argsort(targets)  # sort targets in ascending order

    tmp = []  # create subset of smallest class
    tmp_targets = []  # same for targets

    start = 0
    for count in np.cumsum(class_counts):
        # Add rest of classes
        subset_indices = idxs[start : int(count + start)]
        tmp.append(Subset(trainset, subset_indices))
        
        # Convert targets to tensor before adding
        subset_targets = targets[subset_indices]
        tmp_targets.append(torch.from_numpy(subset_targets))
        
        start += count
        
    sorted_dataset = ConcatDataset(tmp)  # concat dataset
    sorted_dataset.targets = torch.cat(tmp_targets) 
    return sorted_dataset


# pylint: disable=too-many-locals, too-many-arguments
def _power_law_split(
    sorted_trainset: Dataset,
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> Dataset:
    """Partition the dataset following a power-law distribution. It follows the.

    implementation of Li et al 2020: https://arxiv.org/abs/1812.06127 with default
    values set accordingly.

    Parameters
    ----------
    sorted_trainset : Dataset
        The training dataset sorted by label/class.
    num_partitions: int
        Number of partitions to create
    num_labels_per_partition: int
        Number of labels to have in each dataset partition. For
        example if set to two, this means all training examples in
        a given partition will be long to the same two classes. default 2
    min_data_per_partition: int
        Minimum number of datapoints included in each partition, default 10
    mean: float
        Mean value for LogNormal distribution to construct power-law, default 0.0
    sigma: float
        Sigma value for LogNormal distribution to construct power-law, default 2.0

    Returns
    -------
    Dataset
        The partitioned training dataset.
    """
    targets = sorted_trainset.targets
    print(f' targets : {targets}')
    full_idx = list(range(len(targets)))

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)
    labels_cs = [0] + labels_cs[:-1].tolist()
    print(f' targets ctn: {class_counts}')
    partitions_idx: List[List[int]] = []
    num_classes = len(np.bincount(targets))
    hist = np.zeros(num_classes, dtype=np.int32)

    # assign min_data_per_partition
    min_data_per_class = int(min_data_per_partition / num_labels_per_partition)
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            # label for the u_id-th client
            cls = (u_id + cls_idx) % num_classes
            # record minimum data
            indices = list(
                full_idx[
                    labels_cs[cls]
                    + hist[cls] : labels_cs[cls]
                    + hist[cls]
                    + min_data_per_class
                ]
            )
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    # add remaining images following power-law
    '''
    probs = np.random.lognormal(
        mean,
        sigma,
        (num_classes, int(num_partitions / num_classes), num_labels_per_partition),
    )
    '''
    probs = np.random.lognormal(
    mean,
    sigma,
    (num_classes, num_partitions, num_labels_per_partition)  # Each client gets unique group
)
    remaining_per_class = class_counts - hist
    # obtain how many samples each partition should be assigned for each of the
    # labels it contains
    # pylint: disable=too-many-function-args
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )
    
    print(f' targets prob: {probs}')
    print(f"Distribution probabilities shape: {probs.shape}")
    print(f"Example probabilities for first few clients:\n{probs[:, :3, :]}")
    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            #count = int(probs[cls, u_id // num_classes, cls_idx])
            count = int(probs[cls, u_id, cls_idx])  # u_id is the unique group for each client

            # add count of specific class to partition
            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    # construct subsets
    partitions = [Subset(sorted_trainset, p) for p in partitions_idx]
    return partitions

def _dirichlet_split(
    trainset: Dataset,
    num_partitions: int,
    alpha: float = 0.5,
    min_size_ratio: float = 0.1,  # Minimum size of any partition as a ratio of average size
    seed: int = 42
) -> List[Dataset]:
    """
    Partition the dataset using Dirichlet distribution with balanced constraints.
    """
    np.random.seed(seed)
    
    # Get targets
    if isinstance(trainset.targets, torch.Tensor):
        targets = trainset.targets.numpy()
    else:
        targets = np.array(trainset.targets)
    
    num_classes = len(np.unique(targets))
    num_samples = len(targets)
    min_samples_per_partition = int(num_samples * min_size_ratio / num_partitions)
    
    print(f"Total samples: {num_samples}")
    print(f"Minimum samples per partition: {min_samples_per_partition}")
    
    # Keep generating distributions until we get one that satisfies our constraints
    max_attempts = 10
    for attempt in range(max_attempts):
        # Generate Dirichlet distribution for each class
        client_proportions = np.random.dirichlet(alpha=[alpha] * num_partitions, size=num_classes)
        
        # Calculate expected samples per partition
        partition_sizes = np.zeros(num_partitions)
        for class_id in range(num_classes):
            class_indices = np.where(targets == class_id)[0]
            proportions = client_proportions[class_id]
            num_samples_per_client = (proportions * len(class_indices)).astype(int)
            partition_sizes += num_samples_per_client
        
        # Check if distribution satisfies minimum size constraint
        if np.all(partition_sizes >= min_samples_per_partition):
            break
        
        print(f"Attempt {attempt + 1}: Regenerating distribution due to size constraint...")
    
    print(f"\nFinal Dirichlet distribution shape: {client_proportions.shape}")
    print(f"Client proportions per class:")
    for i in range(num_classes):
        print(f"Class {i}: {client_proportions[i]}")
    
    # Initialize partition indices
    partition_indices = [[] for _ in range(num_partitions)]
    
    # Partition data class by class
    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        np.random.shuffle(class_indices)
        
        proportions = client_proportions[class_id]
        num_samples_per_client = (proportions * len(class_indices)).astype(int)
        
        # Adjust for rounding errors
        remaining = len(class_indices) - num_samples_per_client.sum()
        if remaining > 0:
            # Add remaining samples to clients with lowest allocation
            idx = np.argpartition(num_samples_per_client, remaining)[:remaining]
            num_samples_per_client[idx] += 1
        
        # Distribute indices
        start_idx = 0
        for client_id, num_samples in enumerate(num_samples_per_client):
            end_idx = start_idx + num_samples
            partition_indices[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx
    
    # Create partitions
    partitions = []
    for client_id, indices in enumerate(partition_indices):
        partition = Subset(trainset, indices)
        partitions.append(partition)
        
        # Print distribution for this partition
        if isinstance(partition.dataset.targets, torch.Tensor):
            part_targets = partition.dataset.targets[partition.indices].numpy()
        else:
            part_targets = np.array([partition.dataset.targets[j] for j in partition.indices])
            
        dist = np.bincount(part_targets, minlength=num_classes)
        print(f"\nClient {client_id}:")
        print(f"Samples per class: {dist}")
        print(f"Total samples: {sum(dist)}")
        
        # Calculate and print percentages
        percentages = dist / sum(dist) * 100
        print(f"Class distribution: {percentages.round(2)}%")
    
    return partitions