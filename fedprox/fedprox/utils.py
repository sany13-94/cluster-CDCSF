"""Contains utility functions for CNN FL on MNIST."""

import pickle
from pathlib import Path
from secrets import token_hex
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from flwr.server.history import History


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import torch
from torch.utils.data import DataLoader
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import torch
from torch.utils.data import DataLoader
from collections import Counter
import pandas as pd

class LabelDistributionVisualizer:
    def __init__(self, num_clients: int, num_classes: int = 2):
        self.num_clients = num_clients
        self.num_classes = num_classes
        # Use default matplotlib style instead of seaborn
        plt.style.use('default')
        
    def compute_client_distributions(self, trainloaders: List[DataLoader]) -> List[Dict[int, float]]:
        """Compute label distribution for each client."""
        client_distributions = []
        
        for client_id, loader in enumerate(trainloaders):
            # Get all labels for this client
            all_labels = []
            for _, labels in loader:
                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
                if len(labels.shape) > 1:
                    labels = np.squeeze(labels)
                all_labels.extend(labels)
            
            # Count labels
            label_counts = Counter(all_labels)
            total_samples = len(all_labels)
            
            # Calculate distribution
            distribution = {
                label: count/total_samples 
                for label, count in label_counts.items()
            }
            
            # Ensure all classes are represented
            for label in range(self.num_classes):
                if label not in distribution:
                    distribution[label] = 0.0
                    
            client_distributions.append(distribution)
            
        return client_distributions
    
    def plot_label_distributions(self, trainloaders: List[DataLoader], save_path: str = 'label_distributions.png'):
        """Create and save visualization of label distributions."""
        client_distributions = self.compute_client_distributions(trainloaders)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Prepare data for heatmap
        data_for_heatmap = np.zeros((self.num_clients, self.num_classes))
        for i, dist in enumerate(client_distributions):
            for label in range(self.num_classes):
                data_for_heatmap[i, label] = dist[label]
        
        # Plot heatmap
        im = ax1.imshow(data_for_heatmap, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im, ax=ax1)
        
        # Add text annotations to heatmap
        for i in range(self.num_clients):
            for j in range(self.num_classes):
                text = ax1.text(j, i, f'{data_for_heatmap[i, j]:.2%}',
                              ha="center", va="center", color="black")
        
        # Configure heatmap axes
        ax1.set_xticks(range(self.num_classes))
        ax1.set_yticks(range(self.num_clients))
        ax1.set_xticklabels([f'Class {i}' for i in range(self.num_classes)])
        ax1.set_yticklabels([f'Client {i}' for i in range(self.num_clients)])
        ax1.set_title('Label Distribution Heatmap Across Clients')
        
        # Prepare data for bar plot
        client_ids = [f'Client {i}' for i in range(self.num_clients)]
        class_data = {f'Class {i}': [] for i in range(self.num_classes)}
        
        for dist in client_distributions:
            for label in range(self.num_classes):
                class_data[f'Class {label}'].append(dist[label] * 100)
        
        # Plot grouped bar chart
        x = np.arange(len(client_ids))
        width = 0.8 / self.num_classes
        
        for i in range(self.num_classes):
            ax2.bar(x + i * width, 
                   class_data[f'Class {i}'], 
                   width, 
                   label=f'Class {i}')
        
        ax2.set_ylabel('Percentage of Samples')
        ax2.set_title('Label Distribution Per Client')
        ax2.set_xticks(x + width * (self.num_classes - 1) / 2)
        ax2.set_xticklabels(client_ids)
        ax2.legend()
        
        # Add percentage signs to y-axis
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
        
        # Compute and display statistical measures
        total_samples = sum(len(loader.dataset) for loader in trainloaders)
        global_distribution = {i: 0 for i in range(self.num_classes)}
        
        for dist in client_distributions:
            for label, count in dist.items():
                global_distribution[label] += count / self.num_clients
                
        stats_text = "Global Distribution:\n"
        for label, percentage in global_distribution.items():
            stats_text += f"Class {label}: {percentage:.1%}\n"
            
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return client_distributions, global_distribution

    def compute_distribution_metrics(self, client_distributions: List[Dict[int, float]]) -> Dict:
        """Compute statistical metrics for label distributions."""
        metrics = {}
        
        # Convert distributions to array for easier computation
        distributions = np.zeros((self.num_clients, self.num_classes))
        for i, dist in enumerate(client_distributions):
            for j in range(self.num_classes):
                distributions[i, j] = dist[j]
        
        # Compute metrics
        metrics['mean_per_class'] = np.mean(distributions, axis=0)
        metrics['std_per_class'] = np.std(distributions, axis=0)
        metrics['min_per_class'] = np.min(distributions, axis=0)
        metrics['max_per_class'] = np.max(distributions, axis=0)
        
        # Add distribution skewness metric
        metrics['skewness_per_class'] = np.zeros(self.num_classes)
        for j in range(self.num_classes):
            if np.std(distributions[:, j]) > 0:  # Avoid division by zero
                metrics['skewness_per_class'][j] = np.mean(((distributions[:, j] - np.mean(distributions[:, j])) / 
                                                          np.std(distributions[:, j])) ** 3)
        
        return metrics

def visualize_class_domain_shift(trainloaders: List[DataLoader]):
    plt.style.use('default')
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    class_names = ['Benign Mass/Lesion', 'Malignant Mass/Lesion']
    colors = ['blue', 'red', 'green']
    
    for class_idx in range(2):
        for client_id, loader in enumerate(trainloaders):
            class_images = []
            image_count = 0  # Count actual images
            
            for images, labels in loader:
                mask = labels == class_idx
                if mask.any():
                    class_images.append(images[mask].view(-1))
                    image_count += mask.sum().item()  # Count images of this class
            
            if class_images:
                class_pixels = torch.cat(class_images).cpu().numpy()
                sns.kdeplot(
                    data=class_pixels,
                    ax=axes[class_idx],
                    color=colors[client_id],
                    label=f'Client {client_id} ({image_count} images)',
                    linewidth=2
                )
                
                mean_val = np.mean(class_pixels)
                std_val = np.std(class_pixels)
                median_val = np.median(class_pixels)
                
                axes[class_idx].text(
                    0.02, 0.98 - 0.1 * client_id,
                    f'Client {client_id}: Mean={mean_val:.4f}, Std={std_val:.4f}, Median={median_val:.4f}',
                    transform=axes[class_idx].transAxes,
                    fontsize=8,
                    verticalalignment='top'
                )
        
        axes[class_idx].set_title(f'Pixel Intensity Distribution for {class_names[class_idx]}')
        axes[class_idx].set_xlabel('Pixel Value')
        axes[class_idx].set_ylabel('Density')
        axes[class_idx].legend()
        axes[class_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('class_domain_shift.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig