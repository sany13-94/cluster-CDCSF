import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
import seaborn as sns
from torch.utils.data import DataLoader

class StructuredFeatureVisualizer:
    def __init__(self, num_clients: int, num_classes: int, save_dir: str = "feature_visualizations"):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Track best accuracies for each client and overall
        self.best_client_accuracies = {i: 0.0 for i in range(num_clients)}
        self.best_average_accuracy = 0.0

        # Configure t-SNE
        self.tsne = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=1000,
            random_state=42
        )


    def visualize_all_clients_by_class(self,
                                features_dict: Dict[str, np.ndarray],  # Note: str keys
                                labels_dict: Dict[str, np.ndarray],    # Note: str keys
                                accuracies: Dict[str, float],          # Note: str keys
                                epoch: int,
                                stage: str = "validation"):
      """Create a single visualization showing all clients' features organized by class."""
      plt.figure(figsize=(15, 10))

      # Debug: Print initial data
      print("\nDebug Information:")
      print(f"Features dict keys: {features_dict.keys()}")
      print(f"Labels dict keys: {labels_dict.keys()}")

      all_features = []
      all_client_ids = []  # Will store indices (0, 1, 2) instead of string IDs
      all_labels = []

      # Create mapping from string IDs to indices
      client_id_to_idx = {client_id: idx for idx, client_id in enumerate(sorted(features_dict.keys()))}
      print(f"Client ID mapping: {client_id_to_idx}")

      # Collect all features and labels
      for i,(client_id, features) in enumerate(features_dict.items()):
        if features is not None and labels_dict.get(client_id) is not None:
            print(f"\nProcessing Client {client_id}:")
            print(f"Features shape: {features.shape}")
            print(f"Labels shape: {labels_dict[client_id].shape}")

            features_np = np.array(features, dtype=np.float32)
            labels_np = np.array(labels_dict[client_id], dtype=np.int32)
            labels_np = labels_np.reshape(-1)

            #client_idx = client_id_to_idx[client_id]  # Convert string ID to index
            client_idx=i
            all_features.append(features_np)
            all_client_ids.extend([client_idx] * len(features_np))  # Use index instead of string ID
            all_labels.extend(labels_np)

            print(f"Added {len(features_np)} samples for client {client_id} (index {client_idx})")

      if not all_features:
        print("No features to visualize!")
        return

      # Convert to numpy arrays
      all_features = np.vstack(all_features)
      all_client_ids = np.array(all_client_ids)
      all_labels = np.array(all_labels)

      print("\nFinal data shapes:")
      print(f"All features shape: {all_features.shape}")
      print(f"All client IDs shape: {all_client_ids.shape}")
      print(f"All labels shape: {all_labels.shape}")
      print(f"Unique client indices: {np.unique(all_client_ids)}")
      print(f"Unique labels: {np.unique(all_labels)}")

      # Compute t-SNE embedding
      print("\nComputing t-SNE...")
      embedded_features = self.tsne.fit_transform(all_features)
      print(f"t-SNE shape: {embedded_features.shape}")

      # Define distinct markers and colors
      markers = ['o', 's', '^']  # Different marker for each client
      class_colors = ['#2ecc71', '#e74c3c']  # Green for class 0, Red for class 1

      # Plot features
      for class_idx in range(self.num_classes):
        for client_idx in range(len(client_id_to_idx)):  # Use number of actual clients
            mask = (all_labels == class_idx) & (all_client_ids == client_idx)
            num_points = np.sum(mask)
            print(f"\nPlotting Class {class_idx}, Client {client_idx}: {num_points} points")

            if np.any(mask):
                plt.scatter(
                    embedded_features[mask, 0],
                    embedded_features[mask, 1],
                    c=class_colors[class_idx],
                    marker=markers[client_idx],
                    label=f'Client {client_idx}, Class {class_idx}',
                    alpha=0.6,
                    s=100
                )

      # Set plot limits with some padding
      x_min, x_max = embedded_features[:, 0].min(), embedded_features[:, 0].max()
      y_min, y_max = embedded_features[:, 1].min(), embedded_features[:, 1].max()

      # Add 10% padding
      x_padding = (x_max - x_min) * 0.1
      y_padding = (y_max - y_min) * 0.1

      plt.xlim(x_min - x_padding, x_max + x_padding)
      plt.ylim(y_min - y_padding, y_max + y_padding)


      # Create formatted accuracy strings using enumeration for consistency
      accuracy_strings = []
      for i, (client_id, acc) in enumerate(accuracies.items()):
        accuracy_strings.append(f"Client {i}: {acc:.4f}")
      plt.title(f'Feature Space Visualization - All Clients by Class\n'
             f'Epoch {epoch}, {stage}\n'
             f'Average Accuracy: {np.mean(list(accuracies.values())):.4f}\n'
             f'Client Accuracies: {", ".join(accuracy_strings)}')

      plt.xlabel('t-SNE Component 1')
      plt.ylabel('t-SNE Component 2')
      plt.grid(True, alpha=0.3)

      # Create legend
      legend_class = [plt.Line2D([0], [0], color=c, marker='o', linestyle='',
                            label=f'Class {i}') for i, c in enumerate(class_colors)]
      legend_clients = [plt.Line2D([0], [0], color='gray', marker=m, linestyle='',
                               label=f'Client {i}') for i, m in enumerate(markers[:len(client_id_to_idx)])]
      plt.legend(handles=legend_class + legend_clients,
              bbox_to_anchor=(1.05, 1), loc='upper left')

      # Save visualization
      filename = f"features_epoch{epoch}_acc{np.mean(list(accuracies.values())):.4f}.png"
      save_path = os.path.join(self.save_dir, filename)
      plt.savefig(save_path, bbox_inches='tight', dpi=300)
      plt.close()

      print(f"\nVisualization saved to: {save_path}")

    def visualize_global_local_features(self,
                                  client_features_dict: Dict[int, np.ndarray],
                                  global_z: np.ndarray,
                                  client_labels_dict: Dict[int, np.ndarray],
                                  global_labels: np.ndarray,
                                  epoch: int):
      """
      Visualize global generator features vs local client features,
      with both colored by their respective classes.
      """
      plt.figure(figsize=(12, 8))
    
      # Combine all local features and their labels
      all_local_features = []
      all_local_labels = []
      client_indicators = []
    
      for i, (client_id, features) in enumerate(client_features_dict.items()):
        all_local_features.append(features)
        all_local_labels.extend(client_labels_dict[client_id])
        client_indicators.extend([i] * len(features))
    
      all_local_features = np.vstack(all_local_features)
    
      # Combine local and global features for joint t-SNE
      combined_features = np.vstack([all_local_features, global_z])
    
      print(f"Shape of combined features: {combined_features.shape}")
      print(f"Number of local features: {len(all_local_features)}")
      print(f"Number of global features: {len(global_z)}")
    
      # Compute t-SNE embedding
      tsne = TSNE(n_components=2, perplexity=30, random_state=42)
      embedded_features = tsne.fit_transform(combined_features)
    
      # Split back into local and global
      local_embedded = embedded_features[:len(all_local_features)]
      global_embedded = embedded_features[len(all_local_features):]
    
      # Define colors and markers
      colors = ['#2ecc71', '#e74c3c']  # Green for class 0, Red for class 1
      markers = ['o', 's', '^']  # Different marker for each client
    
      # Plot local features
      classes = np.unique(all_local_labels)
      for class_idx in classes:
        for client_idx in range(len(client_features_dict)):
            mask = (np.array(all_local_labels) == class_idx) & (np.array(client_indicators) == client_idx)
            plt.scatter(local_embedded[mask, 0], local_embedded[mask, 1],
                       c=colors[int(class_idx)], marker=markers[client_idx],
                       label=f'Client {client_idx} Class {class_idx}',
                       alpha=0.6)
    
      # Plot global features by class
      print(f"Global labels shape: {global_labels.shape}")
      print(f"Unique global labels: {np.unique(global_labels)}")
    
      for class_idx in classes:
        mask = global_labels == class_idx
        plt.scatter(global_embedded[mask, 0], global_embedded[mask, 1],
                   c=colors[int(class_idx)], marker='*', s=150,
                   label=f'Global Features Class {class_idx}',
                   edgecolor='black', linewidth=1,
                   alpha=0.8)
    
      plt.title(f'Global vs Local Features Comparison - Epoch {epoch}\n'
             f'Features colored by class, Markers by source')
      plt.xlabel('t-SNE Component 1')
      plt.ylabel('t-SNE Component 2')
      plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
      plt.grid(True, alpha=0.3)
    
      # Save visualization
      filename = f"global_local_comparison_epoch{epoch}.png"
      save_path = os.path.join(self.save_dir, filename)
      plt.savefig(save_path, bbox_inches='tight', dpi=300)
      plt.close()
    
      print(f"Visualization saved to: {save_path}")
    def _setup_folders(self) -> str:
        """Create organized folder structure for visualizations."""
        # Base directory for all visualizations
        base_dir = os.path.join("visualizations", self.experiment_name)

        # Create timestamp-based run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, f"run_{timestamp}")

        # Create subfolders for different visualization types
        folders = [
            os.path.join(run_dir, folder)
            for folder in ["tsne_plots", "feature_stats", "best_performances"]
        ]

        # Create all directories
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

        return run_dir

    def _save_visualization(self, fig, accuracy: float, client_id: int,
                          epoch: int, subfolder: str = "tsne_plots"):
        """Save visualization with organized naming."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"client{client_id}_epoch{epoch}_acc{accuracy:.4f}_{timestamp}.png"

        # Save in appropriate subfolder
        save_path = os.path.join(self.base_dir, subfolder, filename)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # If this is a best performance, save a copy in best_performances folder
        if accuracy > self.best_accuracy[client_id]:
            self.best_accuracy[client_id] = accuracy
            best_path = os.path.join(self.base_dir, "best_performances", filename)
            fig.savefig(best_path, bbox_inches='tight', dpi=300)

        return save_path

    def visualize_features(self,
                          features_dict: Dict[int, torch.Tensor],
                          labels_dict: Dict[int, torch.Tensor],
                          accuracy: float,
                          client_id: int,
                          epoch: int,
                          stage: str = "validation"):
        """Create and save feature visualizations."""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Collect all features and labels
        all_features = []
        all_client_ids = []
        all_labels = []

        for cid, features in features_dict.items():
            if features is not None and labels_dict[cid] is not None:
                features_np = features.detach().cpu().numpy()
                labels_np = labels_dict[cid].detach().cpu().numpy()

                all_features.append(features_np)
                all_client_ids.extend([cid] * len(features_np))
                all_labels.extend(labels_np)

        if all_features:
            all_features = np.vstack(all_features)
            all_client_ids = np.array(all_client_ids)
            all_labels = np.array(all_labels)

            # Compute t-SNE
            embedded_features = self.tsne.fit_transform(all_features)

            # Create visualizations
            self._plot_client_wise(ax1, embedded_features, all_client_ids, client_id)
            self._plot_class_wise(ax2, embedded_features, all_labels, all_client_ids)

            # Add overall title
            plt.suptitle(f'Feature Space Visualization - Epoch {epoch}, {stage}\n'
                       f'Client {client_id} Accuracy: {accuracy:.4f}',
                       y=1.02, fontsize=14)

            # Save visualization
            save_path = self._save_visualization(fig, accuracy, client_id, epoch)
            print(f"Saved visualization to: {save_path}")

            # Save feature statistics
            self._save_feature_statistics(
                all_features, all_labels, client_id, epoch, accuracy
            )

    def _save_feature_statistics(self, features: np.ndarray, labels: np.ndarray,
                               client_id: int, epoch: int, accuracy: float):
        """Save statistical information about features."""
        stats = {
            'client_id': client_id,
            'epoch': epoch,
            'accuracy': accuracy,
            'feature_means': features.mean(axis=0).tolist(),
            'feature_stds': features.std(axis=0).tolist(),
            'class_distribution': np.bincount(labels).tolist()
        }

        # Save statistics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stats_client{client_id}_epoch{epoch}_{timestamp}.txt"
        stats_path = os.path.join(self.base_dir, "feature_stats", filename)

        with open(stats_path, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

    def _plot_client_wise(self, ax, embedded_features, client_ids, current_client_id):
        """Plot features colored by client."""
        for cid in range(self.num_clients):
            mask = client_ids == cid
            if np.any(mask):
                ax.scatter(
                    embedded_features[mask, 0],
                    embedded_features[mask, 1],
                    c=[self.client_colors[cid]],
                    label=f'Client {cid}',
                    alpha=0.6,
                    s=50
                )

        ax.set_title('Features by Client')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    def visualize_class_features(self,
                             features_dict: Dict[int, torch.Tensor],
                             labels_dict: Dict[int, torch.Tensor],
                             accuracy: float,
                             client_id: int,
                             epoch: int,
                             stage: str = "validation"):
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Collect all features and labels
        all_features = []
        all_client_ids = []
        all_labels = []

        for cid, features in features_dict.items():
            if features is not None and labels_dict.get(cid) is not None:
                features_np = features.detach().cpu().numpy()
                labels_np = labels_dict[cid].detach().cpu().numpy()

                all_features.append(features_np)
                all_client_ids.extend([cid] * len(features_np))
                all_labels.extend(labels_np)

        if not all_features:
            print("No features to visualize")
            return

        # Convert lists to numpy arrays
        all_features = np.vstack(all_features)
        all_client_ids = np.array(all_client_ids)
        all_labels = np.array(all_labels)

        # Compute t-SNE
        embedded_features = self.tsne.fit_transform(all_features)

        # Plot 1: Client-wise visualization
        for cid in range(self.num_clients):
            mask = all_client_ids == cid
            if np.any(mask):
                ax1.scatter(
                    embedded_features[mask, 0],
                    embedded_features[mask, 1],
                    c=[self.client_colors[cid]],
                    label=f'Client {cid}',
                    alpha=0.6,
                    s=50
                )

        ax1.set_title('Features by Client')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Class-wise visualization
        for class_idx in range(self.num_classes):
            for cid in range(self.num_clients):
                mask = (all_labels == class_idx) & (all_client_ids == cid)
                if np.any(mask):
                    ax2.scatter(
                        embedded_features[mask, 0],
                        embedded_features[mask, 1],
                        c=[plt.cm.Set2(class_idx / self.num_classes)],
                        marker=self.class_markers[cid % len(self.class_markers)],
                        label=f'Class {class_idx}, Client {cid}',
                        alpha=0.6,
                        s=50
                    )

        ax2.set_title('Features by Class and Client')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Add overall title
        plt.suptitle(f'Feature Space Visualization - Epoch {epoch}, {stage}\n'
                   f'Client {client_id} Accuracy: {accuracy:.4f}',
                   y=1.02, fontsize=14)

        # Save visualization
        save_path = self._save_visualization(fig, accuracy, client_id, epoch)
        print(f"Saved visualization to: {save_path}")

        # Save feature statistics
        self._save_feature_statistics(
            all_features, all_labels, client_id, epoch, accuracy
        )

        plt.close(fig)
    def _plot_class_wise(self, ax, embedded_features, labels, client_ids):
        """Plot features colored by class."""
        for class_idx in range(self.num_classes):
            for client_idx in range(self.num_clients):
                mask = (labels == class_idx) & (client_ids == client_idx)
                if np.any(mask):
                    ax.scatter(
                        embedded_features[mask, 0],
                        embedded_features[mask, 1],
                        c=[plt.cm.Set2(class_idx / self.num_classes)],
                        marker=self.class_markers[client_idx % len(self.class_markers)],
                        label=f'Class {class_idx}, Client {client_idx}',
                        alpha=0.6,
                        s=50
                    )

        ax.set_title('Features by Class and Client')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')


def extract_features_and_labels(encoder: torch.nn.Module,
                              data_loader: DataLoader,
                              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
   
    features_list = []
    labels_list = []
    encoder.eval()  # Set encoder to evaluation mode

    with torch.no_grad():  # No need to track gradients
        for batch_idx, (images, labels) in enumerate(data_loader):
            # Move images to device and ensure they're float
            images = images.to(device).float()

            # Get features from encoder
            features = encoder(images)

            # Store features and labels
            features_list.append(features.cpu())  # Move features back to CPU
            labels = labels.squeeze() if labels.dim() > 1 else labels  # Handle different label shapes
            labels_list.append(labels)

    if not features_list:  # If no data was processed
        return None, None

    # Concatenate all features and labels
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return features, labels

# This function can be used for just extracting features without labels if needed
def extract_features(encoder: torch.nn.Module,
                    data_loader: DataLoader,
                    device: torch.device) -> torch.Tensor:
   
    features_list = []
    encoder.eval()

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device).float()
            features = encoder(images)
            features_list.append(features.cpu())

    if not features_list:
        return None

    return torch.cat(features_list, dim=0)

