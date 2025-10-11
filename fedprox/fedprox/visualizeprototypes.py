import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import torch
from typing import List, Dict, Optional
import seaborn as sns
from collections import defaultdict
import os

class ClusterVisualizationForConfigureFit:
    """
    t-SNE visualization for EM clustering in your configure_fit function.
    Integrated directly with your prototype-based clustering approach.
    """
    
    def __init__(self, save_dir: str = "./clustering_visualizations", true_domain_labels: Optional[np.ndarray] = None):
        """
        Args:
            save_dir: Directory to save visualizations
            true_domain_labels: Ground truth domain labels for each client (optional)
                                e.g., np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,3])
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.true_domain_labels = true_domain_labels
        self.history = []  # Store clustering history for evolution plots
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = plt.cm.Set2(np.linspace(0, 1, 10))
    
    def visualize_clustering_from_prototypes(
        self,
        all_prototypes_list: List[Dict],
        client_ids: List[str],
        client_assignments: Dict[str, int],
        server_round: int,
        num_clusters: int,
        perplexity: int = 30,
        save: bool = True
    ):
    
        """
        Visualize clustering results using t-SNE on client prototypes.
        
        This is the MAIN function to call from your configure_fit method.
        
        Args:
            all_prototypes_list: List of prototype dicts from clients
                                 e.g., [{class_0: array, class_1: array, ...}, ...]
            client_ids: List of client IDs corresponding to prototypes
            client_assignments: Dict mapping client_id -> cluster_id from EM
            server_round: Current server round number
            num_clusters: Number of clusters
            perplexity: t-SNE perplexity parameter
            save: Whether to save the figure
        
        Returns:
            fig: Matplotlib figure
            tsne_embedded: 2D t-SNE embeddings
        """
        
        print(f"\n{'='*80}")
        print(f"[t-SNE Visualization] Round {server_round}")
        print(f"{'='*80}")
        
        # Step 1: Convert prototypes to feature vectors
        print(f"  Converting {len(all_prototypes_list)} client prototypes to feature vectors...")
        client_features = self._prototypes_to_feature_vectors(all_prototypes_list)
        
        if client_features is None or len(client_features) == 0:
            print("  ⚠ No valid features extracted. Skipping visualization.")
            return None, None
        
        print(f"  Feature matrix shape: {client_features.shape}")
        
        # Step 2: Extract cluster assignments in order
        predicted_clusters = np.array([
            client_assignments.get(cid, 0) for cid in client_ids
        ])
        
        print(f"  Cluster distribution:")
        unique, counts = np.unique(predicted_clusters, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"    Cluster {cluster_id}: {count} clients")
        
        # Step 3: Run t-SNE
        print(f"  Running t-SNE (perplexity={perplexity})...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(client_features) - 1),
            n_iter=1000,
            random_state=42,
            verbose=0
        )
        tsne_embedded = tsne.fit_transform(client_features)
        print(f"  t-SNE completed.")
        
        # Step 4: Get true domain labels for these clients (if available)
        true_domains = None
        if self.true_domain_labels is not None:
            # Convert client IDs to indices
            true_domains = np.array([
                self._get_true_domain_for_client(cid) 
                for cid in client_ids
            ])
        
        # Step 5: Create visualization
        if true_domains is not None:
            fig = self._create_three_panel_plot(
                tsne_embedded,
                predicted_clusters,
                true_domains,
                client_ids,
                client_features,
                server_round
            )
        else:
            fig = self._create_single_panel_plot(
                tsne_embedded,
                predicted_clusters,
                client_ids,
                client_features,
                server_round
            )
        
        # Step 6: Save and store history
        if save:
            save_path = f"{self.save_dir}/em_clustering_round_{server_round}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved visualization to: {save_path}")
        
        # Store in history
        self.history.append({
            'round': server_round,
            'tsne_embedded': tsne_embedded,
            'predicted_clusters': predicted_clusters,
            'true_domains': true_domains,
            'client_ids': client_ids,
            'client_features': client_features
        })
        
        print(f"{'='*80}\n")
        
        return fig, tsne_embedded
    
    def _prototypes_to_feature_vectors(self, prototypes_list: List[Dict]) -> np.ndarray:
        """
        Convert list of prototype dictionaries to a feature matrix.
        Each client's prototypes are concatenated into a single feature vector.
        """
        feature_vectors = []
        
        for prototypes in prototypes_list:
            if not prototypes:
                continue
            
            # Concatenate all class prototypes for this client
            client_features = []
            for class_id in sorted(prototypes.keys()):
                proto = prototypes[class_id]
                
                # Convert to numpy
                if hasattr(proto, 'numpy'):
                    proto_np = proto.numpy()
                elif hasattr(proto, 'detach'):
                    proto_np = proto.detach().cpu().numpy()
                else:
                    proto_np = np.array(proto)
                
                client_features.append(proto_np.flatten())
            
            if client_features:
                # Concatenate all class prototypes into one vector
                feature_vectors.append(np.concatenate(client_features))
        
        if not feature_vectors:
            return None
        
        # Stack into matrix
        return np.array(feature_vectors)
    
    def _get_true_domain_for_client(self, client_id: str) -> int:
        """
        Extract true domain label for a client ID.
        Assumes client_id format like 'client_0', 'client_1', etc.
        """
        try:
            # Extract numeric ID from client_id string
            client_idx = int(client_id.split('_')[-1])
            if client_idx < len(self.true_domain_labels):
                return self.true_domain_labels[client_idx]
        except:
            pass
        return -1  # Unknown domain
    
    def _create_three_panel_plot(
        self,
        embeddings,
        predicted,
        true_domains,
        client_ids,
        client_features,
        server_round
    ):
        """Create three-panel visualization: Predicted | True | Quality"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Panel 1: Predicted Clusters
        self._plot_clusters(
            embeddings, predicted, client_ids, axes[0],
            f"EM Predicted Clusters\n(Round {server_round})",
            "Predicted Cluster"
        )
        
        # Panel 2: True Domains
        self._plot_clusters(
            embeddings, true_domains, client_ids, axes[1],
            f"True Domain Labels\n(Ground Truth)",
            "True Domain"
        )
        
        # Panel 3: Quality Assessment
        self._plot_quality_assessment(
            embeddings, predicted, true_domains, client_ids, axes[2],
            f"Clustering Quality\n(Round {server_round})"
        )
        
        # Compute and display metrics
        ari = adjusted_rand_score(true_domains, predicted)
        nmi = normalized_mutual_info_score(true_domains, predicted)
        silhouette = silhouette_score(client_features, predicted)
        
        fig.suptitle(
            f"EM Clustering Visualization - Round {server_round}\n"
            f"ARI: {ari:.3f} | NMI: {nmi:.3f} | Silhouette: {silhouette:.3f}",
            fontsize=14, fontweight='bold', y=1.02
        )
        
        print(f"\n  [Clustering Quality Metrics]")
        print(f"    Adjusted Rand Index (ARI): {ari:.3f} (1.0 = perfect)")
        print(f"    Normalized Mutual Info (NMI): {nmi:.3f} (1.0 = perfect)")
        print(f"    Silhouette Score: {silhouette:.3f} (1.0 = best)")
        
        plt.tight_layout()
        return fig
    
    def _create_single_panel_plot(
        self,
        embeddings,
        predicted,
        client_ids,
        client_features,
        server_round
    ):
        """Create single-panel visualization when true labels unavailable"""
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        self._plot_clusters(
            embeddings, predicted, client_ids, ax,
            f"EM Predicted Clusters - Round {server_round}",
            "Cluster"
        )
        
        silhouette = silhouette_score(client_features, predicted)
        
        fig.suptitle(
            f"EM Clustering - Round {server_round} | Silhouette: {silhouette:.3f}",
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        return fig
    
    def _plot_clusters(self, embeddings, labels, client_ids, ax, title, label_name):
        """Plot t-SNE embeddings colored by cluster/domain"""
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip unknown labels
                continue
                
            mask = labels == label
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[self.colors[int(label) % len(self.colors)]],
                label=f"{label_name} {label}",
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5
            )
            
            # Add client ID annotations
            for idx in np.where(mask)[0]:
                # Extract just the number from client_id
                client_num = client_ids[idx].split('_')[-1] if '_' in client_ids[idx] else client_ids[idx]
                ax.annotate(
                    client_num,
                    (embeddings[idx, 0], embeddings[idx, 1]),
                    fontsize=9,
                    ha='center',
                    va='center',
                    fontweight='bold'
                )
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    def _plot_quality_assessment(self, embeddings, predicted, true_domains, client_ids, ax, title):
        """Plot correctness of cluster assignments"""
        
        # Find best alignment between clusters and domains
        alignment = self._find_cluster_alignment(predicted, true_domains)
        
        correct_mask = np.array([
            alignment.get(pred, -1) == true_label
            for pred, true_label in zip(predicted, true_domains)
        ])
        
        # Plot correct assignments
        if correct_mask.sum() > 0:
            ax.scatter(
                embeddings[correct_mask, 0],
                embeddings[correct_mask, 1],
                c='green',
                marker='o',
                s=180,
                alpha=0.6,
                label=f'Correct ({correct_mask.sum()})',
                edgecolors='darkgreen',
                linewidth=2
            )
        
        # Plot incorrect assignments
        if (~correct_mask).sum() > 0:
            ax.scatter(
                embeddings[~correct_mask, 0],
                embeddings[~correct_mask, 1],
                c='red',
                marker='X',
                s=180,
                alpha=0.6,
                label=f'Incorrect ({(~correct_mask).sum()})',
                edgecolors='darkred',
                linewidth=2
            )
        
        # Add annotations
        for idx in range(len(embeddings)):
            color = 'darkgreen' if correct_mask[idx] else 'darkred'
            client_num = client_ids[idx].split('_')[-1] if '_' in client_ids[idx] else client_ids[idx]
            ax.annotate(
                client_num,
                (embeddings[idx, 0], embeddings[idx, 1]),
                fontsize=9,
                ha='center',
                va='center',
                fontweight='bold',
                color=color
            )
        
        accuracy = correct_mask.sum() / len(correct_mask) if len(correct_mask) > 0 else 0
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(f"{title}\nAccuracy: {accuracy:.1%}", fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    def _find_cluster_alignment(self, predicted, true):
        """Find best alignment between clusters and true domains"""
        from scipy.optimize import linear_sum_assignment
        
        n_clusters = len(np.unique(predicted[predicted >= 0]))
        n_domains = len(np.unique(true[true >= 0]))
        
        # Confusion matrix
        confusion = np.zeros((n_clusters, n_domains))
        for pred, true_label in zip(predicted, true):
            if pred >= 0 and true_label >= 0:
                confusion[int(pred), int(true_label)] += 1
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-confusion)
        
        alignment = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
        return alignment
    
    def plot_clustering_statistics(
        self,
        predicted_clusters: np.ndarray,
        true_domains: np.ndarray,
        client_ids: List[str],
        server_round: int,
        save: bool = True
    ):
        """
        Plot detailed clustering statistics (confusion matrix, purity, etc.)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        
        # Remove invalid labels
        valid_mask = (predicted_clusters >= 0) & (true_domains >= 0)
        pred_valid = predicted_clusters[valid_mask]
        true_valid = true_domains[valid_mask]
        
        # 1. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_valid, pred_valid)
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=[f'C{i}' for i in range(cm.shape[1])],
            yticklabels=[f'D{i}' for i in range(cm.shape[0])],
            cbar_kws={'label': 'Count'}
        )
        axes[0, 0].set_title('Confusion Matrix\n(True Domain vs Predicted Cluster)', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('True Domain', fontsize=11)
        axes[0, 0].set_xlabel('Predicted Cluster', fontsize=11)
        
        # 2. Cluster Size Distribution
        unique_clusters, counts = np.unique(pred_valid, return_counts=True)
        axes[0, 1].bar(unique_clusters, counts, color=self.colors[:len(unique_clusters)], 
                      edgecolor='black', linewidth=1.5)
        axes[0, 1].set_xlabel('Cluster ID', fontsize=11)
        axes[0, 1].set_ylabel('Number of Clients', fontsize=11)
        axes[0, 1].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for cluster, count in zip(unique_clusters, counts):
            axes[0, 1].text(cluster, count + 0.1, str(count), 
                          ha='center', va='bottom', fontweight='bold')
        
        # 3. Cluster Purity
        purities = []
        for cluster_id in unique_clusters:
            mask = pred_valid == cluster_id
            domains = true_valid[mask]
            unique, counts = np.unique(domains, return_counts=True)
            purity = counts.max() / counts.sum() if counts.sum() > 0 else 0
            purities.append(purity)
        
        axes[1, 0].bar(unique_clusters, purities, color=self.colors[:len(unique_clusters)],
                      edgecolor='black', linewidth=1.5)
        axes[1, 0].set_xlabel('Cluster ID', fontsize=11)
        axes[1, 0].set_ylabel('Purity', fontsize=11)
        axes[1, 0].set_ylim([0, 1.05])
        axes[1, 0].axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        axes[1, 0].set_title('Cluster Purity (1.0 = perfect)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for cluster, purity in zip(unique_clusters, purities):
            axes[1, 0].text(cluster, purity + 0.02, f'{purity:.2f}',
                          ha='center', va='bottom', fontweight='bold')
        
        # 4. Metrics Summary
        ari = adjusted_rand_score(true_valid, pred_valid)
        nmi = normalized_mutual_info_score(true_valid, pred_valid)
        avg_purity = np.mean(purities)
        
        metrics_text = f"""
Clustering Quality Metrics

Adjusted Rand Index:        {ari:.4f}
  (1.0 = perfect, 0 = random)

Normalized Mutual Info:     {nmi:.4f}
  (1.0 = perfect match)

Average Cluster Purity:     {avg_purity:.4f}
  (1.0 = each cluster pure)

Number of Clusters:         {len(unique_clusters)}
Number of True Domains:     {len(np.unique(true_valid))}
Total Clients:              {len(valid_mask)}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11,
                       verticalalignment='center',
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Clustering Statistics - Round {server_round}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = f"{self.save_dir}/statistics_round_{server_round}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved statistics to: {save_path}")
        
        return fig

