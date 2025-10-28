from pickle import TRUE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import torch
from typing import List, Dict, Optional
import seaborn as sns
from collections import defaultdict
import os
import pandas as pd



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
        
      
        self.history = []  # Store clustering history for evolution plots
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = plt.cm.Set2(np.linspace(0, 1, 10))
    #CLIENTS partiticpants visualization and stragglers figure 4

    def analyze_straggler_detection_with_ground_truth(self, validation_df, 
                                                   ground_truth_stragglers,
                                                   save_path="straggler_validation_gt.png"):
      """
      Comprehensive analysis comparing T_c > T_max against pre-defined stragglers
      """
      import matplotlib.pyplot as plt
      import seaborn as sns
      from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                                 recall_score, f1_score, classification_report)
      import numpy as np
    
      fig = plt.figure(figsize=(20, 12))
      gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
      # ========== 1. CONFUSION MATRIX ==========
      ax1 = fig.add_subplot(gs[0, :2])
    
      y_true = validation_df['ground_truth_straggler'].astype(int)
      y_pred = validation_df['predicted_straggler'].astype(int)
    
      cm = confusion_matrix(y_true, y_pred)
    
      # Create annotated heatmap with percentages
      cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
      annot = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                       for j in range(cm.shape[1])] 
                      for i in range(cm.shape[0])])
    
      sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax1,
                xticklabels=['Fast', 'Straggler'],
                yticklabels=['Fast', 'Straggler'],
                cbar_kws={'label': 'Count'})
    
      ax1.set_xlabel('Predicted by T_c > T_max', fontsize=13, fontweight='bold')
      ax1.set_ylabel('Ground Truth (Pre-defined)', fontsize=13, fontweight='bold')
      ax1.set_title('Straggler Detection Accuracy', fontsize=15, fontweight='bold')
    
      # Calculate metrics
      accuracy = accuracy_score(y_true, y_pred)
      precision = precision_score(y_true, y_pred, zero_division=0)
      recall = recall_score(y_true, y_pred, zero_division=0)
      f1 = f1_score(y_true, y_pred, zero_division=0)
      specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    
      # Add metrics box
      metrics_text = f'Overall Metrics:\n'
      metrics_text += f'━━━━━━━━━━━━━━━━\n'
      metrics_text += f'Accuracy:    {accuracy:.3f}\n'
      metrics_text += f'Precision:   {precision:.3f}\n'
      metrics_text += f'Recall:      {recall:.3f}\n'
      metrics_text += f'Specificity: {specificity:.3f}\n'
      metrics_text += f'F1-Score:    {f1:.3f}'
    
      ax1.text(1.2, 0.5, metrics_text, transform=ax1.transAxes,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6),
             fontsize=11, fontweight='bold', family='monospace')
    
      # ========== 2. DETAILED METRICS TABLE ==========
      ax2 = fig.add_subplot(gs[0, 2:])
      ax2.axis('off')
    
      # Get classification report
      report = classification_report(y_true, y_pred, 
                                   target_names=['Fast Client', 'Straggler'],
                                   output_dict=True)
    
      table_text = "Classification Report\n"
      table_text += "="*50 + "\n\n"
      table_text += f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n"
      table_text += "-"*50 + "\n"
    
      for label in ['Fast Client', 'Straggler']:
        if label in report:
            table_text += f"{label:<15} "
            table_text += f"{report[label]['precision']:.3f}        "
            table_text += f"{report[label]['recall']:.3f}        "
            table_text += f"{report[label]['f1-score']:.3f}\n"
    
      table_text += "-"*50 + "\n"
      table_text += f"{'Accuracy':<15} {accuracy:.3f}\n"
      table_text += f"{'Macro Avg':<15} {report['macro avg']['f1-score']:.3f}\n"
      table_text += f"{'Weighted Avg':<15} {report['weighted avg']['f1-score']:.3f}\n"
    
      ax2.text(0.05, 0.95, table_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4),
             family='monospace')
    
      # ========== 3. T_c vs T_max SCATTER (Colored by Ground Truth) ==========
      ax3 = fig.add_subplot(gs[1, :2])
    
      # Separate by ground truth
      fast_clients = validation_df[~validation_df['ground_truth_straggler']]
      straggler_clients = validation_df[validation_df['ground_truth_straggler']]
    
      # Plot fast clients
      ax3.scatter(fast_clients['T_c'], fast_clients['actual_duration'],
                c='green', alpha=0.5, s=50, label='Ground Truth: Fast',
                edgecolors='black', linewidth=0.5)
    
      # Plot stragglers
      ax3.scatter(straggler_clients['T_c'], straggler_clients['actual_duration'],
                c='red', alpha=0.5, s=50, label='Ground Truth: Straggler',
                edgecolors='black', linewidth=0.5, marker='s')
    
      # Add T_max line
      T_max_mean = validation_df['T_max'].mean()
      ax3.axvline(T_max_mean, color='blue', linestyle='--', linewidth=2,
                label=f'T_max = {T_max_mean:.2f}s', alpha=0.7)
      ax3.axhline(T_max_mean, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    
      # Shade regions
      xlim = ax3.get_xlim()
      ylim = ax3.get_ylim()
      ax3.fill_between([T_max_mean, xlim[1]], ylim[0], ylim[1], 
                     alpha=0.1, color='red', label='Predicted Straggler Region')
    
      ax3.set_xlabel('T_c (EMA Training Time)', fontsize=12, fontweight='bold')
      ax3.set_ylabel('Actual Duration (Current Round)', fontsize=12, fontweight='bold')
      ax3.set_title('EMA vs Actual Duration by Ground Truth Label', fontsize=14, fontweight='bold')
      ax3.legend(loc='upper left', fontsize=9)
      ax3.grid(True, alpha=0.3)
    
      # ========== 4. ACCURACY OVER ROUNDS ==========
      ax4 = fig.add_subplot(gs[1, 2:])
    
      # Calculate per-round accuracy
      round_metrics = validation_df.groupby('round').apply(
        lambda x: pd.Series({
            'accuracy': accuracy_score(x['ground_truth_straggler'], x['predicted_straggler']),
            'precision': precision_score(x['ground_truth_straggler'], x['predicted_straggler'], zero_division=0),
            'recall': recall_score(x['ground_truth_straggler'], x['predicted_straggler'], zero_division=0)
        })
    ).reset_index()
    
      rounds = round_metrics['round']
      ax4.plot(rounds, round_metrics['accuracy'], marker='o', linewidth=2, 
             label='Accuracy', color='steelblue')
      ax4.plot(rounds, round_metrics['precision'], marker='s', linewidth=2,
             label='Precision', color='green', alpha=0.7)
      ax4.plot(rounds, round_metrics['recall'], marker='^', linewidth=2,
             label='Recall', color='orange', alpha=0.7)
    
      ax4.axhline(y=accuracy, color='red', linestyle='--', linewidth=2,
                label=f'Overall Accuracy: {accuracy:.3f}', alpha=0.5)
    
      ax4.set_xlabel('Training Round', fontsize=12, fontweight='bold')
      ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
      ax4.set_title('Detection Metrics Over Training', fontsize=14, fontweight='bold')
      ax4.legend(loc='lower right', fontsize=9)
      ax4.grid(True, alpha=0.3)
      ax4.set_ylim([0, 1.05])
    
      # ========== 5. ERROR ANALYSIS: MISCLASSIFICATIONS ==========
      ax5 = fig.add_subplot(gs[2, :2])
    
      false_positives = validation_df[
        validation_df['predicted_straggler'] & ~validation_df['ground_truth_straggler']
    ]
      false_negatives = validation_df[
        ~validation_df['predicted_straggler'] & validation_df['ground_truth_straggler']
    ]
    
      error_data = {
        'False Positives\n(Fast labeled as Straggler)': len(false_positives),
        'False Negatives\n(Straggler labeled as Fast)': len(false_negatives),
        'True Positives\n(Correct Straggler)': len(validation_df[
            validation_df['predicted_straggler'] & validation_df['ground_truth_straggler']
        ]),
        'True Negatives\n(Correct Fast)': len(validation_df[
            ~validation_df['predicted_straggler'] & ~validation_df['ground_truth_straggler']
        ])
    }
    
      colors = ['red', 'orange', 'green', 'lightgreen']
      bars = ax5.bar(error_data.keys(), error_data.values(), color=colors, 
                   edgecolor='black', alpha=0.7)
    
      # Add count labels on bars
      for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
      ax5.set_ylabel('Count', fontsize=12, fontweight='bold')
      ax5.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
      ax5.grid(axis='y', alpha=0.3)
    
      # ========== 6. CLIENT-LEVEL ACCURACY ==========
      ax6 = fig.add_subplot(gs[2, 2:])
    
      # Calculate per-client accuracy
      client_accuracy = validation_df.groupby('client_id').apply(
        lambda x: (x['predicted_straggler'] == x['ground_truth_straggler']).mean()
    ).sort_values()
    
      # Separate stragglers and fast clients
      straggler_clients_acc = client_accuracy[client_accuracy.index.isin(ground_truth_stragglers)]
      fast_clients_acc = client_accuracy[~client_accuracy.index.isin(ground_truth_stragglers)]
    
      # Plot
      x_pos_stragglers = np.arange(len(straggler_clients_acc))
      x_pos_fast = np.arange(len(straggler_clients_acc), 
                           len(straggler_clients_acc) + len(fast_clients_acc))
    
      ax6.bar(x_pos_stragglers, straggler_clients_acc.values, 
            color='red', alpha=0.6, label='Stragglers', edgecolor='black')
      ax6.bar(x_pos_fast, fast_clients_acc.values,
            color='green', alpha=0.6, label='Fast Clients', edgecolor='black')
    
      ax6.axhline(y=1.0, color='blue', linestyle='--', linewidth=1, alpha=0.5)
      ax6.set_xlabel('Client ID (sorted by accuracy)', fontsize=12, fontweight='bold')
      ax6.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
      ax6.set_title('Per-Client Detection Accuracy', fontsize=14, fontweight='bold')
      ax6.legend(loc='lower right')
      ax6.grid(axis='y', alpha=0.3)
      ax6.set_ylim([0, 1.05])
    
      plt.savefig(save_path, dpi=300, bbox_inches='tight')
      print(f"Validation analysis saved to {save_path}")
      plt.show()
    
      # ========== PRINT STATISTICAL SUMMARY ==========
      print("\n" + "="*70)
      print("STRAGGLER DETECTION VALIDATION SUMMARY")
      print("="*70)
      print(f"Total observations: {len(validation_df)}")
      print(f"Ground truth stragglers: {len(ground_truth_stragglers)} clients")
      print(f"Ground truth fast clients: {len(validation_df['client_id'].unique()) - len(ground_truth_stragglers)} clients")
      print("\n" + "-"*70)
      print("OVERALL PERFORMANCE:")
      print("-"*70)
      print(f"  Accuracy:    {accuracy:.3f} ({accuracy*100:.1f}%)")
      print(f"  Precision:   {precision:.3f} ({precision*100:.1f}%)")
      print(f"  Recall:      {recall:.3f} ({recall*100:.1f}%)")
      print(f"  Specificity: {specificity:.3f} ({specificity*100:.1f}%)")
      print(f"  F1-Score:    {f1:.3f}")
      print("\n" + "-"*70)
      print("ERROR ANALYSIS:")
      print("-"*70)
      print(f"  False Positives: {len(false_positives)} (Fast clients wrongly labeled as stragglers)")
      print(f"  False Negatives: {len(false_negatives)} (Stragglers wrongly labeled as fast)")
      print("\n" + "-"*70)
      print("INTERPRETATION:")
      print("-"*70)
    
      if accuracy >= 0.85:
        print("  ✓ EXCELLENT: T_c > T_max criterion is highly reliable")
      elif accuracy >= 0.75:
        print("  ✓ GOOD: T_c > T_max criterion is reasonably reliable")
      elif accuracy >= 0.65:
        print("  ⚠ MODERATE: T_c > T_max criterion shows moderate reliability")
      else:
        print("  ✗ POOR: T_c > T_max criterion needs improvement")
    
      if precision >= 0.80:
        print("  ✓ Low false positive rate: Fast clients rarely mislabeled")
      else:
        print("  ⚠ Significant false positives: Some fast clients mislabeled as stragglers")
    
      if recall >= 0.70:
        print("  ✓ High detection rate: Most stragglers correctly identified")
      else:
        print("  ⚠ Missing stragglers: Some stragglers not detected")
    
      print("="*70 + "\n")
    
      return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'confusion_matrix': cm
    }

    


    def visualize_clustering_from_prototypes(
        self,
        all_prototypes_list: List[Dict],
        client_ids: List[str],
        true_domain_labels ,
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
        true_domains = true_domain_labels
        print(f"true domains of clients {true_domains}")
       
        
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
    """
    def _get_true_domain_for_client(self, client_id: str) -> int:
        
        try:
            # Extract numeric ID from client_id string
            client_idx = int(client_id.split('_')[-1])
            if client_idx < len(self.true_domain_labels):
                return self.true_domain_labels[client_idx]
        except:
            pass
        return -1  # Unknown domain
    """
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
    '''
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
    '''
    def _find_cluster_alignment(self, predicted, true):
      """Find best alignment between clusters and true domains (robust version)."""
      from scipy.optimize import linear_sum_assignment
      import numpy as np

      predicted = np.array(predicted)
      true = np.array(true)

      # Ensure non-negative valid labels
      valid_mask = (predicted >= 0) & (true >= 0)
      predicted = predicted[valid_mask]
      true = true[valid_mask]

      # Map cluster/domain labels to compact 0..N indices
      unique_pred = np.unique(predicted)
      unique_true = np.unique(true)

      cluster_to_idx = {c: i for i, c in enumerate(unique_pred)}
      domain_to_idx = {d: i for i, d in enumerate(unique_true)}

      n_clusters = len(unique_pred)
      n_domains = len(unique_true)

      # Initialize confusion matrix safely
      confusion = np.zeros((n_clusters, n_domains), dtype=int)

      for pred, true_label in zip(predicted, true):
        i = cluster_to_idx[pred]
        j = domain_to_idx[true_label]
        confusion[i, j] += 1

      # Handle degenerate cases
      if confusion.size == 0:
        print("[WARN] Empty confusion matrix in _find_cluster_alignment.")
        return {}

      # Hungarian algorithm for optimal assignment
      row_ind, col_ind = linear_sum_assignment(-confusion)

      alignment = {int(unique_pred[row]): int(unique_true[col]) for row, col in zip(row_ind, col_ind)}

      # Debug info
      print(f"[DEBUG] Confusion matrix:\n{confusion}")
      print(f"[DEBUG] Alignment mapping: {alignment}")

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

