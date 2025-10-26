import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Set
import json

class ParticipationTracker:
    """
    Tracks client participation metrics for federated learning experiments.
    Compares participation patterns between different selection strategies.
    """
    
    def __init__(self, total_clients: int, total_rounds: int, method_name: str = "FedProto-Fair"):
        self.total_clients = total_clients
        self.total_rounds = total_rounds
        self.method_name = method_name
        
        # Core tracking metrics
        self.participation_history = defaultdict(list)  # client_id -> [round_numbers]
        self.round_participants = {}  # round -> [client_ids]
        self.selection_counts = defaultdict(int)  # client_id -> total_selections
        self.consecutive_absences = defaultdict(int)  # client_id -> consecutive rounds not selected
        self.straggler_flags = defaultdict(list)  # client_id -> [(round, duration)]
        
        # Fairness metrics
        self.gini_coefficients = []  # Gini coefficient per round
        self.jain_fairness_indices = []  # Jain's fairness index per round
        self.coverage_ratios = []  # % of clients participated per round
        
        # Straggler-specific metrics
        self.straggler_participation_rate = []  # % stragglers selected per round
        self.non_straggler_participation_rate = []
        
        # Comparative metrics
        self.participation_variance = []  # Variance in selection counts
        self.participation_entropy = []  # Shannon entropy of selections
        
    def update_round(self, round_num: int, selected_clients: List[str], 
                     training_durations: Dict[str, float] = None,
                     straggler_threshold: float = None):
        """
        Update participation metrics after each round.
        
        Args:
            round_num: Current round number
            selected_clients: List of client IDs selected this round
            training_durations: Dict mapping client_id -> training_time
            straggler_threshold: Time threshold to classify stragglers (e.g., mean + std)
        """
        # Update participation history
        self.round_participants[round_num] = selected_clients
        
        for client_id in selected_clients:
            self.participation_history[client_id].append(round_num)
            self.selection_counts[client_id] += 1
            self.consecutive_absences[client_id] = 0  # Reset absence counter
        
        # Track consecutive absences for non-selected clients
        all_client_ids = [f"client_{i}" for i in range(self.total_clients)]
        for client_id in all_client_ids:
            if client_id not in selected_clients:
                self.consecutive_absences[client_id] += 1
        
        # Identify stragglers if durations provided
        if training_durations and straggler_threshold:
            for client_id, duration in training_durations.items():
                if duration > straggler_threshold:
                    self.straggler_flags[client_id].append((round_num, duration))
        
        # Compute round-level fairness metrics
        self._compute_round_fairness_metrics(round_num)
        
    def _compute_round_fairness_metrics(self, round_num: int):
        """Compute fairness metrics up to current round."""
        
        # Get selection counts up to this round
        counts = np.array([self.selection_counts.get(f"client_{i}", 0) 
                          for i in range(self.total_clients)])
        
        # Coverage ratio: what % of clients have participated so far
        coverage = np.sum(counts > 0) / self.total_clients
        self.coverage_ratios.append(coverage)
        
        # Gini coefficient (inequality measure)
        if np.sum(counts) > 0:
            gini = self._calculate_gini(counts)
            self.gini_coefficients.append(gini)
        else:
            self.gini_coefficients.append(0.0)
        
        # Jain's Fairness Index
        jain_index = self._calculate_jain_fairness(counts)
        self.jain_fairness_indices.append(jain_index)
        
        # Participation variance
        self.participation_variance.append(np.var(counts))
        
        # Shannon entropy (diversity of participation)
        if np.sum(counts) > 0:
            probs = counts / np.sum(counts)
            entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0] + 1e-10))
            self.participation_entropy.append(entropy)
        else:
            self.participation_entropy.append(0.0)
    
    def _calculate_gini(self, counts: np.ndarray) -> float:
        """
        Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality).
        """
        sorted_counts = np.sort(counts)
        n = len(counts)
        cumsum = np.cumsum(sorted_counts)
        
        if cumsum[-1] == 0:
            return 0.0
        
        return (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
    
    def _calculate_jain_fairness(self, counts: np.ndarray) -> float:
        """
        Calculate Jain's Fairness Index (1 = perfect fairness, 1/n = worst case).
        """
        n = len(counts)
        if n == 0 or np.sum(counts) == 0:
            return 0.0
        
        sum_counts = np.sum(counts)
        sum_squares = np.sum(counts ** 2)
        
        return (sum_counts ** 2) / (n * sum_squares) if sum_squares > 0 else 0.0
    
    def track_straggler_participation(self, straggler_ids: Set[str], round_num: int):
        """
        Track participation rate of known stragglers vs non-stragglers.
        
        Args:
            straggler_ids: Set of client IDs identified as stragglers
            round_num: Current round
        """
        selected = self.round_participants.get(round_num, [])
        
        straggler_selected = sum(1 for cid in selected if cid in straggler_ids)
        non_straggler_selected = len(selected) - straggler_selected
        
        total_stragglers = len(straggler_ids)
        total_non_stragglers = self.total_clients - total_stragglers
        
        straggler_rate = straggler_selected / total_stragglers if total_stragglers > 0 else 0
        non_straggler_rate = non_straggler_selected / total_non_stragglers if total_non_stragglers > 0 else 0
        
        self.straggler_participation_rate.append(straggler_rate)
        self.non_straggler_participation_rate.append(non_straggler_rate)
    
    def get_summary_statistics(self) -> Dict:
        """
        Generate comprehensive summary statistics.
        """
        counts = np.array([self.selection_counts.get(f"client_{i}", 0) 
                          for i in range(self.total_clients)])
        
        participated_clients = np.sum(counts > 0)
        never_participated = np.sum(counts == 0)
        
        return {
            "method": self.method_name,
            "total_rounds": self.total_rounds,
            "total_clients": self.total_clients,
            
            # Participation coverage
            "clients_participated": int(participated_clients),
            "clients_never_participated": int(never_participated),
            "coverage_percentage": float(participated_clients / self.total_clients * 100),
            
            # Selection distribution
            "mean_selections": float(np.mean(counts)),
            "std_selections": float(np.std(counts)),
            "min_selections": int(np.min(counts)),
            "max_selections": int(np.max(counts)),
            "median_selections": float(np.median(counts)),
            
            # Fairness metrics (final round)
            "final_gini_coefficient": float(self.gini_coefficients[-1]) if self.gini_coefficients else 0.0,
            "final_jain_fairness_index": float(self.jain_fairness_indices[-1]) if self.jain_fairness_indices else 0.0,
            "avg_gini_coefficient": float(np.mean(self.gini_coefficients)) if self.gini_coefficients else 0.0,
            "avg_jain_fairness_index": float(np.mean(self.jain_fairness_indices)) if self.jain_fairness_indices else 0.0,
            
            # Participation patterns
            "max_consecutive_absences": int(max(self.consecutive_absences.values())) if self.consecutive_absences else 0,
            "avg_participation_entropy": float(np.mean(self.participation_entropy)) if self.participation_entropy else 0.0,
            
            # Straggler metrics
            "identified_stragglers": len([cid for cid, flags in self.straggler_flags.items() if len(flags) > 0]),
        }
    
    def plot_participation_comparison(self, other_tracker: 'ParticipationTracker', 
                                     save_path: str = None):
        """
        Create comprehensive comparison plots between two methods.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Participation Comparison: {self.method_name} vs {other_tracker.method_name}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Selection distribution histogram
        ax = axes[0, 0]
        counts_self = [self.selection_counts.get(f"client_{i}", 0) for i in range(self.total_clients)]
        counts_other = [other_tracker.selection_counts.get(f"client_{i}", 0) for i in range(other_tracker.total_clients)]
        
        ax.hist([counts_self, counts_other], bins=20, alpha=0.7, label=[self.method_name, other_tracker.method_name])
        ax.set_xlabel('Number of Selections', fontsize=12)
        ax.set_ylabel('Number of Clients', fontsize=12)
        ax.set_title('Selection Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Gini coefficient over rounds
        ax = axes[0, 1]
        ax.plot(self.gini_coefficients, label=self.method_name, linewidth=2)
        ax.plot(other_tracker.gini_coefficients, label=other_tracker.method_name, linewidth=2)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Gini Coefficient', fontsize=12)
        ax.set_title('Fairness: Gini Coefficient (lower = better)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Jain's Fairness Index over rounds
        ax = axes[0, 2]
        ax.plot(self.jain_fairness_indices, label=self.method_name, linewidth=2)
        ax.plot(other_tracker.jain_fairness_indices, label=other_tracker.method_name, linewidth=2)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel("Jain's Fairness Index", fontsize=12)
        ax.set_title("Jain's Fairness Index (higher = better)", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Coverage ratio over rounds
        ax = axes[1, 0]
        ax.plot(self.coverage_ratios, label=self.method_name, linewidth=2)
        ax.plot(other_tracker.coverage_ratios, label=other_tracker.method_name, linewidth=2)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Coverage Ratio', fontsize=12)
        ax.set_title('Client Coverage Over Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Participation entropy
        ax = axes[1, 1]
        ax.plot(self.participation_entropy, label=self.method_name, linewidth=2)
        ax.plot(other_tracker.participation_entropy, label=other_tracker.method_name, linewidth=2)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Participation Entropy', fontsize=12)
        ax.set_title('Participation Diversity (higher = better)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Straggler participation rate
        ax = axes[1, 2]
        if self.straggler_participation_rate and other_tracker.straggler_participation_rate:
            ax.plot(self.straggler_participation_rate, label=f'{self.method_name} (Stragglers)', linewidth=2)
            ax.plot(other_tracker.straggler_participation_rate, label=f'{other_tracker.method_name} (Stragglers)', linewidth=2, linestyle='--')
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Participation Rate', fontsize=12)
            ax.set_title('Straggler Participation Rate', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No straggler data', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def export_results(self, filename: str = "participation_results.json"):
        """Export results to JSON file."""
        results = {
            "summary": self.get_summary_statistics(),
            "selection_counts": dict(self.selection_counts),
            "participation_history": {k: v for k, v in self.participation_history.items()},
            "metrics": {
                "gini_coefficients": self.gini_coefficients,
                "jain_fairness_indices": self.jain_fairness_indices,
                "coverage_ratios": self.coverage_ratios,
                "participation_variance": self.participation_variance,
                "participation_entropy": self.participation_entropy,
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {filename}")