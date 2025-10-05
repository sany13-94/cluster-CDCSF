from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
#from MulticoreTSNE import print_function
import flwr
import mlflow
from torch.cuda.amp import autocast, GradScaler
import base64
import pickle
import datetime
from numpy.linalg import norm
from matplotlib import cm
from matplotlib.colors import ListedColormap
from torch.distributions import Dirichlet, Categorical
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from matplotlib import cm
import random
from flwr.common import GetPropertiesIns
import json
from sklearn.manifold import TSNE
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from typing import List, Tuple, Optional, Dict, Callable, Union
from flwr.common.typing import NDArrays, Scalar
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import json
from flwr.server.strategy import Strategy,FedAvg
from fedprox.models import test,test_gpaf 
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from fedprox.features_visualization import extract_features_and_labels,StructuredFeatureVisualizer
import csv
import requests
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
import os


from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
     GetPropertiesIns, GetPropertiesRes
)

class GPAFStrategy(FedAvg):
    def __init__(
        self,
       experiment_name,
        num_classes: int=9,
        fraction_fit: float = 1.0,
        fraction_evaluate=1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients=2,
        min_available_clients=2,
        batch_size=13,
        
   evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
  
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.server_url = "https://add18b7094f7.ngrok-free.app/heartbeat"

        #clusters parameters

        self.num_clusters = 4
        self.client_assignments = {}  # {client_id: cluster_id}
        
        # Initialize as empty dictionaries
        self.cluster_prototypes = {i: {} for i in range(self.num_clusters)}
        self.cluster_class_counts = {i: defaultdict(int) for i in range(self.num_clusters)}
        
        
        # CSMDA Client Selection Parameters (UPDATED)
        self.training_times = defaultdict(float)
        self.selection_counts = defaultdict(int)
        self.accuracy_history = defaultdict(float)
        self._current_accuracies = {}
        # ... existing initialization ...
        # Core parameters from corrected methodology
       
      
        self.clustering_interval = 10  # Cluster every 10 rounds
        self.virtual_cluster_id = 999
        
        # Tracking
        self.training_times = {}
        self.selection_counts = {}
        self.participated_clients = set()
        self.client_assignments = {}
        self.cluster_prototypes = {}
        self.participated_clients = set()
        self.client_assignments = {}
        self.cluster_prototypes = {}
        self.last_round_participants = set()
        # Virtual cluster configuration
        self.use_virtual_cluster = True  # Enable virtual cluster for never-participated clients
        ema_alpha: float = 0.3,  # EMA smoothing for training times
        beta: float = 1.5,  # Penalty strength for reliability score
        initial_alpha1: float = 0.6,  # Initial reliability weight
        initial_alpha2: float = 0.4,  # Initial fairness weight
        phase_threshold: int = 20,  # Round to switch weight emphasis
        total_rounds: int = 3,
      


        # NEW/MODIFIED FAIRNESS ATTRIBUTES
        initial_target_selections= 3
        max_target_selections = 10
        reliability_lambda = 0.05
        acc_drop_threshold  = 0.005
        self.client_targets = defaultdict(lambda: initial_target_selections)
        self.initial_target_selections = initial_target_selections
        self.max_target_selections = max_target_selections
        self.acc_drop_threshold = acc_drop_threshold

        # NEW RELIABILITY ATTRIBUTE
        self.reliability_lambda = reliability_lambda

        self.phase_threshold = 30
        
        # CSMDA Hyperparameters
        self.alpha = 0.3  # EMA decay for training time
        self.beta = 0.2   # fairness boost increment
        self.epsilon = 0.1  # straggler tolerance (10% of T_max)
        self.target_selections = 5  # minimum selections per client
        self.accuracy_eval_interval = 5  # evaluate accuracy every R rounds
        self.phase_threshold = 30  # switch from reliability to fairness focus
        # EMA Training Time Tracking
        self.ema_alpha = ema_alpha
        self.training_times = {}  # T_c(i) - EMA of training times
        
        # Reliability Score Parameters
        self.beta = beta  # Penalty strength parameter
        
        # Fairness Tracking
        self.selection_counts = {}  # v_c - number of times client selected
        self.total_rounds_completed = 0  # T - total rounds
        
        # Weight Adaptation
        self.initial_alpha1 = initial_alpha1
        self.initial_alpha2 = initial_alpha2
        self.phase_threshold = phase_threshold
        self.total_rounds = total_rounds
        
        print(f"[Init] Strategy initialized with α={ema_alpha}, β={beta}")

        # Initialize other components
        self.stat_util = {}
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_avg_accuracy = 0.0
        self.batch_size = batch_size
        self.save_dir = "visualizations"

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
         experiment_id = mlflow.create_experiment(experiment_name)
         print(f"Created new experiment with ID: {experiment_id}")
         experiment = mlflow.get_experiment(experiment_id)
        else:
         print(f"Using existing experiment with ID: {experiment.experiment_id}")
      
        # Store MLflow reference
        self.mlflow = mlflow
        self.client_to_domain={}
        self.num_domains = self.min_fit_clients
        self.batch_size=batch_size
        self.save_dir="visualizations"
        
        print(f'num domain : {self.min_fit_clients}')
       
        #experiment_id = mlflow.create_experiment(experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="server") as run:
         self.server_run_id = run.info.run_id
         # Log server parameters
         mlflow.log_params({
                "num_classes": num_classes,
                "min_fit_clients": min_fit_clients,
                "fraction_fit": fraction_fit
            })
         
        # Initialize the generator and its optimizer here
        self.num_classes =num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_avg_accuracy=0.0
        # Initialize the generator and its optimizer here
       
        self.label_probs = {label: 1.0 / self.num_classes for label in range(self.num_classes)}
        # Store client models for ensemble predictions
        self.client_classifiers = {}
        self.feature_visualizer =StructuredFeatureVisualizer(
        num_clients=2,  # total number of clients
        num_classes=self.num_classes,           # number of classes in your dataset

save_dir="feature_visualizations_gpaf"
         )
         
    def num_evaluate_clients(self, client_manager: ClientManager) -> Tuple[int, int]:
      """Return the sample size and required number of clients for evaluation."""
      num_clients = client_manager.num_available()
      return max(int(num_clients * self.fraction_evaluate), self.min_evaluate_clients), self.min_available_clients
 

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates and update client statistics
        """
        if failures:
            print(f"[Round {server_round}] Failures: {len(failures)}")
        
        if not results:
            print(f"[Round {server_round}] No clients returned results. Skipping aggregation.")
            return None, {}
        
        clients_params_list = []
        num_samples_list = []
        current_round_durations = []
        current_participants = set()
        
        # Process results and update tracking
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            metrics = fit_res.metrics
            
            self.participated_clients.add(client_id)
            current_participants.add(client_id)
            
            # Update EMA training time - Equation (4)
            if "duration" in metrics:
                duration = metrics["duration"]
                
                # Initialize or update T_c(i)
                if client_id not in self.training_times:
                    # First observation: T_c(0) = t_avail_c(0)
                    self.training_times[client_id] = duration
                    print(f"[EMA Init] Client {client_id}: T_c(0) = {duration:.2f}s")
                else:
                    # EMA update: T_c(i) = α * t_avail_c(i) + (1-α) * T_c(i-1)
                    old_ema = self.training_times[client_id]
                    self.training_times[client_id] = (
                        self.ema_alpha * duration + 
                        (1 - self.ema_alpha) * old_ema
                    )
                    print(f"[EMA Update] Client {client_id}: {old_ema:.2f}s → {self.training_times[client_id]:.2f}s (raw: {duration:.2f}s)")
                
                current_round_durations.append(duration)
            
            # Collect parameters for aggregation
            clients_params_list.append(parameters_to_ndarrays(fit_res.parameters))
            num_samples_list.append(fit_res.num_examples)
        
        self.last_round_participants = current_participants
        self.total_rounds_completed = server_round
        
        print(f"\n[Round {server_round}] Participants: {list(current_participants)}")
        print(f"[Round {server_round}] Average raw training time: {np.mean(current_round_durations):.2f}s")
        
        # Perform FedAvg aggregation
        aggregated_params = self._fedavg_parameters(clients_params_list, num_samples_list)
        
        return ndarrays_to_parameters(aggregated_params), {}
    

    def _fedavg_parameters(
        self, params_list: List[List[np.ndarray]], num_samples_list: List[int]
    ) -> List[np.ndarray]:
        """Aggregate parameters using FedAvg (weighted averaging)."""
        if not params_list:
            return []

        print("==== aggregation===")
        total_samples = sum(num_samples_list)

        # Initialize aggregated parameters with zeros
        aggregated_params = [np.zeros_like(param) for param in params_list[0]]

        # Weighted sum of parameters
        for params, num_samples in zip(params_list, num_samples_list):
            for i, param in enumerate(params):
                aggregated_params[i] += param * num_samples

        # Weighted average of parameters
        aggregated_params = [param / total_samples for param in aggregated_params]

        return aggregated_params
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.EvaluateRes]],
        failures: List[Union[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes], Exception]],
    ) -> Tuple[Optional[flwr.common.Parameters], Dict[str, flwr.common.Scalar]]:
        print(f"[Server] Round {server_round}: {len(results)} clients evaluated, {len(failures)} failed evaluation.")
        
        aggregated_accuracy = 0.0
        if results:
            self._current_accuracies = {}
            for client_proxy, res in results:
                client_id = client_proxy.cid
                if "accuracy" in res.metrics:
                    client_accuracy = float(res.metrics["accuracy"])
                    self._current_accuracies[client_id] = client_accuracy
                    
                    with self.mlflow.start_run(run_id=self.server_run_id):
                        self.mlflow.log_metrics({
                            f"accuracy_client_{client_id}": client_accuracy
                        }, step=server_round)
                else:
                    print(f"[Warning] Client {client_id} did not report 'accuracy' in eval_res.metrics.")

                if "features" in res.metrics and "labels" in res.metrics:
                    try:
                        features_np = pickle.loads(base64.b64decode(res.metrics.get("features").encode('utf-8')))
                        labels_np = pickle.loads(base64.b64decode(res.metrics.get("labels").encode('utf-8')))
                        pass
                    except Exception as e:
                        print(f"[Warning] Failed to decode features/labels for client {client_id}: {e}")

            if self._current_accuracies:
                aggregated_accuracy = sum(self._current_accuracies.values()) / len(self._current_accuracies)
                print(f"[Server] Round {server_round}: Aggregated Average Accuracy: {aggregated_accuracy:.4f}")
                with self.mlflow.start_run(run_id=self.server_run_id):
                    self.mlflow.log_metrics({"avg_accuracy_global": aggregated_accuracy}, step=server_round)

            if aggregated_accuracy > self.best_avg_accuracy:
                self.best_avg_accuracy = aggregated_accuracy
            
            log_filename = "server_accuracy_log.csv"
            write_header = not os.path.exists(log_filename)
            with open(log_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(["round", "avg_accuracy"])
                writer.writerow([server_round, aggregated_accuracy])

        else:
            print(f"[Server] Round {server_round}: No evaluation results received.")
            aggregated_accuracy = 0.0
            
        return None, {"accuracy": aggregated_accuracy}
   
  

    def compute_reliability_scores(self, client_ids: List[str]) -> Dict[str, float]:
        """
        Compute reliability scores using the rational formula
        A_s[c] = T_max / (T_c + β * T_max)
        
        Where:
        - T_max: Average of all clients' EMA training times (Equation 5)
        - T_c: Individual client's EMA training time (Equation 4)
        - β: Penalty strength parameter
        """
        reliability_scores = {}
        
        # Calculate T_max - Equation (5): Average of EMA values
        valid_times = [
            self.training_times[cid] 
            for cid in client_ids 
            if cid in self.training_times and self.training_times[cid] > 0.0
        ]
        
        if not valid_times:
            print("[Reliability] Warning: No valid training times available")
            return {cid: 0.5 for cid in client_ids}
        
        # T_max = (1/N) * Σ T_c(i) for all clients
        T_max = np.mean(valid_times)
        
        print(f"\n[Reliability Scores] Round {self.total_rounds_completed}")
        print(f"  T_max (system average EMA): {T_max:.2f}s")
        print(f"  β (penalty strength): {self.beta}")
        print(f"  β * T_max: {self.beta * T_max:.2f}s")
        
        # Calculate reliability score for each client - Equation (6)
        for client_id in client_ids:
            # Get client's EMA training time
            T_c = self.training_times.get(client_id, T_max)
            
            # A_s[c] = T_max / (T_c + β * T_max)
            denominator = T_c + (self.beta * T_max)
            reliability_score = T_max / denominator
            
            # Ensure bounded output [0, 1]
            reliability_scores[client_id] = float(np.clip(reliability_score, 0.0, 1.0))
            
            print(f"  Client {client_id}: T_c={T_c:.2f}s, A_s={reliability_scores[client_id]:.4f}")
        
        return reliability_scores
    
    
    def compute_fairness_scores(self, client_ids: List[str]) -> Dict[str, float]:
        """
        Compute fairness scores using self-regulating mechanism
        f_s[c] = 1 / (1 + R_c)
        
        Where:
        - R_c = v_c / (T/N): Selection ratio (Equation 7)
        - v_c: Number of times client c has been selected
        - T: Total rounds completed
        - N: Total number of clients
        """
        fairness_scores = {}
        
        N = len(client_ids)  # Total number of clients in pool
        T = self.total_rounds_completed  # Total rounds
        
        # Ideal selections per client
        ideal_selections = T / N if T > 0 else 1.0
        
        print(f"\n[Fairness Scores] Round {T}")
        print(f"  Total rounds (T): {T}")
        print(f"  Total clients (N): {N}")
        print(f"  Ideal selections per client (T/N): {ideal_selections:.2f}")
        
        for client_id in client_ids:
            # Get selection count v_c
            v_c = self.selection_counts.get(client_id, 0)
            
            # Calculate selection ratio R_c = v_c / (T/N)
            if ideal_selections > 0:
                R_c = v_c / ideal_selections
            else:
                R_c = 0.0
            
            # Fairness score: f_s = 1 / (1 + R_c) - Equation (8)
            fairness_score = 1.0 / (1.0 + R_c)
            
            fairness_scores[client_id] = float(fairness_score)
            
            print(f"  Client {client_id}: v_c={v_c}, R_c={R_c:.3f}, f_s={fairness_score:.4f}")
        
        return fairness_scores
    
    
    def compute_global_selection_scores(
        self, 
        client_ids: List[str], 
        server_round: int
    ) -> Dict[str, float]:
        """
        Compute global selection scores
        S_c = α₁ * A_s[c] + α₂ * f_s[c]
        
        With adaptive weights based on training phase
        """
        # Compute component scores
        reliability_scores = self.compute_reliability_scores(client_ids)
        fairness_scores = self.compute_fairness_scores(client_ids)
        
        # Adapt weights based on training phase
        alpha_1, alpha_2 = self._adapt_weights(server_round)
        
        # Compute global scores
        final_scores = {}
        for client_id in client_ids:
            reliability = reliability_scores.get(client_id, 0.0)
            fairness = fairness_scores.get(client_id, 0.0)
            
            # Global score: S_c = α₁ * A_s + α₂ * f_s
            global_score = (alpha_1 * reliability) + (alpha_2 * fairness)
            final_scores[client_id] = float(global_score)
        
        # Print summary
        print(f"\n[Global Scores] Round {server_round}")
        print(f"  Weights: α₁(reliability)={alpha_1:.2f}, α₂(fairness)={alpha_2:.2f}")
        print(f"  Top clients by score:")
        
        sorted_clients = sorted(
            final_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:min(10, len(final_scores))]
        
        for cid, score in sorted_clients:
            r_score = reliability_scores.get(cid, 0.0)
            f_score = fairness_scores.get(cid, 0.0)
            v_c = self.selection_counts.get(cid, 0)
            print(f"    {cid}: Score={score:.4f} (R={r_score:.3f}, F={f_score:.3f}, selected={v_c}x)")
        
        return final_scores
    
    
    def _adapt_weights(self, server_round: int) -> Tuple[float, float]:
        """
        Adapt α₁ and α₂ weights based on training progress
        
        Early training: Emphasize reliability (fast convergence)
        Late training: Emphasize fairness (balanced participation)
        """
        print(f'ss {server_round} and ee {self.total_rounds}')
        progress = server_round / self.total_rounds
        
        if progress < 0.2:
            # Early phase (0-20%): Prioritize reliability for stable initial model
            alpha_1, alpha_2 = 0.7, 0.3
        elif progress < 0.8:
            # Middle phase (20-80%): Balanced approach
            alpha_1, alpha_2 = 0.6, 0.4
        else:
            # Late phase (80-100%): Prioritize fairness for comprehensive coverage
            alpha_1, alpha_2 = 0.4, 0.6
        
        return alpha_1, alpha_2
    
    
    def select_clients_from_cluster(
        self,
        cluster_clients: List[str],
        num_clients_to_select: int,
        server_round: int
    ) -> List[str]:
        """
        Select top-k clients from a cluster based on global scores
        
        Args:
            cluster_clients: List of client IDs in the cluster
            num_clients_to_select: Number of clients to select (k)
            server_round: Current round number
        
        Returns:
            List of selected client IDs
        """
        if not cluster_clients:
            return []
        
        # Compute global scores for all clients in cluster
        global_scores = self.compute_global_selection_scores(
            cluster_clients, 
            server_round
        )
        
        # Sort by score (descending) and select top-k
        sorted_clients = sorted(
            global_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        num_to_select = min(num_clients_to_select, len(sorted_clients))
        selected = [cid for cid, _ in sorted_clients[:num_to_select]]
        
        # Update selection counts
        for client_id in selected:
            self.selection_counts[client_id] = self.selection_counts.get(client_id, 0) + 1
        
        print(f"\n[Selection] Round {server_round}: Selected {len(selected)} clients from cluster")
        for cid in selected:
            score = global_scores[cid]
            count = self.selection_counts[cid]
            print(f"  ✓ {cid}: Score={score:.4f}, Total selections={count}")
        
        return selected
    
      
   
    def configure_fit(
    self, 
    server_round: int, 
    parameters: Parameters, 
    client_manager: ClientManager
) -> List[Tuple[ClientProxy, FitIns]]:
      """
      Enhanced client selection with:
      1. Comprehensive clustering on ALL available clients (not just participated)
      2. Virtual cluster for never-participated clients
      3. Resource-aware selection within each cluster using corrected methodology
      """
    
      print(f"\n{'='*80}")
      print(f"[Round {server_round}] CONFIGURE_FIT - Resource-Aware Fair Selection")
      print(f"{'='*80}")
    
      # Get all available clients
      all_clients = client_manager.all()
      available_client_cids = list(all_clients.keys())

      if not available_client_cids:
        print(f"[Round {server_round}] No clients available.")
        return []

      print(f"\n[Client Status]")
      print(f"  Total available clients: {len(available_client_cids)}")
      print(f"  Previously participated: {len(self.participated_clients)}")
    
      # Categorize clients
      participated_available = [cid for cid in available_client_cids 
                             if cid in self.participated_clients]
      never_participated = [cid for cid in available_client_cids 
                         if cid not in self.participated_clients]
    
      print(f"  Available participated clients: {len(participated_available)}")
      print(f"  Available never-participated clients: {len(never_participated)}")

      # =================================================================
      # PHASE 1: COMPREHENSIVE PROTOTYPE COLLECTION (Every N rounds)
      # =================================================================
      clustering_round = (server_round % self.clustering_interval == 0)
    
      if clustering_round and participated_available:
        print(f"\n{'─'*80}")
        print(f"[Clustering Round] Collecting prototypes from ALL available clients")
        print(f"{'─'*80}")
        
        all_prototypes_list = []
        all_client_ids = []
        class_counts_list = []
        clients_with_prototypes = []
        
        # CRITICAL: Collect from ALL available participated clients
        for cid in participated_available:
            client_proxy = all_clients[cid]
            try:
                get_protos_res = client_proxy.get_properties(
                    ins=GetPropertiesIns(config={"request": "prototypes"}), 
                    timeout=15.0,
                    group_id=None
                )
                
                prototypes_encoded = get_protos_res.properties.get("prototypes")
                class_counts_encoded = get_protos_res.properties.get("class_counts")

                if prototypes_encoded and class_counts_encoded:
                    try:
                        prototypes = pickle.loads(base64.b64decode(prototypes_encoded))
                        class_counts = pickle.loads(base64.b64decode(class_counts_encoded))
                        
                        if isinstance(prototypes, dict) and isinstance(class_counts, dict):
                            all_prototypes_list.append(prototypes)
                            all_client_ids.append(cid)
                            class_counts_list.append(class_counts)
                            clients_with_prototypes.append(cid)
                            
                            source = "from round t-1" if cid in self.last_round_participants else "cached"
                            print(f"  ✓ Client {cid}: Prototypes collected [{source}]")
                            
                    except Exception as decode_error:
                        print(f"  ✗ Client {cid}: Decode error - {decode_error}")
                else:
                    print(f"  ⚠ Client {cid}: No prototypes available")
                    
            except Exception as e:
                print(f"  ⚠ Client {cid}: Communication failed - {e}")

        print(f"\n[Prototype Collection Summary]")
        print(f"  Successfully collected: {len(clients_with_prototypes)}/{len(participated_available)} clients")

        # Perform EM clustering if enough clients
        if len(clients_with_prototypes) >= self.num_clusters:
            print(f"\n[EM Clustering] Processing {len(clients_with_prototypes)} clients...")
            
            # Initialize cluster prototypes if first time
            if not self.cluster_prototypes:
                print("  Initializing cluster prototypes...")
                self.cluster_prototypes = self._initialize_clusters(all_prototypes_list)
            
            # E-step: Assign clients to clusters
            global_assignments = self._e_step(all_prototypes_list, all_client_ids)
            
            # M-step: Update cluster prototypes
            self.cluster_prototypes = self._m_step(
                all_prototypes_list, 
                all_client_ids, 
                global_assignments, 
                class_counts_list
            )
            
            # Update cluster assignments for ALL clients
            for client_id, cluster_id in global_assignments.items():
                self.client_assignments[client_id] = cluster_id
            
            print(f"\n[Clustering Results]")
            for cluster_id in range(self.num_clusters):
                cluster_clients = [cid for cid, clust in self.client_assignments.items() 
                                 if clust == cluster_id]
                print(f"  Cluster {cluster_id}: {len(cluster_clients)} clients")
                if cluster_clients:
                    print(f"    Members: {cluster_clients[:5]}{'...' if len(cluster_clients) > 5 else ''}")
        else:
            print(f"\n[Clustering Skipped] Need {self.num_clusters} clients, only have {len(clients_with_prototypes)}")

      # =================================================================
      # PHASE 2: ORGANIZE CLIENTS INTO CLUSTERS
      # =================================================================
      clusters = defaultdict(list)
    
      # Add participated clients to their assigned clusters
      for cid in participated_available:
        if cid in self.client_assignments:
            cluster_id = self.client_assignments[cid]
            clusters[cluster_id].append(cid)
        else:
            # Clients without assignment go to cluster 0 by default
            clusters[0].append(cid)
    
      # Add never-participated clients to virtual cluster
      if never_participated and self.use_virtual_cluster:
        clusters[self.virtual_cluster_id] = never_participated
        print(f"\n[Virtual Cluster {self.virtual_cluster_id}] {len(never_participated)} new clients")
    
      # Display cluster distribution
      print(f"\n[Cluster Distribution]")
      for cluster_id in sorted(clusters.keys()):
        cluster_clients = clusters[cluster_id]
        cluster_type = "Virtual" if cluster_id == self.virtual_cluster_id else "Domain"
        print(f"  Cluster {cluster_id} [{cluster_type}]: {len(cluster_clients)} clients")

      # =================================================================
      # PHASE 3: COMPUTE GLOBAL SELECTION SCORES
      # =================================================================
      print(f"\n{'─'*80}")
      print(f"[Score Computation] Round {server_round}")
      print(f"{'─'*80}")
    
      # Get adaptive weights
      alpha_1, alpha_2 = self._adapt_weights(server_round)
      print(f"Weights: α₁(reliability)={alpha_1:.2f}, α₂(fairness)={alpha_2:.2f}")
    
      # Compute scores for ALL available clients
      all_scores = {}
    
      # Process participated clients (use actual methodology)
      if participated_available:
        print(f"\n[Participated Clients] Computing scores using full methodology...")
        participated_scores = self.compute_global_selection_scores(
            participated_available, 
            server_round
        )
        all_scores.update(participated_scores)
    
      # Process never-participated clients (special handling)
      if never_participated:
        print(f"\n[New Clients] Assigning initial scores...")
        for cid in never_participated:
            # New clients get:
            # - Maximum fairness (never selected): f_s = 1.0
            # - Neutral reliability (unknown): A_s = 0.5 (middle value)
            reliability = 0.5  # Neutral - no history
            fairness = 1.0     # Maximum - never selected
            
            all_scores[cid] = (alpha_1 * reliability) + (alpha_2 * fairness)
            print(f"  Client {cid}: R={reliability:.3f}, F={fairness:.3f}, Score={all_scores[cid]:.3f}")

      # =================================================================
      # PHASE 4: DISTRIBUTE SELECTION BUDGET ACROSS CLUSTERS
      # =================================================================
      print(f"\n{'─'*80}")
      print(f"[Selection Distribution]")
      print(f"{'─'*80}")
    
      if not clusters:
        print("No clusters available for selection")
        return []
    
      total_clusters = len(clusters)
      base_per_cluster = max(1, self.min_fit_clients // total_clusters)
      remaining_budget = self.min_fit_clients - (base_per_cluster * total_clusters)
    
      print(f"Total selection budget: {self.min_fit_clients} clients")
      print(f"Active clusters: {total_clusters}")
      print(f"Base allocation per cluster: {base_per_cluster}")
      print(f"Remaining to distribute: {remaining_budget}")
    
      # Allocate base quota to each cluster
      cluster_allocations = {cluster_id: base_per_cluster for cluster_id in clusters}
    
      # Distribute remaining budget proportionally by cluster size
      if remaining_budget > 0:
        cluster_sizes = {cluster_id: len(clients) for cluster_id, clients in clusters.items()}
        total_size = sum(cluster_sizes.values())
        
        for cluster_id in sorted(clusters.keys(), key=lambda x: cluster_sizes[x], reverse=True):
            if remaining_budget <= 0:
                break
            proportion = cluster_sizes[cluster_id] / total_size if total_size > 0 else 0
            extra = min(remaining_budget, max(1, int(remaining_budget * proportion)))
            cluster_allocations[cluster_id] += extra
            remaining_budget -= extra
    
      print(f"\nFinal allocations:")
      for cluster_id, allocation in sorted(cluster_allocations.items()):
        cluster_type = "Virtual" if cluster_id == self.virtual_cluster_id else "Domain"
        print(f"  Cluster {cluster_id} [{cluster_type}]: {allocation} clients")

      # =================================================================
      # PHASE 5: SELECT CLIENTS FROM EACH CLUSTER
      # =================================================================
      print(f"\n{'─'*80}")
      print(f"[Client Selection by Cluster]")
      print(f"{'─'*80}")
    
      selected_clients_cids = []
    
      for cluster_id in sorted(clusters.keys()):
        cluster_clients = clusters[cluster_id]
        allocation = cluster_allocations[cluster_id]
        cluster_type = "Virtual" if cluster_id == self.virtual_cluster_id else "Domain"
        
        print(f"\n[Cluster {cluster_id} - {cluster_type}]")
        
        # Sort by global score (descending)
        cluster_clients_sorted = sorted(
            cluster_clients,
            key=lambda cid: all_scores.get(cid, 0.0),
            reverse=True
        )
        
        # Select top-k clients
        num_to_select = min(allocation, len(cluster_clients_sorted))
        cluster_selection = cluster_clients_sorted[:num_to_select]
        selected_clients_cids.extend(cluster_selection)
        
        print(f"  Selected {len(cluster_selection)}/{len(cluster_clients)} clients")
        
        # Show detailed scores for top selections
        for i, cid in enumerate(cluster_selection[:5]):
            score = all_scores.get(cid, 0.0)
            status = "NEW" if cid not in self.participated_clients else "participated"
            selections = self.selection_counts.get(cid, 0)
            print(f"    {i+1}. Client {cid:15s} [{status:12s}] Score={score:.4f}, Selected={selections}x")

      # =================================================================
      # PHASE 6: PREPARE INSTRUCTIONS AND UPDATE TRACKING
      # =================================================================
      selected_clients_cids = selected_clients_cids[:self.min_fit_clients]
    
      instructions = []
      for client_id in selected_clients_cids:
        if client_id in all_clients:
            client_proxy = all_clients[client_id]
            client_config = {
                "server_round": server_round,
                "total_rounds": getattr(self, 'total_rounds', 100),
            }
            
            instructions.append((client_proxy, FitIns(parameters, client_config)))
            
            # Update selection counts for fairness tracking
            self.selection_counts[client_id] = self.selection_counts.get(client_id, 0) + 1

      # =================================================================
      # FINAL SUMMARY
      # =================================================================
      print(f"\n{'='*80}")
      print(f"[Round {server_round}] FINAL SELECTION SUMMARY")
      print(f"{'='*80}")
      print(f"Total selected: {len(instructions)} clients")
    
      regular_selected = sum(1 for cid in selected_clients_cids 
                          if cid in self.participated_clients)
      new_selected = sum(1 for cid in selected_clients_cids 
                      if cid not in self.participated_clients)
    
      print(f"  From domain clusters: {regular_selected} clients")
      print(f"  From virtual cluster: {new_selected} clients")
    
      print(f"\nSelection frequency (top 10):")
      top_selected = sorted(self.selection_counts.items(), 
                         key=lambda x: x[1], 
                         reverse=True)[:10]
      for cid, count in top_selected:
        print(f"  {cid:15s}: {count}x")
    
      print(f"{'='*80}\n")

      return instructions


# Helper method to add to your strategy class
def _initialize_clusters(self, prototypes_list):
    """Initialize cluster prototypes using k-means++ style initialization"""
    import numpy as np
    
    # Convert prototypes to vectors for initialization
    proto_vectors = []
    for prototypes in prototypes_list:
        # Flatten all class prototypes into single vector
        all_protos = []
        for class_id in sorted(prototypes.keys()):
            all_protos.append(prototypes[class_id])
        if all_protos:
            proto_vectors.append(np.concatenate(all_protos))
    
    if not proto_vectors:
        return {}
    
    # Simple k-means++ initialization
    proto_array = np.array(proto_vectors)
    n_samples = len(proto_array)
    
    # Random first center
    centers_idx = [np.random.randint(n_samples)]
    
    # Select remaining centers
    for _ in range(self.num_clusters - 1):
        # Compute distances to nearest center
        distances = np.array([
            min(np.linalg.norm(proto_array[i] - proto_array[c]) 
                for c in centers_idx)
            for i in range(n_samples)
        ])
        
        # Select next center with probability proportional to distance
        probs = distances / distances.sum()
        next_center = np.random.choice(n_samples, p=probs)
        centers_idx.append(next_center)
    
    # Convert back to prototype format
    cluster_prototypes = {}
    for k, idx in enumerate(centers_idx):
        cluster_prototypes[k] = prototypes_list[idx]
    
    return cluster_prototypes

    

    def _e_step(self, all_prototypes, client_ids):
      """E-step: Assign clients to clusters."""
      assignments = {}
    
      print(f"[E-step] Assigning {len(client_ids)} clients to {len(self.cluster_prototypes)} clusters")
    
      for client_id, prototypes in zip(client_ids, all_prototypes):
        min_dist = float('inf')
        best_cluster = 0

        for cluster_id in self.cluster_prototypes:
            total_dist = 0
            shared_classes = 0

            for class_id in prototypes:
                if class_id in self.cluster_prototypes[cluster_id]:
                    client_proto = np.array(prototypes[class_id])
                    cluster_proto = np.array(self.cluster_prototypes[cluster_id][class_id])
                    
                    dist = self.cosine_distance(client_proto, cluster_proto)
                    total_dist += dist
                    shared_classes += 1

            avg_dist = total_dist / shared_classes if shared_classes > 0 else 1.0

            if avg_dist < min_dist:
                min_dist = avg_dist
                best_cluster = cluster_id

        assignments[client_id] = best_cluster

      cluster_counts = defaultdict(int)
      for cluster_id in assignments.values():
        cluster_counts[cluster_id] += 1
    
      print(f"[E-step] Assignment summary: {dict(cluster_counts)}")
      return assignments

    def cosine_distance(self, a, b):
      """Compute 1 - cosine similarity."""
      norm_a = np.linalg.norm(a)
      norm_b = np.linalg.norm(b)
    
      if norm_a == 0 or norm_b == 0:
        return 1.0
    
      return 1 - np.dot(a, b) / (norm_a * norm_b)

    def _m_step(self, all_prototypes, client_ids, assignments, class_counts_list):
      """M-step: Update cluster prototypes."""
      sample_proto = None
      for prototypes in all_prototypes:
        if prototypes:
            sample_proto = next(iter(prototypes.values()))
            break
    
      if sample_proto is None:
        return defaultdict(dict)
    
      if hasattr(sample_proto, 'numpy'):
        sample_proto = sample_proto.numpy()
      elif hasattr(sample_proto, 'detach'):
        sample_proto = sample_proto.detach().cpu().numpy()
    
      cluster_weighted_sum = defaultdict(lambda: defaultdict(lambda: np.zeros(sample_proto.shape, dtype=np.float32)))
      cluster_class_counts = defaultdict(lambda: defaultdict(int))

      for i, (client_id, prototypes) in enumerate(zip(client_ids, all_prototypes)):
        cluster_id = assignments[client_id]
        class_counts = class_counts_list[i]

        for class_id, proto in prototypes.items():
            weight = class_counts.get(class_id, 0)
            if weight > 0:
                if hasattr(proto, 'numpy'):
                    proto_np = proto.numpy()
                elif hasattr(proto, 'detach'):
                    proto_np = proto.detach().cpu().numpy()
                else:
                    proto_np = np.array(proto)
                
                proto_np = proto_np.astype(np.float32)
                weighted_proto = weight * proto_np
                cluster_weighted_sum[cluster_id][class_id] += weighted_proto
                cluster_class_counts[cluster_id][class_id] += weight

      new_clusters = defaultdict(dict)
      for cluster_id in range(self.num_clusters):
        if cluster_id in cluster_weighted_sum:
            for class_id in cluster_weighted_sum[cluster_id]:
                count = cluster_class_counts[cluster_id][class_id]
                if count > 0:
                    new_clusters[cluster_id][class_id] = cluster_weighted_sum[cluster_id][class_id] / count
                else:
                    new_clusters[cluster_id][class_id] = np.random.randn(*sample_proto.shape).astype(np.float32)

      self.cluster_class_counts = cluster_class_counts
      print(f"[M-step] Updated {len(new_clusters)} clusters")
      return new_clusters

    def configure_evaluate(
      self, server_round: int, parameters: Parameters, client_manager: ClientManager
) -> List[Tuple[ClientProxy, EvaluateIns]]:
      
      """Configure the next round of evaluation."""
     
      sample_size, min_num_clients = self.num_evaluate_clients(client_manager)
      clients = client_manager.sample(
        num_clients=sample_size, min_num_clients=min_num_clients
    )
      evaluate_config = {"server_round": server_round}  # Pass the round number in config
     
      print(f"Server sending round number: {server_round}")  # Debug print
      evaluate_ins = EvaluateIns(parameters, evaluate_config)
     
      # Return client-EvaluateIns pairs
      return [(client, evaluate_ins) for client in clients]   
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        print(f'===server evaluation=======')
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
      

  

