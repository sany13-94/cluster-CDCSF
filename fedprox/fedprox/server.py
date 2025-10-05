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
        # ... existing initialization ...
        self.participated_clients = set()
        self.client_assignments = {}
        self.cluster_prototypes = {}
        self.last_round_participants = set()
        # Virtual cluster configuration
        self.use_virtual_cluster = True  # Enable virtual cluster for never-participated clients
        self.virtual_cluster_id = -1  # Special ID for never-participated clients

      


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
      """Aggregate with tracking of participants."""
      print(f'results failure: {failures}')
      if not results:
        print("No clients returned results. Aggregation skipped.")
        return None, {}

      clients_params_list = []
      num_samples_list = []
      current_round_durations = []
      current_participants = set()

      for client_proxy, fit_res in results:
        client_id = client_proxy.cid
        metrics = fit_res.metrics
        
        self.participated_clients.add(client_id)
        current_participants.add(client_id)
        
        if "duration" in metrics:
            duration = metrics["duration"]
            ema_alpha = 0.3
            if client_id not in self.training_times:
                self.training_times[client_id] = duration
            else:
                self.training_times[client_id] = (
                    ema_alpha * duration +
                    (1 - ema_alpha) * self.training_times[client_id]
                )
            current_round_durations.append(duration)
        
        if "loss_sq_mean" in metrics and "data_size" in metrics:
            stat_util = metrics["data_size"] * metrics["loss_sq_mean"]
            self.stat_util[client_id] = stat_util
        
        clients_params_list.append(parameters_to_ndarrays(fit_res.parameters))
        num_samples_list.append(fit_res.num_examples)

      self.last_round_participants = current_participants
      print(f"[Aggregate] Round {server_round} participants: {list(current_participants)}")

      if current_round_durations:
        current_avg_duration = sum(current_round_durations) / len(current_round_durations)
        ewma_decay = 0.1
        """
        if self.global_T_max == 0.0:
            self.global_T_max = current_avg_duration
        else:
            self.global_T_max = (1 - ewma_decay) * self.global_T_max + ewma_decay * current_avg_duration
        """
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
   
  
    def _compute_reliability_scores(self, client_ids: List[str]) -> Dict[str, float]:
        reliability_scores = {}
        
        valid_times = [self.training_times[cid] for cid in client_ids 
                       if self.training_times.get(cid, 0.0) > 0.0]

        T_avg = sum(valid_times) / len(valid_times) if valid_times else 1.0

        print(f"[Reliability] T_avg for current selection pool: {T_avg:.2f}s")

        for client_id in client_ids:
            T_c = self.training_times.get(client_id, T_avg)
            penalty_term = max(0.0, T_c - T_avg)
            score = np.exp(-self.reliability_lambda * penalty_term)
            
            reliability_scores[client_id] = float(min(1.0, max(0.0, score)))
            
        return reliability_scores

    def _compute_fairness_scores(self, client_ids: List[str]) -> Dict[str, float]:
        fairness_scores = {}
        for client_id in client_ids:
            v_c = self.selection_counts.get(client_id, 0)
            Target_c = self.client_targets.get(client_id, self.initial_target_selections)

            if Target_c <= 0:
                score = 0.0
            else:
                score = max(0.0, (Target_c - v_c) / Target_c)
            
            fairness_scores[client_id] = float(score)
            
        return fairness_scores


    def _compute_global_selection_scores(self, client_ids: List[str], server_round: int) -> Dict[str, float]:
        reliability_scores = self._compute_reliability_scores(client_ids)
        fairness_scores = self._compute_fairness_scores(client_ids)
        
        alpha_1, alpha_2 = self._adapt_weights(server_round)
        
        final_scores = {}
        for client_id in client_ids:
            reliability = reliability_scores.get(client_id, 0.0)
            fairness = fairness_scores.get(client_id, 0.0)
            final_scores[client_id] = (alpha_1 * reliability) + (alpha_2 * fairness)
        
        print(f"[Global Score] Round {server_round}: Weights: reliability={alpha_1:.2f}, fairness={alpha_2:.2f}")
        for cid in client_ids[:min(5, len(client_ids))]:
            print(f"  Client {cid}: R={reliability_scores.get(cid, 0):.3f}, F={fairness_scores.get(cid, 0):.3f}, Score={final_scores[cid]:.3f}")
        
        return final_scores
    def _adapt_weights(self, server_round: int) -> Tuple[float, float]:
        if server_round <= self.phase_threshold:
            return 0.7, 0.3
        else:
            return 0.4, 0.6

    #fedavg evaluate_fit
      
   
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
     
     """
      Client selection with:
      1. Clustering on participated clients
      2. Virtual cluster for never-participated clients
        3. Resource-aware selection within each cluster (including virtual)
     """
    
     print(f"\n[CSMDA] ========== Configuring round {server_round} ==========")
     all_clients = client_manager.all()
     available_client_cids = list(all_clients.keys())

     if not available_client_cids:
        print(f"[CSMDA] Round {server_round}: No clients available.")
        return []

     print(f"[CSMDA] Total available clients: {len(available_client_cids)}")
     print(f"[CSMDA] Clients who have participated before: {len(self.participated_clients)}")

     # Step 1: Categorize clients
     participated_available = [cid for cid in available_client_cids if cid in self.participated_clients]
     never_participated = [cid for cid in available_client_cids if cid not in self.participated_clients]
    
     print(f"[CSMDA] Available participated clients: {len(participated_available)}")
     print(f"[CSMDA] Available never-participated clients: {len(never_participated)}")

     # Step 2: Collect prototypes from participated clients
     all_prototypes_list = []
     all_client_ids = []
     class_counts_list = []
     clients_with_prototypes = []
    
     if participated_available:
        print(f"\n[CSMDA] Collecting prototypes from {len(participated_available)} participated clients...")
        
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
                            
                            source = "fresh (t-1)" if cid in self.last_round_participants else "saved (earlier)"
                            print(f"[CSMDA] ✅ Client {cid}: Prototypes [{source}]")
                            
                    except Exception as decode_error:
                        print(f"[CSMDA] ❌ Client {cid}: Decode error - {decode_error}")
                else:
                    print(f"[CSMDA] ⚠️ Client {cid}: No prototypes")
                    
            except Exception as e:
                print(f"[CSMDA] ⚠️ Client {cid}: Communication failed - {e}")

     print(f"\n[CSMDA] Prototype collection summary:")
     print(f"  - Participated clients with prototypes: {len(clients_with_prototypes)}")

     # Step 3: Perform clustering on participated clients (if enough available)
     clustering_performed = False
    
     if len(clients_with_prototypes) >= self.num_clusters:
        print(f"\n[CSMDA] 🔄 Performing EM clustering on {len(clients_with_prototypes)} participated clients...")
        
        if not self.cluster_prototypes:
            print("[CSMDA] Initializing cluster prototypes")
            self.cluster_prototypes = self._initialize_clusters(all_prototypes_list)
        
        global_assignments = self._e_step(all_prototypes_list, all_client_ids)
        self.cluster_prototypes = self._m_step(
            all_prototypes_list, 
            all_client_ids, 
            global_assignments, 
            class_counts_list
        )
        
        for client_id, cluster_id in global_assignments.items():
            self.client_assignments[client_id] = cluster_id
        
        print(f"\n[CSMDA] Clustering results:")
        for cluster_id in range(self.num_clusters):
            cluster_clients = [cid for cid, clust in self.client_assignments.items() 
                             if clust == cluster_id]
            print(f"  - Cluster {cluster_id}: {len(cluster_clients)} clients")
        
        clustering_performed = True
     else:
        print(f"\n[CSMDA] ⚠️ Not enough participated clients ({len(clients_with_prototypes)}) "
              f"for clustering (need {self.num_clusters})")

     # Step 4: Organize clients into clusters (including virtual cluster)
     clusters = defaultdict(list)
    
     # Add participated clients to their assigned clusters
     for cid in clients_with_prototypes:
        cluster_id = self.client_assignments.get(cid, 0)
        clusters[cluster_id].append(cid)
    
     # Add never-participated clients to virtual cluster
     if never_participated and self.use_virtual_cluster:
        clusters[self.virtual_cluster_id] = never_participated
        print(f"\n[CSMDA] 📦 Virtual Cluster {self.virtual_cluster_id}: {len(never_participated)} never-participated clients")
    
     # Display all clusters
     print(f"\n[CSMDA] All clusters (including virtual):")
     for cluster_id, cluster_clients in clusters.items():
        cluster_type = "Virtual" if cluster_id == self.virtual_cluster_id else "Regular"
        print(f"  - Cluster {cluster_id} [{cluster_type}]: {len(cluster_clients)} clients")

     # Step 5: Compute selection scores for ALL clients
     print(f"\n[CSMDA] Computing resource-aware scores for all clients...")
    
     all_available_clients = participated_available + never_participated
     alpha_1, alpha_2 = self._adapt_weights(server_round)
    
     # Compute scores
     all_scores = {}
    
     for cid in all_available_clients:
        if cid in self.participated_clients:
            # Participated clients: use actual reliability and fairness
            reliability_scores = self._compute_reliability_scores([cid])
            fairness_scores = self._compute_fairness_scores([cid])
            reliability = reliability_scores.get(cid, 0.5)
            fairness = fairness_scores.get(cid, 0.5)
        else:
            # Never-participated clients: neutral reliability, high fairness
            reliability = 1.0  # Neutral (no history)
            fairness = 1.0     # Maximum fairness (never selected)
        
        all_scores[cid] = (alpha_1 * reliability) + (alpha_2 * fairness)
    
     print(f"[CSMDA] Computed scores for {len(all_scores)} clients")
     print(f"[CSMDA] Score weights: reliability={alpha_1:.2f}, fairness={alpha_2:.2f}")

     # Step 6: Distribute selection budget across clusters (proportional + min guarantee)
     selected_clients_cids = []
    
     if clusters:
        total_clusters = len(clusters)
        base_per_cluster = max(1, self.min_fit_clients // total_clusters)
        remaining_budget = self.min_fit_clients - (base_per_cluster * total_clusters)
        
        print(f"\n[CSMDA] 📊 Selection distribution:")
        print(f"  - Total clusters: {total_clusters}")
        print(f"  - Base allocation per cluster: {base_per_cluster}")
        print(f"  - Remaining budget to distribute: {remaining_budget}")
        
        # First pass: allocate base quota to each cluster
        cluster_allocations = {cluster_id: base_per_cluster for cluster_id in clusters}
        
        # Second pass: distribute remaining budget proportionally by cluster size
        if remaining_budget > 0:
            cluster_sizes = {cluster_id: len(clients) for cluster_id, clients in clusters.items()}
            total_size = sum(cluster_sizes.values())
            
            for cluster_id in sorted(clusters.keys(), key=lambda x: cluster_sizes[x], reverse=True):
                if remaining_budget <= 0:
                    break
                # Give extra slots to larger clusters
                extra = min(remaining_budget, max(1, int(remaining_budget * cluster_sizes[cluster_id] / total_size)))
                cluster_allocations[cluster_id] += extra
                remaining_budget -= extra
        
        # Select clients from each cluster based on scores
        for cluster_id, cluster_clients in clusters.items():
            allocation = cluster_allocations[cluster_id]
            
            # Sort cluster clients by score
            cluster_clients_sorted = sorted(
                cluster_clients,
                key=lambda cid: all_scores.get(cid, 0.0),
                reverse=True
            )
            
            # Select top clients
            num_to_select = min(allocation, len(cluster_clients_sorted))
            cluster_selection = cluster_clients_sorted[:num_to_select]
            selected_clients_cids.extend(cluster_selection)
            
            cluster_type = "Virtual" if cluster_id == self.virtual_cluster_id else "Regular"
            print(f"  - Cluster {cluster_id} [{cluster_type}]: Selected {len(cluster_selection)}/{len(cluster_clients)} clients")
            
            # Show top selections with scores
            for i, cid in enumerate(cluster_selection[:3]):
                score = all_scores.get(cid, 0.0)
                status = "new" if cid not in self.participated_clients else "participated"
                print(f"    {i+1}. Client {cid} [{status}]: score={score:.3f}")
    
     else:
        # Fallback: select all available up to limit
        print(f"\n[CSMDA] ⚠️ No clusters available, selecting from all clients")
        all_sorted = sorted(all_available_clients, key=lambda cid: all_scores.get(cid, 0.0), reverse=True)
        selected_clients_cids = all_sorted[:self.min_fit_clients]

     # Step 7: Prepare instructions
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
            self.selection_counts[client_id] = self.selection_counts.get(client_id, 0) + 1

     print(f"\n[CSMDA] ✅ Round {server_round} FINAL SELECTION:")
     print(f"  - Total selected: {len(instructions)} clients")
     print(f"  - Breakdown:")
     print(f"    * From regular clusters: {sum(1 for cid in selected_clients_cids if cid in self.participated_clients)}")
     print(f"    * From virtual cluster: {sum(1 for cid in selected_clients_cids if cid not in self.participated_clients)}")
     print(f"  - Selection counts (top 5): {dict(sorted(self.selection_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
     print(f"[CSMDA] ========== End of round {server_round} configuration ==========\n")

     return instructions

    def _initialize_clusters(self, all_prototypes):
      """Initialize clusters from most diverse clients."""
      num_clients = len(all_prototypes)
      assert num_clients >= self.num_clusters, \
        f"Need at least {self.num_clusters} clients, got {num_clients}"

      client_diversity = []
      for i, proto_dict in enumerate(all_prototypes):
        diversity = len([v for v in proto_dict.values() if np.linalg.norm(v) > 0])
        client_diversity.append((i, diversity))
    
      client_diversity.sort(key=lambda x: x[1], reverse=True)
      selected_indices = [idx for idx, _ in client_diversity[:self.num_clusters]]

      cluster_prototypes = {}
      for cluster_id, client_idx in enumerate(selected_indices):
        proto_dict = all_prototypes[client_idx]
        cluster_prototypes[cluster_id] = {}
        
        for class_id, proto in proto_dict.items():
            if hasattr(proto, 'numpy'):
                proto_np = proto.numpy()
            elif hasattr(proto, 'detach'):
                proto_np = proto.detach().cpu().numpy()
            else:
                proto_np = np.array(proto)
            
            cluster_prototypes[cluster_id][class_id] = proto_np.astype(np.float32).copy()

      print(f"[Init] Initialized {self.num_clusters} clusters")
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
      

  

