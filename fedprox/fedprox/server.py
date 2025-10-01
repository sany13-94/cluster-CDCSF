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
from fedprox.client_monitoring import CRACS_MDA, load_log_data
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
from fedprox.client_monitoring import update_histories
from fedprox.client_monitoring import T_hat

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
   
    def _initialize_clusters(self, all_prototypes):
      num_clients = len(all_prototypes)
      assert num_clients >= self.num_clusters, \
        f"Need at least {self.num_clusters} clients to initialize clusters"

      # Randomly sample clients (optional: sort by class diversity first)
      sorted_protos = sorted(all_prototypes, key=lambda d: len(d), reverse=True)
      selected_prototypes = sorted_protos[:self.num_clusters]

      cluster_prototypes = {}
      for cluster_id, proto_dict in enumerate(selected_prototypes):
        cluster_prototypes[cluster_id] = {
            class_id: proto.copy()
            for class_id, proto in proto_dict.items()
        }

      print(f"[Init] Cluster prototypes initialized from top-{self.num_clusters} diverse clients")
      return cluster_prototypes

    def cosine_distance(self,a, b):
      """Compute 1 - cosine similarity"""
      if norm(a) == 0 or norm(b) == 0:
        return 1.0  # Maximum distance if one is zero
      return 1 - np.dot(a, b) / (norm(a) * norm(b))

    def _e_step(self, all_prototypes, client_ids):
      """Hard assignment of clients to clusters using cosine similarity"""
      assignments = {}
      for client_id, prototypes in zip(client_ids, all_prototypes):
        min_dist = float('inf')
        best_cluster = 0

        for cluster_id in self.cluster_prototypes:
            total_dist = 0
            class_count = 0

            for class_id in prototypes:
                if class_id in self.cluster_prototypes[cluster_id]:
                    client_proto = prototypes[class_id]
                    cluster_proto = self.cluster_prototypes[cluster_id][class_id]

                    # Use cosine distance instead of L2
                    dist = self.cosine_distance(client_proto, cluster_proto)
                    total_dist += dist
                    class_count += 1

            # Normalize distance by number of shared classes
            if class_count > 0:
                total_dist /= class_count

            if total_dist < min_dist:
                min_dist = total_dist
                best_cluster = cluster_id

        assignments[client_id] = best_cluster
        print(f"Client {client_id} assigned to Cluster {best_cluster}")

      return assignments

    
    def _m_step(self, all_prototypes, client_ids, assignments, class_counts_list):
      cluster_weighted_sum = defaultdict(lambda: defaultdict(lambda: np.zeros_like(next(iter(all_prototypes[0].values())))))
      cluster_class_counts = defaultdict(lambda: defaultdict(int))

      for i, (client_id, prototypes) in enumerate(zip(client_ids, all_prototypes)):
        cluster_id = assignments[client_id]
        class_counts = class_counts_list[i]

        for class_id, proto in prototypes.items():
            weight = class_counts.get(class_id, 0)
            if weight > 0:
                cluster_weighted_sum[cluster_id][class_id] += weight * proto
                cluster_class_counts[cluster_id][class_id] += weight

      new_clusters = defaultdict(dict)
      for cluster_id in cluster_weighted_sum:
        for class_id in cluster_weighted_sum[cluster_id]:
            count = cluster_class_counts[cluster_id][class_id]
            if count > 0:
                new_clusters[cluster_id][class_id] = cluster_weighted_sum[cluster_id][class_id] / count
            else:
                # Optional: fallback to random or default value if no samples
                new_clusters[cluster_id][class_id] = np.random.randn(*proto.shape)

      # Update global class counts
      self.cluster_class_counts = cluster_class_counts  # Used later in configure_fit

      return new_clusters


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, flwr.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate results and update generator."""
        print(f'results faillure {failures}')    
        if not results:
            return None, {}

        # Prepare config for next round
        config = {
            "server_round": server_round,
            
        }
        # Load from the same file saved in the data pipeline
        with open("client_domain_map.json", "r") as f:
          client_domain_map = json.load(f)
        clients_params_list=[]
        print(f'server round is {server_round}')
        num_samples_list=[]
        self.client_prototypes = {}  # <-- ADD THIS LINE
        client_ids=[]
        all_prototypes=[]
        class_counts_list=[]
        client_id_map = {}  # flower_cid -> simulation_index
        for client_proxy, fit_res in results:
                client_id=client_proxy.cid
                # Construct proper input
                props_ins = GetPropertiesIns(config={})
                props = client_proxy.get_properties(props_ins, timeout=10.0, group_id=None)
                # Extract simulation index
                sim_index = props.properties["simulation_index"]
                

                #prototypes = fit_res.metrics.get("prototypes").encode('utf-8')
                #prototypes = pickle.loads(base64.b64decode(prototypes))
                print(f"Flower cid: {client_id}  ↔  Simulation client index: {sim_index}")
                client_id_map[client_id] = sim_index
                metrics = fit_res.metrics
                # Make sure metrics are not empty and contain what you need
                if "loss_sq_mean" in metrics and "data_size" in metrics:
                    stat_util = metrics["data_size"] * metrics["loss_sq_mean"]
                    self.stat_util[client_id] = stat_util
                else:
                    # fallback or warning
                    self.stat_util[client_id] = 1.0  # neutral default

                training_duration = metrics.get("duration")
                if training_duration is not None:
                  ema_alpha = 0.3
                  if client_id not in self.training_times:
                    self.training_times[client_id] = training_duration
                  else:
                    self.training_times[client_id] = (
                        ema_alpha * training_duration +
                        (1 - ema_alpha) * self.training_times[client_id]
                    )
                else:
                  print(f"[Warning] Client {client_id} did not report 'duration' in fit_res.metrics.")

                if "prototypes" not in metrics or "class_counts" not in metrics:
                        print(f"[Warning] Client {client_proxy.cid} returned no prototype info. Skipping.")
                        continue


                client_parameters = parameters_to_ndarrays(fit_res.parameters)
                clients_params_list.append(client_parameters)
                all_prototypes.append(pickle.loads(base64.b64decode(fit_res.metrics["prototypes"])))
                client_ids.append(client_id)
                class_counts_list.append(pickle.loads(base64.b64decode(fit_res.metrics["class_counts"])))  # Dict[class_id] = count
                with open("cid_to_sim_index.json", "w") as f:
                  json.dump(client_id_map, f, indent=2)

                """
                if prototypes:
                    self.client_prototypes[client_id] = prototypes
                """
                num_samples_list.append(fit_res.num_examples)
        # Cluster clients using cosine similarity between prototype vectors
        #aggregated_params = super().aggregate_fit(server_round, client_parameters, failures)
        aggregated_params = self._fedavg_parameters(clients_params_list, num_samples_list)
        #print(f' client ids {client_ids}')

        
        #print(f'prototypes: **** {prototypes} ****')
        # Convert prototypes to numpy arrays
        proto_arrays = []
        for p in all_prototypes:
            proto_arrays.append({
                cls: np.array(proto) 
                for cls, proto in p.items()
            })
        
        # Initialize clusters if first round
        # ROUND 1: Initialize clusters
        if not self.client_assignments:
          # First time clustering → initialize clusters
          print("[Init] Performing first-time EM cluster initialization.")
          self.cluster_prototypes = self._initialize_clusters(proto_arrays)
        # 4. EM Algorithm
        # E-step: Assign clients to clusters
        assignments = self._e_step(proto_arrays, client_ids,)
        # M-step: Update cluster prototypes
        self.cluster_prototypes = self._m_step(proto_arrays, client_ids, assignments, class_counts_list)
        
        # 5. Update client assignments
        self.client_assignments.update(assignments)
        
        # 6. Prepare cluster prototypes for next round
        for cluster_id in self.cluster_prototypes:
            for class_id in self.cluster_prototypes[cluster_id]:
                if isinstance(self.cluster_prototypes[cluster_id][class_id], np.ndarray):
                    self.cluster_prototypes[cluster_id][class_id] = \
                        self.cluster_prototypes[cluster_id][class_id].tolist()

          
        # Visualize every 3 rounds
        print("client_id_map =", client_id_map)
        print("client_domain_map =", client_domain_map)

        for flower_cid in client_id_map:
            sim_index = client_id_map[flower_cid]
            if str(sim_index) not in client_domain_map:
              print(f"[ERROR] sim_index {sim_index} not found in client_domain_map")

        if server_round % 2 == 0:
            true_domain_map = {
    flower_cid: client_domain_map[str(client_id_map[flower_cid])]
    for flower_cid in client_id_map
}
            self._visualize_clusters(all_prototypes, client_ids, server_round, true_domain_map=true_domain_map)
        
    
        # 7. Build per-client config: map cid → cluster-level prototypes
        cluster_proto_map = {}  # Flower client ID -> {class_id: global_proto}
        print("[DEBUG] Assigned clients:")
        print(list(self.client_assignments.keys()))
        
        for cid in client_ids:
          cluster_id = self.client_assignments[cid]
          cluster_protos = self.cluster_prototypes[cluster_id]

          # Convert to list if necessary (ensure JSON serializable)
          serializable_protos = {
            str(cls): proto.tolist() if isinstance(proto, np.ndarray) else proto
            for cls, proto in cluster_protos.items()
          }

          cluster_proto_map[cid] = {
          "cluster_id": cluster_id,
          "cluster_prototypes": serializable_protos,
          }

        # OPTIONAL: Save for debugging
        with open("client_cluster_prototypes.json", "w") as f:
          json.dump(cluster_proto_map, f, indent=2)
        

        try:
            r = requests.post(f"{self.server_url}/save_logs")
            print("[Server] Log save status:", r.json())
        except Exception as e:
            print("[Server] Failed to save logs:", e)
        return ndarrays_to_parameters(aggregated_params),config
    



    def _visualize_clusters(self, prototypes, client_ids, server_round, true_domain_map=None):
      # 1. Flatten prototypes: one vector per client
      prototype_matrix = []
      for client_prototypes in prototypes:
        client_proto = np.mean(list(client_prototypes.values()), axis=0)
        prototype_matrix.append(client_proto)
      prototype_matrix = np.array(prototype_matrix)

      # 2. t-SNE projection
      n_clients = len(prototype_matrix)
      perplexity = min(30, max(1, n_clients - 1))
      tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
      projections = tsne.fit_transform(prototype_matrix)

      # 3. Cluster assignments (predicted by your method)
      cluster_assignments = [self.client_assignments.get(cid, -1) for cid in client_ids]
      unique_clusters = sorted(set(cluster_assignments))
      num_clusters = len(unique_clusters)

      # 4. Color map setup for clusters
      base_cmap = cm.get_cmap("tab20", num_clusters)
      colors = [base_cmap(i) for i in range(num_clusters)]
      color_map = ListedColormap(colors)
      cluster_id_to_color_index = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}
      color_indices = [cluster_id_to_color_index[cid] for cid in cluster_assignments]

      # 5. Marker setup for true domains
      markers = ['o', 's', '^', 'D', 'P', 'X']
      domain_to_marker = {}
      if true_domain_map:
        unique_domains = sorted(set(true_domain_map.get(cid, "unknown") for cid in client_ids))
        domain_to_marker = {dom: markers[i % len(markers)] for i, dom in enumerate(unique_domains)}

      # 6. Begin plotting
      plt.figure(figsize=(12, 8))

      for i, (x, y) in enumerate(projections):
        client_id = client_ids[i]
        cluster_id = cluster_assignments[i]
        color_index = cluster_id_to_color_index[cluster_id]

        if true_domain_map:
            domain = true_domain_map.get(client_id, "unknown")
            marker = domain_to_marker.get(domain, 'o')
        else:
            domain = "unknown"
            marker = 'o'

        plt.scatter(
            x, y,
            c=[colors[color_index]],
            marker=marker,
            edgecolor='k',
            s=100,
            alpha=0.8
        )
        plt.text(x, y, str(client_id), fontsize=7, ha='center', va='bottom')

      # 7. Legends
      # Cluster legend (colors)
      cluster_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cid}',
                   markerfacecolor=colors[idx], markersize=8)
        for cid, idx in cluster_id_to_color_index.items()
    ]

      # Domain legend (markers)
      domain_handles = []
      if true_domain_map:
        for dom, marker in domain_to_marker.items():
            domain_handles.append(
                plt.Line2D([0], [0], marker=marker, color='k', label=f'Domain: {dom}',
                           markerfacecolor='gray', markersize=8, linestyle='None')
            )

      plt.legend(handles=cluster_handles + domain_handles, title="Cluster / Domain", bbox_to_anchor=(1.05, 1), loc='upper left')

      # 8. Plot aesthetics
      plt.title(f"Client Prototypes (Round {server_round})\nColors = Cluster ID, Shapes = True Domain, Labels = Client ID")
      plt.xlabel("t-SNE 1")
      plt.ylabel("t-SNE 2")
      plt.tight_layout()
      plt.savefig(f"clusters_round_{server_round}.png", dpi=300, bbox_inches='tight')
      plt.show()
      plt.close()

      # 9. Optional: Clustering quality metrics
      if true_domain_map:
        predicted_clusters = cluster_assignments
        true_domains = [true_domain_map.get(cid, -1) for cid in client_ids]

        ari = adjusted_rand_score(true_domains, predicted_clusters)
        nmi = normalized_mutual_info_score(true_domains, predicted_clusters)

        print(f"Clustering Quality: ARI = {ari:.3f}, NMI = {nmi:.3f}")


   
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
   
    def _load_client_logs(self, server_round):
        """Load client training logs from heartbeat server"""
        try:
            # Load training time data from your heartbeat logs
            log_file = f"client_logs_round_{server_round-1}.json"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                return logs
            return {}
        except Exception as e:
            print(f"[Warning] Could not load client logs: {e}")
            return {}
   

    def _update_client_targets(self, server_round: int):
        if not self._current_accuracies:
            print(f"[Fairness Update] Round {server_round}: No current accuracies from previous round's evaluation to update targets.")
            return

        total_acc = sum(self._current_accuracies.values())
        num_evaluated_clients = len(self._current_accuracies)
        
        if num_evaluated_clients == 0:
            print(f"[Fairness Update] Round {server_round}: No evaluated clients, cannot compute Avg_Acc_Global for target update.")
            self._current_accuracies = {}
            return
            
        Avg_Acc_Global = total_acc / num_evaluated_clients
        print(f"[Fairness Update] Round {server_round}: Global Average Accuracy (prev round): {Avg_Acc_Global:.4f}")

        for client_id, current_acc in self._current_accuracies.items():
            current_target = self.client_targets[client_id]

            if current_acc < Avg_Acc_Global - self.acc_drop_threshold:
                self.client_targets[client_id] = min(self.max_target_selections, current_target + 1)
                print(f"[Fairness Update] Client {client_id}: Acc {current_acc:.4f} < Avg_Acc {Avg_Acc_Global:.4f}. Target increased from {current_target} to {self.client_targets[client_id]}")
            else:
                print(f"[Fairness Update] Client {client_id}: Acc {current_acc:.4f} >= Avg_Acc {Avg_Acc_Global:.4f}. Target remains {current_target}")

        self._current_accuracies = {}

    
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
      
   
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        print(f"\n[CSMDA] Configuring round {server_round}")
        
        # 1. Update Client Targets (Fairness) based on PREVIOUS round's evaluation accuracies
        # This is CRUCIAL to make the fairness score dynamic based on performance.
        self._update_client_targets(server_round)

        # Get all currently available clients
        available_client_cids = list(client_manager.all().keys())
        
        if not available_client_cids:
            print(f"[CSMDA] Round {server_round}: No clients available for selection.")
            return []

        # 2. First round or no assignments yet: random selection for initialization
        # This part remains to kickstart clustering/assignment.
        if server_round == 1 or not self.client_assignments:
            print("[CSMDA] First round or no assignments - random selection for clustering initialization")
            # Select min_fit_clients or all available if fewer
            selected_clients_cids = available_client_cids[:min(self.min_fit_clients, len(available_client_cids))]
            
            instructions = []
            for client_id in selected_clients_cids:
                client_proxy = client_manager.all()[client_id]
                config = {"server_round": server_round}
                instructions.append((client_proxy, FitIns(parameters, config)))
                self.selection_counts[client_id] += 1 # Update selection count immediately
            print(f"[CSMDA] Round {server_round}: Selected initial clients: {selected_clients_cids}")
            return instructions
        
        # --- Main Selection Logic for subsequent rounds (Cluster-based) ---
        
        # 3. Group clients by cluster assignment
        clusters = defaultdict(list)
        for client_id in available_client_cids:
            if client_id in self.client_assignments:
                cluster_id = self.client_assignments[client_id]
                clusters[cluster_id].append(client_id)
            else:
                # Assign new/unassigned clients to an existing cluster (e.g., the one with fewest members)
                if clusters:
                    # Find the cluster with the fewest *assigned* clients
                    min_cluster_id = min(clusters, key=lambda k: len(clusters[k]))
                    self.client_assignments[client_id] = min_cluster_id
                    clusters[min_cluster_id].append(client_id)
                    print(f"[CSMDA] Client {client_id} (new/unassigned) added to cluster {min_cluster_id}.")
                else:
                    # This case should be rare after round 1, but handles clients if no clusters exist yet.
                    # They will be part of the 'remaining_clients_cids' pool later.
                    print(f"[CSMDA] Client {client_id} unassigned, no clusters exist yet. Will consider in fallback.")

        # 4. Compute Global Selection Scores for *all* available clients
        # These scores are needed for both cluster-wise selection and the fallback.
        global_scores = self._compute_global_selection_scores(available_client_cids, server_round)

        selected_clients_cids = []
        
        # 5. Apply Cluster-based Selection: Select top-scoring clients from each cluster
        active_clusters_with_clients = [c_id for c_id, clients in clusters.items() if clients]
        
        if active_clusters_with_clients:
            # Determine target number of clients to select per cluster
            # Ensure at least 1 client per cluster if min_fit_clients allows, otherwise distribute evenly.
            clients_per_cluster_base = self.min_fit_clients // len(active_clusters_with_clients)
            extra_clients = self.min_fit_clients % len(active_clusters_with_clients) # Distribute remainder
            
            print(f"[CSMDA] Base {clients_per_cluster_base} clients per cluster. Distributing {extra_clients} extras.")

            for i, cluster_id in enumerate(active_clusters_with_clients):
                cluster_clients = clusters[cluster_id]
                
                # Sort clients within this cluster by their global score
                cluster_clients_sorted = sorted(cluster_clients, 
                                                key=lambda cid: global_scores.get(cid, 0.0), 
                                                reverse=True)
                
                # Determine number to select for this specific cluster
                num_to_select_for_cluster = clients_per_cluster_base
                if i < extra_clients: # Distribute extra clients to the first 'extra_clients' clusters
                    num_to_select_for_cluster += 1
                
                # Ensure we don't try to select more clients than are available in the cluster
                num_to_select_for_cluster = min(num_to_select_for_cluster, len(cluster_clients_sorted))
                
                selected_clients_cids.extend(cluster_clients_sorted[:num_to_select_for_cluster])
                print(f"[CSMDA] Cluster {cluster_id}: Selected {num_to_select_for_cluster}/{len(cluster_clients)} clients.")
        else:
            print("[CSMDA] No active clusters with assigned clients to select from. Falling back to global selection.")

        # 6. Fallback: If not enough clients selected from clusters, pick from best overall remaining
       
        # Final sanity check: Ensure we don't select more clients than min_fit_clients
        # (or max available clients, though this should be handled by prior `min` calls)
        selected_clients_cids = selected_clients_cids[:self.min_fit_clients]
        
        if not selected_clients_cids:
            print(f"[CSMDA] No clients selected for round {server_round}. Returning empty list.")
            return []

        # 7. Prepare FitIns for the chosen clients and update selection counts
        instructions = []
        for client_id in selected_clients_cids:
            client_proxy = client_manager.all()[client_id]
            client_config_for_fit = {"server_round": server_round}
            
            # Pass cluster information if available
            if client_id in self.client_assignments:
                cluster_id = self.client_assignments[client_id]
                cluster_protos = {
                    str(cls): proto.tolist() if isinstance(proto, np.ndarray) else proto
                    for cls, proto in self.cluster_prototypes.get(cluster_id, {}).items()
                }
                client_config_for_fit["cluster_id"] = cluster_id
                #client_config_for_fit["cluster_prototypes"] = cluster_protos
                #client_config_for_fit["cluster_class_counts"] = dict(self.cluster_class_counts.get(cluster_id, {}))

            instructions.append((client_proxy, FitIns(parameters, client_config_for_fit)))
            self.selection_counts[client_id] += 1 # IMPORTANT: Increment selection count for fairness
            
        print(f"[CSMDA] Round {server_round}: Final selected clients: {selected_clients_cids}")
        return instructions

  
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
      

  

