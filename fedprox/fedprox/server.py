from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
#from MulticoreTSNE import print_function
import flwr
import mlflow
from torch.cuda.amp import autocast, GradScaler
import base64
import pickle
import datetime
from collections import defaultdict, deque

from numpy.linalg import norm
from matplotlib import cm
from matplotlib.colors import ListedColormap
from torch.distributions import Dirichlet, Categorical
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from matplotlib import cm
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
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
from pathlib import Path
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
from fedprox.visualizeprototypes import ClusterVisualizationForConfigureFit

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
        batch_size=32,
        ground_truth_stragglers=None,
         total_rounds: int = 15,
           fair_window: int = 50,
         
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
        self.warmup_rounds = 5 # Stage 1 duration
        self.num_clusters = 4
        self.client_assignments = {}  # {client_id: cluster_id}
        self.clustering_interval = 8
        # Simple participation counter
        self.client_participation_count = {}  # client_id -> number of times selected
        self.debug=True
        # Initialize as empty dictionaries
        self.cluster_prototypes = {i: {} for i in range(self.num_clusters)}
        self.cluster_class_counts = {i: defaultdict(int) for i in range(self.num_clusters)}
        map_path="client_id_mapping1.csv"
        
        
        self.theta =0.13         # optional threshold for s_c
        self.use_topk = getattr(self, "use_topk", True)    # prefer Top-K when you know |S_gt|

        self.uuid_to_cid = {}     # {"8325...": "client_0"}
        self.cid_to_uuid = {}     # {"client_0": "8325..."}
        self.ground_truth_cids = set(ground_truth_stragglers)  # {"client_0","client_1",...}
        self.ground_truth_flower_ids = set()  # will be filled as clients appear
        self.all_known_clients = set()
        self.fair_window = int(fair_window)
        self.selection_counts = {}          # lifetime counts per client
        self.sel_window = defaultdict(lambda: deque(maxlen=self.fair_window))  # sliding window 0/1
        self.total_rounds_completed = 0

        # mappings

        # straggler ground truth (fill with your logical ids, e.g., {"client_0", ...})
        self._map_written = False
        
        # CSMDA Client Selection Parameters (UPDATED)
        self.training_times = defaultdict(float)
        
        self.accuracy_history = defaultdict(float)
        self._current_accuracies = {}
       
        self.validation_history = []  # Track predictions vs ground truth per round
        
       
        true_domain_labels = np.array([0]*5 + [1]*5 + [2]*4 + [0]*1)  # Adjust to your setup
        self.visualizer = ClusterVisualizationForConfigureFit(
            save_dir="./clustering_visualizations",
            true_domain_labels=true_domain_labels
        )
        self.virtual_cluster_id = 999
        
        # Tracking
        self.training_times = {}
        self.selection_counts = {}
        self.client_assignments = {}
        self.participated_clients = set()
        self.client_assignments = {}
        self.cluster_prototypes = {}
        self.last_round_participants = set()
        # Virtual cluster configuration
        self.use_virtual_cluster = True  # Enable virtual cluster for never-participated clients
        ema_alpha: float = 0.3  # EMA smoothing for training times
        beta: float = 0.05  # Penalty strength for reliability score
        initial_alpha1: float = 0.6  # Initial reliability weight
        initial_alpha2: float = 0.4  # Initial fairness weight
        phase_threshold: int = 20  # Round to switch weight emphasis
       
        self.total_rounds=total_rounds

        # Store ground truth straggler labels
        self.ground_truth_stragglers = ground_truth_stragglers  # Set of client IDs
        
        # NEW/MODIFIED FAIRNESS ATTRIBUTES
      
        reliability_lambda = 0.05
        acc_drop_threshold  = 0.005
       

        # NEW RELIABILITY ATTRIBUTE
        self.reliability_lambda = reliability_lambda

        self.phase_threshold = 30
        
        # CSMDA Hyperparameters
        self.alpha = 0.3  # EMA decay for training time
        self.epsilon = 0.1  # straggler tolerance (10% of T_max)
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
        #self.total_rounds = total_rounds
        
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
        self.map_path = Path(map_path)
        expected_unique=self.min_fit_clients
        self.expected_unique = expected_unique
        # Track what we've already recorded: (client_cid, flower_node_id)
        self._seen= set()

        # If the CSV already exists, preload seen pairs (so we don't duplicate)
        if self.map_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(self.map_path, dtype=str)
                for _, r in df.iterrows():
                    self._seen.add((str(r["client_cid"]), str(r["flower_node_id"])))
            except Exception:
                pass  # if reading fails, start fresh in memory
         
    def num_evaluate_clients(self, client_manager: ClientManager) -> Tuple[int, int]:
      """Return the sample size and required number of clients for evaluation."""
      num_clients = client_manager.num_available()
      return max(int(num_clients * self.fraction_evaluate), self.min_evaluate_clients), self.min_available_clients
    
   
    def _append_rows(self, rows: List[dict]) -> None:
        if not rows:
            return
        header = ["server_cid", "client_cid", "flower_node_id"]
        write_header = not self.map_path.exists()
        with self.map_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerows(rows)
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
      """
      Aggregate model updates and updat
      """
      if failures:
            print(f"[Round {server_round}] Failures: {len(failures)}")
        
      if not results:
            print(f"[Round {server_round}] No clients returned results. Skipping aggregation.")
            return None, {}
      try:  
        clients_params_list = []
        num_samples_list = []
        current_round_durations = []
        current_participants = set()
        new_rows: List[dict] = []

        # Process results and update tracking
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            metrics = fit_res.metrics
            uuid = client_proxy.cid  # Flower internal UUID            
            cid = metrics.get("client_cid")
            node = metrics.get("flower_node_id")
            self.uuid_to_cid[uuid] = cid
            self.cid_to_uuid[cid] = uuid

            print(f'===client id: {cid} and flower id {uuid} and node :{node} ===')

           
            if client_id not in self.client_participation_count:
              self.client_participation_count[client_id] = 0
            self.client_participation_count[client_id] += 1
            
            self.participated_clients.add(client_id)
            current_participants.add(client_id)
            
            # Update EMA training time - Equation (4)
          
            metrics = fit_res.metrics or {}
            if "duration" not in metrics:
                  continue
            dur = float(metrics["duration"])
            prev = self.training_times.get(uuid)
            if prev is None:
                  ema = dur
                  print(f"[EMA Init] {uuid}: T_c(0) = {dur:.2f}s")
            else:
                ema = self.alpha * dur + (1.0 - self.alpha) * prev
                print(f"[EMA Update] {uuid}: {prev:.2f}s → {ema:.2f}s (raw: {dur:.2f}s)")
            self.training_times[uuid] = ema

           
            current_round_durations.append(dur)
            if cid is None or node is None:
                continue
            key = (str(int(cid)), str(node))
            if key not in self._seen:
                self._seen.add(key)
                new_rows.append({
                    "server_cid": client_proxy.cid,        # connection id for reference
                    "client_cid": key[0],
                    "flower_node_id": key[1],
                })
            self._append_rows(new_rows)
            if new_rows:
              print(f"[Server] Recorded {len(new_rows)} new client(s). Total unique: {len(self._seen)}")
            
            
            # Collect parameters for aggregation
            clients_params_list.append(parameters_to_ndarrays(fit_res.parameters))
            num_samples_list.append(fit_res.num_examples)
            '''
            '''
            # The client should report its logical id once in fit metrics
            logical = fit_res.metrics.get("logical_id") if fit_res.metrics else None
            print(f"[Mapping]rtertr ====logical_id={logical} and {self.uuid_to_cid}")
            print(f"[Mapping]ddd ====: {self.cid_to_uuid}")

        
           
        self.last_round_participants = current_participants
        self.total_rounds_completed = server_round

        

        self._validate_straggler_predictions(server_round, results)
        
        print(f"\n[Round {server_round}] Participants: {list(current_participants)}")
        print(f"[Round {server_round}] Average raw training time: {np.mean(current_round_durations):.2f}s")
        
        # Perform FedAvg aggregation
        aggregated_params = self._fedavg_parameters(clients_params_list, num_samples_list)
        
        if server_round == self.total_rounds :
            self.save_client_mapping()
            print("\n" + "="*80)
            print(f"[Round {server_round}] TRAINING COMPLETED - Auto-saving results...")
            print("="*80)
            self._save_all_results()
        return ndarrays_to_parameters(aggregated_params), {}

      except Exception as e:
        print(f"[aggregate_fit] Error processing client {getattr(client_proxy,'cid','?')}: {e}")
        # continue to next client so we still reach the mapping update
        
    # ---- in your Strategy class ----

    def _predict_stragglers_from_score(self, T_max, client_ids):
      """Return set of predicted stragglers using s_c=1-As."""
      # compute scores for current participants only
      scores = {}
      for cid in client_ids:
        T_c = self.training_times.get(cid, 0.0)
        As = T_max / (T_c + self.beta * T_max) if (T_c > 0 and T_max > 0) else 0.0
        s_c = 1.0 - As
        scores[cid] = s_c
      """
      if self.use_topk:
        # Predict exactly as many as we injected (good for clean evaluation)
        k = len(self.ground_truth_flower_ids)  # see mapping below
        # sort by highest score (slowest)
        predicted = set(sorted(scores, key=scores.get, reverse=True)[:k])
      else:
      """
      # Thresholded prediction
      predicted = {cid for cid, s in scores.items() if s >= self.theta}
      return predicted, scores

    def save_client_mapping(self):

      df = pd.DataFrame([
    {"flower_uuid": uuid, "client_cid": cid}
    for uuid, cid in self.uuid_to_cid.items()
])

      df.to_csv("client_id_mapping.csv", index=False)
      print("Saved mapping at:")

      print(df)

    def _save_all_results(self):
      
        self.save_participation_stats()
        self.visualize_client_participation(self.client_participation_count, save_path="participation_chart.png", 
                                )
        self.save_validation_results()
    def _norm(self,label: str) -> str:
      s = str(label).strip()
      return s.replace("client_", "")   # "client_0" -> "0"
    def _validate_straggler_predictions(self, server_round, results):
      # participants
      participants, round_dur = [], {}
      for client_proxy, fit_res in results:
        uuid = client_proxy.cid
        participants.append(uuid)
        if "duration" in fit_res.metrics:
            round_dur[uuid] = float(fit_res.metrics["duration"])

      # compute T_max from EMAs (assume you already updated EMA this round)
      valid_times = [t for t in self.training_times.values() if t is not None]
      if not valid_times:
        return
      T_max = float(np.mean(valid_times))

      # predict (your existing code)
      predicted_set, scores = self._predict_stragglers_from_score(T_max, participants)

      # robust ground-truth check: UUID OR logical label
      gt_uuid_set = self.cid_to_uuid    # UUIDs
      
      gt_logical_set = self.ground_truth_cids             # {"client_0","client_1",...}
      gt_idx_set = {
    int(cid.split("_", 1)[1])
    for cid in gt_logical_set
    if cid.startswith("client_") and cid.split("_", 1)[1].isdigit()
}  
    
      for uuid in participants:
          val = self.uuid_to_cid.get(uuid)  # could be "0" or 0 or None
          try:
            logical_idx = int(val) if val is not None else None
          except (TypeError, ValueError):
            logical_idx = None

          is_gt = (logical_idx is not None) and (logical_idx in gt_idx_set)
          print(f'===== {is_gt} and {self._norm(logical_idx)} and {gt_logical_set}')
          print(f'===== {gt_uuid_set} and {uuid} and {gt_idx_set}')

          rec = {
            "round": server_round,
            "client_id": uuid,
            "logical_id": logical_idx,
            "T_c": self.training_times.get(uuid, float("nan")),
            "T_max": T_max,
            "s_c": scores.get(uuid, float("nan")),
            "actual_duration": round_dur.get(uuid, float("nan")),
            "predicted_straggler": uuid in predicted_set,
            "ground_truth_straggler": is_gt,                         # <-- now correct
        }
          rec["prediction_type"] = self._classify_prediction(rec["predicted_straggler"], rec["ground_truth_straggler"])
          self.validation_history.append(rec)

            
    #strqgglers 


    def _observe_mapping(self, results):
      "Capture mappings from Flower UUID to your logical id when available."
      for client_proxy, fit_res in results:
        uuid = client_proxy.cid
        # Expect client to report its logical id in metrics once (optional, else skip)
        logical = fit_res.metrics.get("logical_id") if "logical_id" in fit_res.metrics else None
        if logical:
            if uuid not in self.uuid_to_cid:
                self.uuid_to_cid[uuid] = logical
                self.cid_to_uuid[logical] = uuid
      # Refresh gt UUID set if we can resolve some cids now
      newly_resolved = {self.cid_to_uuid[c] for c in self.ground_truth_cids if c in self.cid_to_uuid}
      self.ground_truth_flower_ids |= newly_resolved

    def _on_round_end_update_mapping(self, server_round, results):
      self._observe_mapping(results)
      # fall back: if your logical labels already equal Flower ids, this still works
      if not self.ground_truth_flower_ids:
        # if user provided UUIDs directly in ground_truth_cids
        self.ground_truth_flower_ids = set(self.ground_truth_cids)

    def _classify_prediction(self, predicted, actual):
        """Classify prediction type for confusion matrix"""
        if predicted and actual:
            return 'True Positive'  # Correctly identified straggler
        elif not predicted and not actual:
            return 'True Negative'  # Correctly identified fast client
        elif predicted and not actual:
            return 'False Positive'  # Wrongly labeled fast client as straggler
        else:  # not predicted and actual
            return 'False Negative'  # Missed a straggler
    
    def save_validation_results(self, filename="validation_results.csv"):
        """Save validation results"""
        
        
        df = pd.DataFrame(self.validation_history)
        df.to_csv(filename, index=False)
        print(f"Validation results saved to {filename}")
        return df

    
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
            
            log_filename = "server_accuracy_log2.csv"
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
   
  

    
    # ---------------------------------------------------------------------
    # Reliability score (more robust than mean-based)
    # A_s[c] = T_ref / (T_c + β * T_ref), ref = 70th percentile of EMAs
    # ---------------------------------------------------------------------
    def compute_reliability_scores(self, client_ids: List[str]) -> Dict[str, float]:
        reliability_scores: Dict[str, float] = {}

        valid_times = [
            self.training_times[cid]
            for cid in client_ids
            if cid in self.training_times and self.training_times[cid] > 0.0
        ]

        if not valid_times:
            if self.debug:
                print("[Reliability] Warning: No valid training times available")
            return {cid: 0.5 for cid in client_ids}

        T_ref = float(np.percentile(valid_times, 70))  # robust reference
        if self.debug:
            print(f"\n[Reliability Scores] Round {self.total_rounds_completed}")
            print(f"  T_ref (70th pct EMA): {T_ref:.2f}s, β: {self.beta}  (β*T_ref={self.beta*T_ref:.2f})")

        # Precompute median A for clipping bound
        As_tmp = [T_ref / (t + self.beta * T_ref) for t in valid_times]
        medA = float(np.median(As_tmp)) if As_tmp else 0.5
        clip_hi = max(1.0, 2.0 * medA)

        for cid in client_ids:
            T_c = float(self.training_times.get(cid, T_ref))
            denom = T_c + (self.beta * T_ref)
            A = T_ref / max(1e-9, denom)
            reliability_scores[cid] = float(np.clip(A, 0.0, clip_hi))
            if self.debug:
                print(f"  Client {cid}: T_c={T_c:.2f}s, A_s={reliability_scores[cid]:.4f}")

        return reliability_scores

    # ---------------------------------------------------------------------
    # Sliding-window, per-cluster fairness (deficit-based, bounded)
    # E = W_eff * (Kk / nk); deficit = max(0, E - vW); fs = sigmoid(kappa * deficit/E)
    # ---------------------------------------------------------------------
    def _fairness_scores_cluster(self, client_ids_k: List[str], Kk: int, nk: int) -> Dict[str, float]:
        if nk <= 0 or Kk <= 0 or not client_ids_k:
            return {cid: 0.5 for cid in client_ids_k}

        W_eff = max(1, min(self.fair_window, self.total_rounds_completed))
        vW = np.array([sum(self.sel_window.get(cid, [])) for cid in client_ids_k], dtype=float)
        E = W_eff * (Kk / max(1, nk))

        deficit = np.maximum(0.0, E - vW)
        ratio = deficit / (E + 1e-6)  # 0..1-ish
        kappa, fmax = 4.0, 0.7
        fs = 1.0 / (1.0 + np.exp(-kappa * ratio))
        fs = np.minimum(fs, fmax)
        fairness_scores = {cid: float(fs_i) for cid, fs_i in zip(client_ids_k, fs)}

        if self.debug:
            print(f"  [Fairness/cluster] W_eff={W_eff}, E={E:.2f}, Kk={Kk}, nk={nk}")
            for cid, v, f in zip(client_ids_k, vW, fs):
                print(f"    {cid}: vW={v:.0f}, fs={float(f):.3f}")

        return fairness_scores
     # ---------------------------------------------------------------------
    # Stochastic selection with temperature, exploration, recent-cap
    # ---------------------------------------------------------------------
    def _sample_probabilistic(self,
                              cids: List[str],
                              score_map: Dict[str, float],
                              K: int,
                              temp: float = 0.6,
                              p_rand: float = 0.10,
                              recent_cap: int = 3) -> List[str]:
        if K <= 0 or not cids:
            return []
        scores = np.array([float(score_map.get(cid, 0.0)) for cid in cids], dtype=float)

        # recent participation within sliding window
        recent = np.array([sum(self.sel_window.get(cid, [])) for cid in cids], dtype=float)
        ok = recent <= float(recent_cap)
        scores[~ok] = -np.inf  # temporarily ineligible

        if np.any(np.isfinite(scores)):
            s = scores - np.nanmax(scores[np.isfinite(scores)])
        else:
            s = scores

        probs = np.exp(np.where(np.isfinite(s), s, -1e9) / max(1e-6, temp))
        if probs.sum() <= 0:
            probs = np.ones_like(probs)
        probs /= probs.sum()

        idx = np.arange(len(cids))
        n_rand = min(int(round(p_rand * K)), len(cids))
        take_rand = np.random.choice(idx, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=int)

        remaining = max(0, K - len(take_rand))
        pool = np.setdiff1d(idx, take_rand, assume_unique=False)
        if len(pool) == 0:
            chosen = take_rand
        else:
            probs_pool = probs[pool] / probs[pool].sum()
            take_prob = np.random.choice(pool, size=min(remaining, len(pool)), replace=False, p=probs_pool)
            chosen = np.concatenate([take_rand, take_prob])

        return [cids[i] for i in chosen]

    # ---------------------------------------------------------------------
    # Update sliding windows once per round
    # ---------------------------------------------------------------------
    def _update_selection_windows(self, selected_cids: List[str]) -> None:
        touched = set(selected_cids)
        for cid in self.all_known_clients:
            self.sel_window[cid].append(1 if cid in touched else 0)
    
    # ---------------------------------------------------------------------
    # (Optional) Legacy global score (kept for compatibility; now unused)
    # ---------------------------------------------------------------------
    def compute_global_selection_scores(self, client_ids: List[str], server_round: int) -> Dict[str, float]:
        # kept only if you call it elsewhere; not used in per-cluster path
        reliability_scores = self.compute_reliability_scores(client_ids)
        # simple neutral fairness placeholder (not used in new path)
        fairness_scores = {cid: 0.5 for cid in client_ids}
        a1, a2 = self._adapt_weights(server_round)
        final_scores = {cid: float(a1 * reliability_scores.get(cid, 0.0) + a2 * fairness_scores.get(cid, 0.0))
                        for cid in client_ids}
        return final_scores


    def compute_global_selection_scores(
        self, 
        client_ids: List[str], 
        server_round: int
    ) -> Dict[str, float]:
        
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
    
    
    # Schedulers (you can keep yours; here’s a stable, paper-friendly one)
    # ---------------------------------------------------------------------
    def _adapt_weights(self, server_round: int) -> Tuple[float, float]:
        """Reliability/Fairness schedule: early stable, mid coverage, late stable."""
        t = server_round / max(1, self.total_rounds)  # [0,1]
        if t < 0.4:     # early
            a1, a2 = 0.8, 0.2
        elif t < 0.8:   # mid
            a1, a2 = 0.6, 0.4
        else:           # late
            a1, a2 = 0.7, 0.3
        s = a1 + a2
        return a1 / s, a2 / s
       
       
    # Robust normalization helper
    # ---------------------------------------------------------------------
    def _robust01(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.full_like(x, 0.5, dtype=float)
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo)
    
    def select_clients_from_cluster(
        self,
        cluster_clients: List[str],
        num_clients_to_select: int,
        server_round: int
    ) -> List[str]:
       
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
    
      
  
    def _flatten_prototypes(self, proto_dict: dict) -> np.ndarray:
      """
    Turn a dict {class_id: feature_vector} into a single fixed-length vector.
    We concatenate class-wise prototypes in sorted class order. If you prefer
    averaging, replace with: return np.mean(np.stack(list(proto_dict.values())), axis=0)
      """
      # Sort by class key to keep a consistent layout
      keys = sorted(proto_dict.keys())
      vecs = [np.asarray(proto_dict[k]).ravel() for k in keys if proto_dict[k] is not None]
      if not vecs:
        return np.zeros((1,), dtype=float)
      return np.concatenate(vecs, axis=0)


    def _prepare_tsne_inputs(self, all_prototypes_list: list[dict], all_client_ids: list[str]) -> tuple[np.ndarray, list[str]]:
      """
      Map each client's prototype dict -> 1D vector so t-SNE can run on a matrix [n_clients x d_flat].
      Returns (X, ids) where X is float32 matrix for t-SNE and ids are aligned client ids.
      """
      X = [self._flatten_prototypes(p) for p in all_prototypes_list]
      # pad to same length (for safety if some clients miss classes)
      maxd = max(v.shape[0] for v in X)
      Xp = [np.pad(v, (0, maxd - v.shape[0]), mode="constant") if v.shape[0] < maxd else v for v in X]
      return np.asarray(Xp, dtype=np.float32), list(all_client_ids)
    
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
      true_domain_labels = None
    
      if true_domain_map:
        # Convert true_domain_map to labels for metrics
        true_domain_labels = []
        unique_domains = set()
        
        for cid in client_ids:
            if isinstance(true_domain_map, dict):
                # Domain tracker dict
                domain = true_domain_map.get(cid, "unknown")
            else:
                # Array or other
                domain = true_domain_map.get_domain(cid) if hasattr(true_domain_map, 'get_domain') else 0
            
            true_domain_labels.append(domain)
            unique_domains.add(domain)
        
        unique_domains = sorted(unique_domains)
        domain_to_marker = {dom: markers[i % len(markers)] for i, dom in enumerate(unique_domains)}
    
      # 6. Begin plotting
      plt.figure(figsize=(14, 10))
    
      for i, (x, y) in enumerate(projections):
        client_id = client_ids[i]
        cluster_id = cluster_assignments[i]
        color_index = cluster_id_to_color_index[cluster_id]
        
        if true_domain_map and true_domain_labels:
            domain = true_domain_labels[i]
            marker = domain_to_marker.get(domain, 'o')
        else:
            domain = "unknown"
            marker = 'o'
        
        plt.scatter(
            x, y,
            c=[colors[color_index]],
            marker=marker,
            edgecolor='k',
            linewidth=1.5,
            s=150,
            alpha=0.8
        )
        
        # Add client ID label
        plt.text(x, y+2, str(client_id)[:8], fontsize=8, ha='center', va='bottom')
    
      # 7. Legends
      cluster_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cid}',
                   markerfacecolor=colors[idx], markersize=10, markeredgecolor='k')
        for cid, idx in sorted(cluster_id_to_color_index.items())
    ]
    
      domain_handles = []
      if true_domain_map and true_domain_labels:
        for dom, marker in sorted(domain_to_marker.items()):
            domain_handles.append(
                plt.Line2D([0], [0], marker=marker, color='k', label=f'Domain: {dom}',
                           markerfacecolor='gray', markersize=10, linestyle='None', markeredgecolor='k')
            )
    
      plt.legend(
        handles=cluster_handles + domain_handles,
        title="Clusters (Color) / Domains (Shape)",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10
    )
      ###
      # 8. Plot aesthetics
      plt.title(
        f"Client Prototypes Clustering - Round {server_round}\n"
        f"Colors=Predicted Clusters, Shapes=True Domains",
        fontsize=14,
        fontweight='bold'
    )
      plt.xlabel("t-SNE Dimension 1", fontsize=12)
      plt.ylabel("t-SNE Dimension 2", fontsize=12)
      plt.grid(True, alpha=0.3)
      plt.tight_layout()
    
      # Save figure
      output_dir = Path("clustering_visualizations")
      output_dir.mkdir(exist_ok=True)
      plt.savefig(output_dir / f"clusters_round_{server_round}.png", dpi=300, bbox_inches='tight')
      print(f"[Visualization] Saved: clustering_visualizations/clusters_round_{server_round}.png")
    
      plt.close()
    
      # 9. Clustering quality metrics
      if true_domain_map and true_domain_labels:
        try:
            ari = adjusted_rand_score(true_domain_labels, cluster_assignments)
            nmi = normalized_mutual_info_score(true_domain_labels, cluster_assignments)
            
            print(f"\n[Round {server_round}] Clustering Quality Metrics:")
            print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
            print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
    
    def visualize_client_participation(self, participation_dict, save_path="participation_chart.png", 
                                   method_name="FedProto-Fair"):

      # ✅ Load UUID → cid mapping
      mapping_df = pd.read_csv("client_id_mapping1.csv")
      uuid_to_cid = dict(zip(mapping_df["flower_node_id"].astype(str),
                           mapping_df["client_cid"].astype(str)))

      # ✅ Convert participation_dict keys using mapping
      mapped_dict = {}
      for uuid, count in participation_dict.items():
        uuid_str = str(uuid)
        cid = uuid_to_cid.get(uuid_str, f"UNK-{uuid}")  # fallback: unknown
        print('======{cid}====')
        mapped_dict[cid] = count

      # ✅ Sort clients by numeric cid
      sorted_items = sorted(mapped_dict.items(), key=lambda x: int(x[0]))
      client_ids = [f"Client {item[0]}" for item in sorted_items]
      counts = [item[1] for item in sorted_items]

      # ✅ Plot exactly same as before (using client_ids now)
      fig, ax = plt.subplots(figsize=(14, 6))
      bars = ax.bar(range(len(client_ids)), counts)

      for i, count in enumerate(counts):
        if count == 0:
            bars[i].set_color('red')
            bars[i].set_alpha(0.5)

      ax.set_xlabel('Client ID', fontsize=12, fontweight='bold')
      ax.set_ylabel('Number of Participations', fontsize=12, fontweight='bold')
      ax.set_title(f'Client Participation Distribution - {method_name}', fontsize=14, fontweight='bold')
      ax.set_xticks(range(len(client_ids)))
      ax.set_xticklabels(client_ids, rotation=45, ha='right')
      ax.grid(axis='y', alpha=0.3, linestyle='--')

      total_clients = len(client_ids)
      participated = sum(1 for c in counts if c > 0)
      avg_participation = np.mean(counts)
      std_participation = np.std(counts)

      stats_text = f"Total Clients: {total_clients}\nParticipated: {participated} ({participated/total_clients*100:.1f}%)\nAvg Participation: {avg_participation:.2f} ± {std_participation:.2f}"
      ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

      plt.tight_layout()
      plt.savefig(save_path, dpi=300, bbox_inches='tight')
      print(f"Visualization saved to {save_path}")
      plt.show()


    
 
     
    # ---------------------------------------------------------------------
    # CONFIGURE_FIT: integrated with per-cluster fairness & stochastic selection
    # ---------------------------------------------------------------------
    def configure_fit(self,
                      server_round: int,
                      parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:

        print(f"\n{'='*80}")
        print(f"[Round {server_round}] TWO-STAGE RESOURCE-AWARE FAIR SELECTION")
        print(f"{'='*80}")

        # All available clients
        all_clients: Dict[str, ClientProxy] = client_manager.all()
        available_client_cids = list(all_clients.keys())
        if not available_client_cids:
            print(f"[Round {server_round}] No clients available.")
            return []

        # Track "known" clients for window bookkeeping
        for cid in available_client_cids:
            self.all_known_clients.add(cid)

        print(f"\n[Client Status]")
        print(f"  Total available clients: {len(available_client_cids)}")
        print(f"  Previously participated: {len(self.participated_clients)}")

        participated_available = [cid for cid in available_client_cids if cid in self.participated_clients]
        never_participated = [cid for cid in available_client_cids if cid not in self.participated_clients]
        print(f"  Available participated clients: {len(participated_available)}")
        print(f"  Available never-participated clients: {len(never_participated)}")

        # Stage decision
        in_warmup_phase = server_round <= self.warmup_rounds
        if in_warmup_phase:
            print(f"\n[STAGE 1: WARMUP PHASE] Round {server_round}/{self.warmup_rounds} | unified pool")
        else:
            print(f"\n[STAGE 2: DOMAIN-AWARE PHASE] clustering enabled periodically")

        # ----------------------- PHASE 1: CLUSTERING ----------------------
        clusters: Dict[int, List[str]] = defaultdict(list)
        should_cluster = (server_round > self.warmup_rounds) and (self.clustering_interval > 0) \
                         and (server_round % self.clustering_interval == 0)

        if should_cluster and participated_available:
            print(f"\n{'─'*80}")
            print(f"[Clustering Round] Collecting prototypes from participated clients (round {server_round})")

            all_prototypes_list, all_client_ids, class_counts_list, clients_with_prototypes, domains_ids = [], [], [], [], []

            for cid in participated_available:
                client_proxy: ClientProxy = all_clients[cid]
                try:
                    get_protos_res = client_proxy.get_properties(ins=None, timeout=15.0, group_id=None)
                    prototypes_encoded = get_protos_res.properties.get("prototypes")
                    class_counts_encoded = get_protos_res.properties.get("class_counts")
                    domain_id = int(get_protos_res.properties.get("domain_id", -1))

                    if prototypes_encoded and class_counts_encoded:
                        try:
                            protos = pickle.loads(base64.b64decode(prototypes_encoded))
                            cls_counts = pickle.loads(base64.b64decode(class_counts_encoded))
                            if isinstance(protos, dict) and isinstance(cls_counts, dict):
                                all_prototypes_list.append(protos)
                                class_counts_list.append(cls_counts)
                                all_client_ids.append(cid)
                                clients_with_prototypes.append(cid)
                                domains_ids.append(domain_id)
                                if self.debug:
                                    print(f"  ✓ Client {cid}: Prototypes collected")
                        except Exception as dec_err:
                            print(f"  ✗ Client {cid}: Decode error - {dec_err}")
                    else:
                        print(f"  ⚠ Client {cid}: No prototypes available")
                except Exception as e:
                    print(f"  ⚠ Client {cid}: Communication failed - {e}")

            print(f"\n[Prototype Collection] {len(clients_with_prototypes)}/{len(participated_available)} successful")

            if len(clients_with_prototypes) >= self.num_clusters:
                print(f"\n[EM Clustering] Processing {len(clients_with_prototypes)} clients...")
                if not self.cluster_prototypes:
                    print("  Initializing cluster prototypes with k-means++...")
                    self.cluster_prototypes = self._initialize_clusters(all_prototypes_list)  # your impl

                # E & M steps
                global_assignments = self._e_step(all_prototypes_list, all_client_ids)     # your impl
                self.cluster_prototypes = self._m_step(all_prototypes_list, all_client_ids,
                                                       global_assignments, class_counts_list)  # your impl

                for client_id, cluster_id in global_assignments.items():
                    self.client_assignments[client_id] = cluster_id

                print(f"\n[Clustering Results]")
                for k in range(self.num_clusters):
                    cids_k = [cid for cid, cl in self.client_assignments.items() if cl == k]
                    if cids_k:
                        print(f"  Cluster {k}: {len(cids_k)} clients")

                # optional viz
                # === VISUALIZATION ===
            try:
                if len(all_prototypes_list) >= self.num_clusters:
                # Build per-client cluster labels aligned with all_client_ids
                  cluster_ids = [int(global_assignments.get(cid, -1)) for cid in all_client_ids]

                  self._visualize_clusters(
            prototypes=all_prototypes_list,      # or embeddings=X
            client_ids=all_client_ids,
            server_round=server_round,
            cluster_ids=cluster_ids,             # <-- pass cluster assignments for coloring
            true_domain_map=None                 # optionally pass ground-truth domains if you have them
        )
            except Exception as viz_err:
                if self.debug:
                    print(f"[Viz] Skipped t-SNE plot: {viz_err}")
                else:
                    print(f"\n[Clustering Skipped] Need {self.num_clusters} clients with prototypes,"
                      f" have {len(clients_with_prototypes)}")

        # --------------------- PHASE 2: ORGANIZE CLIENTS -------------------
        if in_warmup_phase or not self.client_assignments:
            print(f"\n[Client Organization] UNIFIED POOL MODE")
            clusters[0] = participated_available + never_participated
            print(f"  Single pool: {len(clusters[0])} clients")
        else:
            print(f"\n[Client Organization] DOMAIN-AWARE MODE")
            for cid in participated_available:
                if cid in self.client_assignments:
                    clusters[self.client_assignments[cid]].append(cid)
                else:
                    clusters[0].append(cid)

            # virtual cluster for never-participated (decay late)
            if never_participated and self.use_virtual_cluster:
                progress = server_round / max(1, self.total_rounds)
                if progress <= 0.8:  # suppress late to avoid noise
                    clusters[self.virtual_cluster_id] = never_participated
                    print(f"  Virtual Cluster {self.virtual_cluster_id}: {len(never_participated)} new clients")

            for k in sorted(clusters.keys()):
                ctype = "Virtual" if k == self.virtual_cluster_id else "Domain"
                print(f"  Cluster {k} [{ctype}]: {len(clusters[k])} clients")

        print(f"\n[Active Clusters] {len(clusters)} cluster(s)")

        # ------------------ PHASE 3: DISTRIBUTE BUDGET ---------------------
        print(f"\n{'─'*80}")
        print(f"[Selection Distribution]")
        print(f"{'─'*80}")

        if not clusters:
            print("No clusters available")
            return []

        total_clusters = len(clusters)
        cluster_allocations: Dict[int, int] = {}

        if in_warmup_phase or total_clusters == 1:
            # single pool
            only_id = list(clusters.keys())[0]
            cluster_allocations = {only_id: self.min_fit_clients}
            print(f"Unified pool allocation: {self.min_fit_clients} clients")
        else:
            base_per = max(1, self.min_fit_clients // total_clusters)
            remaining = self.min_fit_clients - base_per * total_clusters
            for cid_k in clusters.keys():
                cluster_allocations[cid_k] = base_per

            if remaining > 0:
                sizes = {cid_k: len(v) for cid_k, v in clusters.items()}
                total_size = sum(sizes.values()) if sizes else 1
                # distribute remaining proportionally by size
                for cid_k in sorted(clusters.keys(), key=lambda x: sizes[x], reverse=True):
                    if remaining <= 0:
                        break
                    prop = sizes[cid_k] / total_size if total_size > 0 else 0
                    extra = min(remaining, max(1, int(round(prop * remaining))))
                    cluster_allocations[cid_k] += extra
                    remaining -= extra

            # Final normalization to ensure exact budget:
            # If over-allocated due to rounding, trim from largest clusters
            alloc_sum = sum(cluster_allocations.values())
            while alloc_sum > self.min_fit_clients:
                # remove one from the currently largest allocation with available clients
                cid_k = max(cluster_allocations, key=cluster_allocations.get)
                if cluster_allocations[cid_k] > 0:
                    cluster_allocations[cid_k] -= 1
                    alloc_sum -= 1
                else:
                    break

            print(f"\nFinal allocations:")
            for cid_k, alloc in sorted(cluster_allocations.items()):
                ctype = "Virtual" if cid_k == self.virtual_cluster_id else "Domain"
                print(f"  Cluster {cid_k} [{ctype}]: {alloc} clients")

        # ------------------ PHASE 4 & 5: SCORE + SELECT --------------------
        print(f"\n{'─'*80}")
        print(f"[Client Selection]")
        print(f"{'─'*80}")

        selected_clients_cids: List[str] = []
        a1, a2 = self._adapt_weights(server_round)
        print(f"Weights: α₁(reliability)={a1:.2f}, α₂(fairness)={a2:.2f}")

        for k in sorted(clusters.keys()):
            cids_k = clusters[k]
            Kk = int(cluster_allocations.get(k, 0))
            if Kk <= 0 or len(cids_k) == 0:
                continue

            ctype = "Unified Pool" if in_warmup_phase else ("Virtual" if k == self.virtual_cluster_id else "Domain")
            print(f"\n[Cluster {k} - type: {ctype}]")

            # per-cluster fairness (windowed)
            F_map = self._fairness_scores_cluster(cids_k, Kk, len(cids_k))

            # reliability for those clients
            R_map = self.compute_reliability_scores(cids_k)

            # normalize per round (scale-compatibility)
            R = np.array([R_map.get(cid, 0.0) for cid in cids_k], dtype=float)
            F = np.array([F_map.get(cid, 0.0) for cid in cids_k], dtype=float)
            Rn, Fn = self._robust01(R), self._robust01(F)

            score_vals = a1 * Rn + a2 * Fn
            score_map = {cid: float(s) for cid, s in zip(cids_k, score_vals)}

            # stochastic selection (prevents lock-in)
            choose = self._sample_probabilistic(
                cids=cids_k, score_map=score_map, K=Kk, temp=0.6, p_rand=0.10, recent_cap=3
            )
            selected_clients_cids.extend(choose)

            print(f"Selected {len(choose)}/{len(cids_k)} clients")
            for i, cid in enumerate(choose[:5]):
                r = R_map.get(cid, 0.0); f = F_map.get(cid, 0.0); s = score_map.get(cid, 0.0)
                status = "NEW" if cid not in self.participated_clients else "participated"
                sel = self.selection_counts.get(cid, 0)
                print(f"    {i+1}. {cid:20s} [{status:12s}] Score={s:.4f} (R={r:.3f}, F={f:.3f}), Selected={sel}x")

        # DO NOT truncate globally; allocations already ensure exact budget.

        # ------------------ PHASE 6: PREPARE INSTRUCTIONS ------------------
        instructions: List[Tuple[ClientProxy, FitIns]] = []
        for cid in selected_clients_cids:
            if cid in all_clients:
                client_proxy = all_clients[cid]
                client_config = {
                    "server_round": server_round,
                    "total_rounds": getattr(self, "total_rounds", 100),

                    # If you simulate stragglers, avoid 100% probability:
                    "simulate_stragglers": "0,1",
                    "delay_base_sec": 5.0,        # milder defaults
                    "delay_jitter_sec": 2.0,
                    "delay_prob": 0.4,            # not everyone is a straggler
                }
                instructions.append((client_proxy, FitIns(parameters, client_config)))

                # lifetime selection count
                self.selection_counts[cid] += 1
                self.participated_clients.add(cid)

        # --------------------- FINAL SUMMARY + BOOKKEEPING -----------------
        print(f"\n{'='*80}")
        print(f"[Round {server_round}] SELECTION SUMMARY")
        print(f"{'='*80}")

        stage_name = "WARMUP" if in_warmup_phase else "DOMAIN-AWARE"
        print(f"Stage: {stage_name}")
        print(f"Total selected: {len(instructions)} clients")

        if not in_warmup_phase:
            regular_selected = sum(1 for cid in selected_clients_cids if cid in self.participated_clients)
            new_selected = sum(1 for cid in selected_clients_cids if cid not in self.participated_clients)
            print(f"  From domain clusters: {regular_selected} clients")
            if self.use_virtual_cluster and self.virtual_cluster_id in clusters:
                print(f"  From virtual cluster: {new_selected} clients")

        print(f"\nSelection frequency (top 10):")
        top_selected = sorted(self.selection_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for cid, count in top_selected:
            print(f"  {cid:20s}: {count}x")
        print(f"{'='*80}\n")

        # update sliding windows + round count
        self._update_selection_windows(selected_clients_cids)
        self.total_rounds_completed += 1

        return instructions

    def save_participation_stats(self, filename="client_participation.csv"):
        """Save participation statistics at the end of training"""
        import pandas as pd
        
        # Create dataframe
        data = []
        for client_id, count in self.client_participation_count.items():
            data.append({
                'client_id': client_id,
                'participation_count': count,
                'participation_rate': count / self.total_rounds_completed
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('participation_count', ascending=False)
        df.to_csv(filename, index=False)
        print(f"Participation stats saved to {filename}")
        return df

 
    def _initialize_clusters(self, prototypes_list):
      """Initialize cluster prototypes using k-means++ style initialization"""
      import numpy as np
    
      # Convert prototypes to vectors
      proto_vectors = []
      for prototypes in prototypes_list:
        all_protos = []
        for class_id in sorted(prototypes.keys()):
            all_protos.append(prototypes[class_id])
        if all_protos:
            proto_vectors.append(np.concatenate(all_protos))
    
      if not proto_vectors:
        return {}
      #nns
      proto_array = np.array(proto_vectors)
      n_samples = len(proto_array)
    
      # k-means++ initialization
      centers_idx = [np.random.randint(n_samples)]
    
      for _ in range(min(self.num_clusters - 1, n_samples - 1)):
        distances = np.array([
            min(np.linalg.norm(proto_array[i] - proto_array[c]) 
                for c in centers_idx)
            for i in range(n_samples)
        ])
        
        if distances.sum() == 0:
            break
            
        probs = distances / distances.sum()
        next_center = np.random.choice(n_samples, p=probs)
        centers_idx.append(next_center)
    
      # Convert back to prototype format
      cluster_prototypes = {}
      for k, idx in enumerate(centers_idx):
        cluster_prototypes[k] = prototypes_list[idx]
    
      return cluster_prototypes


    def _e_step(self, all_prototypes, client_ids):
      """E-step: Assign clients to clusters based on prototype similarity"""
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
                    
                    dist = self._cosine_distance(client_proto, cluster_proto)
                    total_dist += dist
                    shared_classes += 1

            avg_dist = total_dist / shared_classes if shared_classes > 0 else 1.0

            if avg_dist < min_dist:
                min_dist = avg_dist
                best_cluster = cluster_id

        assignments[client_id] = best_cluster

      # Log cluster distribution
      cluster_counts = defaultdict(int)
      for cluster_id in assignments.values():
        cluster_counts[cluster_id] += 1
    
      print(f"[E-step] Distribution: {dict(cluster_counts)}")
      return assignments

    def _cosine_distance(self, a, b):
      """Compute 1 - cosine similarity"""
      norm_a = np.linalg.norm(a)
      norm_b = np.linalg.norm(b)
    
      if norm_a == 0 or norm_b == 0:
        return 1.0
    
      return 1 - np.dot(a, b) / (norm_a * norm_b)


    def _m_step(self, all_prototypes, client_ids, assignments, class_counts_list):
      """M-step: Update cluster prototypes with weighted averaging"""
    
      # Get sample prototype for shape
      sample_proto = None
      for prototypes in all_prototypes:
        if prototypes:
            sample_proto = next(iter(prototypes.values()))
            break
    
      if sample_proto is None:
        return defaultdict(dict)
    
      # Convert to numpy if needed
      if hasattr(sample_proto, 'numpy'):
        sample_proto = sample_proto.numpy()
      elif hasattr(sample_proto, 'detach'):
        sample_proto = sample_proto.detach().cpu().numpy()
    
      # Initialize accumulators
      cluster_weighted_sum = defaultdict(lambda: defaultdict(
        lambda: np.zeros(sample_proto.shape, dtype=np.float32)
    ))
      cluster_class_counts = defaultdict(lambda: defaultdict(int))

      # Accumulate weighted prototypes
      for i, (client_id, prototypes) in enumerate(zip(client_ids, all_prototypes)):
        cluster_id = assignments[client_id]
        class_counts = class_counts_list[i]

        for class_id, proto in prototypes.items():
            weight = class_counts.get(class_id, 0)
            if weight > 0:
                # Convert proto to numpy
                if hasattr(proto, 'numpy'):
                    proto_np = proto.numpy()
                elif hasattr(proto, 'detach'):
                    proto_np = proto.detach().cpu().numpy()
                else:
                    proto_np = np.array(proto)
                
                proto_np = proto_np.astype(np.float32)
                cluster_weighted_sum[cluster_id][class_id] += weight * proto_np
                cluster_class_counts[cluster_id][class_id] += weight

      # Compute new cluster prototypes
      new_clusters = defaultdict(dict)
      for cluster_id in range(self.num_clusters):
        if cluster_id in cluster_weighted_sum:
            for class_id in cluster_weighted_sum[cluster_id]:
                count = cluster_class_counts[cluster_id][class_id]
                if count > 0:
                    new_clusters[cluster_id][class_id] = (
                        cluster_weighted_sum[cluster_id][class_id] / count
                    )

      print(f"[M-step] Updated {len(new_clusters)} cluster prototypes")
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
      

  

