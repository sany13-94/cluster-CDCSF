from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
#from MulticoreTSNE import print_function
import flwr
import mlflow
from torch.cuda.amp import autocast, GradScaler
import base64
import pickle
import datetime
import time
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
        initial_parameters: Optional[Parameters] = None,  # <-- add this

         total_rounds: int = 15,
         save_dir: str = "checkpoints", save_every: int = 10,
         base_round: int = 0,             # <--- NEW
        meta_state: Optional[dict] = None,  # <--- NEW
   evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
  
    ) -> None:
        super().__init__(
        )
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.server_url = "https://add18b7094f7.ngrok-free.app/heartbeat"
        #clusters parameters
        self.warmup_rounds = 20 # Stage 1 duration
        self.num_clusters = 4
        self.client_assignments = {}  # {client_id: cluster_id}
        self.clustering_interval = 8
        # Simple participation counter
        self.client_participation_count = {}  # client_id -> number of times selected
        self.initial_parameters=initial_parameters
        # Initialize as empty dictionaries
        self.cluster_prototypes = {i: {} for i in range(self.num_clusters)}
        self.cluster_class_counts = {i: defaultdict(int) for i in range(self.num_clusters)}
        map_path="client_id_mapping1.csv"
        self.proto_rows = []   # list of dicts: {"round", "client_id", "proto_score", "domain_id"}
        self.client_prototype_cache = {}  # New field
        self.base_round = base_round     # <--- replace old self.base_round = 0
        self.theta =0.13         # optional threshold for s_c
        self.use_topk = getattr(self, "use_topk", True)    # prefer Top-K when you know |S_gt|
        self.uuid_to_cid = {}     # {"8325...": "client_0"}
        self.cid_to_uuid = {}     # {"client_0": "8325..."}
        self.ground_truth_cids = set(ground_truth_stragglers)  # {"client_0","client_1",...}
        self.ground_truth_flower_ids = set()  # will be filled as clients appear
        self._round_t0 = {}       # round -> start wall-clock time
        self.cum_time_sec = 0.0   # accumulated time across rounds
        self.fig3_rows = []       # list of {"round","elapsed_sec","cum_time_sec","avg_acc"}
        self.results_dir=Path("/kaggle/working/cluster-CDCSF/fedprox")
        self._map_written = False
        self.training_times = defaultdict(float)
        self.selection_counts = defaultdict(int)
        self.accuracy_history = defaultdict(float)
        self._current_accuracies = {}
        # ... existing initialization ...
        # Core parameters from corrected methodology

        # Validation tracking
        self.validation_history = []  # Track predictions vs ground truth per round
        self.save_dir_path = save_dir
        self.save_every = save_every
        os.makedirs(self.save_dir_path, exist_ok=True)
        self.round_gt_stragglers = {}   # e.g. { 3: {0, 3, 5}, 4: {3, 7, 10}, ... }
       
        true_domain_labels = np.array([0]*5 + [1]*5 + [2]*4 + [0]*1)  # Adjust to your setup
        
        self.virtual_cluster_id = 999
        self.visualizer = ClusterVisualizationForConfigureFit(
            save_dir="./clustering_visualizations",
            true_domain_labels=true_domain_labels
        )
        # Tracking
        self.training_times = {}
        self.selection_counts = {}
        self.participated_clients = set()
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
        method_name="CDCSF"

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

        # Track what we've already recorded: logical_id values
        self._seen = set()
        if self.map_path.exists():
            try:
                df = pd.read_csv(self.map_path, dtype=str)
                for _, r in df.iterrows():
                    # file has "logical_id" column
                    self._seen.add(str(r["logical_id"]))
            except Exception:
                pass

        # --- Restore persisted meta-state on resume, if available ---
        if meta_state is not None:
            print("[Resume] Restoring meta-state from checkpoint")

            # All keys here are logical ids (lid) as strings
            self.training_times = meta_state.get("training_times", {})
            self.selection_counts = meta_state.get("selection_counts", {})
            self.client_participation_count = meta_state.get("client_participation_count", {})
            self.participated_clients = set(meta_state.get("participated_clients", []))
            self.total_rounds_completed = meta_state.get("total_rounds_completed", 0)

            self.client_assignments = meta_state.get("client_assignments", {})
            self.cluster_prototypes = meta_state.get("cluster_prototypes", {})

            self.fig3_rows = meta_state.get("fig3_rows", [])
            self.cum_time_sec = meta_state.get("cum_time_sec", 0.0)

            self.proto_rows = meta_state.get("proto_rows", [])
            self.validation_history = meta_state.get("validation_history", [])

            # Override base_round if stored
            #self.base_round = meta_state.get("base_round", self.base_round)

            print(f"[Resume] Restored {len(self.training_times)} training_times entries")
            print(f"[Resume] Restored {len(self.selection_counts)} selection_counts entries")
            print(f"[Resume] Restored {len(self.participated_clients)} participated clients")
            print(f"[Resume] base_round = {self.base_round}, total_rounds_completed = {self.total_rounds_completed}")
    

    def _refresh_uuid_mapping(self, client_manager: ClientManager) -> None:
          """Ask each client for its logical_id (stable across runs) and 
          populate uuid_to_cid / cid_to_uuid mapping.
          Safe to call every round; it will skip already-known UUIDs.
          """
          all_clients = client_manager.all()
          for uuid, client_proxy in all_clients.items():
            if uuid in self.uuid_to_cid:
                continue  # already known in this run

            try:
                res = client_proxy.get_properties(
                    ins=GetPropertiesIns(config={"request": "identity"}),
                    timeout=5.0,
                    group_id=None,
                )
                props = res.properties or {}
                lid = (
                    props.get("logical_id")
                    or props.get("client_cid")
                )
                if lid is None:
                    print(f"[Mapping] Client {uuid}: no logical_id in identity response")
                    continue

                lid = str(lid)
                self.uuid_to_cid[uuid] = lid
                self.cid_to_uuid[lid] = uuid

                print(f"[Mapping] Bound uuid={uuid} ↔ lid={lid}")

            except Exception as e:
                print(f"[Mapping] Failed to query identity for uuid={uuid}: {e}")
    
    def initialize_parameters(self, client_manager):
        """Lazy initialization from checkpoint"""
        if self.initial_parameters is not None:
            print(f"✅ Loading checkpoint into strategy (round {self.base_round})")
            # Convert list of ndarrays to Parameters only when needed
            return ndarrays_to_parameters(self.initial_parameters)
        
        print("No checkpoint - initializing from client")
        return None


    def _save_checkpoint(
      self,
    server_round: int,
    aggregated_params: Optional[Parameters] | Optional[list],
    metrics_aggregated: Dict[str, Scalar],
) -> None:
      """Save model and meta-state to disk (one file per *global* round)."""

      # Ensure directory exists
      os.makedirs(self.save_dir_path, exist_ok=True)

      global_round = self.base_round + server_round
      ckpt_path = os.path.join(self.save_dir_path, f"round_{global_round:04d}.pkl")

      try:
        # Normalise to "list of ndarrays"
        if aggregated_params is None:
            params_nd = None
        elif isinstance(aggregated_params, list):
            # Already a list[np.ndarray] from _fedavg_parameters
            params_nd = aggregated_params
        else:
            # It's a Flower Parameters object
            params_nd = parameters_to_ndarrays(aggregated_params)

        data = {
            "server_round": global_round,
            "parameters": params_nd,          # list[np.ndarray] or None
            "metrics": metrics_aggregated,
            "meta_state": self._get_meta_state(),  # if you're using meta-state
        }

        with open(ckpt_path, "wb") as f:
            pickle.dump(data, f)

        print(f"[CheckpointFedProx] ✅ Saved checkpoint: {ckpt_path}")

      except Exception as e:
        print(f"[CheckpointFedProx] FAILED to save checkpoint {ckpt_path}: {e}")

    def _get_meta_state(self):
      return {
        "training_times": self.training_times,
        "selection_counts": self.selection_counts,
        "client_participation_count": self.client_participation_count,
        "participated_clients": list(self.participated_clients),
        "total_rounds_completed": self.total_rounds_completed,
        "client_assignments": self.client_assignments,    # logical IDs only
        "cluster_prototypes": self.cluster_prototypes,
        "fig3_rows": self.fig3_rows,
        "cum_time_sec": self.cum_time_sec,
        "proto_rows": self.proto_rows,
        "validation_history": self.validation_history,
        "client_prototype_cache": self.client_prototype_cache  # Persists!,

        # NEVER STORE UUIDS
    }
    
    def num_evaluate_clients(self, client_manager: ClientManager) -> Tuple[int, int]:
      """Return the sample size and required number of clients for evaluation."""
      num_clients = client_manager.num_available()
      return max(int(num_clients * self.fraction_evaluate), self.min_evaluate_clients), self.min_available_clients
    
   
    def _append_rows(self, rows: List[dict]) -> None:
        if not rows:
            return
        header = ["logical_id", "server_uuid"]
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
        """Aggregate model updates and update meta-state."""
        if failures:
            print(f"[Round {server_round}] Failures: {len(failures)}")

        if not results:
            print(f"[Round {server_round}] No clients returned results. Skipping aggregation.")
            return None, {}

        try:
            clients_params_list = []
            num_samples_list = []
            current_round_durations = []
            current_participants = set()   # set of logical ids
            new_rows: List[dict] = []

            # Process results and update tracking
            for client_proxy, fit_res in results:
                uuid = client_proxy.cid          # Flower internal UUID
                metrics = fit_res.metrics or {}

                # Logical id from client metrics (our stable key)
                logical_id = metrics.get("client_cid")
                if logical_id is None:
                    logical_id = metrics.get("simulation_index")

                if logical_id is None:
                    print(f"[aggregate_fit] WARNING: no logical_id for uuid={uuid}, skipping meta tracking")
                    lid = None
                else:
                    lid = str(logical_id)

                    # Maintain mapping uuid <-> logical id
                    self.uuid_to_cid[uuid] = lid
                    self.cid_to_uuid[lid] = uuid

                node = metrics.get("flower_node_id")
                print(f"=== logical_id: {logical_id}, lid: {lid}, uuid: {uuid}, node: {node} ===")

                # Participation counts and sets (by logical id)
                if lid is not None:
                    if lid not in self.client_participation_count:
                        self.client_participation_count[lid] = 0
                    self.client_participation_count[lid] += 1

                    self.participated_clients.add(lid)
                    current_participants.add(lid)

                # EMA training time (also by logical id)
                if "duration" not in metrics:
                    continue

                dur = float(metrics["duration"])
                if lid is not None:
                    prev = self.training_times.get(lid)
                    if prev is None:
                        ema = dur
                        print(f"[EMA Init] {lid}: T_c(0) = {dur:.2f}s")
                    else:
                        ema = self.alpha * dur + (1.0 - self.alpha) * prev
                        print(f"[EMA Update] {lid}: {prev:.2f}s → {ema:.2f}s (raw: {dur:.2f}s)")
                    self.training_times[lid] = ema

                current_round_durations.append(dur)

                # Write mapping CSV (logical_id, server_uuid) once per logical id
                if lid is not None and lid not in self._seen:
                    self._seen.add(lid)
                    new_rows.append({
                        "logical_id": lid,
                        "server_uuid": uuid,
                    })

                self._append_rows(new_rows)
                if new_rows:
                    print(f"[Server] Recorded {len(new_rows)} new logical client(s). Total unique: {len(self._seen)}")

                # Collect parameters for aggregation
                clients_params_list.append(parameters_to_ndarrays(fit_res.parameters))
                num_samples_list.append(fit_res.num_examples)

            self.last_round_participants = current_participants
            self.total_rounds_completed = server_round

            # Validate straggler predictions (uses logical ids now)
            self._validate_straggler_predictions(server_round, results)

            print(f"\n[Round {server_round}] Participants (logical ids): {list(current_participants)}")
            if current_round_durations:
                print(f"[Round {server_round}] Average raw training time: {np.mean(current_round_durations):.2f}s")

            # FedAvg aggregation
            aggregated_params = self._fedavg_parameters(clients_params_list, num_samples_list)

            # Prototype logging
            self._log_prototypes_after_fit(server_round, results)

            # Save checkpoint periodically
            #if server_round % self.save_every == 0:
            metrics_aggregated = {}
            self._save_checkpoint(server_round, aggregated_params, metrics_aggregated)

            # Finalization at last round
            global_round=self.base_round+server_round
            if global_round == self.total_rounds:
                self.save_client_mapping()
                print("\n" + "=" * 80)
                print(f"[Round {server_round}] TRAINING COMPLETED - Auto-saving results...")
                print("=" * 80)
                self._save_all_results()
                self._save_prototype_heatmap()

            return ndarrays_to_parameters(aggregated_params), {}

        except Exception as e:
            print(f"[aggregate_fit] Error: {e}")
            return None, {} 
    
    def _log_prototypes_after_fit(
    self,
    server_round: int,
    results: list,
):
     if not results:
        return

     print(f"[ProtoLog] Round {server_round}: logging prototypes for {len(results)} clients")

     for client_proxy, fit_res in results:
        metrics = fit_res.metrics or {}
        cid = metrics.get("client_cid", None)  # logical id from client.fit()
        if cid is None:
            print(f"  [ProtoLog] client_proxy.cid={client_proxy.cid}: no client_cid in metrics, skipping")
            continue

        try:
            cid_int = int(cid)
        except Exception:
            cid_int = cid  # keep as string if needed

        try:
            # Ask client for its prototypes
            get_protos_res = client_proxy.get_properties(
                ins=GetPropertiesIns(config={"request": "prototypes"}),
                timeout=15.0,
                group_id=None,
            )

            props = get_protos_res.properties
            prototypes_encoded = props.get("prototypes")
            class_counts_encoded = props.get("class_counts")
            domain_id_raw = props.get("domain_id", None)
            client_id_prop = props.get("client_id", None)  # just for sanity check

            try:
                domain_id = int(domain_id_raw) if domain_id_raw is not None else -1
            except Exception:
                domain_id = -1

            if not prototypes_encoded or not class_counts_encoded:
                print(f"  [ProtoLog] Client {cid_int}: no prototypes/class_counts, skipping")
                continue

            try:
                prototypes = pickle.loads(base64.b64decode(prototypes_encoded))
                # class_counts = pickle.loads(base64.b64decode(class_counts_encoded))  # optional
            except Exception as e:
                print(f"  [ProtoLog] Client {cid_int}: decode error: {e}")
                continue

            if not isinstance(prototypes, dict):
                print(f"  [ProtoLog] Client {cid_int}: prototypes not a dict, skipping")
                continue

            proto_score = self._compute_proto_score_from_dict(prototypes)

            self.proto_rows.append({
                "round": int(server_round),
                "client_id": cid_int,
                "proto_score": float(proto_score),
                "domain_id": domain_id,
            })

            print(f"  [ProtoLog] Round {server_round} | Client {cid_int} | domain={domain_id} | proto_score={proto_score:.4f}")

        except Exception as e:
            print(f"  [ProtoLog] client_proxy.cid={client_proxy.cid}: get_properties failed: {e}")

    def _predict_stragglers_from_score(self, T_max, client_ids):
      """Return set of predicted stragglers using s_c = 1 - As."""
      scores = {}
      for cid in client_ids:
        T_c = self.training_times.get(cid, 0.0)
        As = T_max / (T_c + self.beta * T_max) if (T_c > 0 and T_max > 0) else 0.0
        s_c = 1.0 - As
        scores[cid] = s_c

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
        self.visualize_client_participation(self.client_participation_count, save_path="participation_chart.png", )
        self.save_validation_results()
    def _norm(self,label: str) -> str:
      s = str(label).strip()
      return s.replace("client_", "")   # "client_0" -> "0"

    def _validate_straggler_predictions(self, server_round, results):
      """Validate straggler predictions using logical ids and per-round ground truth."""
      participants, round_dur = [], {}

      for client_proxy, fit_res in results:
        metrics = fit_res.metrics or {}

        logical_id = (
            metrics.get("client_cid")
            or metrics.get("logical_id")
            or metrics.get("simulation_index")
        )
        if logical_id is None:
            continue

        lid_str = str(logical_id)
        participants.append(lid_str)

        if "duration" in metrics:
            round_dur[lid_str] = float(metrics["duration"])

      # compute T_max from EMAs (assume you already updated EMA this round)
      valid_times = [t for t in self.training_times.values() if t is not None]
      if not valid_times:
        return

      T_max = float(np.mean(valid_times))

      # predict using logical ids (keys are the same string lids as in participants)
      predicted_set, scores = self._predict_stragglers_from_score(T_max, participants)

      gt_lid_set = self.round_gt_stragglers.get(server_round, set())

      for lid_str in participants:
        # try to convert "client_3" or "3" -> 3
        logical_idx = None
        try:
            if lid_str.startswith("client_"):
                logical_idx = int(lid_str.split("_", 1)[1])
            else:
                logical_idx = int(lid_str)
        except Exception:
            logical_idx = None

        # ground truth: was this logical index designated as straggler this round?
        is_gt = (logical_idx is not None) and (logical_idx in gt_lid_set)

        rec = {
            "round": server_round,
            "client_id": lid_str,  # logical id as string
            "logical_id": logical_idx,
            "T_c": self.training_times.get(lid_str, float("nan")),
            "T_max": T_max,
            "s_c": scores.get(lid_str, float("nan")),
            "actual_duration": round_dur.get(lid_str, float("nan")),
            "predicted_straggler": lid_str in predicted_set,
            "ground_truth_straggler": is_gt,
        }
        rec["prediction_type"] = self._classify_prediction(
            rec["predicted_straggler"], rec["ground_truth_straggler"]
        )
        self.validation_history.append(rec)

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
        
        # === NEW: accumulate wall-clock time and log for Figure 3 ===
        t0 = self._round_t0.pop(server_round, None)
        elapsed = (time.time() - t0) if t0 is not None else 0.0
        self.cum_time_sec += elapsed

        # Store one row: X=cum_time_sec, Y=avg_acc
        self.fig3_rows.append({
        "round": int(server_round),
        "elapsed_sec": float(elapsed),
        "cum_time_sec": float(self.cum_time_sec),
        "avg_acc": float(aggregated_accuracy),
    })

        # Save CSV + PNG (can be every round; last one is your final figure)
        self._save_fig3_time_vs_acc()
            
        return None, {"accuracy": aggregated_accuracy}
   
  
    def _save_fig3_time_vs_acc(self):
      """Save/refresh Figure 3: Avg accuracy vs cumulative training time."""
      if not self.fig3_rows:
        return

      df = pd.DataFrame(self.fig3_rows).sort_values("round")
      # safety: recompute cum_time if needed
      if "cum_time_sec" not in df.columns:
        df["cum_time_sec"] = df["elapsed_sec"].cumsum()

      csv_path = self.results_dir / f"fig3_time_vs_acc_CDCSF.csv"
      png_path = self.results_dir / f"fig3_time_vs_acc_CDCSF.png"
      df.to_csv(csv_path, index=False)

      # X = cumulative time, Y = avg accuracy
      plt.figure(figsize=(9, 5))
      plt.plot(df["cum_time_sec"], df["avg_acc"], marker="o", linewidth=2)
      plt.xlabel("Cumulative training time (s)")
      plt.ylabel("Average accuracy")
      plt.title(f"Average Accuracy vs Cumulative Time — CDCSF")
      plt.grid(True, linestyle="--", alpha=0.4)
      plt.tight_layout()
      plt.savefig(png_path, dpi=200)
      plt.close()

      print(f"[Fig3] Saved CSV -> {csv_path}")
      print(f"[Fig3] Saved PNG -> {png_path}")

    def compute_reliability_scores(self, client_ids: List[str]) -> Dict[str, float]:
        
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
      # Global counts
      T = max(1, int(self.total_rounds_completed))  # rounds completed so far
      # total registered/known clients (fixed-ish); prefer a fixed n if you have one
      n = max(1, len(getattr(self, "all_known_clients", client_ids)))

      # Average K per round so far (actual selections)
      total_selections = int(sum(self.selection_counts.values()))
      K_bar = max(1e-6, total_selections / T)

      # Ideal selections per client up to now
      ideal = max(1e-6, T * (K_bar / n))

      fairness_scores = {}
      for cid in client_ids:
        v_c = int(self.selection_counts.get(str(cid), 0))
        R_c = v_c / ideal
        f_s = 1.0 / (1.0 + R_c)   # Eq. (8)
        fairness_scores[str(cid)] = float(f_s)

        # optional debug
        # print(f"[Fair(Global)] {cid}: v={v_c}, ideal={ideal:.2f}, R={R_c:.2f}, f={f_s:.3f}")

      return fairness_scores   

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

    def _adapt_weights(self, global_round: int) -> Tuple[float, float]:
      progress = global_round / self.total_rounds
    
      if progress < 0.1:        # First 10%: EXPLORATION
        alpha_1, alpha_2 = 0.3, 0.7  # Prioritize FAIRNESS to discover clients
      elif progress < 0.4:      # 10-40%: TRANSITION
        alpha_1, alpha_2 = 0.6, 0.4  # Balanced
      elif progress < 0.8:      # 40-80%: EXPLOITATION
        alpha_1, alpha_2 = 0.8, 0.2  # Prioritize RELIABILITY
      else:                     # Final 20%: FAIRNESS
        alpha_1, alpha_2 = 0.3, 0.7  # Ensure fair participation
    
      return alpha_1, alpha_2

    
    
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

          # participation_dict: {lid: count}
          mapped_dict = {}
          for lid, count in participation_dict.items():
            mapped_dict[str(lid)] = count

          sorted_items = sorted(mapped_dict.items(), key=lambda x: int(x[0]))
          client_ids = [f"Client {item[0]}" for item in sorted_items]
          counts = [item[1] for item in sorted_items]

          fig, ax = plt.subplots(figsize=(14, 6))
          bars = ax.bar(range(len(client_ids)), counts)

          for i, count in enumerate(counts):
            if count == 0:
                bars[i].set_color('red')
                bars[i].set_alpha(0.5)

          ax.set_xlabel('Client ID (logical)', fontsize=12, fontweight='bold')
          ax.set_ylabel('Number of Participations', fontsize=12, fontweight='bold')
          ax.set_title(f'Client Participation Distribution - {method_name}', fontsize=14, fontweight='bold')
          ax.set_xticks(range(len(client_ids)))
          ax.set_xticklabels(client_ids, rotation=45, ha='right')
          ax.grid(axis='y', alpha=0.3, linestyle='--')

          total_clients = len(client_ids)
          participated = sum(1 for c in counts if c > 0)
          avg_participation = np.mean(counts) if counts else 0.0
          std_participation = np.std(counts) if counts else 0.0

          stats_text = (
            f"Total Clients: {total_clients}\n"
            f"Participated: {participated} ({(participated/total_clients*100 if total_clients else 0):.1f}%)\n"
            f"Avg Participation: {avg_participation:.2f} ± {std_participation:.2f}"
        )
          ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

          plt.tight_layout()
          plt.savefig(save_path, dpi=300, bbox_inches='tight')
          print(f"Visualization saved to {save_path}")
          plt.close()
    
    # In your Strategy class, add this visualization:
    def _save_selection_heatmap(self):
      """Visualize client selection pattern over rounds."""
      rounds = []
      client_ids = []
      for round_num, selected_clients in self.round_selections.items():
        for client_id in selected_clients:
            rounds.append(round_num)
            client_ids.append(client_id)
    
      df = pd.DataFrame({'round': rounds, 'client_id': client_ids})
      heatmap_data = df.pivot_table(
        index='client_id', 
        columns='round', 
        aggfunc='size', 
        fill_value=0
    )
    
      plt.figure(figsize=(12, 8))
      sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Selected'})
      plt.xlabel('Training Round')
      plt.ylabel('Client ID')
      plt.title('Client Selection Pattern: Exploration → Exploitation → Fairness')
    
      # Add phase annotations
      plt.axvline(x=self.total_rounds * 0.1, color='blue', 
                linestyle='--', label='Exploration End')
      plt.axvline(x=self.total_rounds * 0.75, color='green', 
                linestyle='--', label='Fairness Start')
      plt.legend()
      plt.savefig(self.results_dir / 'selection_heatmap.png', dpi=300, bbox_inches='tight')

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        effective_round = self.base_round + server_round

        print(f"\n{'='*80}")
        print(f"[Round {server_round}] TWO-STAGE RESOURCE-AWARE FAIR SELECTION (global={effective_round})")
        print(f"{'='*80}")

        # 1) Refresh mapping: uuid -> logical_id
        self._refresh_uuid_mapping(client_manager)

        # Get all available clients (UUIDs)
        all_clients = client_manager.all()
        available_uuids = list(all_clients.keys())

        # ---- start-of-round timer ----
        self._round_t0[server_round] = time.time()
        if not available_uuids:
            print(f"[Round {server_round}] No clients available.")
            return []

        # Build UUID -> logical id mapping for currently known clients
        uuid_to_lid: Dict[str, Optional[str]] = {}
        for uuid in available_uuids:
            lid = self.uuid_to_cid.get(uuid)
            uuid_to_lid[uuid] = lid

        print(f"\n[Client Status]")
        print(f"  Total available clients: {len(available_uuids)}")
        print(f"  Previously participated (logical ids): {len(self.participated_clients)}")

        # Categorize clients (by uuid, but using lid membership)
        participated_available = []   # uuids with known lid that has participated
        never_participated = []       # uuids whose lid not in participated set (or lid unknown)

        for uuid in available_uuids:
            lid = uuid_to_lid.get(uuid)
            if lid is not None and lid in self.participated_clients:
                participated_available.append(uuid)
            else:
                never_participated.append(uuid)

        print(f"  Available participated clients: {len(participated_available)}")
        print(f"  Available never-participated clients: {len(never_participated)}")

        # =================================================================
        # DETERMINE STAGE: WARMUP vs DOMAIN-AWARE
        # =================================================================
        in_warmup_phase = server_round <= self.warmup_rounds

        effective_round = self.base_round + server_round

        in_warmup_phase = effective_round <= self.warmup_rounds

        # For logging:
        if in_warmup_phase:
            print(f"\n[STAGE 1: WARMUP PHASE] Global Round {effective_round}/{self.warmup_rounds}")
        else:
            print(f"\n[STAGE 2: DOMAIN-AWARE PHASE] Global Round {effective_round}")
        # =================================================================
        # PHASE 1: CLUSTERING (unchanged logic, still uses UUIDs)
        # =================================================================
        clusters = defaultdict(list)

        if effective_round % 2 != 0 and participated_available and not in_warmup_phase:
            print(f"\n{'─'*80}")
            print(f"[Clustering Round] Collecting prototypes from ALL participated clients IN ROUND {server_round}")

            all_prototypes_list = []
            all_client_ids = []
            class_counts_list = []
            clients_with_prototypes = []
            domains_ids = []

            for uuid in participated_available:
                client_proxy = all_clients[uuid]

                try:
                    get_protos_res = client_proxy.get_properties(
                        ins=GetPropertiesIns(config={"request": "prototypes"}),
                        timeout=15.0,
                        group_id=None
                    )

                    props = get_protos_res.properties
                    prototypes_encoded = props.get("prototypes")
                    class_counts_encoded = props.get("class_counts")
                    domain_id = int(props.get("domain_id", -1))

                    if prototypes_encoded and class_counts_encoded:
                        try:
                            prototypes = pickle.loads(base64.b64decode(prototypes_encoded))
                            class_counts = pickle.loads(base64.b64decode(class_counts_encoded))

                            if isinstance(prototypes, dict) and isinstance(class_counts, dict):
                                #all_prototypes_list.append(prototypes)
                                #all_client_ids.append(uuid)
                                #domains_ids.append(domain_id)
                                #class_counts_list.append(class_counts)
                                #clients_with_prototypes.append(uuid)
                                #print(f"  ✓ Client {uuid}: Prototypes collected")
                                # ========== CACHE ON SERVER SIDE ==========
                                self.client_prototype_cache[lid] = {
                                    "prototypes": prototypes,
                                    "class_counts": class_counts,
                                    "last_round": effective_round,
                                    "domain_id": domain_id
                                }
                                print(f"  ✓ {lid}: Collected & cached prototypes")

                        except Exception as decode_error:
                            print(f"  ✗ Client {uuid}: Decode error - {decode_error}")
                    else:
                        print(f"  ⚠ Client {uuid}: No prototypes available")

                except Exception as e:
                    print(f"  ⚠ Client {uuid}: Communication failed - {e}")

            print(f"\n[Prototype Collection] {len(clients_with_prototypes)}/{len(participated_available)} successful")

            # Get all clients that have cached prototypes (current + previous rounds)
            for lid in self.participated_clients:
                if lid in self.client_prototype_cache:
                    cache = self.client_prototype_cache[lid]
                    all_prototypes_list.append(cache["prototypes"])
                    all_client_ids.append(lid)
                    class_counts_list.append(cache["class_counts"])
                    domains_ids.append(cache.get("domain_id", -1))
            
            if len(clients_with_prototypes) >= self.num_clusters:
                print(f"\n[EM Clustering] Processing {len(clients_with_prototypes)} clients...")

                if not self.cluster_prototypes:
                    print("  Initializing cluster prototypes with k-means++...")
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

                print(f"\n[Clustering Results]")
                for cluster_id in range(self.num_clusters):
                    cluster_clients = [cid for cid, clust in self.client_assignments.items()
                                       if clust == cluster_id]
                    if cluster_clients:
                        print(f"  Cluster {cluster_id}: {len(cluster_clients)} clients")

                if len(all_prototypes_list) >= self.num_clusters:
                    self._visualize_clusters(
                        prototypes=all_prototypes_list,
                        client_ids=all_client_ids,
                        server_round=server_round,
                        true_domain_map=None
                    )
            else:
                print(f"\n[Clustering Skipped] Need {self.num_clusters} clients, have {len(clients_with_prototypes)}")
                print(f"  Will use unified pool selection")

        # =================================================================
        # PHASE 2: ORGANIZE CLIENTS INTO CLUSTERS OR UNIFIED POOL (UUIDs)
        # =================================================================
        if in_warmup_phase or not self.client_assignments:
            print(f"\n[Client Organization] UNIFIED POOL MODE")
            clusters[0] = participated_available + never_participated
            print(f"  Single pool: {len(clusters[0])} clients")
        else:
            print(f"\n[Client Organization] DOMAIN-AWARE MODE")

            for uuid in participated_available:
                if uuid in self.client_assignments:
                    cluster_id = self.client_assignments[uuid]
                    clusters[cluster_id].append(uuid)
                else:
                    clusters[0].append(uuid)

            if never_participated and self.use_virtual_cluster:
                clusters[self.virtual_cluster_id] = never_participated
                print(f"  Virtual Cluster {self.virtual_cluster_id}: {len(never_participated)} new clients")

            for cluster_id in sorted(clusters.keys()):
                cluster_clients = clusters[cluster_id]
                cluster_type = "Virtual" if cluster_id == self.virtual_cluster_id else "Domain"
                print(f"  Cluster {cluster_id} [{cluster_type}]: {len(cluster_clients)} clients")

        print(f"\n[Active Clusters] {len(clusters)} cluster(s)")

        # =================================================================
        # PHASE 3: COMPUTE GLOBAL SELECTION SCORES (LIDs -> map back to UUIDs)
        # =================================================================
        print(f"\n{'─'*80}")
        print(f"[Score Computation] Round {server_round}")
        print(f"{'─'*80}")

        alpha_1, alpha_2 = self._adapt_weights(effective_round)
        print(f"Weights: α₁(reliability)={alpha_1:.2f}, α₂(fairness)={alpha_2:.2f}")

        all_scores: Dict[str, float] = {}  # uuid -> score

        # 1) Participated clients: use full method on logical ids
        if participated_available:
            print(f"\n[Participated Clients] Computing reliability + fairness scores...")
            lids = []
            lid_by_uuid = {}
            for uuid in participated_available:
                lid = uuid_to_lid.get(uuid)
                if lid is not None:
                    lids.append(lid)
                    lid_by_uuid[uuid] = lid

            if lids:
                lid_scores = self.compute_global_selection_scores(lids, server_round)
                for uuid, lid in lid_by_uuid.items():
                    all_scores[uuid] = lid_scores.get(lid, 0.0)

        # 2) New / unmapped clients: neutral reliability, max fairness
       
        # Score new clients (neutral reliability, max fairness)
        if never_participated:
          for uuid in never_participated:
            reliability = 0.8  # Assume fast until proven slow
            fairness = 1.0
            all_scores[uuid] = (alpha_1 * reliability) + (alpha_2 * fairness)
    
        # =================================================================
        # PHASE 4: DISTRIBUTE SELECTION BUDGET ACROSS CLUSTERS (UUIDs)
        # =================================================================
        print(f"\n{'─'*80}")
        print(f"[Selection Distribution]")
        print(f"{'─'*80}")

        if not clusters:
            print("No clusters available")
            return []

        total_clusters = len(clusters)

        if in_warmup_phase or total_clusters == 1:
            cluster_allocations = {list(clusters.keys())[0]: self.min_fit_clients}
            print(f"Unified pool allocation: {self.min_fit_clients} clients")
        else:
            base_per_cluster = max(1, self.min_fit_clients // total_clusters)
            remaining_budget = self.min_fit_clients - (base_per_cluster * total_clusters)

            print(f"Total selection budget: {self.min_fit_clients} clients")
            print(f"Active clusters: {total_clusters}")
            print(f"Base per cluster: {base_per_cluster}")
            print(f"Remaining: {remaining_budget}")

            cluster_allocations = {cluster_id: base_per_cluster for cluster_id in clusters}

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
        # PHASE 5: SELECT CLIENTS FROM EACH CLUSTER (UUIDs, but log with LIDs)
        # =================================================================
        print(f"\n{'─'*80}")
        print(f"[Client Selection]")
        print(f"{'─'*80}")

        selected_clients_uuids: List[str] = []

        for cluster_id in sorted(clusters.keys()):
            cluster_clients = clusters[cluster_id]
            allocation = cluster_allocations.get(cluster_id, 0)

            if allocation == 0:
                continue

            cluster_type = "Unified Pool" if in_warmup_phase else (
                "Virtual" if cluster_id == self.virtual_cluster_id else "Domain"
            )

            print(f"\n[Cluster {cluster_id} - type: {cluster_type}]")

            cluster_clients_sorted = sorted(
                cluster_clients,
                key=lambda uuid: all_scores.get(uuid, 0.0),
                reverse=True
            )

            num_to_select = min(allocation, len(cluster_clients_sorted))
            cluster_selection = cluster_clients_sorted[:num_to_select]
            selected_clients_uuids.extend(cluster_selection)

            print(f"Selected {len(cluster_selection)}/{len(cluster_clients)} clients")

            for i, uuid in enumerate(cluster_selection[:5]):
                score = all_scores.get(uuid, 0.0)
                lid = uuid_to_lid.get(uuid)
                status = "NEW" if (lid is None or lid not in self.participated_clients) else "participated"
                lid_str = f"lid={lid}" if lid is not None else "lid=UNK"
                lid_sel = self.selection_counts.get(lid, 0) if lid is not None else 0
                print(f"    {i+1}. {uuid:20s} [{status:12s}] {lid_str} Score={score:.4f}, Selected={lid_sel}x")

        # =================================================================
        # PHASE 6: PREPARE INSTRUCTIONS (UUIDs) + update selection_counts (LIDs)
        # =================================================================
        selected_clients_uuids = selected_clients_uuids[:self.min_fit_clients]

        # Which logical clients are allowed to be stragglers?
        STRAGGLER_LIDS = {0, 3, 5, 7, 10, 12}
        NUM_STRAGGLERS_PER_ROUND = 3

        instructions: List[Tuple[ClientProxy, FitIns]] = []

        # 1) choose which logical ids will be stragglers this round
        eligible_lids = [lid for lid in STRAGGLER_LIDS if lid in uuid_to_lid.values()]
        num_to_pick = min(NUM_STRAGGLERS_PER_ROUND, len(eligible_lids))
        round_straggler_lids = set(random.sample(eligible_lids, num_to_pick))

        print(f"[Round {effective_round}] chosen straggler LIDs: {round_straggler_lids}")

        self.round_gt_stragglers[effective_round] = set(round_straggler_lids)

        for uuid in selected_clients_uuids:
          if uuid in all_clients:
            client_proxy = all_clients[uuid]

            lid = uuid_to_lid.get(uuid)   # logical id (int)
            is_straggler = lid in round_straggler_lids

            client_config = {
            "server_round": effective_round,
            "simulate_delay": 1 if is_straggler else 0,
            "delay_base_sec": 20.0,
            "delay_jitter_sec": 3.0,
            "delay_prob": 1.0,
        }

            instructions.append((client_proxy, FitIns(parameters, client_config)))

            if lid is not None:
                self.selection_counts[lid] = self.selection_counts.get(lid, 0) + 1

        print(f"\n{'='*80}")
        print(f"[Round {server_round}] SELECTION SUMMARY")
        print(f"{'='*80}")

        stage_name = "WARMUP" if in_warmup_phase else "DOMAIN-AWARE"
        print(f"Stage: {stage_name}")
        print(f"Total selected: {len(instructions)} clients")

        print(f"\nSelection frequency (top 10) [by logical id]:")
        top_selected = sorted(self.selection_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for lid, count in top_selected:
            print(f"  lid {lid:8s}: {count}x")

        print(f"{'='*80}\n")

        return instructions
   
    def _compute_proto_score_from_dict(self, prototypes: dict) -> float:
     """
     Collapse a dict[class_id] -> embedding vector into a single scalar.
     Simple: mean of all coordinates across all prototype vectors.
     """
     if not prototypes:
        return 0.0

     vecs = []
     for v in prototypes.values():
        if v is None:
            continue
        v_np = np.asarray(v)
        if v_np.ndim == 1:
            vecs.append(v_np)
        else:
            vecs.append(v_np.reshape(-1))

     if not vecs:
        return 0.0

     all_vecs = np.stack(vecs, axis=0)
     return float(all_vecs.mean())

    def _save_prototype_heatmap(self):
     if not self.proto_rows:
        print("[ProtoHeatmap] No prototype logs recorded, skipping.")
        return

     df = pd.DataFrame(self.proto_rows)
     # pivot: rows = rounds, columns = client_id, value = proto_score
     table = df.pivot_table(
        index="round",
        columns="client_id",
        values="proto_score",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)

     #tag = self.method_name.replace(" ", "_")
     csv_path = self.results_dir / f"proto_heatmap_CDCSF.csv"
     table.to_csv(csv_path)
     print(f"[ProtoHeatmap] Saved matrix CSV -> {csv_path}")

     # (optional) what domain each client mostly belongs to
     dom_agg = (df.groupby("client_id")["domain_id"]
                 .agg(lambda x: np.bincount([d for d in x if d >= 0]).argmax()
                      if any(np.array(x) >= 0) else -1))

     dom_csv = self.results_dir / f"proto_client_domains_CDCSF.csv"
     dom_agg.to_csv(dom_csv, header=["domain_id"])
     print(f"[ProtoHeatmap] Saved client→domain map -> {dom_csv}")

     # Make heatmap
     plt.figure(figsize=(10, 6))
     im = plt.imshow(
        table.values.T,     # shape = (num_clients, num_rounds)
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
     plt.colorbar(im, label="Prototype score")
     plt.xticks(
        np.arange(len(table.index)),
        table.index,
        rotation=45,
        ha="right",
    )
     plt.yticks(
        np.arange(len(table.columns)),
        [f"Client {c}" for c in table.columns],
    )
     plt.xlabel("Round")
     plt.ylabel("Client ID (logical)")
     plt.title(f"Prototype-based selection pattern ")
     plt.tight_layout()

     png_path = self.results_dir / f"proto_heatmap_CDCSF.png"
     plt.savefig(png_path, dpi=200)
     plt.close()
     print(f"[ProtoHeatmap] Saved heatmap PNG -> {png_path}")

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
      #nn
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
      

  