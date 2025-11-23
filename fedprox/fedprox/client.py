"""Defines the MNIST Flower Client and a function to instantiate it."""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
from flwr.common import Context
import flwr as fl
import numpy as np
import torch
import copy
from pathlib import Path
from datetime import datetime
from collections import defaultdict
# ==== imports & setup ====
import os, gc, glob
import csv
import torch
from torch.cuda.amp import autocast, GradScaler
import math
import torch.nn.functional as F
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
import csv
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.utils.data import DataLoader
import json
from flwr.client import NumPyClient, Client,  NumPyClient
import matplotlib.pyplot as plt # Utilisé pour la visualisation
import time
from flwr.common import Status, Code, parameters_to_ndarrays ,ConfigsRecord, MetricsRecord, ParametersRecord ,Context, ConfigRecord
# Importer la classe d'analyse des pixels
from fedprox.pixeldistributionanalyzer import PixelDistributionAnalyzer # NOUVEAU

from  mlflow.tracking import MlflowClient
import base64
import pickle
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
import os
from fedprox.models import train_gpaf,test_gpaf,ModelCDCSF,train_moon
from fedprox.features_visualization import extract_features_and_labels,StructuredFeatureVisualizer
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, net: ModelCDCSF, 
     data,validset,
     local_epochs,
     client_id,  
            device,
            context=None):
        self.net = net
        self.context = context  # Flower runtime context
        self.traindata = data
        self.validdata=validset
        self.device = device
        print(f'device gpu {self.device}')
        self.local_epochs=local_epochs
        self.client_id=client_id
        self.num_classes=9
        self.num_clients=2
       
        # Move models to device
        self.net.to(self.device)
        
        # Initialize optimizers
        self.optimizer= torch.optim.Adam(self.net.parameters())
        # Initialize dictionaries to store features and labels
        self.client_features = {}  # Add this
        self.client_labels = {}    # Add this
        
        # Prototype storage setup
        self.prototype_dir = Path("prototype_cache")
        self.prototype_dir.mkdir(exist_ok=True)
        self.prototype_file = self.prototype_dir / f"client_{self.client_id}_prototypes.pkl"
        self.counts_file = self.prototype_dir / f"client_{self.client_id}_counts.pkl"
        


        BASE_DIR = Path("/kaggle/working/cluster-CDCSF/fedprox")
        self.prototype_dir = BASE_DIR / "prototype_cache"
        self.prototype_dir.mkdir(parents=True, exist_ok=True)
        self.prototype_file = self.prototype_dir / f"client_{self.client_id}_prototypes.pkl"
        self.counts_file = self.prototype_dir / f"client_{self.client_id}_counts.pkl"


        # Initialize prototype storage
        self.prototypes_from_last_round = None
        self.class_counts_from_last_round = None
        
        # Load existing prototypes if available
        self._load_prototypes_from_disk()

        # Optional: if you added your previous run's "Notebook Output" via Add Data,
        # set this path to that dataset so we can resume across versions automatically.
        # Example: "/kaggle/input/your-notebook-name/ckpts"
        self.PERSIST_INPUT = os.environ.get("KAGGLE_PERSIST_INPUT", "").strip()  # or hardcode the path
        
        
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        
        try:

          self.net.load_state_dict(state_dict, strict=True)

        except Exception as e:
          print("❌ Parameter loading FAILED")
          print("Model keys:", list(self.net.state_dict().keys()))
          print("Checkpoint keys:", list(state_dict.keys()))
          print("Shape mismatch error:", e)
          raise e

   

    def get_parameters(self, config=None):
      return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    #second and call set_para  
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        print(f'===evaluate client=== {type(parameters)}')
        self.set_parameters(parameters)

        for batch_idx, (data, target) in enumerate(self.validdata):
          print(f"evaluate dd Batch {batch_idx}, data shape: {data.shape}, target shape: {target.shape}")
          break  # J
        loss, accuracy = test_gpaf(self.net, self.validdata, self.device)
      
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        return float(loss), len(self.validdata), {"accuracy": float(accuracy),
        "cid": self.client_id  # ✅ send local client id to server

        }
    def fit(self, parameters, config):
        """Train local model and extract prototypes (NumPyClient interface)."""
        try: 
            import random
            round_number = config.get("server_round", -1)
            is_straggler = config.get("simulate_delay", False)  # ← Simple check

            simulate_delay = is_straggler 
            uuid=self.client_id
            print(f"Client {self.client_id} starting fit() for round {round_number}")
            uuid = str(self.client_id)  # force to string
            simulate_ids = {str(s).strip() for s in (config.get("simulate_stragglers") or "").split(",") if s}
            #simulate_delay = (uuid in simulate_ids) and (random.random() < config.get("delay_prob", 1.0))

            print(f"Client simulate_ids: {simulate_ids}tttt starting fit() for round {round_number}")

            print(f"[Client {self.client_id}] Is straggler: {simulate_delay}")

          
            start_time = time.time()
            # On ne le fait que pour la première ronde pour des raisons de performance.
            if round_number == 1 :
              print(f'=== visualize =====')
              # Il prend le client_id et le DataLoader d'entraînement pour analyser la distribution locale.
              self.pixel_analyzer = PixelDistributionAnalyzer(
            client_id=self.client_id, 
            traindata_loader=self.traindata # Utilisation du DataLoader d'entraînement
        )
            # Update model with global parameters
            self.set_parameters(parameters)
            
            # Train the model
            print(f"Client {self.client_id} starting training...")
            self.train(self.net, self.traindata, self.client_id, epochs=self.local_epochs,cfg=config, simulate_delay=simulate_delay)
            print(f"Client {self.client_id} completed training")
           
            # Extract and cache prototypes after training
            #print(f"Client {self.client_id} extracting prototypes...")
            self._extract_and_cache_prototypes(round_number)
            
            training_duration = time.time() - start_time
            
            num_examples = len(self.traindata.dataset) if hasattr(self.traindata, 'dataset') else len(self.traindata)
           
            return  (self.get_parameters(), num_examples, {
                "data_size": num_examples,
                "duration": training_duration,
                     "client_cid": self.client_id,           # your logical ID
            "flower_node_id": str(self.context.node_id),   # stringify for CSV safety
                "logical_id": f"client_{self.client_id}",  # ensures "client_0"

            })
      
        except Exception as e:
            print(f"ERROR: Client {self.client_id} FAILURE during fit round {round_number}: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
       
        req = (config or {}).get("request", None)
        print(f"Client {self.client_id} - get_properties called with request={req}")

        # ------------------------------------------------------------------
        # 1) IDENTITY REQUEST (used by _refresh_uuid_mapping)
        # ------------------------------------------------------------------
        if req == "identity":
            # MUST be super cheap and MUST NOT fail
            lid = str(self.client_id)
            return {
                "logical_id": lid,
                "client_cid": lid,
                "simulation_index": lid,
            }

        # ------------------------------------------------------------------
        # 2) PROTOTYPES REQUEST (used by your clustering stage)
        # ------------------------------------------------------------------
        if req == "prototypes":
            print(f"Client {self.client_id} - Server requesting prototypes")

            # Only touch disk when prototypes are actually requested
            if self.prototypes_from_last_round is None or self.class_counts_from_last_round is None:
                # Try loading from disk
                self._load_prototypes_from_disk()

            has_protos = self.prototypes_from_last_round is not None
            has_counts = self.class_counts_from_last_round is not None

            print(
                f"Client {self.client_id} - prototype file exists: "
                f"{self.prototype_file.exists()}, counts file exists: {self.counts_file.exists()}"
            )

            if has_protos and has_counts:
                try:
                    prototypes_bytes = pickle.dumps(self.prototypes_from_last_round)
                    class_counts_bytes = pickle.dumps(self.class_counts_from_last_round)

                    prototypes_b64 = base64.b64encode(prototypes_bytes).decode("utf-8")
                    class_counts_b64 = base64.b64encode(class_counts_bytes).decode("utf-8")

                    return {
                        "prototypes": prototypes_b64,
                        "class_counts": class_counts_b64,
                        "client_cid": str(self.client_id),
                        "flower_node_id": str(self.context.node_id) if self.context else "",
                    }
                except Exception as e:
                    print(f"ERROR: Client {self.client_id} - Encoding prototypes failed: {e}")

            # No prototypes available: still return something small and safe
            domain_id = getattr(getattr(self.traindata, "dataset", None), "domain_id", -1)
            return {
                "domain_id": str(domain_id),
                "client_cid": str(self.client_id),
            }

        # ------------------------------------------------------------------
        # 3) DEFAULT: return minimal identity
        # ------------------------------------------------------------------
        domain_id = getattr(getattr(self.traindata, "dataset", None), "domain_id", -1)
        return {
            "domain_id": str(domain_id),
            "simulation_index": str(self.client_id),
        }

    def _extract_and_cache_prototypes(self, round_number):
        """Extract prototypes from trained model and cache them."""
        print(f"Client {self.client_id} - Starting prototype extraction for round {round_number}")
        
        self.net.eval()
        class_embeddings = defaultdict(list)
        class_counts = defaultdict(int)
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.traindata):
                images, labels = batch
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
                
                # Forward pass to get embeddings
                # Adjust this based on your network architecture
                # Assuming net returns (embeddings, logits, aux_output) or similar
                h, _, _ = self.net(images)
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    # Convert to CPU and numpy immediately
                    class_embeddings[label].append(h[i].detach().cpu().numpy())
                    class_counts[label] += 1
                    total_samples += 1
        
        print(f"Client {self.client_id} - Processed {total_samples} samples")
        print(f"Client {self.client_id} - Classes found: {list(class_embeddings.keys())}")
        print(f"Client {self.client_id} - Class counts: {dict(class_counts)}")
        
        # Compute prototypes as NumPy arrays
        prototypes = {}
        embedding_dim = None
        
        for class_id in range(self.num_classes):
            if class_id in class_embeddings:
                # Stack as numpy arrays and compute mean
                stacked = np.stack(class_embeddings[class_id])
                prototypes[class_id] = stacked.mean(axis=0).astype(np.float32)
                if embedding_dim is None:
                    embedding_dim = prototypes[class_id].shape[0]
            else:
                # Handle missing classes with numpy zeros
                if embedding_dim is not None:
                    prototypes[class_id] = None
                """
                elif len(class_embeddings) > 0:
                    # Get dimension from any existing embedding
                    sample_embedding = next(iter(class_embeddings.values()))[0]
                    prototypes[class_id] = None
                    if embedding_dim is None:
                        embedding_dim = sample_embedding.shape[0]
                
                else:
                    print(f"ERROR: Client {self.client_id} - No embeddings found!")
                    return
                """
        
        # Cache prototypes and class counts IN MEMORY
        self.prototypes_from_last_round = prototypes
        self.class_counts_from_last_round = dict(class_counts)
        
        # IMMEDIATELY save to disk for persistence
        self._save_prototypes_to_disk()
        
        print(f"Client {self.client_id} - Successfully cached and saved prototypes for {len(prototypes)} classes")
        print(f"Client {self.client_id} - Prototype shapes: {[(k, v.shape) for k, v in list(prototypes.items())[:3]]}")
        
        # Verify persistence
        has_prototypes = hasattr(self, 'prototypes_from_last_round')
        has_class_counts = hasattr(self, 'class_counts_from_last_round')
        prototypes_not_none = has_prototypes and self.prototypes_from_last_round is not None
        counts_not_none = has_class_counts and self.class_counts_from_last_round is not None
        
        print(f"Client {self.client_id} - POST-SAVE verification:")
        print(f"  - has_prototypes: {has_prototypes}")
        print(f"  - has_class_counts: {has_class_counts}")
        print(f"  - prototypes_not_none: {prototypes_not_none}")
        print(f"  - counts_not_none: {counts_not_none}")
        print(f"  - prototype file exists: {self.prototype_file.exists()}")
        print(f"  - counts file exists: {self.counts_file.exists()}")

    def _save_prototypes_to_disk(self):
        """Save prototypes and class counts to disk for persistence."""
        try:
            if hasattr(self, 'prototypes_from_last_round') and self.prototypes_from_last_round is not None:
                with open(self.prototype_file, 'wb') as f:
                    pickle.dump(self.prototypes_from_last_round, f)
                print(f"Client {self.client_id} - Saved prototypes to {self.prototype_file}")
            
            if hasattr(self, 'class_counts_from_last_round') and self.class_counts_from_last_round is not None:
                with open(self.counts_file, 'wb') as f:
                    pickle.dump(self.class_counts_from_last_round, f)
                print(f"Client {self.client_id} - Saved class counts to {self.counts_file}")
                
        except Exception as e:
            print(f"ERROR: Client {self.client_id} - Failed to save prototypes: {e}")
    
    def _load_prototypes_from_disk(self):
        """Load prototypes and class counts from disk."""
        try:
            if self.prototype_file.exists():
                with open(self.prototype_file, 'rb') as f:
                    self.prototypes_from_last_round = pickle.load(f)
                print(f"Client {self.client_id} - Loaded prototypes from disk")
            else:
                self.prototypes_from_last_round = None
                print(f"Client {self.client_id} - No existing prototypes on disk")
            
            if self.counts_file.exists():
                with open(self.counts_file, 'rb') as f:
                    self.class_counts_from_last_round = pickle.load(f)
                print(f"Client {self.client_id} - Loaded class counts from disk")
            else:
                self.class_counts_from_last_round = None
                print(f"Client {self.client_id} - No existing class counts on disk")
                
        except Exception as e:
            print(f"ERROR: Client {self.client_id} - Failed to load prototypes: {e}")
            self.prototypes_from_last_round = None
            self.class_counts_from_last_round = None

    def train(self, net, trainloader, client_id, epochs,cfg, simulate_delay=False):
        """Train the network on the training set."""

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        num_classes=9
        #net.to(self.device)
        net.train()
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        SAVE_EVERY_STEPS   = 2000   # keep this fairly large
        PRINT_EVERY_STEPS  = 100
        KEEP_LAST          = 3      # rotate checkpoints
        MAX_STEPS          = int(1e9)  # effectively "no cap"; epochs control duration
        amp_enabled=True
        # Metrics (binary classification)
        accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        precision = Precision(task="multiclass", num_classes=num_classes, average='macro').to(self.device)
        recall = Recall(task="multiclass", num_classes=num_classes, average='macro').to(self.device)
        f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(self.device)
  
        # ——— Prepare CSV logging ———
        log_filename = f"client_cluster_train_{client_id}_loss_log.csv"
        write_header = not os.path.exists(log_filename)
        with open(log_filename, 'a', newline='') as csvfile:
          writer = csv.writer(csvfile)
          if write_header:
            writer.writerow([
                "epoch","train_loss",
                "accuracy","precision","recall","f1"
            ])
     
        for epoch in range(epochs):
            accuracy.reset()
            precision.reset()
            recall.reset()
            f1_score.reset()
            correct, total, epoch_loss ,loss_sumi ,loss_sum = 0, 0, 0.0 , 0 , 0
            for batch_idx, (images, labels) in enumerate(trainloader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                # ---- Mixed-precision forward & loss ----
                with autocast(enabled=amp_enabled):
                  h, _, outputs = net(images)                 # adapt if your net returns differently
                  loss = criterion(outputs, labels)
                
                # ---- Scaled backward + optimizer step ----
                scaler.scale(loss).backward()
                # (Optional) gradient clipping:
                scaler.step(optimizer)
                scaler.update()

                # ---- Bookkeeping ----
                bs = images.size(0)
                epoch_loss += float(loss.detach().cpu().item()) * bs

                preds = torch.argmax(outputs, dim=1)
                accuracy.update(preds, labels)
                precision.update(preds, labels)
                recall.update(preds, labels)
                f1_score.update(preds, labels)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

            # ---- End of epoch metrics ----
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = accuracy.compute().item()
            epoch_prec = precision.compute().item()
            epoch_rec  = recall.compute().item()
            epoch_f1   = f1_score.compute().item()

            print(f"local Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f} (Client {client_id})")
            print(f"Acc={epoch_acc:.4f}, Precision={epoch_prec:.4f}, Recall={epoch_rec:.4f}, F1={epoch_f1:.4f} (Client {client_id})")

            # ---- Log to CSV ----
            with open(log_filename, 'a', newline='') as csvfile:
              writer = csv.writer(csvfile)
              writer.writerow([epoch+1, epoch_loss, epoch_acc])

        if simulate_delay:
          import random
          base = float(cfg.get("delay_base_sec", 10.0))
          jitter = float(cfg.get("delay_jitter_sec", 3.0))
          delay = base + random.uniform(0.0, jitter)
          uuid=client_id
          print(f"[client {uuid}] Simulating straggler delay: {delay:.2f}s")
          time.sleep(delay)

def get_client_signature(dataset):
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    return md5(str(all_labels).encode()).hexdigest()


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    model=None,
experiment_name =None,
strategy='fedavg',
domain_assignment=None,
cfg=None  ,
 device= torch.device("cuda")

) -> Callable[[Context], Client]:  # pylint: disable=too-many-arguments
    # be a straggler. This is done so at each round the proportion of straggling
    client = MlflowClient()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def client_fn(context: Context) -> Client:
        # Destroy any left-over state records that might have bad TTLs
       
        # Access the client ID (cid) from the context
        cid = context.node_config["partition-id"]
     
        input_dim = 28  # Example: 28x28 images flattened
        hidden_dim = 128
        latent_dim = 64
        num_classes = 9
        num_epochs=3
        #the same model for all methods
        model = ModelCDCSF(out_dim=256, n_classes=9).to(device)

        trainloader = trainloaders[int(cid)]

        images, labels = next(iter(trainloader))
        #print(f"Saved sample image for client {cid} (label={images[0]})")
        # Access the original client_id stored in the dataset
        dataset_client_id = trainloader.dataset.client_id
        #print(f"[Client {cid}] dataset_client_id = {dataset_client_id}")
        valloader = valloaders[int(cid)]  
        numpy_client =  FederatedClient(
           model,
            trainloader,
            valloader,
            num_epochs,
            cid
            ,
            device,
            context

          )
        return numpy_client.to_client()

       
       
      
    return client_fn

