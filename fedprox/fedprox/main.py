"""Runs CNN federated learning for MNIST dataset."""

from typing import Dict, Union,Tuple
import mlflow
import flwr as fl
import hydra
import ray
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from fedprox import client, server, utils
from fedprox.client import gen_client_fn 
from fedprox.dataset import load_datasets
from fedprox.utils import LabelDistributionVisualizer,visualize_class_domain_shift
import mlflow
from  mlflow.tracking import MlflowClient
import time
import nest_asyncio
from flwr.common import ConfigsRecord, MetricsRecord, ParametersRecord
from fedprox.strategy import FedAVGWithEval
import os
import subprocess
#from fedprox.mlflowtracker import setup_tracking
from fedprox.features_visualization import StructuredFeatureVisualizer
#from fedprox.models import Generator
FitConfig = Dict[str, Union[bool, float]]
import mlflow
import subprocess
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import List
from torch.utils.data import DataLoader
import pandas as pd

from fedprox.visualizeprototypes import ClusterVisualizationForConfigureFit
strategy="gpaf"
 # Create or get experiment
experiment_name = "fedgpaf_Fed_FL32"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment with ID: {experiment_id}")
else:
        print(f"Using existing experiment with ID: {experiment.experiment_id}")
backend_config = {"client_resources": {"num_cpus":1 , "num_gpus": 0.0}}
# When running on GPU, assign an entire GPU for each client
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
# partition dataset and get dataloaders
def prepare_image_for_display(img_tensor):
    """Helper function to prepare tensor for matplotlib display."""
    img_np = img_tensor.detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    if img_np.shape[0] == 3:  # RGB
        return np.transpose(img_np, (1, 2, 0))
    else:  # Grayscale
        return img_np[0]

def visualize_client_participation(participation_dict, save_path="participation_chart.png", 
                                   method_name="FedProto-Fair"):
      """
      Simple bar chart showing how many times each client participated
    
      Args:
        participation_dict: Dictionary {client_id: count}
        save_path: Where to save the plot
        method_name: Name of your method for the title
      """
    
      # Sort clients by participation count
      sorted_items = sorted(participation_dict.items(), key=lambda x: int(x[0].split('_')[1]))
      client_ids = [item[0] for item in sorted_items]
      counts = [item[1] for item in sorted_items]
    
      # Create figure
      fig, ax = plt.subplots(figsize=(14, 6))
    
      # Create bar chart
      bars = ax.bar(range(len(client_ids)), counts, color='steelblue', edgecolor='black', alpha=0.7)
    
      # Highlight clients with 0 participation in red
      for i, count in enumerate(counts):
        if count == 0:
            bars[i].set_color('red')
            bars[i].set_alpha(0.5)
    
      # Customize plot
      ax.set_xlabel('Client ID', fontsize=12, fontweight='bold')
      ax.set_ylabel('Number of Participations', fontsize=12, fontweight='bold')
      ax.set_title(f'Client Participation Distribution - {method_name}', 
                 fontsize=14, fontweight='bold')
      ax.set_xticks(range(len(client_ids)))
      ax.set_xticklabels(client_ids, rotation=45, ha='right')
      ax.grid(axis='y', alpha=0.3, linestyle='--')
    
      # Add statistics text
      total_clients = len(client_ids)
      participated = sum(1 for c in counts if c > 0)
      avg_participation = np.mean(counts)
      std_participation = np.std(counts)
    
      stats_text = f'Total Clients: {total_clients}\n'
      stats_text += f'Participated: {participated} ({participated/total_clients*100:.1f}%)\n'
      stats_text += f'Avg Participation: {avg_participation:.2f} ± {std_participation:.2f}'
    
      ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
      plt.tight_layout()
      plt.savefig(save_path, dpi=300, bbox_inches='tight')
      print(f"Visualization saved to {save_path}")
      plt.show()

def visualize_from_train_loaders(train_loaders, k=15, d=3, image_idx=0):
   
    num_train_clients = k - 1
    clients_per_domain = num_train_clients // d
    
    # Build domain assignment (same logic as in make_pathmnist_clients_with_domains)
    domain_assignment = []
    for domain_id in range(d):
        for _ in range(clients_per_domain):
            domain_assignment.append(domain_id)
    
    # Handle remaining clients
    for i in range(num_train_clients - len(domain_assignment)):
        domain_assignment.append(i % d)
    
    # Select one representative client from each training domain
    representative_clients = []
    for domain_id in range(d):
        # Find first client in this domain
        client_id = domain_assignment.index(domain_id)
        representative_clients.append(client_id)
    
    # Add test domain client (last client, k-1)
    representative_clients.append(k - 1)
    
    print(f"\n[Visualization] Selected representative clients:")
    for i, client_id in enumerate(representative_clients[:-1]):
        print(f"  Domain {i}: Client {client_id}")
    print(f"  Test Domain (unshifted): Client {representative_clients[-1]}")
    
    # Equipment names for display
    equipment_names = ['High-End', 'Mid-Range', 'Older Model', 'High-End (Test)']
    
    # Create figure
    fig, axes = plt.subplots(1, d + 1, figsize=(5 * (d + 1), 5))
    if d + 1 == 1:
        axes = [axes]
    
    images = []
    labels = []
    
    # Collect images from each representative client
    for idx, client_id in enumerate(representative_clients):
        loader = train_loaders[client_id]
        
        # Get first batch
        batch_imgs, batch_labels = next(iter(loader))
        
        # Get the image at image_idx
        if image_idx >= len(batch_imgs):
            image_idx = 0  # Fallback to first image if index too large
        
        img = batch_imgs[image_idx]
        label = batch_labels[image_idx]
        
        images.append(img)
        labels.append(label.item())
        
        # Display image
        ax = axes[idx]
        img_display = prepare_image_for_display(img)
        ax.imshow(img_display)
        ax.axis('off')
        
        # Create title
        if idx < d:
            title = f'Domain {idx}\n({equipment_names[idx]})\n'
        else:
            title = f'Test Domain\n(unshifted)\n'
        title += f'Label: {label.item()}'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Print statistics
        domain_name = f"Domain {idx}" if idx < d else "Test Domain"
        print(f"\n  {domain_name} (Client {client_id}):")
        print(f"    Label: {label.item()}")
        print(f"    Image range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"    Mean: {img.mean():.3f}, Std: {img.std():.3f}")
    
    plt.suptitle(f'Preprocessed Images from Train Loaders (Batch Index: {image_idx})', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    return images, labels

def visualize_intensity_distributions(trainloaders: List[DataLoader], num_clients: int):
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    stats = {}
    
    for client_id in range(num_clients):
        try:
            # Get the complete dataset for this client
            client_dataset = trainloaders[client_id].dataset
            all_images = []
            
            # Collect ALL images for this client
            for idx in range(len(client_dataset)):
                image, _ = client_dataset[idx]
                all_images.append(image)
            
            # Convert to tensor and flatten
            images_flat = torch.stack(all_images).float().cpu().numpy().flatten()
            
            # Calculate statistics using all data points
            mean_val = np.mean(images_flat)
            std_val = np.std(images_flat)
            median_val = np.median(images_flat)
            
            stats[f'Client {client_id}'] = {
                'mean': mean_val,
                'std': std_val,
                'median': median_val
            }
            
            # Plot distribution using all data points
            sns.kdeplot(
                data=images_flat,
                ax=ax1,
                color=colors[client_id % len(colors)],
                label=f'Client {client_id}',
                linewidth=2
            )
            
        except Exception as e:
            print(f"Error processing client {client_id}: {str(e)}")
            continue
    
    ax1.set_title('Pixel Intensity Distributions Across Clients', fontsize=12)
    ax1.set_xlabel('Pixel Value', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Create statistics table
    stats_data = np.array([[
        stats[f'Client {i}']['mean'],
        stats[f'Client {i}']['std'],
        stats[f'Client {i}']['median']
    ] for i in range(num_clients)])
    
    sns.heatmap(
        stats_data.T,
        ax=ax2,
        xticklabels=[f'Client {i}' for i in range(num_clients)],
        yticklabels=['Mean', 'Std Dev', 'Median'],
        cmap='YlOrRd',
        annot=True,
        fmt='.4f',
        cbar_kws={'label': 'Value'}
    )
    ax2.set_title('Statistical Measures of Pixel Distributions', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('intensity_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    
def evaluate_metrics_aggregation_fn(eval_metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """Aggregate evaluation metrics from multiple clients."""
        # Unpack the evaluation metrics from each client
        losses = []
        accuracies = []
        for _, metrics in eval_metrics:
            losses.append(metrics["loss"])
            accuracies.append(metrics["accuracy"])
        
        # Aggregate the metrics
        return {
            "loss": sum(losses) / len(losses),
            "accuracy": sum(accuracies) / len(accuracies),
        }
def get_on_evaluate_config_fn():
    """Return a function which returns training configurations."""

    def evaluate_config(server_round: int):
        print('server round sanaa'+str(server_round))
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "server_round": str(server_round),
        }
        return config

    return evaluate_config

import os, glob, pickle
from flwr.common import Parameters

def load_latest_checkpoint(checkpoint_dir: str) -> dict | None:
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "round_*.pkl")))
    if not ckpts:
        print(f"[Checkpoint] No checkpoints found in {checkpoint_dir}, starting from scratch.")
        return None

    latest = ckpts[-1]
    print(f"[Checkpoint] Loading checkpoint: {latest}")
    with open(latest, "rb") as f:
        data = pickle.load(f)
    return data  # {"server_round": int, "parameters": Parameters, "metrics": dict}

def get_server_fn(mlflow=None):
 """Create server function with MLflow tracking."""
 def server_fn(context: Context) -> ServerAppComponents:
    global strategy

    initial_parameters = None

    # 1) Directory where previous version outputs are mounted (read-only)
    prev_ckpt_dir = "/kaggle/input/checkpoints9"

    # 2) Directory where this run will write new checkpoints
    curr_ckpt_dir = "/kaggle/working/cluster-CDCSF/fedprox/checkpoints"
    os.makedirs(curr_ckpt_dir, exist_ok=True)

    ckpt = load_latest_checkpoint(prev_ckpt_dir)

    initial_parameters = None
    base_round = 0  # default: brand new training

    # Total global rounds across ALL runs
    global_total_rounds = 100  #8 or cfg.num_rounds

    if ckpt is not None:
            initial_parameters = ckpt["parameters"]
            #base_round = ckpt["server_round"]   # e.g., 4
            print(f"[Resume] Resuming from global round {base_round}")

    # Number of rounds to run in THIS process
    remaining_rounds = max(1, global_total_rounds - base_round)
    print(f"[Config] global_total_rounds={global_total_rounds}, "
              f"base_round={base_round}, remaining_rounds={remaining_rounds}")
    if strategy=="fedavg":
      
      strategyi = FedAVGWithEval(
      fraction_fit=1.0,  # Train with 50% of available clients
      fraction_evaluate=1,  # Evaluate with all available clients
      min_fit_clients=4,
      min_evaluate_clients=4,
      min_available_clients=4,
      evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,  # Add this

      on_evaluate_config_fn=get_on_evaluate_config_fn(),
)
      print(f'strategy ggg {strategyi}')
    else: 
      print(f'strategy of method {strategy}')
      # Define stragglers
      ground_truth_stragglers = {f'client_{i}' for i in range(2)}
      strategyi = server.GPAFStrategy(
        experiment_name,
        fraction_fit=1.0,  # Ensure all clients participate in training
        #fraction_evaluate=1.0,
        min_fit_clients=  6,  # Set minimum number of clients for training
        min_evaluate_clients=6,
        min_available_clients=6,
         ground_truth_stragglers=ground_truth_stragglers,
     total_rounds=global_total_rounds,
   
  initial_parameters=initial_parameters,  # <-- use checkpoint
  save_dir=curr_ckpt_dir,   
    save_every=1,
    base_round=base_round,                       # <--- NEW
    meta_state=ckpt["meta_state"] if ckpt else None,  # <--- NEW
   
      )

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=remaining_rounds)
    return ServerAppComponents(strategy=strategyi, config=config)
 return server_fn

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    global experiment_name
    # In your main training script
    visualizer = StructuredFeatureVisualizer(
    num_clients=3,  # your number of clients
    num_classes=2,  # number of classes in your data
    
    )
    server_fn = get_server_fn()
    # Create mlruns directory
    os.makedirs("mlruns", exist_ok=True)
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    
   
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    trainloaders, valloaders,domain_assignment=data_load(cfg)

    #visualize_from_train_loaders(trainloaders, k=10, d=3, image_idx=0)
     
    #print(f'2: {valloaders[0]}')
    client_fn = gen_client_fn(
        num_clients=cfg.num_clients,
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        valloaders=valloaders,
        num_rounds=cfg.num_rounds,
        learning_rate=cfg.learning_rate,
        experiment_name=experiment_name,
        strategy=strategy,
        domain_assignment=domain_assignment
       )
    
    client = ClientApp(client_fn=client_fn)
    
    device = cfg.server_device
    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # resolve and convert to python dict
            fit_config: FitConfig = OmegaConf.to_container(  # type: ignore
                cfg.fit_config, resolve=True
            )
            fit_config["curr_round"] = server_round  # add round info
            return fit_config

        return fit_config_fn
   
    # Start simulation
    print(f'gpu number {cfg.client_resources.num_gpus}')
    if not ray.is_initialized():
        print("Initialisation du moteur Ray...")
        # L'argument 'num_gpus' ici FORCE Ray à réserver un GPU pour son propre fonctionnement 
        # et à l'enregistrer dans son système de ressources.
        # Nous utilisons torch.cuda.device_count() pour que Ray sache combien il y a de GPU.
        ray.init(log_to_driver=True, logging_level=30, num_gpus=torch.cuda.device_count()) 
        print(f"Ressources Ray disponibles: {ray.available_resources()}")
    # --
    server= ServerApp(server_fn=server_fn)
    history = run_simulation(
        client_app=client,
        server_app=server ,
          num_supernodes=cfg.num_clients,
      backend_config= {
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
            'client_resources': {'num_cpus': 2, 'num_gpus':cfg.client_resources.num_gpus }
            ,"_system_config": {
        "object_timeout_milliseconds": 600000,  # 10 minutes
        "num_heartbeats_timeout": 300,
    }
            }
       
      
    )
    # generate plots using the `history`
    
    save_path = HydraConfig.get().runtime.output_dir
    

    # Load saved validation data
    ground_truth_stragglers = {f'client_{i}' for i in range(2)}

    per_round_df = pd.read_csv("validation_results.csv")
    visualizer = ClusterVisualizationForConfigureFit()    
    print("[1/3] Generating straggler detection analysis...")
    visualizer.analyze_straggler_detection_with_ground_truth(
            validation_df=per_round_df,
            ground_truth_stragglers=ground_truth_stragglers,
            save_path="straggler_validation_detailed.png"
        )
        
   

    #save_results_as_pickle(history, file_path=save_path, extra_results={})
    
def data_load(cfg: DictConfig):
  trainloaders, valloaders,domain_assignment = load_datasets(
        config=cfg.dataset_config,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        domain_shift=True
    )
  print(f'1: {valloaders[0]}')
  return trainloaders, valloaders ,domain_assignment 
if __name__ == "__main__":
    
    main()
    