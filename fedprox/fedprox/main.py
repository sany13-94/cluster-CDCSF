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
from fedprox.client import gen_client_fn , FlowerClient
from fedprox.dataset import load_datasets
from fedprox.utils import LabelDistributionVisualizer,visualize_class_domain_shift
import mlflow
from  mlflow.tracking import MlflowClient
import time
import nest_asyncio
from flwr.common import ConfigsRecord, MetricsRecord, ParametersRecord
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

def visualize_domains_same_image(train_loaders, val_loaders, k=15, d=3, sample_idx=42):
    """
    Visualize the same image across all domains (0, 1, 2) and the test domain.
    
    Args:
        train_loaders: List of train DataLoaders for all clients
        val_loaders: List of val DataLoaders for all clients
        k: Total number of clients
        d: Number of training domains
        sample_idx: Index of the sample to visualize
    """
    
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
    
    # Select one representative client from each domain
    representative_clients = []
    for domain_id in range(d):
        # Find first client in this domain
        client_id = domain_assignment.index(domain_id)
        representative_clients.append(client_id)
    
    # Add test domain client (last client, client k-1)
    representative_clients.append(k - 1)
    
    print(f"[Visualization] Selected clients:")
    for i, client_id in enumerate(representative_clients[:-1]):
        print(f"  Domain {i}: Client {client_id}")
    print(f"  Test Domain (unshifted): Client {representative_clients[-1]}")
    
    # Create figure
    fig, axes = plt.subplots(1, d + 1, figsize=(4 * (d + 1), 4))
    
    # Collect images and labels
    images = []
    labels = []
    
    for idx, client_id in enumerate(representative_clients):
        # Get the train loader for this client
        loader = train_loaders[client_id]
        
        # Find the specific sample
        found = False
        for batch_imgs, batch_labels in loader:
            if len(batch_imgs) > sample_idx:
                # Get the sample
                img = batch_imgs[sample_idx]
                label = batch_labels[sample_idx]
                
                images.append(img)
                labels.append(label.item())
                found = True
                break
        
        if not found:
            # If sample_idx is too large, just get first sample
            for batch_imgs, batch_labels in loader:
                img = batch_imgs[0]
                label = batch_labels[0]
                images.append(img)
                labels.append(label.item())
                break
    
    # Plot images
    for idx, (img, label) in enumerate(zip(images, labels)):
        ax = axes[idx] if d + 1 > 1 else axes
        
        # Denormalize image for visualization
        # Assuming normalization was applied with mean=0.5, std=0.5 for each channel
        img_np = img.cpu().numpy()
        
        # Denormalize: img = (img * std) + mean
        # Reverse: img_denorm = img * 0.5 + 0.5
        img_denorm = img_np * 0.5 + 0.5
        
        # Clip to [0, 1] range
        img_denorm = np.clip(img_denorm, 0, 1)
        
        # Convert from CHW to HWC for display
        if img_denorm.shape[0] == 3:  # RGB
            img_display = np.transpose(img_denorm, (1, 2, 0))
        else:  # Grayscale
            img_display = img_denorm[0]
        
        ax.imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
        ax.axis('off')
        
        # Set title
        if idx < d:
            ax.set_title(f'Domain {idx} | Label: {label}', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Test Domain (unshifted) | Label: {label}', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Same Image Across All Domains (Sample Index from Batch)', 
                 fontsize=14, fontweight='bold', y=1.02)
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
def get_server_fn(mlflow=None):
 """Create server function with MLflow tracking."""
 def server_fn(context: Context) -> ServerAppComponents:
    global strategy
    if strategy=="fedavg":
      
      strategyi = FedAVGWithEval(
      fraction_fit=1.0,  # Train with 50% of available clients
      fraction_evaluate=0.5,  # Evaluate with all available clients
      min_fit_clients=8,
      min_evaluate_clients=2,
      min_available_clients=3,
      evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,  # Add this

      on_evaluate_config_fn=get_on_evaluate_config_fn(),
)
      print(f'strategy ggg {strategyi}')
    else: 
      print(f'strategy of method {strategy}')
      strategyi = server.GPAFStrategy(
        experiment_name,
        fraction_fit=1.0,  # Ensure all clients participate in training
        #fraction_evaluate=1.0,
        min_fit_clients=8,  # Set minimum number of clients for training
        min_evaluate_clients=8,
        min_available_clients=8,

        #on_fit_config_fn=fit_config_fn,
     
      )

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=15)
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

    trainloaders, valloaders=data_load(cfg)
    print("🔍 Visualizing same image index across clients to inspect domain shifts...")
    visualize_domains_same_image(trainloaders, valloaders, k=15, d=3, sample_idx=0)

    # Print data distribution before visualization
   
        
    #visualize_intensity_distributions(trainloaders, cfg.num_clients) 
    #visualize_class_domain_shift(trainloaders)    # Visualize label distributions
    #visualizer = LabelDistributionVisualizer(
    #    num_clients=cfg.num_clients,
    #    num_classes=2  # For binary classification in breast cancer dataset
    #)
    # Create visualization directory
    """
    viz_dir = os.path.join(os.getcwd(), 'visualizations')
    # Generate and save visualizations
    #save_path = os.path.join(viz_dir, 'initial_label_distribution.png')
    #client_distributions, global_distribution = visualizer.plot_label_distributions(
    #    trainloaders,
    #    save_path=save_path
    #)
    
    # Log distribution metrics
    #distribution_metrics = visualizer.compute_distribution_metrics(client_distributions)
    """
    CLIENT_RESOURCES = {
    "num_cpus": 2, 
    "num_gpus": 2.0 # Alloue une unité GPU à l'acteur Ray
} 

    if strategy=="gpaf":
      # à chaque client Ray dans la simulation.
     
      print(f'2: {valloaders[0]}')
      client_fn = gen_client_fn(
        num_clients=cfg.num_clients,
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        valloaders=valloaders,
        num_rounds=cfg.num_rounds,
        learning_rate=cfg.learning_rate,
        experiment_name=experiment_name,
        strategy=strategy
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
            }
       
      
    )
    # generate plots using the `history`
    
    save_path = HydraConfig.get().runtime.output_dir

    #save_results_as_pickle(history, file_path=save_path, extra_results={})
    
def data_load(cfg: DictConfig):
  trainloaders, valloaders = load_datasets(
        config=cfg.dataset_config,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        domain_shift=True
    )
  print(f'1: {valloaders[0]}')
  return trainloaders, valloaders   
if __name__ == "__main__":
    
    main()
    
