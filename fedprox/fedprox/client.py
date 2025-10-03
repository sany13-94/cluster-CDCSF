"""Defines the MNIST Flower Client and a function to instantiate it."""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
from flwr.common import Context
import flwr as fl
import numpy as np
import torch
import copy
from datetime import datetime
from collections import defaultdict
import torch.nn.functional as F
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import json
from flwr.client import NumPyClient, Client,  NumPyClient

  
from flwr.common import Status, Code, parameters_to_ndarrays ,ConfigsRecord, MetricsRecord, ParametersRecord ,Context, ConfigRecord

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
from fedprox.models import train_gpaf,test_gpaf,Encoder,Classifier,Discriminator,GlobalGenerator,LocalDiscriminator,init_net,train_moon,test_moon,Decoder,save_client_model,load_client_model
from fedprox.dataset_preparation import compute_label_counts, compute_label_distribution
from fedprox.features_visualization import extract_features_and_labels,StructuredFeatureVisualizer
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, encoder: Encoder, classifier: Classifier, discriminator: Discriminator,
    decoder,
     data,validset,
     local_epochs,
     client_id,
      mlflow,
      run_id,
      feature_visualizer
      ,
            device):
        self.encoder = encoder
        self.classifier = classifier
        self.discriminator = discriminator 
        self.decoder=decoder
        self.traindata = data
        self.validdata=validset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_epochs=local_epochs
        self.client_id=client_id
        self.num_classes=9
        self.num_clients=2
        for batch_idx, (data, target) in enumerate(self.validdata):
          print(f"dd Batch {batch_idx}, data shape: {data.shape}, target shape: {target.shape}")
          break  # J
        # Move models to device
        self.encoder.to(self.device)
        self.classifier.to(self.device)
        self.discriminator.to(self.device)
        #self.global_generator = GlobalGenerator(noise_dim=62, label_dim=2, hidden_dim=256  , output_dim=64)
        self.domain_dim=32
        self.global_generator = GlobalGenerator(noise_dim=62, label_dim=9,hidden_dim=256  , output_dim=64)
        # Initialize server discriminator with GRL
        # Initialize server discriminator with GRL
        self.domain_discriminator = LocalDiscriminator(
            feature_dim=64, 
            num_domains=self.num_clients
        ).to(self.device)
        self. mlflow= mlflow
        # Initialize optimizers
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters())
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters())
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters())
        self.run_id=run_id
        self.feature_visualizer=feature_visualizer
        # Initialize dictionaries to store features and labels
        self.client_features = {}  # Add this
        self.client_labels = {}    # Add this
        # Generator will be updated from server state
        #self.generator = None
    
    def get_parameters(self,config: Dict[str, Scalar] = None) -> List[np.ndarray]:
      """Return the parameters of the current encoder and classifier to the server.
        Exclude 'num_batches_tracked' from the parameters.
      """
      #print(f'Classifier state from server: {self.classifier.state_dict().keys()}')

      # Extract parameters and exclude 'num_batches_tracked'
      encoder_params = [val.cpu().numpy() for key, val in self.encoder.state_dict().items() if "num_batches_tracked" not in key]
      classifier_params = [val.cpu().numpy() for key, val in self.classifier.state_dict().items() if "num_batches_tracked" not in key]
      discriminator_params = [val.cpu().numpy() for key, val in self.domain_discriminator.state_dict().items() if "num_batches_tracked" not in key]
      parameters = encoder_params + classifier_params + discriminator_params

      #print(f' send client para format {type(parameters)}')

      return parameters
    #three run
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
      """Set the parameters of the encoder and classifier.
      Exclude 'num_batches_tracked' from the parameters.
      """
 
      num_encoder_params =len([key for key in self.encoder.state_dict().keys() if "num_batches_tracked" not in key])

      num_classifier_params = len([key for key in self.classifier.state_dict().keys() if "num_batches_tracked" not in key])
      num_discriminator_params = len([key for key in self.domain_discriminator.state_dict().keys() if "num_batches_tracked" not in key])
        
      # Extract encoder parameters
      encoder_params = parameters[:num_encoder_params]
      #print(f'encoder_params {encoder_params}')
      encoder_param_names = [key for key in self.encoder.state_dict().keys() if "num_batches_tracked" not in key]    
      params_dict_en = dict(zip(encoder_param_names, encoder_params))
      # Update encoder parameters
      encoder_state = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict_en.items()}
      )
      self.encoder.load_state_dict(encoder_state, strict=True)
      
      
      # Extract classifier parameters
      classifier_params = parameters[num_encoder_params : num_encoder_params + num_classifier_params]
      classifier_param_names = list(self.classifier.state_dict().keys())
      params_dict_cls = dict(zip(classifier_param_names, classifier_params))
      #print(f'classifier_params {classifier_params}')
      # Update classifier parameters
      classifier_state = OrderedDict(
          {k: torch.tensor(v) for k, v in params_dict_cls.items()}
      )

      self.classifier.load_state_dict(classifier_state, strict=False)

      discriminator_params = parameters[num_encoder_params + num_classifier_params : num_encoder_params + num_classifier_params + num_discriminator_params]

      # Load discriminator
      discriminator_param_names = [key for key in self.domain_discriminator.state_dict().keys() if "num_batches_tracked" not in key]
      params_dict_disc = dict(zip(discriminator_param_names, discriminator_params))
      discriminator_state = OrderedDict({k: torch.tensor(v) for k, v in params_dict_disc.items()})
      self.domain_discriminator.load_state_dict(discriminator_state, strict=True)
      
      
      print(f'Classifier parameters updated')


    #second and call set_para  
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        print(f'===evaluate client=== {type(parameters)}')
        self.set_parameters(parameters)

        for batch_idx, (data, target) in enumerate(self.validdata):
          print(f"evaluate dd Batch {batch_idx}, data shape: {data.shape}, target shape: {target.shape}")
          break  # J
        loss, accuracy = test_gpaf(self.encoder,self.classifier, self.validdata, self.device)
        #get the round in config
        # Log evaluation metrics using mlflow directly

        # Extract features and labels
        val_features, val_labels = extract_features_and_labels(
        self.encoder,
        self.validdata,
        self.device
           )
    
        if val_features is not None:
          self.client_features[self.client_id] = val_features
          self.client_labels[self.client_id] = val_labels

        with self.mlflow.start_run(run_id=self.run_id):  
            print(f' config client {config.get("server_round")}')
            self.mlflow.log_metrics({
                f"client_{self.client_id}/eval_loss": float(loss),
                f"client_{self.client_id}/eval_accuracy": float(accuracy),
                #f"client_round":float(round_number),
               # f"client_{self.client_id}/eval_samples": samples
            }, step=config.get("server_round"))
            # Also log in format for easier plotting
        
        #visualize all clients features per class
        features_np = val_features.detach().cpu().numpy()
        labels_np = val_labels.detach().cpu().numpy().reshape(-1)  # Ensure 1D array
        # In client:
        features_serialized = base64.b64encode(pickle.dumps(features_np)).decode('utf-8')
        labels_serialized = base64.b64encode(pickle.dumps(labels_np)).decode('utf-8')
        print(f"Client {self.client_id} sending features shape: {features_np.shape}")
        print(f"Client {self.client_id} sending labels shape: {labels_np.shape}")
         
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        return float(loss), len(self.validdata), {"accuracy": float(accuracy),
         "features": features_serialized,
            "labels": labels_serialized,
        }
    
    
    
    

    def fit(self, ins: FitIns) -> FitRes:
      try:
        parameters = parameters_to_ndarrays(ins.parameters)
        config = ins.config
        round_number = config.get("server_round", -1)
        simulate_delay = config.get("simulate_delay", False)
        
        print(f"🔥 DEBUG: Client {self.client_id} starting fit() for round {round_number}")
        
        # Send join timestamp
        self.send_status(f"{self.server_url}/join", {
            "client_id": self.client_id,
            "round": round_number,
            "timestamp": datetime.now().isoformat()
        })
        
        start_time = time.time()
        self.set_parameters(parameters)
        
        print(f"🔥 DEBUG: Client {self.client_id} starting training...")
        self.train(self.net, self.traindata, self.client_id, epochs=self.local_epochs, simulate_delay=simulate_delay)
        print(f"🔥 DEBUG: Client {self.client_id} completed training")
        
        # Send leave timestamp
        self.send_status(f"{self.server_url}/leave", {
            "client_id": self.client_id,
            "round": round_number,
            "timestamp": datetime.now().isoformat()
        })
        
        # === CRITICAL: Always extract and cache prototypes after training ===
        print(f"🔥 DEBUG: Client {self.client_id} starting prototype extraction...")
        self._extract_and_cache_prototypes_debug(round_number)
        
        training_duration = time.time() - start_time
        status = Status(code=Code.OK, message="Success")
        
        return FitRes(
            status=status,
            parameters=self.get_parameters(config).parameters,
            num_examples=len(self.traindata),
            metrics={
                "data_size": len(self.traindata),
                "duration": training_duration,
            }
        )
      except Exception as e:
        print(f"🔥 ERROR: Client {self.client_id} CRITICAL FAILURE during fit round {round_number}: {e}")
        import traceback
        traceback.print_exc()
        raise e

    def _extract_and_cache_prototypes(self):
      """Extract and cache prototypes from current model state."""
      self.net.eval()
      class_embeddings = defaultdict(list)
      class_counts = defaultdict(int)
    
      with torch.no_grad():
        for batch in self.traindata:
            images, labels = batch
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            h, _, _ = self.net(images)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_embeddings[label].append(h[i].cpu())
                class_counts[label] += 1
    
      # Compute prototypes
      prototypes = {}
      for class_id in range(self.num_classes):
        if class_id in class_embeddings:
            stacked = torch.stack(class_embeddings[class_id])
            prototypes[class_id] = stacked.mean(dim=0)
        else:
            # Use zero vector for missing classes
            if len(class_embeddings) > 0:
                # Get prototype dimension from existing embeddings
                sample_embedding = next(iter(class_embeddings.values()))[0]
                prototypes[class_id] = torch.zeros_like(sample_embedding)
            else:
                # Fallback: extract one sample to get dimensions
                with torch.no_grad():
                    sample_batch = next(iter(self.traindata))
                    sample_images = sample_batch[0][:1].to(self.device, dtype=torch.float32)
                    sample_h, _, _ = self.net(sample_images)
                    prototypes[class_id] = torch.zeros_like(sample_h[0].cpu())
    
      # Cache prototypes and class counts
      self.prototypes_from_last_round = prototypes
      self.class_counts_from_last_round = dict(class_counts)  # Convert defaultdict to regular dict
    
      print(f"Client {self.client_id} successfully cached prototypes for {len(prototypes)} classes.")
      print(f"Class counts: {dict(class_counts)}")

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
      """Send prototypes and class counts to server when requested."""
      status = Status(code=Code.OK, message="Success")
    
      if ins.config.get("request") == "prototypes":
        # Check if we have cached prototypes from previous training
        if (hasattr(self, 'prototypes_from_last_round') and 
            self.prototypes_from_last_round is not None and
            hasattr(self, 'class_counts_from_last_round') and
            self.class_counts_from_last_round is not None):
            
            try:
                # Encode prototypes and class counts
                prototypes_encoded = base64.b64encode(
                    pickle.dumps(self.prototypes_from_last_round)
                ).decode('utf-8')
                
                class_counts_encoded = base64.b64encode(
                    pickle.dumps(self.class_counts_from_last_round)
                ).decode('utf-8')
                
                print(f"Client {self.client_id}: Successfully sending prototypes and class counts to server.")
                
                return GetPropertiesRes(
                    status=status,
                    properties={
                        "prototypes": prototypes_encoded,
                        "class_counts": class_counts_encoded,
                    }
                )
                
            except Exception as e:
                print(f"Client {self.client_id}: Error encoding prototypes: {e}")
                return GetPropertiesRes(status=status, properties={})
        else:
            print(f"Client {self.client_id}: No prototypes available yet (hasn't participated in training).")
            return GetPropertiesRes(status=status, properties={})
    
      # Default response for other property requests
      return GetPropertiesRes(
        status=status, 
        properties={"simulation_index": str(self.client_id)}
    )



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
cfg=None  ,
 device= torch.device("cpu")

) -> Callable[[Context], Client]:  # pylint: disable=too-many-arguments
    import mlflow
    # be a straggler. This is done so at each round the proportion of straggling
    client = MlflowClient()
    
    def client_fn(context: Context) -> Client:
        # Access the client ID (cid) from the context
      cid = context.node_config["partition-id"]
      # Create or get experiment
      experiment = mlflow.get_experiment_by_name(experiment_name)
      if "mlflow_id" not in context.state.config_records:
            context.state.config_records["mlflow_id"] = ConfigRecord()
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      #check the client id has a run id in the context.state
      run_ids = context.state.config_records["mlflow_id"]

      if str(cid) not in run_ids:
            run = client.create_run(experiment.experiment_id)
            run_ids[str(cid)] = [run.info.run_id]
    
      with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_ids[str(cid)][0],nested=True) as run:
        run_id = run.info.run_id
        print(f"Created MLflow run for client {cid}: {run_id}")
        
        input_dim = 28  # Example: 28x28 images flattened
        hidden_dim = 128
        latent_dim = 64
        num_classes = 9
        num_epochs=3
        
               
        if strategy=="gpaf":
          
          img_shape=(28,28)
          encoder = Encoder(latent_dim).to(device)
          classifier = Classifier(latent_dim=64, num_classes=num_classes).to(device)
          #print(f' clqssifier intiliation {classifier}')
          discriminator = Discriminator(latent_dim=64).to(device)
          decoder = Decoder(latent_dim).to(device)
          # Note: each client gets a different trainloader/valloader, so each client
          # will train and evaluate on their own unique data
          trainloader = trainloaders[int(cid)]
          # Initialize the feature visualizer for all clients
          feature_visualizer = StructuredFeatureVisualizer(
        num_clients=num_clients,  # total number of clients
num_classes=num_classes,
save_dir="feature_visualizations"
          )
          #print(f'  ffghf {trainloader}')
          valloader = valloaders[int(cid)]
          for batch_idx, (data, target) in enumerate(valloader):
            print(f"Batch {batch_idx}, data shape: {data.shape}, target shape: {target.shape}")
            break  # Just check the first batch
          numpy_client =  FederatedClient(
            encoder,
            classifier,
            discriminator,
            
            decoder,
            trainloader,
            valloader,
            num_epochs,
            cid,
            mlflow
            ,
            run_id,
            feature_visualizer,
            device

          )

         
        elif strategy =="moon":
          
          trainloader = trainloaders[int(cid)]
          testloader = valloaders[int(cid)]
          return MOONFlowerClient(
            int(cid),
            cfg.output_dim,
            trainloader,
            testloader,
            device,
            num_epochs,        
            cfg.mu,
            cfg.temperature,
                
          )

        else:
          # Load model
          trainloader = trainloaders[int(cid)]
          valloader = valloaders[int(cid)]
          numpy_client = FlowerClient(
            model, trainloader, valloader,num_epochs,
           cid,run_id,mlflow)

        return numpy_client.to_client()
      
    return client_fn


# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus":1 , "num_gpus": 0.0}}
# When running on GPU, assign an entire GPU for each client
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`
class FlowerClient(NumPyClient):

    def __init__(self, net, trainloader, valloader,local_epochs,partition_id,run_id,mlflow):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs=local_epochs
        self.client_id=partition_id
        self.run_id=run_id
        self.mlflow=mlflow
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #update the local model with parameters received from the server
    def set_parameters(self,net, parameters: List[np.ndarray]):
      params_dict = zip(net.state_dict().keys(), parameters)
      state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
      net.load_state_dict(state_dict, strict=True)

    #get the updated model parameters from the local model return local model parameters
    
    def get_parameters(self , config: Dict[str, Scalar] = None):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    #get parameters from server train with local data end return the updated local parameter to the server
    def fit(self, parameters, config):

        self.set_parameters(self.net, parameters)
        self.train(self.net, self.trainloader,self.client_id,epochs=self.local_epochs)
        # Log the model after training
        """
        with mlflow.start_run(run_id=self.run_ids[str(self.client_id)][0], nested=True) as run:
            mlflow.pytorch.log_model(self.net, f"model_client_{self.client_id}")
        """
        return self.get_parameters(self.net), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
       
          server_round = config["server_round"]
          #print(f"Client {self.client_id} round id after training: {server_round}")
          self.set_parameters(self.net, parameters)
          loss, accuracy = self.test(self.net, self.valloader)
          print(f"Client {self.client_id} round id {server_round} , val accuracy: {accuracy}")
          #print(f'****evaluation**** {mlflow}')
          with self.mlflow.start_run(run_id=self.run_id):  
            self.mlflow.log_metrics({
                f"client_{self.client_id}/eval_loss": float(loss),
                f"client_{self.client_id}/eval_accuracy": float(accuracy),
               
            }, step=config.get("server_round"))
            # Also log in format for easier plotting
          print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
          # Extract features and labels
          val_features, val_labels = extract_features_and_labels(
          self.net,
         self.valloader,
          self.device
           )
          #visualize all clients features per class
          features_np = val_features.detach().cpu().numpy()
          labels_np = val_labels.detach().cpu().numpy().reshape(-1)  # Ensure 1D array
          # In client:
          features_serialized = base64.b64encode(pickle.dumps(features_np)).decode('utf-8')
          labels_serialized = base64.b64encode(pickle.dumps(labels_np)).decode('utf-8')
          #print(f"Client {self.client_id} sending features shape: {features_np.shape}")
          #print(f"Client {self.client_id} sending labels shape: {labels_np.shape}")
         
          #print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
          return float(loss), len(self.valloader), {"accuracy": float(accuracy),
         "features": features_serialized,
            "labels": labels_serialized,
          }
    
    def train(self,net, trainloader, client_id,epochs: int, verbose=False):
      """Train the network on the training set."""
      criterion = torch.nn.CrossEntropyLoss()
      lr=0.00013914064388085564
      optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
      net.train()
      for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:

            images, labels = batch
            labels=labels.long()
            #print(f'labels shape hh {labels.shape}')
            
            # Remove any squeeze operation since labels are already 1D
            if len(labels.shape) == 1:
              labels = labels.to(self.device)  # Just move to device
            else:
              labels=labels.squeeze(1)
            #print(f'after labels shape hh {labels.shape}')
            #print(labels)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        save_client_model_moon(client_id, net)
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc} of client : {client_id}")


    def test(self,net, testloader):
      """Evaluate the network on the entire test set."""
      criterion = torch.nn.CrossEntropyLoss()
      correct, total, loss = 0, 0, 0.0
      num_classes=2
      net.eval()
      # Initialize metrics
      accuracy = Accuracy(task="binary", num_classes=num_classes)
      precision = Precision(task="binary", num_classes=num_classes, average='macro')
      recall = Recall(task="binary", num_classes=num_classes, average='macro')
      f1_score = F1Score(task="binary", num_classes=num_classes, average='macro')
      print(f' ==== client test func')
      with torch.no_grad():
        for batch in testloader:
            images, labels = batch
            labels=labels.squeeze(1)
            
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy.update(predicted, labels)
            precision.update(predicted, labels)
            recall.update(predicted, labels)
            f1_score.update(predicted, labels)
            
      print(f"Test Accuracy: {accuracy.compute():.4f}")
      print(f"Test Precision: {precision.compute():.4f}")
      print(f"Test Recall: {recall.compute():.4f}")
      print(f"Test F1 Score: {f1_score.compute():.4f}")
      loss /= len(testloader.dataset)
      accuracy = correct / total
      return loss, accuracy   

#monn client side

class MOONFlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        # net: torch.nn.Module,
        net_id: int,
       
        output_dim: int,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        mu: float,
        temperature: float,
    ):  # pylint: disable=too-many-arguments
 
        self.net = init_net(output_dim)
        self.net_id = net_id
        #self.dataset = dataset
        
        self.output_dim = output_dim
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = 0.00013914064388085564
        self.mu = mu  # pylint: disable=invalid-name
        self.temperature = temperature
        self.model_dir="moon"
        self.client_id=net_id
        #self.model_dir = model_dir
        #self.alg = alg
        self.global_net=init_net(output_dim)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)
        print(f'model output :{self.output_dim}')
        prev_net = init_net(self.output_dim)
      
        if not os.path.exists(os.path.join(self.model_dir, str(self.net_id))):
            prev_net = copy.deepcopy(self.net)
        else:
            # load previous model from model_dir
            prev_net.load_state_dict(
                torch.load(
                    os.path.join(self.model_dir, str(self.net_id), "prev_net.pt")
                )
            )
        global_net = init_net(self.output_dim)
        global_net.load_state_dict(self.net.state_dict())
        self.global_net=global_net
        train_moon(
                self.net,
                global_net,
                prev_net,
                self.trainloader,
                self.num_epochs,
                self.learning_rate,
                self.mu,
                self.temperature,
                self.device,
                self.client_id
                )
        
        
        if not os.path.exists(os.path.join(self.model_dir, str(self.net_id))):
            os.makedirs(os.path.join(self.model_dir, str(self.net_id)))
        torch.save(
            self.net.state_dict(),
            os.path.join(self.model_dir, str(self.net_id), "prev_net.pt"),
        )
     
        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        # skip evaluation in the client-side
        loss = 0.0
        accuracy = 0.0
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        # Extract local features and labels
        val_features, val_labels = extract_features_and_labels(
          self.net,
         self.valloader,
          self.device
           )

        
        # Extract global features and labels
        if self.global_net is not None:
          print("========global features exist==========")
          global_features,global_labels = extract_features_and_labels(
          self.global_net,
         self.valloader,
          self.device
           )
          global_features_np = global_features.detach().cpu().numpy()
          global_features_serialized = base64.b64encode(pickle.dumps(global_features_np)).decode('utf-8')
          global_labels_np = global_labels.detach().cpu().numpy().reshape(-1)  # Ensure 1D array
          global_labels_serialized = base64.b64encode(pickle.dumps(global_labels_np)).decode('utf-8')

        else:
          global_features_serialized=""
        accuracy , loss = test_moon(self.net, self.valloader, device="cpu")

        #visualize all clients features per class
        features_np = val_features.detach().cpu().numpy()
        labels_np = val_labels.detach().cpu().numpy().reshape(-1)  # Ensure 1D array
        # In client:
        features_serialized = base64.b64encode(pickle.dumps(features_np)).decode('utf-8')
        labels_serialized = base64.b64encode(pickle.dumps(labels_np)).decode('utf-8')
        
        

        
        
        return float(loss), len(self.valloader), {"accuracy": float(accuracy),
         "features": features_serialized,
         "global_features": global_features_serialized,
            "labels": labels_serialized,
            "global_labels":global_labels_serialized
        }
       


  # Save the trained model to MLflow.    

  # Save the trained model to MLflow.    
