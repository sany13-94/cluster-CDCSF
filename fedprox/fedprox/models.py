"""CNN model architecture, training, and testing functions for MNIST."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
#GLOBAL Generator 
from torchmetrics import Accuracy, Precision, Recall, F1Score
# use a Generator Network with reparametrization trick
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.swin_transformer import SwinTransformer
#model vit
#from vit_pytorch.vit_for_small_dataset import ViT
import sys
import os
from torch.cuda.amp import autocast, GradScaler
# Get the path to the nested repo relative to your current script
nested_repo_path = os.path.join(os.path.dirname(__file__),  "..","Swin-Transformer-fed")
sys.path.append(os.path.abspath(nested_repo_path))
print(f'gg: {nested_repo_path}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch

Tensor = torch.FloatTensor
   
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetPathMNIST(nn.Module):
    """ResNet backbone adapted for PathMNIST (28x28 RGB images)"""
    
    def __init__(self, block, layers, num_classes=9):
        super().__init__()
        
        # Initial channel is 3 for RGB images (PathMNIST)
        self.inplanes = 32  # Reduced from 64 for smaller 28x28 images
        
        # First conv layer for 28x28 RGB input
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Main ResNet layers
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18_pathmnist():
    """ResNet-18 model adapted for PathMNIST dataset"""
    return ResNetPathMNIST(BasicBlock, [2, 2, 2, 2])

# =========== PathMnist DATASET ======

class ModelCDCSF(nn.Module):

    """
    Simplified model for federated learning with prototype extraction.
    
    Architecture:
    - Feature extractor: ResNet18 backbone
    - Classification head: Linear layer for final prediction
    
    Returns:
    - h: Feature embeddings from backbone (for prototype extraction)
    - _: Placeholder (kept for compatibility)
    - y: Final predictions (for classification)
    """

    def __init__(self, out_dim=256, n_classes=9):
        """
        Args:
            out_dim: Not used, kept for compatibility
            n_classes: Number of output classes
        """
        super().__init__()

        # Base ResNet18 for PathMNIST
        basemodel = resnet18_pathmnist()
        
        # Feature extractor (all layers except final FC)
        self.features = nn.Sequential(*list(basemodel.children())[:-1])
        
        # Get the feature dimension from the base model
        num_ftrs = basemodel.fc.in_features  # Should be 256 for ResNet18

        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Classification head (directly from features to classes)
        self.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        # Feature extraction
        h = self.features(x)
        h = h.squeeze()  # Remove spatial dimensions: [B, 256, 1, 1] -> [B, 256]
        
        # Handle case where batch size is 1
        if h.dim() == 1:
            h = h.unsqueeze(0)

        #features for clustering
        z = self.projection(h)
        #normalization for cosine similairty
        z = F.normalize(self.projection(h), dim=1)

        # Classification
        y = self.fc(h)
        
        # Return (features, placeholder, predictions)
        # Placeholder is None but maintains 3-return compatibility
        return z , None, y

'''
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetBreastMNIST(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super().__init__()
        
        # Initial channel is 1 for grayscale images
        self.inplanes = 32  # Reduced from 64 to handle smaller images
        
        # First conv layer modified for 28x28 grayscale input
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Main layers
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18_breastmnist():
    """ResNet-18 model adapted for BreastMNIST dataset"""
    return ResNetBreastMNIST(BasicBlock, [2, 2, 2, 2])

class ModelCDCSF(nn.Module):
    """Model for MOON."""

    def __init__(self,out_dim, n_classes):
        super().__init__()

        basemodel = resnet18_breastmnist()
        self.features = nn.Sequential(*list(basemodel.children())[:-1])
        num_ftrs = basemodel.fc.in_features
        self.feature_dim = num_ftrs  # ← ADD THIS LINE
      

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except KeyError as err:
            raise ValueError("Invalid model name.") from err

    def forward(self, x):
        """Forward."""
        h = self.features(x)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x,y
'''
import torch
import os

def train_gpaf( encoder: nn.Module,
classifier,
discriminator,
    trainloader: DataLoader,
    device: torch.device,
    client_id,
    epochs: int,
   global_generator,domain_discriminator
   ,
            decoder,batch_size
    ):

# j
    learning_rate=0.01
        
    train_one_epoch_gpaf(
        encoder,
classifier,discriminator , trainloader, device,client_id,
            epochs,global_generator,domain_discriminator
,
            decoder,batch_size
        )


import csv
#we must add a classifier that classifier into a binary categories
#send back the classifier parameter to the server
def train_one_epoch_gpaf(encoder,classifier,discriminator,trainloader, DEVICE,client_id, epochs,global_generator,local_discriminator,decoder,batch_size,verbose=False):
    """Train the network on the training set."""
    #criterion = torch.nn.CrossEntropyLoss()
    lr=0.00013914064388085564
    print(f" batch size at local model {batch_size}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Model on device: { DEVICE}')

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA available'}")


    encoder.to(DEVICE)
    classifier.to(DEVICE)
    discriminator.to(DEVICE)
    local_discriminator.to(DEVICE)
    decoder.to(DEVICE)
    global_generator.to(DEVICE)  # If used during training

    num_clients=2
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    optimizer_U = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-4)
    optimizer_L = torch.optim.Adam(local_discriminator.parameters(), lr=0.0002)

  
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    criterion_cls = nn.CrossEntropyLoss().to(DEVICE)  # Classification loss (for binary classification)
    criterion_mse = nn.MSELoss(reduction='mean')
    encoder.train()
    classifier.train()
    discriminator.train()
    local_discriminator.train()
    decoder.train()
    num_classes=9
    # Metrics (binary classification)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    precision = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    lambda_align = 1.0   # Full weight to alignment loss
    lambda_adv = 0.1  
    lambda_vae=1.0
    lambda_contrast = 0.9
    lambda_confusion = 1.0
    lambda_contrast = 0.9
                     
    # ——— Prepare CSV logging ———
    log_filename = f"client_gpaf_train_{client_id}_loss_log.csv"
    write_header = not os.path.exists(log_filename)
    with open(log_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "epoch","train_loss",
                "accuracy","precision","recall","f1"
            ])
    scaler = GradScaler()    
    for  epoch in range(epochs):
        print('==start local training ==')
        # Reset metrics for epoch
        accuracy.reset()
        precision.reset()
        recall.reset()
        f1_score.reset()
        correct, total, epoch_loss ,loss_sumi ,loss_sum = 0, 0, 0.0 , 0 , 0
        
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            images, labels = images.to(DEVICE , dtype=torch.float32 , non_blocking=True), labels.to(DEVICE  , dtype=torch.long , non_blocking=True)
         
            
            if labels.dim() > 1:
                labels = labels.squeeze()
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)  # Handle single sample
            
            real_imgs = images.to(DEVICE)
            # Generate global z representation
            batch_size = batch_size
            noise = torch.randn(batch_size, 64, dtype=torch.float32).to(DEVICE)
            labels_onehot = F.one_hot(labels.long(), num_classes=num_classes).float()
            noise = torch.tensor(noise, dtype=torch.float32)
            with autocast():
             with torch.no_grad():
              global_z = global_generator(noise, labels_onehot.to(DEVICE))
          
             optimizer_D.zero_grad()            
             if global_z is not None:
                    real_labels = torch.ones(global_z.size(0), 1, device=DEVICE, dtype=torch.float32)  # Real labels
                    #print(f' z shape on train {real_labels.shape}')
                    real_loss = criterion(discriminator(global_z), real_labels)
                    #print(f' dis glob z shape on train {discriminator(global_z).shape}')

             else:
                    real_loss = 0

             local_features = encoder(real_imgs)
            
             fake_labels = torch.zeros(real_imgs.size(0), 1 , dtype=torch.float32 , device=DEVICE)  # Fake labels
             fake_loss = criterion(discriminator(local_features.detach()), fake_labels)
           
             # Total discriminator loss
             d_loss = 0.5 * (real_loss + fake_loss)
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_D)
            scaler.update()
           

            optimizer_E.zero_grad()
            optimizer_C.zero_grad()
            optimizer_U.zero_grad()
            optimizer_L.zero_grad()
             
            # Get fresh features for encoder training
            with autocast():
             local_features = encoder(images)
             local_features.requires_grad_(True)
             reconstructed = decoder(local_features)
             recon_loss = criterion_mse(reconstructed, images)
             vae_loss=recon_loss
             g_loss = criterion(discriminator(local_features), real_labels)
             # Classification loss
             logits = classifier(local_features)  # Detach to avoid affecting encoder
             cls_loss = criterion_cls(logits, labels)
             local_features = encoder(images)          
         
             grl_features = GradientReversalLayer()(local_features)
             confusion_logits = local_discriminator(grl_features)
             # Create uniform distribution target
             uniform_target = torch.full(
                (batch_size, num_clients), 
             1.0/num_clients,
             device=device
             )
             confusion_loss = F.kl_div(
             F.log_softmax(confusion_logits, dim=1),
             uniform_target,
             reduction='batchmean'
             )
             # Add contrastive loss
             contrast_loss = contrastive_loss(local_features, global_z, temperature=0.5)

             total_loss =lambda_vae * vae_loss + lambda_adv * g_loss  + cls_loss+lambda_confusion * confusion_loss + lambda_contrast * contrast_loss 

            # backward + step all optimizers in one go
            scaler.scale(total_loss).backward()
            scaler.step(optimizer_E)
            scaler.step(optimizer_C)
            scaler.step(optimizer_U)
            scaler.step(optimizer_L)
            scaler.update()            
           
            # Update metrics
            preds = torch.argmax(logits, dim=1)
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1_score.update(preds, labels) 
            # Accumulate loss
            epoch_loss += total_loss.item()
            #loss += loss * labels.size(0)
            # Compute accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        epoch_acc = accuracy.compute().item()
        epoch_precision = precision.compute().item()
        epoch_recall = recall.compute().item()
        epoch_f1 = f1_score.compute().item()
        print(f"local Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f} (Client {client_id})")
        print(f"Accuracy = {epoch_acc:.4f}, Precision = {epoch_precision:.4f}, Recall = {epoch_recall:.4f}, F1 = {epoch_f1:.4f} (Client {client_id})")    
        save_client_model(client_id, encoder, classifier, decoder, save_dir="client_models")
        # log to CSV
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch+1, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1])
  
    
    print(f"local Epoch {epoch+1}: Loss_local/-discriminator = {loss_sum:.4f}, for (Client {client_id})")

    #return grads


def test_gpaf(net, testloader,device,num_classes=9):
        """Evaluate the network on the entire test set."""
        net.to(device)
       

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        # Initialize metrics
        accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        precision = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
        recall = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
        f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
        print(f' ==== client test func')
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device , dtype=torch.float32 , non_blocking=True), labels.to(device ,dtype=torch.long , non_blocking=True)
                if not num_classes==9: 
                  labels=labels.squeeze(1)
                #labels_onehot = F.one_hot(labels.long(), num_classes=num_classes).float()
                """
                print("Input shape:", inputs.shape)
                print("Labels:", labels)
                print("Labels dtype:", labels.dtype)
                print("Labels min/max:", labels.min().item(), labels.max().item())
                """
                # Forward pass
                h, _, outputs = net(inputs)

                # Compute loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy.update(predicted, labels)
                precision.update(predicted, labels)
                recall.update(predicted, labels)
                f1_score.update(predicted, labels)

        # Compute average loss and accuracy
        avg_loss = total_loss / len(testloader)
        avg_accuracy = correct / total
        # Print results
        print(f"Test Accuracy: {accuracy.compute():.4f}")
        print(f"Test Precision: {precision.compute():.4f}")
        print(f"Test Recall: {recall.compute():.4f}")
        print(f"Test F1 Score: {f1_score.compute():.4f}")
        return avg_loss, avg_accuracy

def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
  
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

#monn train and test
def save_client_model(client_id, encoder, classifier, decoder, save_dir="client_models"):
    """Save the state dictionaries of the client's models."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(save_dir, f"encoder_client_{client_id}.pth"))
    torch.save(classifier.state_dict(), os.path.join(save_dir, f"classifier_client_{client_id}.pth"))
    torch.save(decoder.state_dict(), os.path.join(save_dir, f"decoder_client_{client_id}.pth"))


def load_client_model(client_id, encoder, classifier, decoder, save_dir="client_models"):
    """Load the saved state dictionaries into the client's models."""
    encoder.load_state_dict(torch.load(os.path.join(save_dir, f"encoder_client_{client_id}.pth")))
    classifier.load_state_dict(torch.load(os.path.join(save_dir, f"classifier_client_{client_id}.pth")))
    decoder.load_state_dict(torch.load(os.path.join(save_dir, f"decoder_client_{client_id}.pth")))
    encoder.eval()
    classifier.eval()
    decoder.eval()
def init_net(output_dim, device="cpu"):
   
    n_classes=2
    net = ModelMOON(output_dim, n_classes)
    

    return net

def init_net(output_dim, device="cpu"):
   
    n_classes=2
    net = ModelMOON(output_dim, n_classes)
    

    return net

def save_client_model_moon(client_id, net, save_dir="client_models"):
    """Save the state dictionary of the client's model."""
    import os
    import torch
    
    # Create the save directory if it doesn’t exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the entire model’s state dictionary
    save_path = os.path.join(save_dir, f"model_client_moon_{client_id}.pth")
    torch.save(net.state_dict(), save_path)

def load_client_model_moon(client_id, net, save_dir="client_models"):
    """Load the saved state dictionary into the client's model."""
    import torch
    
    # Load the state dictionary from the file
    load_path = os.path.join(save_dir, f"model_client_moon_{client_id}.pth")
    net.load_state_dict(torch.load(load_path))
    
    # Set the model to evaluation mode
    net.eval()

def train_moon(
    net,
    global_net,
    previous_net,
    train_dataloader,
    epochs,
    lr,
    mu,
    temperature,
    device="cpu",
    client_id=None
):
    """Training function for MOON."""
    net.to(device)
    global_net.to(device)
    previous_net.to(device)
    #train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-5,
    )

    criterion = torch.nn.CrossEntropyLoss()
    

    previous_net.eval()
    for param in previous_net.parameters():
        param.requires_grad = False
    previous_net
    num_classes=2
    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)
    # Initialize metrics
    """
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    precision = Precision(task="multiclass", num_classes=num_classes).to(device)
    recall = Recall(task="multiclass", num_classes=num_classes).to(device)
    f1_score = F1Score(task="multiclass", num_classes=num_classes).to(device)
    """
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for _, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            """
            if len(target.shape) == 1:
                target = target.unsqueeze(1)
            else:
                  target=target.squeeze(1)

            """
            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            # pro1 is the representation by the current model (Line 14 of Algorithm 1)
            _, pro1, out = net(x)
            # pro2 is the representation by the global model (Line 15 of Algorithm 1)
            _, pro2, _ = global_net(x)
            # posi is the positive pair
            posi = cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            previous_net.to(device)
            # pro 3 is the representation by the previous model (Line 16 of Algorithm 1)
            _, pro3, _ = previous_net(x)
            # nega is the negative pair
            nega = cos(pro1, pro3)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

            previous_net.to("cpu")
            logits /= temperature
            labels = torch.zeros(x.size(0)).long()
            # compute the model-contrastive loss (Line 17 of Algorithm 1)
            loss2 = mu * criterion(logits, labels)
            # compute the cross-entropy loss (Line 13 of Algorithm 1)
            loss1 = criterion(out, target)
            # compute the loss (Line 18 of Algorithm 1)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())
            

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)

       
        print(
            "Epoch: %d Loss: %f Loss1: %f Loss2: %f"
            % (epoch, epoch_loss, epoch_loss1, epoch_loss2)
        )
        
    previous_net.to("cpu")
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    save_client_model_moon(client_id, net)
    print(f">> Training accuracy: %f of client : {client_id}" % train_acc)
    net.to("cpu")
    global_net.to("cpu")
    print(" ** Training complete **")
    return net ,global_net
    
def test_moon(net, test_dataloader, device="cpu" ,load_model=False, client_id=None):
    """Test function."""
    net.to(device)

    """Test function with optional model loading."""
    if load_model and client_id is not None:
        # Load the model if specified
        load_client_model_moon(client_id, net)
        test_acc, loss,test_acc,  test_prec, test_rec, test_f1= compute_accuracy(net, test_dataloader, device=device, load_model=load_model)
        print(f">> Test accuracy: {test_acc:.4f}")
        print(f">> Test precision: {test_prec:.4f}")
        print(f">> Test recall: {test_rec:.4f}")
        print(f">> Test F1 score: {test_f1:.4f}")
    else:

        test_acc, loss = compute_accuracy(net, test_dataloader, device=device)
    print(">> Test accuracy: %f" % test_acc)
    net.to("cpu")
    
    return test_acc, loss

def compute_accuracy(model, dataloader, device="cpu", load_model=False ,multiloader=False):
    """Compute accuracy."""
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    
    criterion = torch.nn.CrossEntropyLoss()
    # Initialize metrics
    num_classes=2
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    precision = Precision(task="multiclass", num_classes=num_classes).to(device)
    recall = Recall(task="multiclass", num_classes=num_classes).to(device)
    f1_score = F1Score(task="multiclass", num_classes=num_classes).to(device)
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for _, (x, target) in enumerate(loader):
                  
                    _, _, out = model(x)
                    if len(target) == 1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                   
                    pred_labels_list = np.append(
                    pred_labels_list, pred_label.numpy()
                        )
                    true_labels_list = np.append(
                            true_labels_list, target.data.numpy()
                        )
                    '''
                    else:
                        pred_labels_list = np.append(
                            pred_labels_list, pred_label.cpu().numpy()
                        )
                        true_labels_list = np.append(
                            true_labels_list, target.data.cpu().numpy()
                        )
                    '''
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for _, (x, target) in enumerate(dataloader):
                # print("x:",x)
                if not  was_training and not load_model:
                  
                  if len(target.shape) == 1:
                    target = target.unsqueeze(1)
                  else:
                    target=target.squeeze(1)

        
                _, _, out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())

                _, pred_label = torch.max(out, 1)
                accuracy.update(pred_label, target)
                precision.update(pred_label, target)
                recall.update(pred_label, target)
                f1_score.update(pred_label, target)
                '''
                else:
                    pred_labels_list = np.append(
                        pred_labels_list, pred_label.cpu().numpy()
                    )
                    true_labels_list = np.append(
                        true_labels_list, target.data.cpu().numpy()
                    )
                '''
            avg_loss = sum(loss_collector) / len(loss_collector)
            acc = accuracy.compute()
            prec = precision.compute()
            rec = recall.compute()
            f1 = f1_score.compute()



    if  was_training:
        model.train()

        return correct / float(total), avg_loss
    elif not load_model :
       return correct / float(total), avg_loss
    else:

    
      return correct / float(total), avg_loss,acc,prec,rec,f1

