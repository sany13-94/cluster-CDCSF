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
def get_model(model_name):
  if model_name == 'vit':
    model = ViT(
    image_size=28,        # specify image size
    patch_size=14,
    num_classes=2,        # specify the number of output classes
    dim=128,               # embedding dimension
    depth=8,               # number of transformer layers
    heads=4,               # number of attention heads
    mlp_dim=512,          # MLP hidden layer dimension
    pool='mean',            # 'cls' or 'mean' pooling
    channels=1,            # number of input channels (e.g., 3 for RGB images)
    dim_head=64,           # dimension per attention head
    dropout=0.3,
    #emb_dropout=0.1        # embedding dropout rate
    ).to(device)
  elif model_name == 'swim':
    layernorm = nn.LayerNorm
    USE_CHECKPOINT=False
    FUSED_WINDOW_PROCESS=False
    IMG_SIZE=28
    IN_CHANS=1
    NUM_CLASSES=2
    DEPTHS= [4,6]
    NUM_HEADS=[12,24]
    WINDOW_SIZE=7
    MLP_RATIO=4
    PATCH_SIZE=2
    EMBED_DIM=96
    QKV_BIAS=True
    QK_SCALE=None
    DROP_RATE=0.1
    DROP_PATH_RATE=0.2
    APE=False
    PATCH_NORM=True
    model = SwinTransformer(img_size=IMG_SIZE,
                                patch_size=PATCH_SIZE,
                                in_chans=IN_CHANS,
                                num_classes=NUM_CLASSES,
                                embed_dim=EMBED_DIM,
                                depths=DEPTHS,
                                num_heads=NUM_HEADS,
                                window_size=WINDOW_SIZE,
                                mlp_ratio=MLP_RATIO,
                                qkv_bias=QKV_BIAS,
                                qk_scale=QK_SCALE,
                                drop_rate=DROP_RATE,
                                drop_path_rate=DROP_PATH_RATE,
                                ape=APE,
                                norm_layer=layernorm,
                                patch_norm=PATCH_NORM,
                                use_checkpoint=USE_CHECKPOINT,
                                fused_window_process=FUSED_WINDOW_PROCESS)

  elif model_name =='resnet':
         model= resnet18_breastmnist()  # Using the ResNet model we defined
  return model
Tensor = torch.FloatTensor
# First, let's define the GRL layer for client side

class GradientReversalFunction(torch.autograd.Function):
    """
    Custom autograd function for gradient reversal.
    Forward: Acts as identity function
    Backward: Reverses gradient by multiplying by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        # Store lambda for backward pass
        ctx.lambda_ = lambda_
        # Forward pass is identity function
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient during backward pass
        # grad_output: gradient from subsequent layer
        # -lambda * gradient gives us gradient reversal
        return ctx.lambda_ * grad_output.neg(), None

class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer.
    Implements gradient reversal for adversarial training.
    """
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
        




class GlobalGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, hidden_dim, output_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        
        # Initial projection for noise
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Initial projection for labels
        self.label_proj = nn.Sequential(
            nn.Linear(label_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Mu and logvar projections
        self.mu_proj = nn.Linear(2 * hidden_dim, output_dim)
        self.logvar_proj = nn.Linear(2 * hidden_dim, output_dim)
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, noise, labels, return_distribution=False):
        # Project noise and labels to same dimension
        noise_feat = self.noise_proj(noise)  # [batch_size, hidden_dim]
        label_feat = self.label_proj(labels)  # [batch_size, hidden_dim]
        #print(f"noise_feat shape: {noise_feat.shape}")
        #print(f"label_feat shape: {label_feat.shape}")
        
        label_feat = label_feat.squeeze(1)  # Converts [32, 1, 256] → [32, 256]
        # Combine features
        combined = torch.cat([noise_feat, label_feat], dim=1)  # [batch_size, 2*hidden_dim]
        
        # Generate mu and logvar
        mu = self.mu_proj(combined)
        logvar = self.logvar_proj(combined)
        
        # Apply reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Final output projection
        features = self.output_proj(z)
        
        if return_distribution:
            return features, mu, logvar
        return features



class GlobalDiscriminator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.model(z)


def reparameterize(mu, logvar):

    std = torch.exp(0.5 * logvar)  # Standard deviation
    eps = torch.randn_like(std)    # Random noise from N(0, I)
    z = mu + eps * std             # Reparameterized sample
    return z
def sample_labels(batch_size, label_probs):
  
    #print(f'lqbel prob {label_probs}')
    # Extract probabilities from the dictionary
    probabilities = list(label_probs.values())
    
    # Extract labels from the dictionary
    labels = list(label_probs.keys())
    sampled_labels = np.random.choice(labels, size=batch_size, p=probabilities)
    return torch.tensor(sampled_labels, dtype=torch.long)

def generate_feature_representation(generator, noise, labels_one_hot):
   
    z = generator(noise, labels_one_hot)
    return z
#in our GPAF we will train a VAE-GAN local model in each client
img_shape=(28,28)

def reparameterization(mu, logvar,latent_dim):
    std = torch.exp(logvar / 2)
    #sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    sampled_z = torch.randn_like(mu)  # Sample from standard normal distribution
    z = sampled_z * std + mu
    return z


#replace shalow feature extractor with swim architecture 
'''
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        layernorm = nn.LayerNorm
        USE_CHECKPOINT=False
        FUSED_WINDOW_PROCESS=False
        IMG_SIZE=28
        IN_CHANS=1
        NUM_CLASSES=2
        DEPTHS= [4,6]
        NUM_HEADS=[12,24]
        WINDOW_SIZE=7
        MLP_RATIO=4
        PATCH_SIZE=2
        EMBED_DIM=96
        QKV_BIAS=True
        QK_SCALE=None
        DROP_RATE=0.1
        DROP_PATH_RATE=0.2
        APE=False
        PATCH_NORM=True
        # Replace sequential model with Swin Transformer
        self.swin = SwinTransformer(img_size=IMG_SIZE,
                                patch_size=PATCH_SIZE,
                                in_chans=IN_CHANS,
                                num_classes=NUM_CLASSES,
                                embed_dim=EMBED_DIM,
                                depths=DEPTHS,
                                num_heads=NUM_HEADS,
                                window_size=WINDOW_SIZE,
                                mlp_ratio=MLP_RATIO,
                                qkv_bias=QKV_BIAS,
                                qk_scale=QK_SCALE,
                                drop_rate=DROP_RATE,
                                drop_path_rate=DROP_PATH_RATE,
                                ape=APE,
                                norm_layer=layernorm,
                                patch_norm=PATCH_NORM,
                                use_checkpoint=USE_CHECKPOINT,
                                fused_window_process=FUSED_WINDOW_PROCESS)
  
   
        # Remove the classification head
        delattr(self.swin, 'head')
        
        # Add a feature processing layer (similar to your original 512 dims)
        self.feature_process = nn.Sequential(
            nn.Linear(self.swin.num_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512)
        )
        
        # Keep the same mu and logvar projections
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        # Get Swin features
        features = self.swin.forward_features(img)
        
        # Process features
        x = self.feature_process(features)
        
        # Get mu and logvar
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        # Sample using reparameterization
        z = reparameterization(mu, logvar, self.latent_dim)
        
        return z
'''
#resnet for fedavg

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
    def __init__(self, block, layers, num_classes=9):
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
"""
class globalDiscriminator(nn.Module):
    def __init__(self, noise_dim, feat_dim, hidden_dim):
        super().__init__()
        
        # Network to process noise vector
        self.noise_net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Network to process features
        self.feat_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, noise, features):
        noise_feat = self.noise_net(noise)
        z_feat = self.feat_net(features)
        
        # Concatenate both feature vectors
        combined = torch.cat([noise_feat, z_feat], dim=1)
        
        # Output prediction
        return self.output(combined)

class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim=latent_dim
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        #print(f"Encoder input shape (img): {img.shape}")  # Debug: Print input shape

        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        #print(f"Encoder model output shape (x): {x.shape}")  # Debug: Print model output shape

        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar,self.latent_dim)
        #print(f"Encoder output shape (z): {z.shape}")  # Debug: Print output shape

        #self._register_hooks()
        return z
    


# =========== bREASTmnIST DATASET ======
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        img_shape=(3,28,28)
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * (img_shape[1] // 4) * (img_shape[2] // 4), 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        x = self.model(img)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, self.latent_dim)
        return z # Return mu, logvar for KL loss

class LocalDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_domains):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 3)  # Output logits for each domain
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self,latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        # Register hooks to track shapes
        #self._register_hooks()

    def forward(self, z):
        validity = self.model(z)
        return validity
    

class Decoder(nn.Module):
    def __init__(self, latent_dim):
       
        super(Decoder, self).__init__()
        img_shape=(3,28,28)
        self.latent_dim = latent_dim
        self.img_shape = img_shape  # e.g., (1, 28, 28) for channels, height, width

        # Project latent vector to a spatial feature map
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)  # Initial feature map: 512 channels, 7x7

        # Transposed convolutional layers for upsampling
        self.conv_transpose_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0,1], matching your normalization
        )

    def forward(self, z):
        # Project latent vector to spatial dimensions
        x = self.fc(z)  # Shape: (batch_size, 512 * 7 * 7)
        x = x.view(-1, 512, 7, 7)  # Reshape to (batch_size, 512, 7, 7)
        x = self.conv_transpose_layers(x)  # Upsample to (batch_size, C, 28, 28)
        return x



class Classifier(nn.Module):
    def __init__(self,latent_dim,num_classes=2):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes),  # Output layer for multi-class classification
      
        )
        self._register_hooks()

    def forward(self, z):
        logits = self.model(z)
        return logits
  
#for moon model 

class ModelMOON(nn.Module):

    def __init__(self,out_dim, n_classes):
        super().__init__()

        basemodel = resnet18_breastmnist()
        self.features = nn.Sequential(*list(basemodel.children())[:-1])
        num_ftrs = basemodel.fc.in_features
      

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
        h = self.features(x)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x, y
"""
# =========== PathMnist DATASET ======
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        img_shape = (3, 28, 28)  # PathMNIST is RGB (3 channels)
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * (img_shape[1] // 4) * (img_shape[2] // 4), 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)
 
    def forward(self, img):
        x = self.model(img)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, self.latent_dim)
        return z  # Return z for feature alignment

class LocalDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_domains=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_domains)  # Output logits for each domain
        )
   
    def forward(self, x):
        return self.model(x)
 
class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
 
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        img_shape = (3, 28, 28)  # PathMNIST is RGB (3 channels)
        self.latent_dim = latent_dim
        self.img_shape = img_shape
 
        # Project latent vector to a spatial feature map
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)  # Initial feature map: 512 channels, 7x7
 
        self.conv_transpose_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0,1], matching your normalization
        )

    def forward(self, z):
        # Project latent vector to spatial dimensions
        x = self.fc(z)  # Shape: (batch_size, 512 * 7 * 7)
        x = x.view(-1, 512, 7, 7)  # Reshape to (batch_size, 512, 7, 7)
        x = self.conv_transpose_layers(x)  # Upsample to (batch_size, C, 28, 28)
        return x

class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes=9):  # PathMNIST has 9 classes
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes),
        )
 
    def forward(self, z):
        logits = self.model(z)
        return logits


# Save client models
def save_client_model(client_id, encoder, classifier, decoder, save_dir="client_models"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(save_dir, f"encoder_client_{client_id}.pth"))
    torch.save(classifier.state_dict(), os.path.join(save_dir, f"classifier_client_{client_id}.pth"))
    torch.save(decoder.state_dict(), os.path.join(save_dir, f"decoder_client_{client_id}.pth"))

# Load client models
def load_client_model(client_id, encoder, classifier, decoder, save_dir="client_models"):
    encoder.load_state_dict(torch.load(os.path.join(save_dir, f"encoder_client_{client_id}.pth")))
    classifier.load_state_dict(torch.load(os.path.join(save_dir, f"classifier_client_{client_id}.pth")))
    decoder.load_state_dict(torch.load(os.path.join(save_dir, f"decoder_client_{client_id}.pth")))
    encoder.eval()
    classifier.eval()
    decoder.eval()

import torch
import os

def print_gpu_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    os.system('nvidia-smi | grep MiB')  # Light, readable GPU info

#contrastive loss for gpaf
def contrastive_loss(local_features, global_features, temperature=0.5):
   cos = torch.nn.CosineSimilarity(dim=-1)
   
   # Local-global alignment (positive pairs)
   positive_sim = cos(local_features, global_features)
   positive_loss = torch.mean(1 - positive_sim)
   
   # Feature diversity (negative pairs)
   batch_size = local_features.size(0)
   feature_sims = cos(local_features.unsqueeze(1), local_features.unsqueeze(0))
   # Remove diagonal (self-similarity)
   mask = ~torch.eye(batch_size, dtype=torch.bool, device=local_features.device)
   negative_loss = torch.mean(feature_sims[mask])
   
   return positive_loss - temperature * negative_loss

# Function to save the model checkpoint
def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")
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


def test_gpaf(encoder,classifier, testloader,device,num_classes=9):
        """Evaluate the network on the entire test set."""
        encoder.to(device)
        classifier.to(device)
        encoder.eval()
        classifier.eval()

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
                features = encoder(inputs)
                outputs = classifier(features)

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

