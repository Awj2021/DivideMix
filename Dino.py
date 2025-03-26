import torch
import torch.nn as nn
from typing import Union
from collections.abc import Iterable


def get_dino(
    n_classes: int, 
    repo_or_dir: str="facebookresearch/dinov2",
    model: str="dinov2_vits14",
    n_hidden_neurons: Union[int, list] = 128,
    dropout: float = 0.5,
    freeze_backbone: bool = True):
    """
    Create a DINO model for fine-tuning on classification tasks
    
    Args:
        n_classes: Number of output classes
        repo_or_dir: Repository or directory containing the DINO model
        model: Name of the DINO model variant to use
        n_hidden_neurons: Number of neurons in hidden layers (single int or list of ints)
        dropout: Dropout rate for the classification head
        freeze_backbone: Whether to freeze the DINO backbone during training
    """
    # Get the DINO backbone
    dino = torch.hub.load(repo_or_dir, model)
    
    # Get the feature dimension from the backbone
    n_features = dino.embed_dim
    
    # Create list of neurons for the classification head
    neuron_list = [n_features]
    if isinstance(n_hidden_neurons, int) and n_hidden_neurons > 0:
        n_hidden_neurons = [n_hidden_neurons]
    
    for n_hidden in n_hidden_neurons:
        neuron_list.append(n_hidden)
    
    n_last_layer_neurons = neuron_list[-1]

    def get_embed_x():
        module_list = []
        for i in range(len(neuron_list) - 1):
            module_list.append(nn.Linear(neuron_list[i], neuron_list[i+1]))
            module_list.append(nn.BatchNorm1d(num_features=neuron_list[i+1]))
            module_list.append(nn.ReLU())
            module_list.append(nn.Dropout(dropout))
        
        if len(module_list) > 1:
            return nn.Sequential(*module_list)
        else:
            return nn.Identity()
        
    def get_output():
        return nn.Linear(n_last_layer_neurons, n_classes)

    # Create the classification head
    classification_head = nn.Sequential(
        get_embed_x(),
        get_output()
    )

    # Create the complete model
    class DinoClassifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
            
            # Freeze the backbone if specified
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        
        def forward(self, x):
            # Get features from the backbone
            features = self.backbone(x)
            # Pass through the classification head
            return self.head(features)

    # Create and return the model
    model = DinoClassifier(dino, classification_head)
    return model




    
    