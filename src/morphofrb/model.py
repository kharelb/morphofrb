import torch
import torchvision
from torch import nn


class CustomConvnext(nn.Module):
    """Convnext model with a modified classifier head for binary classification.
    """
    
    def __init__(self, weight=None):
        """
        Parameters:
        ----------
            weight : str
                The weight to use for the pretrained model. Default is None which only loads the model
                architecture without weight. If you want to include pretrained weights then include
                one of these ["DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"].
        """
        if (weight is not None) and (weight not in ["DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"]):
            raise ValueError(
                "Invalid weight. Choose from 'DEFAULT', 'IMAGENET1K_V1', or 'IMAGENET1K_V2' or None to load the model without pretrained weights."
            )

        super().__init__()
        self.pretrained = torchvision.models.convnext_base(weights=weight)

		# Freeze all layers
        for param in self.pretrained.parameters():
            param.requires_grad = False

		# Modify the classifier head to have 1 output for binary classification
        self.pretrained.classifier[2] = nn.Linear(1024, 1, bias=True)

		#  Again freeze all the layers as the last layer is modified
        # All the layers are frozen for consistency. 
        for param in self.pretrained.parameters():
            param.requires_grad = False


    def forward(self, x):
        """Forward pass of the model.
		
		Parameters:
		----------
			x : torch.Tensor
				Input tensor of shape (batch_size, channels, height, width).
		
		Returns:
		-------
			torch.Tensor
				Output tensor after passing through the Convnext model.
		"""

        return self.pretrained(x)