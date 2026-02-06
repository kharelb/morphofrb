import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, Dict, List


class NpyDataset(Dataset):
    """
    A custom Dataset class to load numpy array 2D datasets.
    """
    def __init__(self, target_dir:str, transform=None):
        """
        Initialize the custom datareader class 

        Parameters:
        ----------
            target_dir : directory containing subdirectories of training and test datasets
            transform  : transformations to be applied to the dataset
        """
        self.target_dir = target_dir
        self.transform = transform

        # Create a list of paths for all the files that are in train and test directory
        self.paths = list(Path(target_dir).glob("*/*.npy"))


    def find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the  data classes folder names in a target (train/test) directory and creates
        classes (categories) with index labels.
        """


        # Get the class names by scanning the target dictionary
        classes = sorted(entry.name for entry in  os.scandir(self.target_dir) if entry.is_dir())

        # Raise an error if class names could not be found
        if not classes or len(classes) == 1:
            raise FileNotFoundError(f"Couldn't find any classes in {self.target_dir}.."
                                    f"..please check file structure")

        # Create a dictionary with index and labels
        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

        return classes, class_to_idx
    
    def class_sizes(self) -> Dict:
		# Get the class size in dictionary format
        classes = self.find_classes()[0]
        size_of_classes = {class_n: len(list((Path(self.target_dir) / class_n).rglob("*.npy"))) for class_n in classes}
        return size_of_classes

    # Override the default __getitem__() method from Pytorh's Dataset class
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns the data and label at the given index.
        Loads one image at a time and returns a tuple of data with its label.
        Loading of multiple images will be handled by DataLoader.
        """
        file = self.paths[index]
        img = np.load(file) # Loading an image for the given index.

        image_tensor = torch.from_numpy(img).unsqueeze(dim=0).to(torch.float32) # Convert the numpy file to pytorch tensor

        # Get the label
        class_name = file.parent.name
        classes_with_index = self.find_classes()[1]
        label = classes_with_index[class_name]

        # Perform image transformation if any
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label

    def __len__(self) -> int:
        """
        Returns the total number of samples.
        """
        return len(self.paths)


    def get_labels(self):
        """
        Returns the labels of the dataset on a given directory(train, test or validation).
        """
        class_labels = self.find_classes()[0]
        return [class_labels.index(self.paths[i].parent.name) for i in range(len(self.paths))]
