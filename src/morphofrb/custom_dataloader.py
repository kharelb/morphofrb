import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .custom_datareader import NpyDataset



def create_dataloader(
		target_dir: str,
		transform: transforms.Compose,
		batch_size: int = 32,
		num_workers: int = 0,
		shuffle: bool = True, 
		set_workers: bool = False,
		r_sampling: bool = True,
		w_sampler: bool = False,
		pin_memory: bool = False
		):
    """
	Creates a custom dataloader for numpy data set. 
	
	Parameters:
	----------
        target_dir    :  Path to the directory (train/test/validation) to load data.
		transform     :  Transformation to be applied to the images
		batch_size    :  Mini batch size
		num_workers   :  Number of sub-process to fetch and preprocess data
		shuffle       :  Randomly shuffle dataset. The shuffling should be disabled for weighted random sampling
		set_workers   :  If we want to change the value of num_workers from default value of 0
		r_sampling    :  Replacement sampling for weighted random sampler
		w_sampler     :  Weighted sampling while data loading
		pin_memory    :  Allocating memory in CPU when fetching data for faster data transfer between devices
		
    Return:
	------
        Returns a tuple of dataloader and class names i.e. names of categories.
    """
    
    dataset = NpyDataset(target_dir, transform=transform)    # Using NpyReader to create custom dataset

	# Get class names
    class_names = dataset.find_classes()[0]

	# setting number of workers will give an error on mac so set is optional
    num_workers_value = num_workers if set_workers else 0
	
    if w_sampler:
        # Create a weighted sampler for the unbalanced training data
        class_counts = [val for val in dataset.class_sizes().values()]
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[dataset.get_labels()]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                        replacement=r_sampling)
		
        
        shuffle = False  # If weighted sampler enabled then disable the shuffling
    
    else:
        sampler = None

	# Turn images into data loaders
    dataloader = DataLoader(
        dataset=dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		pin_memory=pin_memory,
		num_workers=num_workers_value,
		sampler=sampler
	)
    
    return dataloader, class_names
