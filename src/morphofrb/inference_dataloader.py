import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader


transforms_val = v2.Compose([
        v2.Resize((224, 224)),
        v2.Grayscale(num_output_channels=3),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])


class CustomDataset(Dataset):
    def __init__(self, target_dir, transform=None):
        self.target_dir = target_dir
        self.transform = transform

        # Create a list of paths for all the files that are in train and test directory
        self.paths = list(Path(target_dir).glob("*.npy"))

        if len(self.paths) == 0:
            raise ValueError(
                f"Directory: {target_dir} contains no '.npy' files."
            )

        # Override the default __getitem__() method from Pytorh's Dataset class
    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Returns the data at the given index.
        Loading of multiple images will be handled by DataLoader.
        """
        file = self.paths[index]
        img = np.load(file) # Loading an image for the given index.

        image_tensor = torch.from_numpy(img).unsqueeze(dim=0).to(torch.float32) # Convert the numpy file to pytorch tensor

        # Perform image transformation if any
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, file.name

    def __len__(self) -> int:
        """
        Returns the total number of samples.
        """
        return len(self.paths)


def load_data(target_path, transform=transforms_val, batch_size=5):
    target_path = Path(target_path)
    if target_path.is_dir():
        dataset = CustomDataset(target_dir=target_path, transform=transforms_val)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    elif target_path.is_file() and target_path.suffix == '.npy':
        load_file = np.load(target_path)
        torch_img = torch.from_numpy(load_file).unsqueeze(dim=0).to(torch.float32)
        transformed = transform(torch_img)
        return transformed.unsqueeze(dim=0), target_path.name

    else:
        raise ValueError(
            f"Invalid path: {target_path}. Must be a directory containing '.npy' files or a '.npy' file."
        )