import json
import torch
import pandas as pd
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from torch.optim import lr_scheduler
from importlib.resources import files
from torchvision.transforms import v2

from .engine import train
from .config import Config
from .misc import FocalLoss
from . import custom_dataloader
from .model import CustomConvnext
from .weights import get_weights_path



def fine_tune(train_dir, val_dir, out_dir=Path.cwd(), focal_gamma=0, config=Config()):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{out_dir}/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Setup device agnostic code
    if torch.cuda.is_available():
        device = torch.device('cuda')  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = 'mps'   # Mac Metal M-Chip
    else:
        device = torch.device('cpu')
    print(f"--> Using device: {device}")
    
    if focal_gamma != 0:
        print(f"--> Implementing focal loss with gamma value: {focal_gamma}")

    else:
        print(f"--> Implementing binary crossentropy loss as gamma value of focal loss is 0.")
    
    
    
    if config.early_stopping:
        print(f"--> Early Stopping Enabled with patience: {config.patience}")
    
    
    # Define transformation
    transforms_train = v2.Compose([
        v2.Resize((224, 224)),  # ImageNet standard size
        v2.RandomAffine(degrees=0, translate=(0.1, 0.00)),
        v2.Grayscale(num_output_channels=3),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])  # ConvNext standard normalization
    ])
    
    transforms_val = v2.Compose([
        v2.Resize((224, 224)),
        v2.Grayscale(num_output_channels=3),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    
    
    
    # Create the dataloaders
    train_dl, classes = custom_dataloader.create_dataloader(target_dir=train_dir,
                                                            transform=transforms_train,
                                                            set_workers=config.set_workers,
                                                            num_workers=config.num_workers,
                                                            batch_size=config.batch_size,
                                                            w_sampler = config.weighted_sampling,
                                                            shuffle=config.shuffle_train,
                                                            r_sampling=config.replacement_sampling,
                                                            pin_memory=config.pin_memory
                                                           )

    val_dl, classes = custom_dataloader.create_dataloader(target_dir=val_dir,
                                                          transform=transforms_val,
                                                          set_workers=config.set_workers,
                                                          num_workers=config.num_workers,
                                                          batch_size=config.batch_size,
                                                          w_sampler = False,
                                                          shuffle=config.shuffle_val,
                                                          r_sampling=False,
                                                          pin_memory=config.pin_memory
                                                         )


    if config.fine_tune_chime_cattwo:
        # Load the architecture
        print("<<< Fine Tuning On Already Fine Tuned Model on CHIME/FRB Catalog 2 >>>")
        model = CustomConvnext(weight=None)
        DEFAULT_MODEL_PATH: Path = get_weights_path()
        model.load_state_dict(torch.load(DEFAULT_MODEL_PATH, weights_only=True, map_location=device))

    else:
        # The model layers are already frozen by default. 
        model = CustomConvnext(weight=config.model_weight)

    
    # Unfreeze the layers
    for layer in config.feature_layers_to_unfreeze.keys():
        module = model.pretrained.features[layer]
        if config.feature_layers_to_unfreeze.get(layer) == []:
            for param in module.parameters():
                param.requires_grad = True
        else:
            for item in config.feature_layers_to_unfreeze.get(layer):
                for param in module[item].parameters():
                    param.requires_grad = True

    if config.unfreeze_classification_layer:
        for params in model.pretrained.classifier.parameters():
                params.requires_grad = True
    
    
    # Define the loss function and optimizer
    loss_fn = FocalLoss(gamma=focal_gamma)  # with gamma 0 it is = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.pretrained.parameters()), lr=config.learning_rate)

    if config.learning_rate_scheduler:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=config.learning_rate_milestones,
            gamma=config.scheduler_gamma
        )
        print(f"--> Learning rate is scheduled with gamma value of {config.scheduler_gamma}")
    else:
        scheduler = None
    
    
    #Move the model to the device
    model.to(device)
    
    # Train the model
    print("--> Training the model...")
    loss_hist_train, loss_hist_val, acc_hist_train, acc_hist_val = train(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=config.num_epochs,
        verbose=config.print_training_details,
        early_stop=config.early_stopping,
        patience=config.patience,
        scheduler=scheduler,
        output_path=out_dir
    )
    
    # Store the result in a dictionary
    result = {
        'loss_hist_train': loss_hist_train,
        'loss_hist_val': loss_hist_val,
        'acc_hist_train': acc_hist_train,
        'acc_hist_val': acc_hist_val
    }
    
    
    print("--> Saving the final model...")
    # Save the model state dict to a file
    torch.save(model.state_dict(), str(out_dir / 'final_trained_model.pth'))
    print("--> Training completed successfully.....")
    print(f"--> Checkpoints and training result are saved in: {out_dir}")
    
    # Save the train results as a csv file
    df = pd.DataFrame(result)
    df.to_csv(str(out_dir / 'train_results.csv'), index=False)
    
    print(f"--> Model Training Completed...")

    

def main(argv=None):
    # Parse the arguments
    parser = ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help="Directory containing training files" )
    parser.add_argument('--val_dir', type=str, required=True, help="Directory containing validation files")
    parser.add_argument('--out_dir', type=str, help='Output directory to store the results')
    parser.add_argument('--focal_gamma', type=float, help='Gamma value for Focal Loss')
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    args = parser.parse_args(argv)
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    focal_gamma = args.focal_gamma if args.focal_gamma else 0
    if args.config:
        with open(args.config, 'r') as f:
            update_config = json.load(f)

        config = Config(**update_config)

    else:
        config = Config()

    fine_tune(
        train_dir=train_dir,
        val_dir=val_dir,
        out_dir=out_dir, 
        focal_gamma=focal_gamma, 
        config=config
        )

    return 0
    
if __name__ == "__main__":
    raise SystemExit(main())
    