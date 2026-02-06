import os
import tqdm
import torch
from torch import nn
from pathlib import Path
from typing import Dict, List, Tuple


def train(
        model: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int,
        verbose: bool = True,
        early_stop: bool = False,
        patience: int = 5,
        scheduler: torch.optim.lr_scheduler = None,
        output_path: Path = None
        ) -> Tuple[List[float], List[float], List[float], List[float]]:
    
    """A function to train/fine tune a neural network model.
    
    Parameters:
    ----------
        model       :   NN to be trained
        train_dl    :   Train data loader
        val_dl      :   Validation data loader
        loss_fn     :   Objective function to be minimized
        optimizer   :   Algorithm to minimize the loss function
        device      :   CPU/GPU to be used for optimizing
        num_epochs  :   Target number of iterations to optimize the model
        verbose     :   Option to print out loss and accuracies during optimization
        early_stop  :   Option to implement early stopping during optimization to prevent overfitting
        patience    :   Threshold number of epochs without improvement on the validation data
        scheduler   :   Algorithm to update the learnig rate during optimization
        output_path :   Output path to save the trained model

    Return:
    -------
        Returns a tuple of Lists : ([Train Loss], [Validation Loss], [Train Accuracy], [Validation Accuracy])
    """
    
    

    # Create empty lists to store the training and validation metrics
    loss_hist_train = [0] * num_epochs
    acc_hist_train = [0] * num_epochs
    loss_hist_val = [0] * num_epochs
    acc_hist_val = [0] * num_epochs

    # Set the counter for early stopping
    counter = 0
    # track epoch where best model was saved
    best_val_acc_epoch_tracker = []

    for epoch in tqdm.tqdm(range(num_epochs), leave=False, dynamic_ncols=True):
        # set the counter for early stopping
        # Put the model in train mode
        model.train()
        for x_batch, y_batch in train_dl:
            # Send data to target device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).to(torch.float32)

            # 1. Zero the gradients
            optimizer.zero_grad()

            # 2. Forward pass
            pred = model(x_batch)[:, 0]

            sig_pred = torch.sigmoid(pred)

            # 3. Compute loss
            loss = loss_fn(pred, y_batch)  # Because our loss function expects logits not probability

            # 4. Compute gradients
            loss.backward()

            # 5. Update weights
            optimizer.step()

            # Loss for the batch
            loss_hist_train[epoch] += loss.item()*len(y_batch)

            # Number of correct predictions for the batch
            is_correct = ((sig_pred >= 0.5).float() == y_batch).float()
            acc_hist_train[epoch] += is_correct.sum().item()

        # Calculate the average loss and accuracy for the epoch
        loss_hist_train[epoch] /= len(train_dl.dataset)
        acc_hist_train[epoch] /= len(train_dl.dataset)

        # Put the model in eval mode for validation
        model.eval()

        # Turn off gradients for validation
        with torch.inference_mode():
            for x_batch, y_batch in val_dl:
                # Send data to target device
                x_batch, y_batch = x_batch.to(device), y_batch.to(device).to(torch.float32)

                # 1. Forward pass
                pred = model(x_batch)[:, 0]

                sig_pred = torch.sigmoid(pred)   # Apply sigmoid to the output

                # 2. Compute loss
                loss = loss_fn(pred, y_batch)

                # Loss for the batch
                loss_hist_val[epoch] += loss.item()*len(y_batch)

                # Number of correct predictions for the batch
                is_correct = ((sig_pred >= 0.5).float() == y_batch).float()
                acc_hist_val[epoch] += is_correct.sum().item()

            # Calculate the average loss and accuracy for the epoch
            loss_hist_val[epoch] /= len(val_dl.dataset)
            acc_hist_val[epoch] /= len(val_dl.dataset)

        if scheduler is not None:
            scheduler.step()
            if epoch % 10 == 0:
                print(f"\tLearning rate: {scheduler.get_last_lr()[-1]}")

        if verbose:
            print(f"Epoch: {epoch} | {num_epochs - 1}")
            print(f"{'-' * 60}")
            print(f"\tTrain Loss   :   {loss_hist_train[epoch]:.4f}    |  Train Acc  :  {acc_hist_train[epoch]:.4f}")
            print(f"\tVal Loss     :   {loss_hist_val[epoch]:.4f}    |  Val Acc    :  {acc_hist_val[epoch]:.4f}")
        if epoch == 0:
            best_val_acc = acc_hist_val[epoch]
            best_val_acc_epoch_tracker.append(epoch)
            print(f"\tSaving the model at epoch: {epoch}...")
            try:
                torch.save(model.state_dict(), str(output_path / f'trained_model_{epoch}.pth'))
                print(f"\tModel was saved successfully...")
            except Exception as e:
                print(f"\tModel was not saved due to error: {e}")

        else:
            if (acc_hist_val[epoch] > best_val_acc):
                best_val_acc = acc_hist_val[epoch]
                print(f"\tSaving the model at epoch: {epoch}...")
                try:
                    torch.save(model.state_dict(), str(output_path / f'trained_model_{epoch}.pth'))
                    print(f"\tModel was saved successfully...\n")
                except Exception as e:
                    print(f"\tModel was not saved due to error: {e}")
                
                print("\tNow removing previous best model.....\n")
                if Path(output_path / f'trained_model_{best_val_acc_epoch_tracker[-1]}.pth').exists():
                    print(f"\tDeleting:trained_model_{best_val_acc_epoch_tracker[-1]}.pth")
                    os.remove(str(output_path / f'trained_model_{best_val_acc_epoch_tracker[-1]}.pth'))
                best_val_acc_epoch_tracker.append(epoch)
                counter = 0
            else:
                counter += 1

        if early_stop:
            print(f"\tEarly Stop Counter: {counter}")
            print(f"\tBest Validation Accuracy: {best_val_acc}")
    
            if counter >= patience:
                print(f"\tEarly stopping at epoch {epoch}")
                return loss_hist_train[:epoch+1], loss_hist_val[:epoch+1], acc_hist_train[:epoch+1], acc_hist_val[:epoch+1]

    return loss_hist_train, loss_hist_val, acc_hist_train, acc_hist_val