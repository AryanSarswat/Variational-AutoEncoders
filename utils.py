import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import gc
import numpy as np

import matplotlib.pyplot as plt

def modelSummary(model, verbose=False):
    """Method provides a description of a model and its parameters

    Args:
        model (nn.Module): The model to summarize
        verbose (bool, optional): Describes the model with specification for each layers. Defaults to False.
    """
    if verbose:
        print(model)
    
    total_parameters = 0
        
    for name, param in model.named_parameters():
        num_params = param.size()[0]
        total_parameters += num_params
        if verbose:
            print(f"Layer: {name}")
            print(f"\tNumber of parameters: {num_params}")
            print(f"\tShape: {param.shape}")
    
    if total_parameters > 1e5:
        print(f"Total number of parameters: {total_parameters/1e6:.2f}M")
    else:
        print(f"Total number of parameters: {total_parameters/1e3:.2f}K") 

def train_epoch(model: nn.Module, device: torch.device, train_dataloader: DataLoader, training_params: dict, metrics: dict):
    """Method to train a model for one epoch

    Args:
        model (nn.Module): Model to be trained by
        device (str): device to be trained on
        train_dataloader (nn.data.DataLoader): Dataloader object to load batches of dataset
        training_params (dict): Dictionary of training parameters containing "batch_size", "loss_function"
                                "optimizer".
        metrics (dict): Dictionary of functional methods that would compute the metric value

    Returns:
        run_results (dict): Dictionary of metrics computed for the epoch
    """
    OPTIMIZER = training_params["optimizer"]
    
    model = model.to(device)
    model.train()
    
    # Dictionary holding result of this epoch
    run_results = dict()
    for metric in metrics:
        run_results[metric] = 0.0
    run_results["loss"] = 0.0
    
    # Iterate over batches
    num_batches = 0
    for x, target in train_dataloader:
        num_batches += 1

        # Move tensors to device
        input = x.to(device)
        
        # Forward pass
        output = model(input)
        
        # Compute loss
        loss = ((output - input)**2).sum() + model.encoder.kl
        
        # Backward pass
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        
        # Update metrics
        run_results["loss"] += loss.detach().item()
        for key, func in metrics.items():
            run_results[key] += func(output, input).detach().item()
            
        # Clean up memory
        del loss
        del input
        del output
        del x 
        del target
        
    for key in run_results:
        run_results[key] /= num_batches
    
    return run_results


def evaluate_epoch(model: nn.Module, device: torch.device, validation_dataloader: DataLoader, training_params: dict, metrics: dict):
    """Method to evaluate a model for one epoch

    Args:
        model (nn.Module): model to evaluate
        device (str): device to evaluate on
        validation_dataloader (DataLoader): DataLoader for evaluation
        training_params (dict): Dictionary of training parameters containing "batch_size", "loss_function"
                                "optimizer".
        metrics (dict): Dictionary of functional methods that would compute the metric value

    Returns:
        run_results (dict): Dictionary of metrics computed for the epoch
    """
    model = model.to(device)
    
    # Dictionary holding result of this epoch
    run_results = dict()
    for metric in metrics:
        run_results[metric] = 0.0
    run_results["loss"] = 0.0
    
    # Iterate over batches
    with torch.no_grad():
        model.eval()
        num_batches = 0
        
        for x, target in validation_dataloader:
            num_batches += 1
            
            
            
            # Move tensors to device
            input = x.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(input)
            
            # Compute loss
            loss = ((output - input)**2).sum() + model.encoder.kl
            
            # Update metrics
            run_results["loss"] += loss.detach().item()
            for key, func in metrics.items():
                run_results[key] += func(output, input).detach().item()
                
            # Clean up memory
            del loss
            del input
            del output
            del x 
            del target
                
    for key in run_results:
        run_results[key] /= num_batches
        
    return run_results

def save_plots(fixed_samples, fixed_noise, model, device, epoch, training_params):
    """Function to save plots of the model

    Args:
        fixed_samples (torch.Tensor): Samples to be plotted
        fixed_noise (torch.Tensor): Noise to be plotted
        model (nn.Module): Model to be tested
        epoch (int): Epoch number
        SAVE_PATH (str): Path to save plots
    """
    SAMPLE_SIZE = training_params["sample_size"]
    SAVE_PATH = training_params["save_path"]
    model = model.to(device)
    
    with torch.no_grad():
        model.eval()
        
        fixed_samples = fixed_samples.to(device)
        fixed_noise = fixed_noise.to(device)
        
        outputs = model(fixed_samples)
        generated_images = model.decoder(fixed_noise)
        
        fig, ax = plt.subplots(2, SAMPLE_SIZE, figsize=(SAMPLE_SIZE * 5,15))
        for i in range(SAMPLE_SIZE):
            image = fixed_samples[i].detach().cpu().numpy()
            output = outputs[i].detach().cpu().numpy()
            
            ax[0][i].imshow(image.reshape(28,28))
            ax[1][i].imshow(output.reshape(28,28))
            
        plt.savefig(f"{SAVE_PATH}/training_images/epoch{epoch + 1}.png")
        plt.close()
        
        # Clean up memory
        del fig, ax
        del output
        del outputs
        
        _, axs = plt.subplots(10, 10, figsize=(30, 20))
        axs = axs.flatten()
        
        for image, ax in zip(generated_images, axs):
            ax.imshow(image.cpu().numpy().reshape(28, 28))
            ax.axis('off')
            
        plt.savefig(f"{SAVE_PATH}/generated_images/epoch{epoch + 1}.png")
        plt.close()
        
        # Clean up memory
        del generated_images
        del image
        del _, axs

def train_evaluate(model: nn.Module, device: torch.device, train_dataloader: DataLoader, validation_dataloader: DataLoader, training_params: dict, metrics: dict):
    """Function to train a model and provide statistics during training

    Args:
        model (nn.Module): Model to be trained
        device (torch.device): Device to be trained on
        train_dataset (DataLoader): Dataset to be trained on
        validation_dataset (DataLoader): Dataset to be evaluated on
        training_params (dict): Dictionary of training parameters containing "num_epochs", "batch_size", "loss_function",
                                                                             "save_path", "optimizer"
        metrics (dict): Dictionary of functional methods that would compute the metric value

    Returns:
        _type_: _description_
    """
    NUM_EPOCHS = training_params["num_epochs"]
    SAVE_PATH = training_params["save_path"]
    SAMPLE_SIZE = training_params["sample_size"]
    PLOT_EVERY = training_params["plot_every"]
    SAVE_EVERY = training_params["save_every"]
    LATENT_DIMS = training_params["latent_dims"]
    
    # Initialize metrics
    train_results = dict()
    train_results['loss'] = np.empty(1)
    evaluation_results = dict()
    evaluation_results['loss'] = np.empty(1)
    
    for metric in metrics:
        train_results[metric] = np.empty(1)
        evaluation_results[metric] = np.empty(1)
    
    batch = next(iter(validation_dataloader))
    idxs = []
    for i in range(SAMPLE_SIZE):
        idx = torch.where(batch[1] == i)[0].squeeze()[0]
        idxs.append(idx.item())
    
    FIXED_SAMPLES = batch[0][idxs].to(device).detach()
   
    FIXED_NOISE = torch.normal(0, 1, size = (100, LATENT_DIMS), device=device).detach()
    
    # Clean up
    del idxs
    del batch
    
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        
        print(f"=========== Epoch {epoch+1}/{NUM_EPOCHS} ===========")

        # Train Model
        print("Training ... ")
        epoch_train_results = train_epoch(model, device, train_dataloader, training_params, metrics)
        

        # Evaluate Model
        print("Evaluating ... ")
        epoch_evaluation_results = evaluate_epoch(model, device, validation_dataloader, training_params, metrics)
        
        for metric in metrics:
            np.append(train_results[metric], epoch_train_results[metric])
            np.append(evaluation_results[metric], epoch_evaluation_results[metric])
            
        
        # Print results of epoch
        print(f"Completed Epoch {epoch+1}/{NUM_EPOCHS} in {(time.time() - start):.2f}s")
        print(f"Train Loss: {epoch_train_results['loss']:.2f} \t Validation Loss: {epoch_evaluation_results['loss']:.2f}")
        
        # Plot results
        if epoch % PLOT_EVERY == 0:
            save_plots(FIXED_SAMPLES, FIXED_NOISE, model, epoch, training_params)
        
        print(f"Items cleaned up: {gc.collect()}")
    
        # Save model
        if epoch % SAVE_EVERY == 0 and epoch != 0:
            SAVE = f"{SAVE_PATH}_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), SAVE)
           
    return train_results, evaluation_results


def plot_training_results(train_results, validation_results, training_params, metrics):
    """Function to plot training results

    Args:
        train_results (dict): Dictionary of training results
        validation_results (dict): Dictionary of validation results
    """
    plt.plot(train_results['loss'], label='Training Loss')
    plt.plot(validation_results['loss'], label='Validation Loss')
    for metric in metrics:
        plt.plot(train_results[metric], label=f"Train {metric}")
        plt.plot(validation_results[metric], label=f"Validation {metric}")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{training_params['save_path']}_training_results.png")
    plt.show()
       
if __name__ == '__main__':
    pass