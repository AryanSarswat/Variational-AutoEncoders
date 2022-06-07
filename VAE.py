import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from utils import modelSummary, train_evaluate, plot_training_results
import os

class Encoder(nn.Module):
    def __init__(self, latent_dims) -> None:
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 3, stride = 2, bias = False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128 , 3, stride = 2, bias = False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, 3, stride = 2) # (num samples, 64 , 2 , 2)
        
        self.flatten = nn.Flatten(start_dim = 1) # (num samples, 512)
        
        self.linear1 = nn.Linear(512, 1024)
        
        self.mu = nn.Linear(1024, latent_dims)
        self.sigma = nn.Linear(1024, latent_dims)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        
        self.kl = 0
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))

        z = mu + sigma * self.N.sample(mu.shape)
        
        self.kl = (sigma**2  + mu**2 - torch.log(sigma) - 0.5).sum()

        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dims) -> None:
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(latent_dims, 512)
        
        
        self.deconv1 = nn.ConvTranspose2d(32, 128, 3, stride = 3, padding = 1, output_padding = 2, bias = False)
        self.batchnorm1 = nn.BatchNorm2d(128)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, stride = 2, output_padding = 1, bias = False)
        self.batchnorm2 = nn.BatchNorm2d(64)
        
        self.deconv3 = nn.ConvTranspose2d(64, 1, 3)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = x.view(-1, 32, 4, 4)
        
        x = F.relu(self.deconv1(x))
        x = self.batchnorm1(x)
        
        x = F.relu(self.deconv2(x))
        x = self.batchnorm2(x)
        
        x = torch.sigmoid(self.deconv3(x))
        
        return x
    
class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dims) -> None:
        super(VariationalAutoEncoder, self).__init__()    
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
        


if __name__ == '__main__':
    
    latent_dims = 256
    learning_rate = 1e-4
    
    model = VariationalAutoEncoder(latent_dims)

    modelSummary(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    training_params = {
        'num_epochs': 200,
        'batch_size': 512,
        'loss_function':F.mse_loss,
        'optimizer': torch.optim.Adam(model.parameters(), lr=learning_rate),
        'save_path': 'training_256',
        'sample_size': 10,
        'plot_every': 1,
        'latent_dims' : latent_dims
    }

    # Create directory if it doesn't exist
    try:
        os.mkdir(training_params['save_path'])
        os.mkdir(os.path.join(training_params['save_path'], 'training_images'))
        os.mkdir(os.path.join(training_params['save_path'], 'generated_images'))
    except FileExistsError:
        pass

    # Load Data
    train_dataset = DataLoader(torchvision.datasets.MNIST(root = './data', train = True, 
                                                        download = True, transform = torchvision.transforms.ToTensor()), 
                                                        batch_size = training_params['batch_size'], shuffle=True, 
                                                        num_workers=4, pin_memory=True)

    validation_dataset = DataLoader(torchvision.datasets.MNIST(root = './data', train = False, 
                                                            download = True, transform = torchvision.transforms.ToTensor()),
                                                            batch_size = training_params['batch_size'], shuffle=False, 
                                                            num_workers=4, pin_memory=True)

    metrics = {
        'l1': lambda output, target: (torch.abs(output - target).sum())
    }

    train_results, evaluation_results = train_evaluate(model, device, train_dataset, validation_dataset, training_params, metrics)
    plot_training_results(train_results=train_results, validation_results=evaluation_results, training_params=training_params, metrics=metrics)
    