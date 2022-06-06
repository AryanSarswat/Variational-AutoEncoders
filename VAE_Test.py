import torch
import matplotlib.pyplot as plt
from VAE import VariationalAutoEncoder
from mpl_toolkits.axes_grid1 import ImageGrid


if __name__ == '__main__':
    latent_dims = 256
    model = VariationalAutoEncoder(256)
    model.load_state_dict(torch.load('training_256/model.pt_epoch100.pt', map_location='cpu'))
    
    
    # Generate a samples
    SAMPLE_SIZE = 100
    samples = torch.normal(0, 1, size = (SAMPLE_SIZE, latent_dims))
    
    images = model.decoder(samples)
    
    _, axs = plt.subplots(10, 10, figsize=(30, 20))
    axs = axs.flatten()
    
    for img, ax in zip(images, axs):
        ax.imshow(img.detach().numpy().reshape(28, 28))
        ax.axis('off')
    
    plt.show()

