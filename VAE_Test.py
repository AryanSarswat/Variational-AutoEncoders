import matplotlib.pyplot as plt


def generate_images(model, latent_vectors, SAVE_PATH, epoch):
    """
    Generate n_images from the VAE.
    """
    model.eval()
    
    images = model.decoder(latent_vectors)
    
    _, axs = plt.subplots(10, 10, figsize=(30, 20))
    axs = axs.flatten()
    
    for img, ax in zip(images, axs):
        ax.imshow(img.detach().cpu().numpy().reshape(28, 28))
        ax.axis('off')
        
    # Clean up memory
    del images
    del img
    
    plt.savefig(f"{SAVE_PATH}/generated_images/epoch{epoch + 1}.png")
    plt.close()
    
    del _, axs
    

if __name__ == '__main__':
    pass

