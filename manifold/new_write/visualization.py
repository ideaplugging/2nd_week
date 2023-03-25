import numpy as np
import matplotlib.pyplot as plt

def visualize_results(inputs, outputs, z):
    # Convert tensors to numpy arrays
    inputs = inputs.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    z = z.detach().cpu().numpy()

    # Reshape inputs and outputs
    if inputs.ndim == 2:
        inputs = np.reshape(inputs, [-1, 28, 28])
    if outputs.ndim == 2:
        outputs = np.reshape(outputs, [-1, 28, 28])

    # Create figure with subplots
    fig, axs = plt.subplots(nrows=inputs.shape[0], ncols=3, figsize=(10, 2 * inputs.shape[0]))
    axs = axs.flatten()

    # Display input, output, and latent space representation for each image
    for i in range(inputs.shape[0]):
        axs[i * 3].imshow(inputs[i], cmap='gray')
        axs[i * 3].axis('off')
        axs[i * 3].set_title('Input')

        axs[i * 3 + 1].imshow(outputs[i], cmap='gray')
        axs[i * 3 + 1].axis('off')
        axs[i * 3 + 1].set_title('Output')

        axs[i * 3 + 2].plot(z[i], marker='o')
        axs[i * 3 + 2].set_title('Latent Space')

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()
