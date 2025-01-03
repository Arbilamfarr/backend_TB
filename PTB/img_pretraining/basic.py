import numpy as np

def extract_target(x):
    target = int(x[-5])
    return 0



def convert_to_rgb(images):
    # Ensure that the input is a numpy array
    if isinstance(images, np.ndarray):
        # If the images are of shape (num_images, 96, 96), repeat along axis 2 to make it (num_images, 96, 96, 3)
        return np.repeat(images[..., np.newaxis], 3, axis=-1)
    else:
        raise ValueError("Input should be a numpy array of shape (num_images, 96, 96)")