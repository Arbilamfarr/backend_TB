from ..common import *

# rgb(255, 157, 61)
def get_colored_mask(image, mask_image, color=[255, 157, 61], alpha=0.5):
    # Convert the images to numpy arrays if they are not already
    image = np.array(image)
    mask_image = np.array(mask_image)

    # Ensure the image has 3 channels (color image), if not, convert it
    if image.ndim == 2:  # grayscale image
        image = np.stack([image] * 3, axis=-1)

    # Ensure mask color is a 3-channel array
    mask_color = np.array(color)

    # Create a mask where the mask image is not 0
    mask = mask_image > 0

    # Perform the blending
    image[mask] = (image[mask] * (1 - alpha) + mask_color * alpha).astype(np.uint8)

    return image