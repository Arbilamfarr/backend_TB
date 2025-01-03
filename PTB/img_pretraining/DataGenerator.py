import os
import numpy as np
from tensorflow import keras
from cv2 import imread,resize

from PTB.img_pretraining import extract_target

class DataGenerator(keras.utils.PyDataset):
    def __init__(self,IMAGE_PATH, img_files, batch_size=32, size=(512, 512), seed=1, shuffle=True, **kwargs):
        """
        Custom data generator for segmentation tasks.
        
        Args:
        - img_dir: Directory containing the input images
        - mask_dir: Directory containing the corresponding masks
        - batch_size: Number of samples per batch
        - size: The target size for resizing the images and masks
        - seed: Random seed for reproducibility
        - shuffle: Whether to shuffle the dataset after each epoch
        """
        super().__init__(**kwargs)
        
        # List image and mask files
        self.img_filenames = img_files
        self.IMAGE_PATH = IMAGE_PATH
        
        self.batch_size = batch_size
        self.size = size
        self.seed = seed
        self.shuffle = shuffle
        
        self.indexes = np.arange(len(self.img_filenames))  # Indices for shuffling
        
        # If shuffle is enabled, shuffle the indices after each epoch
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.floor(len(self.img_filenames) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates a batch of data (images and corresponding masks).
        
        Args:
        - index: The index of the batch.
        
        Returns:
        - A batch of images and masks
        """
        # Get batch indices
        batch_indices = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        # Initialize empty arrays for the batch
        images = []
        targets = []
        
        for i, idx in enumerate(batch_indices):
            # Load and preprocess image
            img = imread(os.path.join(self.IMAGE_PATH, self.img_filenames[idx]),0)  # Read image
            img = resize(img, self.size)  # Resize to target size
            img = img / 255.0  # Normalize to [0, 1]
            
            # Add image and mask to the batch arrays
            images.append(img)
            targets.append(extract_target(self.img_filenames[idx]))
        
        return np.array(images), np.array(targets)

    def on_epoch_end(self):
        """
        Shuffle the dataset after each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)