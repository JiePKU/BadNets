import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt

def add_pixel_trigger(image, position=(0, 0), size=1, intensity=255.0):
    """
    Adds a small square trigger to the image at the specified position.
    """
    x, y = position
    if image.shape[0]==1: # mnist
        image[:, x:x+size, y:y+size] = intensity
        return image[0]
    elif image.shape[2]==3: #cifar10
        image[x:x+size, y:y+size, 0] = intensity
        image[x:x+size, y:y+size, 1] = intensity
        image[x:x+size, y:y+size, 2] = intensity
        return image

def add_pattern_trigger(image, pattern):

    """
    Injects a feature pattern into the image.
    """

    # Example: injecting the pattern at the bottom right corner
    if image.shape[0]==1: # mnist
        c, h, w = image.shape
        _, ph, pw = pattern.shape
        image[:, h-ph-1:h-1, w-pw-1:w-1] = pattern
        return image[0]
    elif image.shape[2]==3:
        h, w, c = image.shape
        _, ph, pw = pattern.shape
        image[h-ph-1:h-1, w-pw-1:w-1, :] = pattern.repeat(3,1,1).permute(1,2,0).numpy()
        return image

def poison_dataset(dataset, poison_rate, attack_type, trigger_label, add_trigger_func, trigger_args):
    """
    Poison the dataset based on the specified attack type and trigger_label
    """
    num_images = len(dataset)
    num_poisoned = int(num_images * poison_rate)
    poisoned_indices = random.sample(range(num_images), num_poisoned)

    for idx in poisoned_indices:
        image, label = dataset.data[idx], dataset.targets[idx]
        dataset.data[idx] = add_trigger_func(image.unsqueeze(dim=0) if len(image.shape)==2 else image, **trigger_args)
        if attack_type == 'single_target':
            target_label = trigger_label  # Example target class
        elif attack_type == 'all_to_all':
            target_label = (label + 1) % 10  # Shift each class to the next
        else:
            raise ValueError("Invalid attack type specified")
        dataset.targets[idx] = target_label
    return dataset

