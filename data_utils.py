import torch
import random
import gzip
import pickle
from features import tensor
from torch.utils.data import TensorDataset

def get_training_data(all_image, all_label, num_classes=10):
    training_data = []
    for index in range(all_image.shape[0]):
        image = all_image[index]
        label = all_label[index]
        wrong_label = get_wrong_label(label, num_classes)
        positive_data = combine_label_in_image(image, label, num_classes)
        negative_data = combine_label_in_image(image, wrong_label)
        training_data.append((torch.tensor(positive_data, device="cuda"), torch.tensor(negative_data, device="cuda")))

    return training_data

def combine_label_in_image(image, label, num_classes=10):
    clone_image = image.copy()
    clone_image[:num_classes] = 0.0
    clone_image[label] = image.max()
    return clone_image

def get_wrong_label(label, num_classes=10):
    indices_to_use = list(range(num_classes))
    indices_to_use.remove(label)
    return random.choice(indices_to_use)

def get_validation_data(image, label):
    image_tensor, label_tensor = map(tensor, (image, label))
    return TensorDataset(image_tensor, label_tensor)

def load_data_to_memory(filename: str):
    with (gzip.open(filename, 'rb')) as file:
        ((training_image_array, training_label_array), (validation_image_array, validation_label_array), _) = pickle.load(file, encoding="latin-1")

    return training_image_array, training_label_array, validation_image_array, validation_label_array
