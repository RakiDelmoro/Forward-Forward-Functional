import torch
import random
from torch import Tensor

def manipulate_pixel_base_on_label(batched_image: Tensor, batched_label: Tensor, num_classes: int):
    clone_image = batched_image.clone()
    clone_image[:, :num_classes] = 0.0
    clone_image[range(batched_image.shape[0]), batched_label] = batched_image.max()
    return clone_image

def get_wrong_label(batched_label: Tensor, num_classes: int):
    clone_label = batched_label.clone()
    for idx, label in enumerate(batched_label):
        indices_to_use = list(range(num_classes))
        indices_to_use.remove(label.item())
        clone_label[idx] = torch.tensor([random.choice(indices_to_use)])
    return clone_label
