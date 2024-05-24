import torch
from functools import partial

tensor = partial(torch.tensor, device="cuda")