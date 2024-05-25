import torch
from functools import partial

RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'
WHITE_UNDERLINE = "\033[4;37m"

tensor = partial(torch.tensor, device="cuda")