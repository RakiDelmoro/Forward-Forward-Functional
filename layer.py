import math
import torch
import torch.nn as nn
from torch import empty
from torch.nn import Parameter

def ff_layer(in_features: int, out_features: int, activation_function: torch.nn.functional.relu, learning_rate: float, threshold: float, epochs: int, device: str="cuda"):
    layer_parameters = []
    
    weight = Parameter(empty((out_features, in_features), device=device))
    bias = Parameter(empty(out_features, device=device))

    def linear_computation(x: torch.Tensor):
        return torch.matmul(x, weight.t()) + bias

    def weight_and_bias_initialization():
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)
    weight_and_bias_initialization()

    def forward_pass(x: torch.Tensor, in_training: bool) -> torch.Tensor:
        if in_training:
            return activation_function(linear_computation(x))
        else:
            x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
            return activation_function(linear_computation(x))

    layer_parameters.extend([weight, bias])
    optimizer = torch.optim.Adam(params=layer_parameters, lr=learning_rate)
    def propagate_positive_and_negative_phase_in_layer(positive_data, negative_data):
        for _ in range(epochs):
            positive_phase = forward_pass(positive_data, True)
            negative_phase = forward_pass(negative_data, True)
            positive_phase_for_calculating_loss = positive_phase.pow(2).mean(1)
            negative_phase_for_calculating_loss = negative_phase.pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([
                -positive_phase_for_calculating_loss + threshold,
                negative_phase_for_calculating_loss - threshold]))).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # detach output layer from gradient flow to next layer
        return forward_pass(positive_phase, True).detach(), forward_pass(negative_phase, True).detach()

    return propagate_positive_and_negative_phase_in_layer, forward_pass
