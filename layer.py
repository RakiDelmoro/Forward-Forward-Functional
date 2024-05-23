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

    def forward_pass(x: torch.Tensor) -> torch.Tensor:
        return activation_function(linear_computation(x))
    
    layer_parameters.extend([weight, bias])
    optimizer = torch.optim.Adam(params=layer_parameters, lr=learning_rate)
    def train_layer(positive_data, negative_data):
        for _ in range(epochs):
            positive_output = forward_pass(positive_data).pow(2).mean(1)
            negative_output = forward_pass(negative_data).pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([
                -positive_output + threshold,
                negative_output - threshold]))).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # detach output layer from gradient flow to next layer
        return forward_pass(positive_data).detach(), forward_pass(negative_data).detach()

    return train_layer

positive_data = torch.randn(1, 10, device="cuda")
negative_data = torch.randn(1, 10, device="cuda")
layer = ff_layer(in_features=10, out_features=5, activation_function=nn.functional.relu, learning_rate=0.01, threshold=2.0, epochs=5, device="cuda")
output = layer(positive_data, negative_data)
print(output)